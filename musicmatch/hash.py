import acoustid

from musicmatch.debug import debug


def compute_fingerprint(filepath: str, maxlength: int = 120) -> str | None:
    try:
        _, fp = acoustid.fingerprint_file(filepath, maxlength)
        if not fp:
            debug(f"No fingerprint generated for {filepath}", tag="hash")
            return None
        return fp
    except Exception as e:
        debug(f"Failed to fingerprint {filepath}: {e}", tag="hash")
        return None


def init_hash_table(con):
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS track_hashes (
            filepath TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL
        )"""
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_th_hash ON track_hashes(content_hash)")
    con.commit()


def store_hash(con, filepath: str, content_hash: str) -> None:
    con.cursor().execute(
        "INSERT OR REPLACE INTO track_hashes (filepath, content_hash) VALUES (?, ?)",
        (filepath, content_hash),
    )


def get_filepath_by_hash(con, content_hash: str) -> str | None:
    row = (
        con.cursor()
        .execute(
            "SELECT filepath FROM track_hashes WHERE content_hash = ? LIMIT 1",
            [content_hash],
        )
        .fetchone()
    )
    return row[0] if row else None


def get_filepaths_missing_hashes(con) -> list[str]:
    rows = (
        con.cursor()
        .execute(
            """SELECT DISTINCT t.filepath
               FROM track_chunks t
               LEFT JOIN track_hashes h ON t.filepath = h.filepath
               WHERE h.filepath IS NULL"""
        )
        .fetchall()
    )
    return [r[0] for r in rows]


def get_hashes_map(con, filepaths: list[str]) -> dict[str, str | None]:
    if not filepaths:
        return {}
    placeholders = ",".join("?" * len(filepaths))
    rows = (
        con.cursor()
        .execute(
            f"SELECT filepath, content_hash FROM track_hashes WHERE filepath IN ({placeholders})",
            filepaths,
        )
        .fetchall()
    )
    result: dict[str, str | None] = {fp: None for fp in filepaths}
    for fp, h in rows:
        result[fp] = h
    return result


def dedup_by_hash(con, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not scored:
        return []

    hashes = get_hashes_map(con, [fp for fp, _ in scored])

    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    kept: list[tuple[str, float]] = []

    for fp, dist in scored:
        h = hashes.get(fp)
        if h is None:
            if fp not in seen_paths:
                seen_paths.add(fp)
                kept.append((fp, dist))
        elif h not in seen_hashes:
            seen_hashes.add(h)
            seen_paths.add(fp)
            kept.append((fp, dist))

    return kept
