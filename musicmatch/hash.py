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


def find_or_create(con, content_hash: str) -> int:
    cur = con.cursor()
    row = cur.execute(
        "SELECT id FROM chromaprints WHERE hash = ?", [content_hash]
    ).fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO chromaprints (hash) VALUES (?)", [content_hash])
    return cur.lastrowid


def dedup_by_hash(con, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not scored:
        return []

    seen: set[int] = set()
    kept: list[tuple[str, float]] = []
    cur = con.cursor()

    for fp, dist in scored:
        row = cur.execute(
            "SELECT chromaprint_id FROM files WHERE path = ?", [fp]
        ).fetchone()
        cid = row[0] if row else None
        if cid is None or cid not in seen:
            if cid is not None:
                seen.add(cid)
            kept.append((fp, dist))

    return kept
