import numpy as np

from musicmatch.config import DB_PATH, EMBEDDING_DIM


def init_db():
    import turso

    with turso.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute("DROP TABLE IF EXISTS track_chunks")
        cur.execute("DROP TABLE IF EXISTS track_hashes")
        cur.execute("""CREATE TABLE IF NOT EXISTS chromaprints (
            id    INTEGER PRIMARY KEY,
            hash  TEXT UNIQUE
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS files (
            id             INTEGER PRIMARY KEY,
            chromaprint_id INTEGER NOT NULL REFERENCES chromaprints(id) ON DELETE RESTRICT,
            path           TEXT NOT NULL UNIQUE,
            duration       REAL
        )""")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
            id             INTEGER PRIMARY KEY,
            chromaprint_id INTEGER NOT NULL REFERENCES chromaprints(id) ON DELETE CASCADE,
            chunk_index    INTEGER NOT NULL,
            embedding      F32_BLOB({EMBEDDING_DIM}),
            UNIQUE(chromaprint_id, chunk_index)
        )""")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_chromaprint_id ON files(chromaprint_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_chromaprint_id ON chunks(chromaprint_id)"
        )
        con.commit()


def insert_chunk(con, chromaprint_id: int, chunk_index: int, embedding):
    con.cursor().execute(
        "INSERT INTO chunks (chromaprint_id, chunk_index, embedding) VALUES (?, ?, vector32(?))",
        (chromaprint_id, chunk_index, str(embedding.tolist())),
    )


def is_empty(con) -> bool:
    row = con.cursor().execute("SELECT COUNT(*) FROM files").fetchone()
    return row[0] == 0


def search(con, query_embedding, top_k: int = 5) -> list[dict]:
    embedding_json = str(query_embedding.tolist())
    cur = con.cursor()

    sql = """SELECT f.path, c.chunk_index * 10.0 AS start_time,
                    c.chromaprint_id,
                    vector_distance_cos(c.embedding, vector32(?)) AS distance
             FROM chunks c
             JOIN files f ON f.chromaprint_id = c.chromaprint_id
                      AND f.id = (SELECT MIN(f2.id) FROM files f2 WHERE f2.chromaprint_id = c.chromaprint_id)
             ORDER BY distance ASC LIMIT ?"""

    rows = cur.execute(sql, [embedding_json, top_k]).fetchall()
    return [
        {"filepath": r[0], "start_time": r[1], "chromaprint_id": r[2], "distance": r[3]}
        for r in rows
    ]


def get_file_embedding(con, filepath: str):
    row = (
        con.cursor()
        .execute("SELECT chromaprint_id FROM files WHERE path = ?", [filepath])
        .fetchone()
    )
    if not row:
        return None
    chromaprint_id = row[0]

    rows = (
        con.cursor()
        .execute(
            "SELECT embedding FROM chunks WHERE chromaprint_id = ? ORDER BY chunk_index",
            [chromaprint_id],
        )
        .fetchall()
    )
    if not rows:
        return None

    embeddings = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    return np.mean(embeddings, axis=0)


def group_and_score(results: list[dict], top_k: int = 5) -> list[tuple[str, float]]:
    groups: dict[str, list[float]] = {}
    for r in results:
        groups.setdefault(r["filepath"], []).append(r["distance"])
    return sorted(
        [(fp, sum(dists) / len(dists)) for fp, dists in groups.items()],
        key=lambda x: x[1],
    )[:top_k]
