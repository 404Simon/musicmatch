import numpy as np

from musicmatch.config import DB_PATH


def init_db():
    import turso

    with turso.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS track_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_time REAL NOT NULL,
                embedding F32_BLOB(512)
            )"""
        )
        con.commit()


def insert_chunk(
    con,
    filepath: str,
    chunk_index: int,
    start_time: float,
    embedding,
):
    embedding_json = str(embedding.tolist())
    cur = con.cursor()
    cur.execute(
        "INSERT INTO track_chunks (filepath, chunk_index, start_time, embedding) VALUES (?, ?, ?, vector32(?))",
        (filepath, chunk_index, start_time, embedding_json),
    )


def get_file_embedding(con, filepath: str) -> np.ndarray | None:
    rows = (
        con.cursor()
        .execute(
            "SELECT embedding FROM track_chunks WHERE filepath = ? ORDER BY chunk_index",
            [filepath],
        )
        .fetchall()
    )
    if not rows:
        return None
    embeddings = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    return np.mean(embeddings, axis=0)


def search(
    con, query_embedding, top_k: int = 5, exclude_filepath: str | None = None
) -> list[dict]:
    embedding_json = str(query_embedding.tolist())
    cur = con.cursor()
    sql = """SELECT filepath, start_time, vector_distance_cos(embedding, vector32(?)) AS distance
             FROM track_chunks"""
    params: list = [embedding_json]
    if exclude_filepath:
        sql += " WHERE filepath != ?"
        params.append(exclude_filepath)
    sql += " ORDER BY distance ASC LIMIT ?"
    params.append(top_k)
    rows = cur.execute(sql, params).fetchall()
    return [{"filepath": r[0], "start_time": r[1], "distance": r[2]} for r in rows]


def is_empty(con) -> bool:
    row = con.cursor().execute("SELECT COUNT(*) FROM track_chunks").fetchone()
    return row[0] == 0


def group_and_score(results: list[dict], top_k: int = 5) -> list[tuple[str, float]]:
    groups: dict[str, list[float]] = {}
    for r in results:
        groups.setdefault(r["filepath"], []).append(r["distance"])
    return sorted(
        [(fp, sum(dists) / len(dists)) for fp, dists in groups.items()],
        key=lambda x: x[1],
    )[:top_k]
