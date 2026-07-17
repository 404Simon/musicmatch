import json
import os
import subprocess
from glob import glob

import click

from musicmatch.config import (
    AUDIO_EXTENSIONS,
    DB_PATH,
    MAX_DURATION_MINUTES,
    MPD_MUSIC_DIR,
    SAMPLE_RATE,
    TOP_K,
)
from musicmatch.debug import debug, rss, set_verbose as set_debug_verbose


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Enable debug output")
@click.option("--no-ignore", is_flag=True, help="Disable .matchignore filtering")
def cli(verbose: bool, no_ignore: bool):
    if verbose:
        set_debug_verbose(True)
        debug("Verbose mode enabled", tag="cli")
    if no_ignore:
        from musicmatch import matchignore

        matchignore.set_enabled(False)


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False),
    default=os.path.expanduser("~/Music"),
    required=False,
)
@click.option(
    "--full",
    is_flag=True,
    help="Delete all data and re-index from scratch",
)
def index(directory: str, full: bool):
    from musicmatch import hash as hashmod
    from musicmatch import matchignore
    from musicmatch.audio import load_audio, chunk_audio
    from musicmatch.db import init_db, insert_chunk
    from musicmatch.model import get_audio_embeddings
    import turso
    from tqdm import tqdm

    debug(f"RSS at start: {rss()}", tag="index")

    files: list[str] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

    if not files:
        click.echo("No audio files found.")
        raise SystemExit(0)

    debug(f"Found {len(files)} audio files. RSS: {rss()}", tag="index")
    init_db()

    with turso.connect(DB_PATH) as con:
        if full:
            click.echo("Full re-index: clearing all data...")
            con.cursor().execute("DELETE FROM chunks")
            con.cursor().execute("DELETE FROM files")
            con.cursor().execute("DELETE FROM chromaprints")
            con.commit()

        abs_dir = os.path.abspath(directory)
        scanned = set(files)

        known = {
            r[0] for r in con.cursor().execute("SELECT path FROM files").fetchall()
        }

        stale = [p for p in known if p.startswith(abs_dir + "/") and p not in scanned]
        if stale:
            click.echo(f"Pruning {len(stale)} stale file(s)...")
            for path in stale:
                con.cursor().execute("DELETE FROM files WHERE path = ?", [path])
            con.cursor().execute(
                "DELETE FROM chromaprints WHERE id NOT IN (SELECT chromaprint_id FROM files)"
            )
            con.cursor().execute(
                "DELETE FROM chunks WHERE chromaprint_id NOT IN (SELECT id FROM chromaprints)"
            )
            con.commit()

        new_files = sorted(
            f
            for f in files
            if f not in known
            and not matchignore.is_ignored(os.path.relpath(f, directory))
        )
        if not new_files:
            click.echo("All files already indexed.")
            return

        debug(f"New files to index: {len(new_files)}. RSS: {rss()}", tag="index")

        max_sec = MAX_DURATION_MINUTES * 60
        indexed = 0

        for filepath in tqdm(new_files, desc="Indexing"):
            fp_hash = hashmod.compute_fingerprint(filepath)

            if fp_hash:
                existing = (
                    con.cursor()
                    .execute(
                        "SELECT c.id FROM chromaprints c JOIN chunks ch ON ch.chromaprint_id = c.id WHERE c.hash = ? LIMIT 1",
                        [fp_hash],
                    )
                    .fetchone()
                )
                if existing:
                    con.cursor().execute(
                        "INSERT INTO files (chromaprint_id, path) VALUES (?, ?)",
                        (existing[0], filepath),
                    )
                    con.commit()
                    indexed += 1
                    continue

            try:
                audio = load_audio(filepath)
            except Exception as e:
                click.echo(f"  Skipping {filepath}: {e}", err=True)
                continue

            duration = len(audio) / SAMPLE_RATE
            if duration > max_sec:
                debug(
                    f"Skipping {filepath}: {duration / 60:.1f}min > {MAX_DURATION_MINUTES}min limit",
                    tag="index",
                )
                continue

            if fp_hash:
                chromaprint_id = hashmod.find_or_create(con, fp_hash)
            else:
                con.cursor().execute("INSERT INTO chromaprints (hash) VALUES (NULL)")
                chromaprint_id = con.cursor().lastrowid

            con.cursor().execute(
                "INSERT INTO files (chromaprint_id, path, duration) VALUES (?, ?, ?)",
                (chromaprint_id, filepath, duration),
            )

            chunks = chunk_audio(audio)
            chunk_arrays = [c for _, _, c in chunks]
            debug(f"{filepath}: {len(chunks)} chunks. RSS: {rss()}", tag="index")
            embeddings = get_audio_embeddings(chunk_arrays)
            for (chunk_idx, _, _), emb in zip(chunks, embeddings):
                insert_chunk(con, chromaprint_id, chunk_idx, emb)

            con.commit()
            indexed += 1

    debug(f"Final RSS: {rss()}", tag="index")
    click.echo(f"Done. Indexed {indexed} file(s).")


@cli.command()
def list_files():
    from musicmatch.db import init_db
    import turso

    init_db()
    with turso.connect(DB_PATH) as con:
        rows = (
            con.cursor()
            .execute(
                """SELECT f.path, COUNT(c.id) AS chunks
                   FROM files f
                   LEFT JOIN chunks c ON c.chromaprint_id = f.chromaprint_id
                   GROUP BY f.id
                   ORDER BY f.path"""
            )
            .fetchall()
        )
    if not rows:
        click.echo("No files indexed yet.")
        return
    for r in rows:
        click.echo(f"{r[0]}  ({r[1]} chunks)")


@cli.command()
@click.argument("query")
def search(query: str):
    from musicmatch import matchignore
    from musicmatch.db import init_db, is_empty, search as db_search
    from musicmatch.model import get_text_embedding
    import turso

    init_db()
    embedding = get_text_embedding(query)

    with turso.connect(DB_PATH) as con:
        if is_empty(con):
            click.echo("No files indexed yet. Run `musicmatch index` first.")
            return
        rows = db_search(con, embedding, TOP_K)

    if not rows:
        click.echo("No results found.")
        return

    mpd_root = os.path.expanduser(MPD_MUSIC_DIR)
    rows = [
        r
        for r in rows
        if not matchignore.is_ignored(os.path.relpath(r["filepath"], mpd_root))
    ]
    if not rows:
        click.echo("No results found.")
        return

    for r in rows:
        click.echo(
            f"{r['filepath']}  @ {r['start_time']:.1f}s  (distance: {r['distance']:.4f})"
        )


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
def similar(filepath: str):
    import numpy as np

    from musicmatch import hash as hashmod
    from musicmatch import matchignore
    from musicmatch.audio import load_and_chunk
    from musicmatch.db import (
        init_db,
        is_empty,
        search as db_search,
        group_and_score,
        get_file_embedding,
    )
    from musicmatch.model import get_audio_embeddings
    import turso

    init_db()

    with turso.connect(DB_PATH) as con:
        if is_empty(con):
            click.echo("No files indexed yet. Run `musicmatch index` first.")
            return

        avg_embedding = get_file_embedding(con, filepath)

    if avg_embedding is None:
        chunks = load_and_chunk(filepath)
        embeddings = get_audio_embeddings([c for _, _, c in chunks])
        avg_embedding = np.mean(embeddings, axis=0)

    with turso.connect(DB_PATH) as con:
        rows = db_search(con, avg_embedding, top_k=100)

        query_cid = (
            con.cursor()
            .execute("SELECT chromaprint_id FROM files WHERE path = ?", [filepath])
            .fetchone()
        )
        if query_cid:
            rows = [r for r in rows if r["chromaprint_id"] != query_cid[0]]

        if not rows:
            click.echo("No similar files found.")
            return

        scored = group_and_score(rows, TOP_K)
        scored = hashmod.dedup_by_hash(con, scored)

    mpd_root = os.path.expanduser(MPD_MUSIC_DIR)
    scored = [
        (fp, d)
        for fp, d in scored
        if not matchignore.is_ignored(os.path.relpath(fp, mpd_root))
    ]

    if not scored:
        click.echo("No similar files found.")
        return

    for fp, d in scored:
        click.echo(f"{fp}  (distance: {d:.4f})")


@cli.command()
@click.argument("amount", type=int, default=5, required=False)
@click.option(
    "--all",
    "-a",
    "all_flag",
    is_flag=True,
    help="Find similar songs for every entry in the queue",
)
def rmpc(amount: int, all_flag: bool):
    """Find similar songs to the currently playing song and add them to the rmpc queue."""
    import numpy as np

    from musicmatch import hash as hashmod
    from musicmatch import matchignore
    from musicmatch.audio import load_and_chunk
    from musicmatch.db import (
        init_db,
        get_file_embedding,
        search as db_search,
        group_and_score,
    )
    from musicmatch.model import get_audio_embeddings
    import turso

    result = subprocess.run(["rmpc", "queue"], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo("Failed to get current queue from rmpc.", err=True)
        raise SystemExit(1)

    queue = json.loads(result.stdout)
    if not queue:
        click.echo("Queue is empty.", err=True)
        raise SystemExit(1)

    mpd_root = os.path.expanduser(MPD_MUSIC_DIR)
    queue_files = {entry["file"] for entry in queue}
    entries = queue if all_flag else [queue[0]]

    all_scored = []
    init_db()

    with turso.connect(DB_PATH) as con:
        for entry in entries:
            rel_path = entry["file"]
            abs_path = os.path.join(mpd_root, rel_path)

            avg_embedding = get_file_embedding(con, abs_path)
            if avg_embedding is None:
                if not os.path.exists(abs_path):
                    click.echo(f"File not in index and not found: {abs_path}", err=True)
                    continue
                chunks = load_and_chunk(abs_path)
                embeddings = get_audio_embeddings([c for _, _, c in chunks])
                avg_embedding = np.mean(embeddings, axis=0)

            rows = db_search(con, avg_embedding, top_k=100)

            query_cid = (
                con.cursor()
                .execute("SELECT chromaprint_id FROM files WHERE path = ?", [abs_path])
                .fetchone()
            )
            if query_cid:
                rows = [r for r in rows if r["chromaprint_id"] != query_cid[0]]

            if not rows:
                click.echo(f"No similar files found for {rel_path}.")
                continue

            scored = group_and_score(rows, amount)
            scored = [
                (fp, d)
                for fp, d in scored
                if not matchignore.is_ignored(os.path.relpath(fp, mpd_root))
                and os.path.relpath(fp, mpd_root) not in queue_files
            ]

            if scored:
                all_scored.extend(scored)

        if not all_scored:
            click.echo("No similar files found.")
            return

        deduped = hashmod.dedup_by_hash(con, all_scored)

    rel_paths = [os.path.relpath(fp, mpd_root) for fp, _ in deduped]
    subprocess.run(["rmpc", "add", *rel_paths])
    added = len(rel_paths)
    click.echo(f"Added {added} song(s) to queue.")


@cli.command()
@click.option("--output", "-o", default="musicmatch_3d.html", help="Output HTML path")
@click.option("--limit", "-n", type=int, default=None, help="Randomly sample N songs")
def visualize(output: str, limit: int | None):
    from musicmatch.db import init_db, is_empty
    from musicmatch.viz import get_all_file_embeddings, pca, scatter_3d
    import turso

    init_db()
    with turso.connect(DB_PATH) as con:
        if is_empty(con):
            click.echo("No files indexed yet. Run `musicmatch index` first.")
            return
        filepaths, embeddings = get_all_file_embeddings(con)

    if len(filepaths) < 3:
        click.echo("Need at least 3 songs for 3D visualization.")
        return

    click.echo(f"Reducing {len(filepaths)} songs from 512D to 3D via PCA...")
    coords = pca(embeddings, n_components=3)

    click.echo(f"Rendering interactive 3D plot to {output} ...")
    scatter_3d(coords, filepaths, output, limit=limit)
    click.echo(f"Done. Open {output} in your browser.")
