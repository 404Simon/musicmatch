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
def index(directory: str):
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
        missing_hash = hashmod.get_filepaths_missing_hashes(con)
        if missing_hash:
            click.echo(f"Backfilling hashes for {len(missing_hash)} file(s)...")
            for fp in tqdm(missing_hash, desc="Backfilling hashes"):
                fp_hash = hashmod.compute_fingerprint(fp)
                if fp_hash:
                    hashmod.store_hash(con, fp, fp_hash)
            con.commit()

        existing = {
            r[0]
            for r in con.cursor()
            .execute("SELECT DISTINCT filepath FROM track_chunks")
            .fetchall()
        }
        new_files = sorted(f for f in files if f not in existing)
        skipped = len(files) - len(new_files)
        if skipped:
            click.echo(f"Skipping {skipped} already-indexed files.")
        new_files = [
            f
            for f in new_files
            if not matchignore.is_ignored(os.path.relpath(f, directory))
        ]
        if not new_files:
            click.echo("All files already indexed.")
            return

        debug(f"New files to index: {len(new_files)}. RSS: {rss()}", tag="index")

        max_sec = MAX_DURATION_MINUTES * 60
        indexed = 0
        for filepath in tqdm(new_files, desc="Indexing"):
            fp_hash = hashmod.compute_fingerprint(filepath)
            if fp_hash:
                dup = hashmod.get_filepath_by_hash(con, fp_hash)
                if dup and dup != filepath:
                    click.echo(f"  Skipping {filepath}: duplicate of {dup}", err=True)
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
            chunks = chunk_audio(audio)
            chunk_arrays = [c for _, _, c in chunks]
            debug(f"{filepath}: {len(chunks)} chunks. RSS: {rss()}", tag="index")
            embeddings = get_audio_embeddings(chunk_arrays)
            for (chunk_idx, start_time, _), emb in zip(chunks, embeddings):
                insert_chunk(con, filepath, chunk_idx, start_time, emb)
            if fp_hash:
                hashmod.store_hash(con, filepath, fp_hash)
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
                "SELECT filepath, COUNT(*) as chunks FROM track_chunks GROUP BY filepath ORDER BY filepath"
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
    from musicmatch.db import init_db, is_empty, search as db_search, group_and_score
    from musicmatch.model import get_audio_embeddings
    import turso

    init_db()
    chunks = load_and_chunk(filepath)
    embeddings = get_audio_embeddings([c for _, _, c in chunks])
    avg_embedding = np.mean(embeddings, axis=0)

    with turso.connect(DB_PATH) as con:
        if is_empty(con):
            click.echo("No files indexed yet. Run `musicmatch index` first.")
            return
        rows = db_search(con, avg_embedding, top_k=100, exclude_filepath=filepath)

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

            rows = db_search(con, avg_embedding, top_k=100, exclude_filepath=abs_path)
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
