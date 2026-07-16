import json
import os
import subprocess
from glob import glob

import click
import numpy as np
import turso
from tqdm import tqdm

from musicmatch.audio import load_audio, chunk_audio, load_and_chunk
from musicmatch.config import AUDIO_EXTENSIONS, DB_PATH, MAX_DURATION_MINUTES, MPD_MUSIC_DIR, SAMPLE_RATE, TOP_K
from musicmatch.db import get_file_embedding, group_and_score, init_db, insert_chunk, is_empty, search as db_search
from musicmatch.debug import debug, rss, set_verbose as set_debug_verbose
from musicmatch.model import get_audio_embeddings, get_text_embedding


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug output")
def cli(verbose: bool):
    if verbose:
        set_debug_verbose(True)
        debug("Verbose mode enabled", tag="cli")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False), default=os.path.expanduser("~/Music"), required=False)
def index(directory: str):
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
        if not new_files:
            click.echo("All files already indexed.")
            return

        debug(f"New files to index: {len(new_files)}. RSS: {rss()}", tag="index")

        max_sec = MAX_DURATION_MINUTES * 60
        indexed = 0
        for filepath in tqdm(new_files, desc="Indexing"):
            try:
                audio = load_audio(filepath)
            except Exception as e:
                click.echo(f"  Skipping {filepath}: {e}", err=True)
                continue
            duration = len(audio) / SAMPLE_RATE
            if duration > max_sec:
                debug(f"Skipping {filepath}: {duration/60:.1f}min > {MAX_DURATION_MINUTES}min limit", tag="index")
                continue
            chunks = chunk_audio(audio)
            chunk_arrays = [c for _, _, c in chunks]
            debug(f"{filepath}: {len(chunks)} chunks. RSS: {rss()}", tag="index")
            embeddings = get_audio_embeddings(chunk_arrays)
            for (chunk_idx, start_time, _), emb in zip(chunks, embeddings):
                insert_chunk(con, filepath, chunk_idx, start_time, emb)
            con.commit()
            indexed += 1

    debug(f"Final RSS: {rss()}", tag="index")
    click.echo(f"Done. Indexed {indexed} file(s).")


@cli.command()
def list_files():
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

    for r in rows:
        click.echo(f"{r['filepath']}  @ {r['start_time']:.1f}s  (distance: {r['distance']:.4f})")


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
def similar(filepath: str):
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

    for fp, d in scored:
        click.echo(f"{fp}  (distance: {d:.4f})")


@cli.command()
@click.argument("amount", type=int, default=5, required=False)
def rmpc(amount: int):
    """Find similar songs to the currently playing song and add them to the rmpc queue."""
    result = subprocess.run(["rmpc", "queue"], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo("Failed to get current queue from rmpc.", err=True)
        raise SystemExit(1)

    queue = json.loads(result.stdout)
    if not queue:
        click.echo("Queue is empty.", err=True)
        raise SystemExit(1)

    rel_path = queue[0]["file"]
    mpd_root = os.path.expanduser(MPD_MUSIC_DIR)
    abs_path = os.path.join(mpd_root, rel_path)

    init_db()
    with turso.connect(DB_PATH) as con:
        avg_embedding = get_file_embedding(con, abs_path)
        if avg_embedding is None:
            if not os.path.exists(abs_path):
                click.echo(f"File not in index and not found: {abs_path}", err=True)
                raise SystemExit(1)
            chunks = load_and_chunk(abs_path)
            embeddings = get_audio_embeddings([c for _, _, c in chunks])
            avg_embedding = np.mean(embeddings, axis=0)

        rows = db_search(con, avg_embedding, top_k=100, exclude_filepath=abs_path)

    if not rows:
        click.echo("No similar files found.")
        return

    scored = group_and_score(rows, amount)

    if not scored:
        click.echo("No similar files found.")
        return

    rel_paths = [os.path.relpath(fp, mpd_root) for fp, _ in scored]
    subprocess.run(["rmpc", "add", *rel_paths])
    added = len(rel_paths)
    click.echo(f"Added {added} song(s) to queue.")
