# MusicMatch

CLI tool for music similarity search using CLAP embeddings and vector similarity search via Turso/libsql.

## Commands

- **`index [DIRECTORY]`**: Index audio files (mp3, wav, opus, flac). Incremental by default: adds new files, prunes stale paths, skips duplicates via chromaprint fingerprint. Use `--full` to wipe and re-index everything.
- **`list-files`**: List indexed files with chunk counts
- **`search <QUERY>`**: Find music matching a text description (e.g. "upbeat jazz with saxophone")
- **`similar <FILEPATH>`**: Find audio files similar to a given file
- **`rmpc [AMOUNT]`**: Find songs similar to the currently playing MPD track and add them to the queue
- **`visualize [--output, -n]`**: Generate an interactive 3D scatter plot of the music library via PCA

## Database Architecture

Three tables:

```text
files -- chromaprints -- chunks
```

- **`chromaprints`**: Unique audio fingerprint (AcoustID hash). Stable identifier for a track regardless of filename.
- **`files`**: File paths on disk. Multiple files can point to the same chromaprint (duplicates, different formats, re-encodes).
- **`chunks`**: 10-second audio segments with 512-dim CLAP embeddings. Keyed by chromaprint, not by file. Duplicate content shares one set of chunks.

This design means renaming a file is a single `UPDATE` on the `files` table -- chunks are untouched. Duplicate content at different paths is stored once.

## Index Behavior

Default `index` is incremental:

1. **Scan**: finds all audio files in the directory
2. **Prune**: removes DB entries for paths that no longer exist on disk
3. **Add**: fingerprints each new file; if content is already indexed (matching chromaprint), only a `files` row is added, no embedding recomputation
4. **Commit per file**: Ctrl+C is safe, only the current file's work is lost

`index --full` deletes all data and re-indexes from scratch.

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `MUSICMATCH_DB_PATH` | `~/music_vectors.db` | Path to the vector database |
| `MUSICMATCH_TOP_K` | `5` | Default number of results |
| `MUSICMATCH_MPD_MUSIC_DIR` | `~/Music` | Music directory root for MPD |
| `MUSICMATCH_MAX_DURATION_MINUTES` | `12` | Skip files longer than this when indexing |
| `MUSICMATCH_MATCHIGNORE_PATH` | `~/.matchignore` | Path to the matchignore file |

## Content-Hash Dedup

MusicMatch computes an [AcoustID](https://acoustid.org/) fingerprint (chromaprint) for each indexed file. This allows the same audio content to be recognized even if it exists under different filepaths and formats.

- During `index`, fingerprints are computed. Matching content skips embedding computation -- only a file path reference is added.
- In `similar` and `rmpc` results, duplicates by content hash are removed -- only the result with the best distance per unique track is kept.

**System requirement:** `libchromaprint` (or `fpcalc` binary). Install via your package manager:

- Debian/Ubuntu: `sudo apt install libchromaprint-tools`
- macOS: `brew install chromaprint`
- Arch: `sudo pacman -S chromaprint`

## `.matchignore`

**Usage:** `musicmatch --no-ignore` disables it globally.

Patterns that match a file's relative path exclude it from `index`, `search`, `similar`, and `rmpc` results.

Format: one regex per line, blank lines and `#` comments ignored.

```text
# Exclude entire directories
^coding/
^lernen/

# Exclude specific files
theo_banenenbrot\.mp3$

# Exclude by extension
\.wav$
```
