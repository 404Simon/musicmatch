# MusicMatch

CLI tool for music similarity search using CLAP embeddings and vector similarity search via Turso/libsql.

## Commands

- **`index [DIRECTORY]`** — Index audio files (mp3, wav, opus) by computing CLAP embeddings for 10-second chunks
- **`list-files`** — List indexed files with chunk counts
- **`search <QUERY>`** — Find music matching a text description (e.g. "upbeat jazz with saxophone")
- **`similar <FILEPATH>`** — Find audio files similar to a given file
- **`rmpc [AMOUNT]`** — Find songs similar to the currently playing MPD track and add them to the queue

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `MUSICMATCH_DB_PATH` | `music_vectors.db` | Path to the vector database |
| `MUSICMATCH_TOP_K` | `5` | Default number of results |
| `MUSICMATCH_MPD_MUSIC_DIR` | `~/Music` | Music directory root for MPD |
| `MUSICMATCH_MAX_DURATION_MINUTES` | `12` | Skip files longer than this when indexing |
| `MUSICMATCH_MATCHIGNORE_PATH` | `~/.matchignore` | Path to the matchignore file |

## Content-Hash Dedup

MusicMatch computes an [AcoustID](https://acoustid.org/) fingerprint (chromaprint) for each indexed file. This allows the same audio content to be recognized even if it exists under different filepaths and formats.

- During `index`, fingerprints are computed and stored. If a file's fingerprint matches an already-indexed file, indexing is skipped with a warning.
- In `similar` and `rmpc` results, duplicates by content hash are removed — only the result with the best (lowest) distance per unique track is kept.

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
