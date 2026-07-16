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
