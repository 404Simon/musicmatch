import os
import re
import sys

from musicmatch.config import MATCHIGNORE_PATH as _CONFIG_PATH

_path = None
_patterns: list[re.Pattern] | None = None
_mtime = 0.0
_enabled = True


def set_enabled(val: bool) -> None:
    global _enabled
    _enabled = val


def _load() -> list[re.Pattern]:
    global _path, _patterns, _mtime
    path = os.path.expanduser(_CONFIG_PATH)
    if path != _path:
        _path = path
        _mtime = 0
    try:
        cur = os.path.getmtime(path)
    except OSError:
        _patterns = []
        return _patterns
    if cur == _mtime:
        return _patterns
    _mtime = cur
    _patterns = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                _patterns.append(re.compile(line))
            except re.error as e:
                print(f"musicmatch: {path}:{i}: {e}", file=sys.stderr)
    return _patterns


def is_ignored(rel_path: str) -> bool:
    if not _enabled:
        return False
    return any(p.search(rel_path) for p in _load())
