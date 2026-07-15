import sys

_VERBOSE = False


def set_verbose(v: bool = True):
    global _VERBOSE
    _VERBOSE = v


def verbose() -> bool:
    return _VERBOSE


def debug(msg: str, tag: str = "debug"):
    if _VERBOSE:
        print(f"[{tag}] {msg}", file=sys.stderr)


def rss() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.strip().split()[1] + "kB"
    except OSError:
        pass
    return "?kB"
