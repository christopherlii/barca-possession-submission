"""Refuse supplied-match paths in synthetic tooling.

The synthetic generator and the experiment harness must never read or write the supplied match:
not its public ground truth, not the gated data bundle, not the video proxies. Every input and
output path they take goes through check_paths() first. A path is refused when, after resolving
symlinks and '..', it lies inside

  - challenge/evaluation/gt/   the public ground truth (never a tuning set)
  - data/                      the gated match bundle (and MTAH_DATA_DIR, if set)
  - media/                     match video proxies

or when any component of it is named with the supplied match's "BAR-ATH" stem.

    python scripts/path_guard.py PATH [PATH ...]    # exits 2 naming the first refused path
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "challenge"))
from games import DATA_DIR, GAME  # noqa: E402

EXIT_FORBIDDEN = 2


class ForbiddenPathError(ValueError):
    pass


def _resolve(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def forbidden_roots():
    """(description, resolved directory) pairs no synthetic tool may touch."""
    roots = [
        ("public ground truth", os.path.join(ROOT, "challenge", "evaluation", "gt")),
        ("match data", os.path.join(ROOT, "data")),
        ("match media", os.path.join(ROOT, "media")),
        ("match data", DATA_DIR),
    ]
    if os.environ.get("MTAH_DATA_DIR"):
        roots.append(("match data", os.environ["MTAH_DATA_DIR"]))
    return [(name, _resolve(folder)) for name, folder in roots]


def forbidden_reason(path):
    """None if the path may be used, else a one-line reason why not."""
    resolved = _resolve(path)
    for name, root in forbidden_roots():
        if os.path.commonpath([resolved, root]) == root:
            return "{} is inside the {} directory {}".format(path, name, root)
    stem = GAME.lower()
    names = [os.path.basename(os.path.normpath(str(path)))] + resolved.split(os.sep)
    if any(part.lower().startswith(stem) for part in names if part):
        return "{} is named with the supplied match stem {}".format(path, GAME)
    return None


def check_paths(paths):
    """Raise ForbiddenPathError for the first refused path."""
    for path in paths:
        if path is None:
            continue
        reason = forbidden_reason(path)
        if reason:
            raise ForbiddenPathError(reason)


def guard_paths(paths):
    """check_paths() for command-line tools: print the reason and exit non-zero."""
    try:
        check_paths(paths)
    except ForbiddenPathError as error:
        print("refused: {}".format(error), file=sys.stderr)
        sys.exit(EXIT_FORBIDDEN)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: path_guard.py PATH [PATH ...]", file=sys.stderr)
        sys.exit(1)
    guard_paths(sys.argv[1:])
