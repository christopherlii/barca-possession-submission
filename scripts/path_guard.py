"""Frozen submission inference: path guard."""
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from games import DATA_DIR, GAME


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

