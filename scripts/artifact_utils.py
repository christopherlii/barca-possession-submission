"""Frozen submission inference: artifact utils."""
import hashlib


import subprocess


from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Generated data must remain under ignored outputs/')
    if subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=ROOT).returncode:
        raise ValueError('Generated data must be Git-ignored')
    return path

