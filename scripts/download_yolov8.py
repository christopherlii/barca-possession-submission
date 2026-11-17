"""Pinned official January-2024 YOLOv8 intake; passive checkpoint inspection."""
import hashlib
import json
from pathlib import Path
import pickletools
import urllib.request
import zipfile

from path_guard import check_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/yolov8-source-intake'
WEIGHTS_SHA = '31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95'
WHEEL_SHA = '9571a74cb51f58f49f7b29e545a36b7fc79f42b43a9e3e9d93e54183d0980b0c'
ARTIFACTS = {
    'yolov8n.pt': ('https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt', 6534387, WEIGHTS_SHA),
    'ultralytics-8.1.0-py3-none-any.whl': ('https://files.pythonhosted.org/packages/57/2e/9d58b392f3cb41f75e1f8e274e528c48c46e47caf581e3e34147780c26dc/ultralytics-8.1.0-py3-none-any.whl', 699197, WHEEL_SHA),
}


def fetch(path, url, size, digest):
    check_paths([path])
    if path.exists():
        data = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={'User-Agent': 'barca-possession-research'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(size+1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Pinned source artifact differs: '+path.name)
    if not path.exists():
        temporary = path.with_suffix(path.suffix+'.part')
        temporary.write_bytes(data)
        temporary.replace(path)


def run(out=OUT):
    out = Path(out)
    check_paths([out])
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, size, digest) in ARTIFACTS.items():
        fetch(out/name, url, size, digest)
    with zipfile.ZipFile(out/'yolov8n.pt') as archive:
        members = archive.infolist()
        if sum(v.file_size for v in members) > 20_000_000 or any(v.file_size > 15_000_000 for v in members):
            raise ValueError('Unexpected checkpoint ZIP structure')
        if archive.testzip() is not None:
            raise ValueError('Checkpoint ZIP CRC failed')
        names = [v.filename for v in members if v.filename.endswith('/data.pkl')]
        if len(names) != 1:
            raise ValueError('Expected one passive pickle record')
        operations = list(pickletools.genops(archive.read(names[0])))
        if any(op.name == 'STACK_GLOBAL' for op, _, _ in operations):
            raise ValueError('Unexpected dynamic pickle-global format')
        globals_ = sorted({value.replace(' ', '.') for op, value, _ in operations if op.name == 'GLOBAL'})
        if len(globals_) != 23 or any(not n.startswith(('torch.', 'ultralytics.', 'collections.', '__builtin__.')) for n in globals_):
            raise ValueError('Unexpected checkpoint global constructors')
    with zipfile.ZipFile(out/'ultralytics-8.1.0-py3-none-any.whl') as wheel:
        license_data = wheel.read('ultralytics-8.1.0.dist-info/LICENSE')
        if b'GNU AFFERO GENERAL PUBLIC LICENSE' not in license_data[:100]:
            raise ValueError('Expected upstream AGPL license')
        (out/'LICENSE.ultralytics').write_bytes(license_data)
        files = {n: hashlib.sha256(wheel.read(n)).hexdigest() for n in wheel.namelist() if n.endswith('.py')}
    receipt = dict(schema=1, author='Ultralytics', version='8.1.0', code_license='AGPL-3.0',
        weights_repository_license='AGPL-3.0',
        weights_license_source='https://raw.githubusercontent.com/ultralytics/assets/v8.1.0/LICENSE',
        release_url='https://github.com/ultralytics/assets/releases/tag/v8.1.0',
        release_asset_id=144705711, asset_created='2024-01-10T00:31:16Z', asset_updated='2024-01-10T00:31:17Z',
        wheel_published='2024-01-10T09:59:58.067451Z',
        artifacts={n: dict(url=url, bytes=size, sha256=digest) for n, (url, size, digest) in ARTIFACTS.items()},
        checkpoint_checksum_note='SHA256 pinned at first official-asset acquisition; publisher supplies size/date/id but no independent digest.',
        wheel_checksum_note='SHA256 independently matches PyPI release metadata.',
        license_sha256=hashlib.sha256(license_data).hexdigest(), wheel_python_sha256=files,
        checkpoint_static_globals=globals_, checkpoint_pickle_executed=False,
        model_inference=False, model_fitting=False, supplied_match_read=False,
        data_lineage_note='Publisher asset creation predates supplied match; training metadata still requires restricted loading.')
    (out/'source-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(version='8.1.0', weights_sha256=WEIGHTS_SHA, wheel_sha256=WHEEL_SHA,
        passive_globals=len(globals_), model_executed=False)), flush=True)
    return receipt


if __name__ == '__main__':
    run()
