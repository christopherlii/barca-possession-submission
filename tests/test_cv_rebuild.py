import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rebuild_v2_cv as rebuild


def test_recipe_pins_and_fresh_paths():
    recipe=rebuild.verified_recipe()
    steps=rebuild.commands(recipe,Path('/private/new-engine'),Path('/licensed/tactical.mp4'),'cpu')
    assert len(steps)==23
    assert steps[0]['command'][0]==sys.executable
    assert '/licensed/tactical.mp4' in steps[0]['command']
    assert '/private/new-engine/outputs/two-stage-cv-models/yolo11x.pt' in steps[0]['command']
    for step in steps:
        assert not any('{'+k in arg for arg in step['command'] for k in ('run','engine','python','video'))
        assert not any('/Users/chrisli/' in arg for arg in step['command'][1:])
        assert 'mps' not in step['command']
    assert recipe['raw_video_reproduction_verified'] is False


def test_source_tampering_fails(tmp_path,monkeypatch):
    monkeypatch.setattr(rebuild,'ROOT',tmp_path)
    root=tmp_path/'cv_rebuild';(root/'snapshot/scripts').mkdir(parents=True)
    p=root/'snapshot/scripts/a.py';p.write_text('original')
    (root/'recipe.json').write_text(json.dumps({'source_sha256':{'scripts/a.py':hashlib.sha256(p.read_bytes()).hexdigest()}}))
    rebuild.verified_recipe()
    p.write_text('modified')
    with pytest.raises(ValueError,match='source mismatch'):rebuild.verified_recipe()


def test_modified_checkpoint_fails(tmp_path):
    (tmp_path/'model.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='checkpoint differs'):
        rebuild.check_assets({'models':{'model.pt':'0'*64}},tmp_path)
