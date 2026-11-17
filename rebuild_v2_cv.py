"""Rebuild the v2 CV stages, or verify/replay the historical CV cache.

No fitting, answer keys, or full-match CV. A mismatching rebuilt CSV fails;
it never silently replaces the frozen submission. All outputs stay private.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
EXPECTED_CACHE={'02-fused.npz':'4cfc59eb494a000bd7b53271acd1f45ab07ee9877a8574d97950d90f2b28323a',
    '12-final/detections.jsonl':'4c4a79fed27e495c3a5ffb2de38588308124ca936c307bf99554fe77cda71b25',
    '12-final/config.json':'0262e703a85dc0661b20a5c17351bff9528b9943edfb776045f92b117ad4b1a7'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verified_recipe():
    recipe=json.loads((ROOT/'cv_rebuild/recipe.json').read_text())
    for name,expected in recipe['source_sha256'].items():
        if digest(ROOT/'cv_rebuild/snapshot'/name)!=expected:
            raise ValueError('Frozen CV source mismatch: '+name)
    return recipe


def commands(recipe,engine,video,device):
    substitutions={'{python}':sys.executable,'{engine}':str(engine),
                   '{run}':str(engine/'outputs/exhaustive'),'{video}':str(video)}
    result=[]
    for step in recipe['steps']:
        command=[]
        for arg in step['command']:
            for key,value in substitutions.items():arg=arg.replace(key,value)
            command.append(device if arg=='mps' else arg)
        result.append(dict(name=step['name'],command=command))
    return result


def check_assets(recipe,assets):
    for name,expected in recipe['models'].items():
        if digest(assets/name)!=expected:raise ValueError('CV checkpoint differs: '+name)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=assets/'sam2-source',text=True).strip()
    if revision!=recipe['sam_source_revision']:raise ValueError('SAM source revision differs')


def run(args):
    started=time.monotonic();recipe=verified_recipe();out=args.out.resolve();data=args.data_dir.resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Use a private output path under outputs/')
    if subprocess.run(['git','check-ignore','-q',str(out)],cwd=ROOT).returncode:raise ValueError('Output must be ignored')
    if out.exists():raise ValueError('Use a new output directory')
    needed=['BAR-ATH_tactical_atd.txt','BAR-ATH_tactical_atd.xml','BAR-ATH_tactical_homography.csv','BAR-ATH_tactical_smart_tagging.xml']
    if args.rebuild or args.check:needed.append('BAR-ATH_tactical.mp4')
    for name in needed:
        if not (data/'tactical'/name).is_file():raise FileNotFoundError('Missing tactical input: '+name)
    if args.rebuild or args.check:check_assets(recipe,args.assets_dir.resolve())
    if args.replay_cache:
        cache=args.replay_cache.resolve()
        for name,expected in EXPECTED_CACHE.items():
            if digest(cache/name)!=expected:raise ValueError('Historical CV cache mismatch: '+name)
    out.mkdir(parents=True)
    report=dict(action='rebuild' if args.rebuild else 'check' if args.check else 'historical_cache_replay',
        recipe_sha256=digest(ROOT/'cv_rebuild/recipe.json'),answers_read=False,training=False,
        raw_video_reproduction_verified=False,device=args.device,commands=[])
    def save():
        (out/'rebuild-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    def execute(command,label):
        report['commands'].append(command);save()
        print(label,flush=True)
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.check:
        report.update(status='source_and_asset_preflight_passed',
            note='This checks files and hashes, not CV execution or raw-video byte equality.',
            planned_steps=commands(recipe,out/'engine',data/'tactical/BAR-ATH_tactical.mp4',args.device))
        save();print(report['status']);return
    if args.rebuild:
        engine=out/'engine';shutil.copytree(ROOT/'cv_rebuild/snapshot',engine)
        (engine/'data').symlink_to(data,target_is_directory=True)
        (engine/'.venv').symlink_to(Path(sys.executable).absolute().parent.parent,target_is_directory=True)
        (engine/'outputs').mkdir()
        assets=args.assets_dir.resolve()
        for name in ['two-stage-cv-models','sam2-models','sam2-source']:
            (engine/'outputs'/name).symlink_to(assets/name,target_is_directory=True)
        if args.runtime_dir:
            for name in ['two-stage-cv-runtime','sam2-runtime']:
                (engine/'outputs'/name).symlink_to(args.runtime_dir.resolve()/name,target_is_directory=True)
        cache=engine/'outputs/exhaustive';cache.mkdir()
        for step in commands(recipe,engine,data/'tactical/BAR-ATH_tactical.mp4',args.device):
            execute(step['command'],step['name'])
    # Rebuild adapters from observed CV evidence. Do not reuse labels or a
    # pre-qualified possession input to make the replay appear successful.
    execute([sys.executable,str(ROOT/'scripts/cv_to_possession.py'),'--features',str(cache/'02-fused.npz'),
        '--cache',str(cache/'12-final'),'--data-dir',str(data),'--out',str(out/'adapter')],'adapter')
    execute([sys.executable,str(ROOT/'scripts/cv_confidence_adapter.py'),
        '--manifest',str(out/'adapter/manifest.json'),'--features',str(cache/'02-fused.npz'),
        '--cache',str(cache/'12-final'),'--data-dir',str(data),'--unknown-airborne-policy','allow_uncertain',
        '--out',str(out/'qualified')],'qualify')
    execute([sys.executable,str(ROOT/'run_submission.py'),'--data-dir',str(data),'--model','ensemble',
        '--cv-manifest',str(out/'qualified/manifest.json'),'--out',str(out/'prediction')],'predict')
    actual=digest(out/'prediction/remontada.csv')
    report.update(csv_sha256=actual,expected_csv_sha256=recipe['expected_submission_sha256'],
        csv_byte_exact=actual==recipe['expected_submission_sha256'],runtime_seconds=time.monotonic()-started)
    report['raw_video_reproduction_verified']=bool(args.rebuild and report['csv_byte_exact'])
    report['status']='passed' if report['csv_byte_exact'] else 'failed_csv_mismatch';save()
    if not report['csv_byte_exact']:raise ValueError('Rebuilt file differs from frozen v2; see receipt. It was not promoted.')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--rebuild',action='store_true')
    group.add_argument('--replay-cache',type=Path)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,default=ROOT/'outputs/cv-assets')
    p.add_argument('--runtime-dir',type=Path,help='Optional existing pinned private dependency directories')
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    run(p.parse_args())
