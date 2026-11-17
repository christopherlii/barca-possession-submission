"""Label-blind local inference of the already-frozen cycle-018 tracking model.

No fitting, candidate selection, answer paths or target inputs. Preserve the
original float64 base coordinates / float32 motion coordinate conventions.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ball_control_features import ball_control_features
from control_motion import tracking_arrays
from controller_dynamics import motion_features
from controller_symmetry import swap_teams,average_predictions
from direct_weight_model import features_from_arrays
from learned_controller import contextualize
from player_track_features import player_track_features,swap_player_features
from player_possession_tcn import load_weights,probabilities
from possession_boundary import decode_team_switch
from possession_candidates import sha256
from sequence_decoder import team_of_each_track,write_submission

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'challenge'))
from games import asset_paths,GAME,GAMES,half_video_blocks,source_frames


def features(players,ball,teams,pitch,blocks,fps=25.):
    base=features_from_arrays(players,ball,teams,pitch,blocks,fps)
    p=np.asarray(players,np.float32);b=np.asarray(ball,np.float32)
    global_features=np.concatenate([base,motion_features(p,b,teams,blocks,fps),
        ball_control_features(b,blocks,fps,.04),ball_control_features(b,blocks,fps,.2)],axis=1).astype(np.float32)
    actors,presence=player_track_features(p,b,pitch,blocks,fps)
    return global_features,actors,presence,np.array([0 if t=='H' else 1 for t in teams],np.int8)


def predict(raw,meta,n,pitch,blocks,fps,weights_path,tree_path,receipt_path,return_components=False,return_branch_scores=False):
    teams=team_of_each_track(meta)
    p,b=tracking_arrays(raw,teams,n,pitch,dtype=np.float64)
    return predict_arrays(p,b,teams,pitch,blocks,fps,weights_path,tree_path,receipt_path,
                          return_components,return_branch_scores)


def predict_arrays(p,b,teams,pitch,blocks,fps,weights_path,tree_path,receipt_path,return_components=False,return_branch_scores=False):
    """Same frozen forward pass for metric observations supplied by a CV adapter."""
    import joblib
    receipt=json.loads(Path(receipt_path).read_text())
    if (receipt['chosen']!='boundary_scale_1' or sha256(weights_path)!=receipt['model_sha256'] or
        sha256(tree_path)!=receipt['tree_sha256']):raise ValueError('Frozen model artifacts differ')
    if fps != 25.:
        raise ValueError('Frozen player ensemble requires its native 25 fps clock')
    print('Computing frozen observed tracking features locally.',flush=True)
    g,a,m,t=features(p,b,teams,pitch,blocks,fps)
    del p,b
    w=load_weights(weights_path);tree=joblib.load(tree_path)
    if list(tree.classes_)!=[0,1,2]:raise ValueError('Unexpected learned class order')
    original,q=probabilities(g,a,m,t,w,blocks)
    changed=swap_teams(g)
    mirrored,qs=probabilities(changed,swap_player_features(a),m,1-t,w,blocks)
    pooled=average_predictions(tree.predict_proba(contextualize(g[:,:40],blocks,fps)),
        tree.predict_proba(contextualize(changed[:,:40],blocks,fps)))
    temporal=average_predictions(original,mirrored)
    combined=.5*pooled+.5*temporal
    switch=.5*(q+qs)
    labels=decode_team_switch(combined,switch,blocks,1.,16.)
    if return_components:
        result=dict(prediction=labels,probabilities=combined,switch=switch)
        if return_branch_scores:result.update(tree_probabilities=pooled,temporal_probabilities=temporal)
        return result
    return labels


def run(out):
    out=Path(out).resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Private predictions must be under outputs/')
    if subprocess.run(['git','check-ignore','-q',str(out)],cwd=ROOT).returncode:
        raise ValueError('Private prediction output must be ignored')
    paths=asset_paths(data_dir=str(ROOT/'data'))
    weights=ROOT/'outputs/goal95/cycle-018-player-tracks/weights.json'
    tree=ROOT/'outputs/goal95/cycle-005/model.joblib'
    receipt=ROOT/'outputs/goal95/cycle-018-player-tracks/results.json'
    inputs=[Path(paths['atd_raw']),Path(paths['atd_meta']),weights,tree,receipt]
    frozen=dict(schema=1,model='frozen-cycle018-boundary-scale1',training_performed=False,
        answers_read=False,inputs_sha256={str(p):sha256(p) for p in inputs},
        source_sha256={p.name:sha256(p) for p in (ROOT/'scripts').glob('*.py')},
        configuration=dict(tree_weight=.5,network_weight=.5,team_symmetry=True,team_scale=1.,dead_cost=16.))
    out.parent.mkdir(parents=True,exist_ok=True)
    Path(str(out)+'.receipt.json').write_text(json.dumps(frozen,indent=2)+'\n')
    labels=predict(paths['atd_raw'],paths['atd_meta'],source_frames('tactical'),GAMES[GAME]['pitch'],
        half_video_blocks(GAME),GAMES[GAME]['fps'],weights,tree,receipt)
    if any(sha256(ROOT/'scripts'/name)!=h for name,h in frozen['source_sha256'].items()):
        raise ValueError('Inference implementation changed during prediction')
    write_submission(str(out),labels)
    frozen.update(status='completed',prediction_sha256=sha256(out))
    Path(str(out)+'.receipt.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(frames=len(labels),prediction_sha256=frozen['prediction_sha256'],answers_read=False)),flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/public-five-minute-method-check/predictions/learned_cycle018.csv')
    try:run(parser.parse_args().out)
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print('Frozen inference failed ('+type(error).__name__+'); private details suppressed.',file=sys.stderr)
        sys.exit(1)
