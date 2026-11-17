"""Frozen submission inference: predict frozen tracking."""
import json


from pathlib import Path


import numpy as np


from ball_control_features import ball_control_features


from controller_dynamics import motion_features


from controller_symmetry import swap_teams, average_predictions


from direct_weight_model import features_from_arrays


from learned_controller import contextualize


from player_track_features import player_track_features, swap_player_features


from player_possession_tcn import load_weights, probabilities


from possession_boundary import decode_team_switch


from artifact_utils import digest as sha256


def features(players,ball,teams,pitch,blocks,fps=25.):
    base=features_from_arrays(players,ball,teams,pitch,blocks,fps)
    p=np.asarray(players,np.float32);b=np.asarray(ball,np.float32)
    global_features=np.concatenate([base,motion_features(p,b,teams,blocks,fps),
        ball_control_features(b,blocks,fps,.04),ball_control_features(b,blocks,fps,.2)],axis=1).astype(np.float32)
    actors,presence=player_track_features(p,b,pitch,blocks,fps)
    return global_features,actors,presence,np.array([0 if t=='H' else 1 for t in teams],np.int8)


def predict_arrays(p,b,teams,pitch,blocks,fps,weights_path,tree_path,receipt_path,return_components=False,return_branch_scores=False):
    """Frozen forward pass for automatic tactical tracking observations."""
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

