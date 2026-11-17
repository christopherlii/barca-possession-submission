"""Regenerate Remontada possession predictions without reading any answer key."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'challenge'), str(ROOT/'challenge/starter'),
               str(ROOT/'challenge/evaluation')]

import numpy as np
import pandas as pd
import possession_metrics as pm
from control_motion import tracking_arrays
from games import GAME, GAMES, asset_paths, half_video_blocks
from sequence_decoder import team_of_each_track
from artifact_utils import digest, private_path


def run(data_dir, out):
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = str(args[0]).lower()
            if '/evaluation/gt/' in name or 'possession_gt' in name or name.endswith('/targets.csv'):
                raise PermissionError('Answer files are not inference inputs')
    sys.addaudithook(guard)
    started = time.monotonic()
    out = private_path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    paths = asset_paths(data_dir=str(Path(data_dir).resolve()), source='tactical')
    game = GAMES[GAME]
    n, fps, pitch, blocks = game['n_video_frames'], game['fps'], game['pitch'], half_video_blocks(GAME)
    teams = team_of_each_track(paths['atd_meta'])
    players, ball = tracking_arrays(paths['atd_raw'], teams, n, pitch, dtype=np.float64)
    backbone_observed = np.isfinite(ball).all(1)
    input_files = [Path(paths[k]) for k in ('atd_raw', 'atd_meta')]
    from predict_frozen_tracking import predict_arrays
    folder = ROOT/'models/ensemble'
    input_files.extend(folder/name for name in ('weights.json', 'tree.joblib', 'receipt.json'))
    pins = {str(p): digest(p) for p in input_files}
    code = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *(ROOT/'scripts').glob('*.py')]}
    out.mkdir(parents=True)
    protocol = dict(model='ensemble', input_sha256=pins, code_sha256=code, fps=fps, pitch=pitch, blocks=blocks,
                    answers_read=False, training=False, class_order=['H', 'A', 'D'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    result = predict_arrays(players, ball, teams, pitch, blocks, fps,
                            folder/'weights.json', folder/'tree.joblib', folder/'receipt.json', True)
    labels, probabilities = np.asarray(result['prediction']), result['probabilities']
    active = np.zeros(n, bool)
    for _, first, last in blocks:
        active[first:last+1] = True
    if np.any((labels == 'X') & active):
        raise ValueError('Unexpected active-period abstention')
    # Declared pre-match, halftime and post-match frames are inactive. This is
    # supplied period metadata, not a missing-ball decision.
    labels[~active] = 'D'
    if (probabilities.shape != (n, 3) or not np.isfinite(probabilities).all() or
            (probabilities < 0).any() or (probabilities > 1).any() or
            not np.allclose(probabilities.sum(1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Invalid H/A/D probabilities')
    frame_ids = np.arange(n)
    csv = out/'remontada.csv'
    pd.DataFrame(dict(frame=frame_ids, label=labels)).to_csv(csv, index=False)
    reason = pm.validate_submission(pm.read_label_csv(str(csv)), n)
    if reason or not np.isin(labels, ['H', 'A', 'D']).all():
        raise ValueError('Submission validation failed: '+str(reason))
    np.savez_compressed(out/'probabilities.npz', probabilities=probabilities, frame_ids=frame_ids, labels=['H', 'A', 'D'])
    np.savez_compressed(out/'availability.npz', frame_ids=frame_ids, backbone_observed=backbone_observed,
                        model_observed=np.isfinite(ball).all(1))
    if any(digest(p) != h for p, h in pins.items()) or any(digest(ROOT/p) != h for p, h in code.items()):
        raise ValueError('Input or code changed during inference')
    counts = {k: int((labels == k).sum()) for k in ('H', 'A', 'D')}
    report = dict(protocol, rows=n, frame_start=0, frame_end=n-1, label_counts=counts,
                  transitions=int(np.count_nonzero(labels[1:] != labels[:-1])),
                  inactive_period_rule='D outside supplied half bounds',
                  backbone_observed=int(backbone_observed.sum()), model_observed=int(np.isfinite(ball).all(1).sum()),
                  official_format_valid=True,
                  runtime_seconds=time.monotonic()-started, csv_sha256=digest(csv),
                  probability_sha256=digest(out/'probabilities.npz'))
    (out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('rows', 'label_counts', 'runtime_seconds', 'csv_sha256')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', choices=['ensemble'], default='ensemble')
    args = parser.parse_args()
    run(args.data_dir, args.out)
