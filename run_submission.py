"""Regenerate Remontada possession predictions without reading any answer key."""
import argparse
from dataclasses import asdict
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
from stage1_cv_baseline import digest, private_path
from possession_observation_quality import QualityConfig


def attach_cv(players, ball, teams, path):
    """Use only qualified observed fills; never replace an available ATD point."""
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    if m.get('training_eligible') is not False or m.get('camera') != 'tactical' or m['fps'] != 25:
        raise ValueError('Expected inference-only tactical CV at 25 fps')
    if m.get('quality_policy_sha256') != digest(ROOT/'scripts/possession_observation_quality.py'):
        raise ValueError('Expected the externally tested, frozen quality policy')
    if m.get('quality_policy') != asdict(QualityConfig(unknown_airborne_policy='allow_uncertain')):
        raise ValueError('Submission v2 requires the allow_uncertain policy; regenerate the adapter with that explicit option')
    files = [path]
    for name, expected in m['quality_outputs_sha256'].items():
        p = path.parent/name
        if digest(p) != expected:
            raise ValueError('CV adapter artifact hash mismatch')
        files.append(p)
    with np.load(path.parent/m['files']['observations'], allow_pickle=False) as d:
        ids, xy = d['frame_ids'], d['ball']
        if (not np.array_equal(ids, m['frame_start']+np.arange(m['n_frames'])) or
                ids[0] < 0 or ids[-1] >= len(ball) or not np.array_equal(d['teams'], teams) or
                not np.allclose(d['players'], players[ids], atol=1e-4, rtol=0, equal_nan=True)):
            raise ValueError('CV timeline, player coordinates or team mapping differs')
        if not np.array_equal(d['ball_observed'], np.isfinite(xy).all(1)):
            raise ValueError('CV visibility differs from coordinates')
    with np.load(path.parent/m['quality_sidecar'], allow_pickle=False) as q:
        if (not np.array_equal(q['frame_ids'], ids) or
                not np.allclose(q['baseline_ball'], ball[ids], atol=1e-4, rtol=0, equal_nan=True)):
            raise ValueError('CV and full-match backbone differ')
        use = q['cv_used'].copy()
        if np.any(use & (np.isfinite(ball[ids]).all(1) | ~np.isfinite(xy).all(1))):
            raise ValueError('CV may fill only missing backbone observations')
        raw_sources = q['cv_source'].copy()
        observed = q['image_observed'].copy()
        uncertain = q['ground_projection_uncertain'].copy()
    combined = ball.copy()
    combined[ids[use]] = xy[use]
    masks = dict(cv_available=np.zeros(len(ball), bool), cv_observed=np.zeros(len(ball), bool),
                 cv_used=np.zeros(len(ball), bool), cv_source=np.full(len(ball), 'uncovered', dtype='<U80'))
    masks['cv_available'][ids] = True
    masks['cv_observed'][ids] = observed
    masks['cv_used'][ids] = use
    masks['cv_source'][ids] = raw_sources
    masks['cv_projection_uncertain'] = np.zeros(len(ball), bool)
    masks['cv_projection_uncertain'][ids] = uncertain
    return combined, masks, files


def run(data_dir, out, model='ensemble', cv_manifest=None, backbone_only=False):
    if model == 'ensemble' and not cv_manifest and not backbone_only:
        raise ValueError('Updated submission requires --cv-manifest; use --backbone-only explicitly for the preserved baseline')
    if cv_manifest and backbone_only:
        raise ValueError('Choose CV-supported submission or explicit backbone-only baseline')
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
    masks = dict(cv_available=np.zeros(n, bool), cv_observed=np.zeros(n, bool), cv_used=np.zeros(n, bool))
    masks['cv_projection_uncertain'] = np.zeros(n, bool)
    if cv_manifest:
        ball, masks, cv_files = attach_cv(players, ball, teams, cv_manifest)
        input_files.extend(cv_files)
    if model == 'ensemble':
        from predict_frozen_tracking import predict_arrays
        folder = ROOT/'models/ensemble'
        input_files.extend(folder/name for name in ('weights.json', 'tree.joblib', 'receipt.json'))
    elif model == 'gbdt':
        folder = ROOT/'models/gbdt'
        input_files.extend(folder/name for name in ('model.txt', 'model.json'))
        input_files.append(Path(paths['smart_tagging']))
    pins = {str(p): digest(p) for p in input_files}
    code = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *(ROOT/'scripts').glob('*.py')]}
    out.mkdir(parents=True)
    protocol = dict(model=model, input_sha256=pins, code_sha256=code, fps=fps, pitch=pitch, blocks=blocks,
                    answers_read=False, training=False, class_order=['H', 'A', 'D'], cv_manifest_used=bool(cv_manifest))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if model == 'ensemble':
        result = predict_arrays(players, ball, teams, pitch, blocks, fps,
                                folder/'weights.json', folder/'tree.joblib', folder/'receipt.json', True)
        labels, probabilities = np.asarray(result['prediction']), result['probabilities']
    elif model == 'gbdt':
        from possession_gbdt import load_input, feature_table, predict
        from tagging_evidence import restart_events, team_names, Restart
        import xml.etree.ElementTree as ET
        events = [dict(code=e.findtext('code'), start_frame=round(float(e.findtext('start'))*fps),
                       end_frame=round(float(e.findtext('end'))*fps),
                       tags={t.findtext('group'): t.findtext('text') for t in e.findall('label')})
                  for e in ET.parse(paths['smart_tagging']).iter('instance')]
        restarts = [Restart(max(0, r.start), min(n, r.end), r.kind, r.team)
                    for r in restart_events(events, team_names(paths['atd_meta'])) if r.start < n and r.end > 0]
        restarts.sort(key=lambda r: (r.start, r.end))
        table = dict(x=feature_table(players.astype(np.float32), ball.astype(np.float32), teams, pitch, blocks, fps, restarts),
                     blocks=blocks, fps=fps)
        probabilities, labels = predict(table, folder)
        labels = np.asarray(labels)
    else:
        import baseline
        if cv_manifest:
            raise ValueError('Official baseline is tracking-only')
        labels = baseline.smooth(baseline.nearest_player_labels(paths['atd_raw'],
                  np.array([0 if t == 'H' else 1 for t in teams]), n, pitch), 50)
        probabilities = np.column_stack([labels == k for k in ('H', 'A', 'D')]).astype(float)
    active = np.zeros(n, bool)
    for _, first, last in blocks:
        active[first:last+1] = True
    if np.any((labels == 'X') & active):
        raise ValueError('Unexpected active-period abstention')
    # Declared pre-match, halftime and post-match frames are inactive. This is
    # supplied period metadata, not a missing-CV or missing-ball decision.
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
                        model_observed=np.isfinite(ball).all(1), **masks)
    if any(digest(p) != h for p, h in pins.items()) or any(digest(ROOT/p) != h for p, h in code.items()):
        raise ValueError('Input or code changed during inference')
    counts = {k: int((labels == k).sum()) for k in ('H', 'A', 'D')}
    report = dict(protocol, rows=n, frame_start=0, frame_end=n-1, label_counts=counts,
                  transitions=int(np.count_nonzero(labels[1:] != labels[:-1])),
                  inactive_period_rule='D outside supplied half bounds; never inferred from missing CV',
                  backbone_observed=int(backbone_observed.sum()), model_observed=int(np.isfinite(ball).all(1).sum()),
                  cv_available=int(masks['cv_available'].sum()), cv_observed=int(masks['cv_observed'].sum()),
                  cv_used=int(masks['cv_used'].sum()), official_format_valid=True,
                  uncertain_cv_used=int((masks['cv_used'] & masks['cv_projection_uncertain']).sum()),
                  runtime_seconds=time.monotonic()-started, csv_sha256=digest(csv),
                  probability_sha256=digest(out/'probabilities.npz'))
    (out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('rows', 'label_counts', 'cv_used', 'runtime_seconds', 'csv_sha256')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', choices=['ensemble', 'gbdt', 'baseline'], default='ensemble')
    parser.add_argument('--cv-manifest')
    parser.add_argument('--backbone-only', action='store_true')
    args = parser.parse_args()
    run(args.data_dir, args.out, args.model, args.cv_manifest, args.backbone_only)
