"""Adapt the tactical CV fusion/recovery pipeline to the GBDT input contract."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stage1_cv_baseline import ROOT, digest, private_path
from fuse_tactical_cv import transform
from games import GAME, GAMES, asset_paths, half_video_blocks
from tagging_evidence import restart_events, team_names
from sequence_decoder import team_of_each_track

OBSERVATION_SOURCES = {
    'smoothed_detection', 'gated_detection', 'short_gap_detection',
    'blank_recovery_observation', 'bidirectional_observation', 'aerial_observed_path',
    'sam2_retry_bidirectional', 'sam2_retry_cv_supported', 'sam2_optical_flow_supported',
    'sam2_cv_supported', 'sam2_tight_crop_cv_supported', 'sam2_multiscale_flow_supported',
    'optical_gap_tracking',
    'sam2_extended_verified', 'sam2_reacquired',
}


def recovery_point(row):
    """Respect the final observation decision, including explicit rejection."""
    if (row.get('penalty_spot_guard') or {}).get('rejected'):
        return None, 'missing'
    if row.get('local_motion') and not row['local_motion'].get('accepted', False):
        return None, 'missing'
    if 'evidence_position' in row:
        point = row['evidence_position']
        source = row.get('evidence_source', 'missing')
        if source in {'sam2_extended_verified', 'sam2_reacquired'} and not row.get('sam2_mask'):
            return None, 'missing'
        return (point, source) if source in OBSERVATION_SOURCES else (None, 'missing')
    # Earlier recovery stages expose their accepted image observation here.
    from recover_blank_tracks import trusted
    point = trusted(row)
    return point, row.get('context_source', 'recovery') if point is not None else 'missing'


def project_recovery(rows, frames, homography, pitch):
    if [r['frame'] for r in rows] != frames.tolist():
        raise ValueError('Recovery cache and features must have identical video frames')
    matrices = {}
    with Path(homography).open() as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame'])
            if frames[0] <= frame <= frames[-1]:
                if frame in matrices:
                    raise ValueError('Repeated homography frame')
                h = np.array([[float(row[f'h{i}{j}']) for j in range(3)] for i in range(3)])
                usable = (int(row['valid']) == 1 and np.isfinite(float(row['reprojection_error']))
                          and float(row['reprojection_error']) <= 2 and np.isfinite(h).all()
                          and abs(np.linalg.det(h)) > 1e-12)
                matrices[frame] = h if usable else None
    if len(matrices) != len(frames):
        raise ValueError('Homography must cover every cached video frame')
    ball = np.full((len(frames), 2), np.nan)
    sources = np.full(len(frames), 'missing', dtype='<U80')
    for i, row in enumerate(rows):
        point, source = recovery_point(row)
        if point is None:
            continue
        point = np.asarray(point, float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Invalid recovery position')
        matrix = matrices[int(frames[i])]
        if matrix is None:
            sources[i] = 'invalid_homography'
            continue
        ball[i] = transform([point], np.linalg.inv(matrix))[0] * pitch
        sources[i] = source
    return ball, sources


def local_blocks(frames, segments, halves):
    """Split period blocks at camera cuts, with local array indices."""
    blocks = []
    for half, first, last in halves:
        start = max(first, int(frames[0])) - int(frames[0])
        stop = min(last, int(frames[-1])) - int(frames[0]) + 1
        if start >= stop:
            continue
        cuts = np.flatnonzero(np.diff(segments[start:stop]) != 0) + start + 1
        edges = [start, *cuts.tolist(), stop]
        blocks.extend([[half, a, b-1] for a, b in zip(edges[:-1], edges[1:])])
    return blocks


def clipped_restarts(path, names, fps, first, last):
    events = []
    for event in ET.parse(path).iter('instance'):
        events.append(dict(code=event.findtext('code'),
            start_frame=round(float(event.findtext('start'))*fps),
            end_frame=round(float(event.findtext('end'))*fps),
            tags={tag.findtext('group'): tag.findtext('text') for tag in event.findall('label')}))
    result = []
    for event in restart_events(events, names):
        start, end = max(first, event.start), min(last+1, event.end)
        if start < end:
            result.append(dict(start=start-first, end=end-first, kind=event.kind, team=event.team))
    return result


def adapt(features, output, cache=None, data_dir=None):
    features, output = Path(features).resolve(), private_path(output)
    if output.exists():
        raise ValueError('Use a new adapter output directory')
    paths = asset_paths(data_dir=str(data_dir or ROOT/'data'), source='tactical')
    with np.load(features, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta.get('feature_mode') != 'tactical_cv_fusion':
            raise ValueError('Expected the current pipeline 02-fused.npz cache')
        if Path(meta['source']).resolve() != Path(paths['video']).resolve():
            raise ValueError('Features must reference the same tactical video')
        frames, segments = data['frame_ids'].copy(), data['segments'].copy()
        if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
                or not np.all(np.diff(frames) == 1) or frames[0] < 0
                or frames[-1] >= GAMES[GAME]['n_video_frames']
                or segments.shape != frames.shape):
            raise ValueError('Invalid cached video frame interval or segments')
        players, teams = data['atd_players_m'].copy(), data['atd_slot_teams'].copy()
        ball, flags = data['fused_ball_m'].copy(), data['source_flags'].copy()
    pitch = np.asarray(GAMES[GAME]['pitch'])
    if meta['fps'] != GAMES[GAME]['fps']:
        raise ValueError('Feature and tactical video frame rates differ')
    if (players.shape != (len(frames), len(teams), 2) or not np.isin(teams, [0, 1]).all()
            or np.isinf(players).any() or ball.shape != (len(frames), 2)
            or flags.shape != (len(frames), 12)):
        raise ValueError('Invalid fused tracking arrays')
    provenance = meta['fusion_input_sha256']
    for key in ('homography', 'atd_meta'):
        if digest(paths[key]) != provenance[key]:
            raise ValueError('Tactical calibration/team metadata differs from the fused cache')
    expected_teams = np.array([0 if t == 'H' else 1 for t in team_of_each_track(paths['atd_meta'])])
    if not np.array_equal(teams, expected_teams):
        raise ValueError('Cached H/A slot mapping differs from tactical metadata')
    inputs = dict(features_sha256=digest(features), homography_sha256=provenance['homography'],
                  atd_meta_sha256=provenance['atd_meta'])
    if cache:
        cache = Path(cache).resolve()
        config = json.loads((cache/'config.json').read_text())
        if config['features_sha256'] != inputs['features_sha256']:
            raise ValueError('Recovery cache belongs to a different fused feature file')
        recovery_hash = digest(cache/'detections.jsonl')
        rows = [json.loads(line) for line in (cache/'detections.jsonl').read_text().splitlines()]
        ball, sources = project_recovery(rows, frames, paths['homography'], pitch)
        if digest(cache/'detections.jsonl') != recovery_hash:
            raise ValueError('Recovery cache changed during adaptation; retry after it finishes')
        inputs.update(recovery_sha256=recovery_hash,
                      recovery_config_sha256=digest(cache/'config.json'))
    else:
        observed = flags[:, 2].astype(bool) & ~flags[:, 11].astype(bool)
        ball[~observed] = np.nan
        sources = np.where(observed, np.where(flags[:, 9], 'atd', 'cv'), 'missing')
    blocks = local_blocks(frames, segments, half_video_blocks(GAME))
    tags = clipped_restarts(paths['smart_tagging'], team_names(paths['atd_meta']),
                            meta['fps'], int(frames[0]), int(frames[-1]))
    inputs['tagging_sha256'] = digest(paths['smart_tagging'])
    manifest = dict(format='cv-possession-v1', match_id=GAME, supplied_match=True,
        training_eligible=False, camera='tactical', tagging_camera='tactical',
        n_frames=len(frames), frame_start=int(frames[0]), fps=meta['fps'],
        pitch=pitch.tolist(), blocks=blocks,
        files=dict(observations='observations.npz', tagging='restarts.json'),
        provenance=inputs, adapter_sha256=digest(__file__),
        player_source='fused cache automatic tactical player tracks',
        ball_source='accepted recovery observations' if cache else 'observed fused tracking')
    output.mkdir(parents=True)
    np.savez_compressed(output/'observations.npz', players=players, ball=ball,
        teams=np.where(teams == 0, 'H', 'A'), frame_ids=frames,
        ball_observed=np.isfinite(ball).all(axis=1), ball_source=sources)
    (output/'restarts.json').write_text(json.dumps(tags)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summary = dict(frames=len(frames), frame_start=int(frames[0]), frame_end=int(frames[-1]),
        observed_ball_frames=int(np.isfinite(ball).all(axis=1).sum()),
        blocks=len(blocks), restart_windows=len(tags), accuracy_measured=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return output/'manifest.json', summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='Completed ball-recovery directory')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Also run this frozen GBDT model')
    args = parser.parse_args()
    manifest, summary = adapt(args.features, args.out, args.cache, args.data_dir)
    if args.model:
        from possession_gbdt import write_predictions
        summary['prediction'] = write_predictions(manifest, args.model, args.out/'prediction')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
