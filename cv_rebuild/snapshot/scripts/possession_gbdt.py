"""Feature-engineered H/A/D LightGBM with automatic restart evidence and Viterbi.

Inputs are bundle manifests with ATD files or metric CV arrays. Coordinates in
CV NPZs are metres from the pitch corner; team IDs must already be H/A mapped.
All temporal context is isolated to the manifest's inclusive period blocks.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from control_motion import tracking_arrays
from direct_weight_model import (
    FEATURE_NAMES as TRACKING_NAMES, _blocks, features_from_arrays, predict_from_logits,
)
from path_guard import check_paths
from sequence_decoder import team_of_each_track
from stage1_cv_baseline import digest, private_path
from tagging_evidence import Restart, TaggingConfig, apply_restarts, restart_events, team_names

LABELS = np.array(['H', 'A', 'D'])
SCHEMA = 'possession-gbdt-v1'


@dataclass(frozen=True)
class Config:
    trees: int = 120
    leaves: int = 15
    learning_rate: float = .05
    min_leaf: int = 40
    switch_cost: float = 4.
    seed: int = 17

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.trees, self.leaves, self.min_leaf)):
            raise ValueError('Tree counts and sizes must be positive integers')
        if not np.isfinite([self.learning_rate, self.switch_cost]).all() or not 0 < self.learning_rate <= 1 or self.switch_cost < 0:
            raise ValueError('Invalid learning rate or transition cost')


def feature_table(players, ball, teams, pitch, blocks, fps, restarts=()):
    """Observed geometry, missingness, local context and soft tagging features."""
    base = features_from_arrays(players, ball, list(teams), pitch, blocks, fps)
    blocks = _blocks(blocks, len(ball))
    players, ball = np.asarray(players), np.asarray(ball)
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    columns = {name: base[:, i] for i, name in enumerate(TRACKING_NAMES)}
    nearest = []
    for team, prefix in [('H', 'home'), ('A', 'away')]:
        values = np.sort(distance[:, np.asarray(teams) == team], axis=1)
        for rank in range(2):
            d = values[:, rank].copy() if values.shape[1] > rank else np.full(len(ball), np.inf)
            d[~np.isfinite(d)] = np.nan
            columns[f'{prefix}_distance_{rank+1}'] = d
        nearest.append(columns[f'{prefix}_distance_1'])
        columns[f'{prefix}_nearby_count'] = (values < 3.).sum(axis=1)
        columns[f'{prefix}_visible_count'] = np.isfinite(players[:, np.asarray(teams) == team]).all(axis=2).sum(axis=1)
    columns['distance_advantage_m'] = nearest[1] - nearest[0]
    evidence = np.full(len(ball), None, dtype=object)
    home_distance, away_distance = [np.where(np.isfinite(d), d, np.inf) for d in nearest]
    close = np.minimum(home_distance, away_distance) <= 3.
    evidence[close & (home_distance < away_distance)] = 'H'
    evidence[close & (away_distance < home_distance)] = 'A'
    outside = np.isfinite(ball).all(axis=1) & ((ball < -1).any(axis=1) | (ball > np.asarray(pitch)+1).any(axis=1))
    evidence[outside] = 'D'
    # Scale existing tagging timescales from their 25 fps convention.
    tag_config = asdict(TaggingConfig())
    for key in ('settle_frames', 'reappear_gap', 'speed_max_gap', 'in_play_frames',
                'min_stoppage', 'max_stoppage', 'max_kickoff_stoppage', 'prior_frames'):
        tag_config[key] = max(1, round(tag_config[key]*fps/25))
    positions = [tuple(p) if np.isfinite(p).all() else None for p in ball]
    tagged, _ = apply_restarts(evidence, positions, blocks, restarts, TaggingConfig(**tag_config), fps)
    tagged = np.asarray(tagged)
    columns['tag_available'] = np.full(len(ball), bool(restarts), dtype=float)
    for label in LABELS:
        columns[f'tag_added_{label}'] = ((tagged == label) & (evidence != label)).astype(float)
    frame = pd.DataFrame(columns)
    # Centred context is intentional: this is offline match processing.
    for name in ('home_distance_1', 'away_distance_1', 'distance_advantage_m',
                 'ball_speed', 'ball_observed', 'outside_distance'):
        for seconds in (.4, 2.):
            out = np.full(len(ball), np.nan)
            for _, first, last in blocks:
                out[first:last+1] = frame[name].iloc[first:last+1].rolling(
                    max(1, round(seconds*fps)), center=True, min_periods=1).mean()
            frame[f'{name}_mean_{seconds}s'] = out
    return frame


def load_input(manifest_path, training=False):
    """Load observation files only; targets are accessed separately for fitting."""
    path = Path(manifest_path).resolve()
    if training:
        check_paths([path])
    manifest = json.loads(path.read_text())
    files = {key: (path.parent / value).resolve() for key, value in manifest['files'].items()}
    if training:
        check_paths(files.values())
        synthetic = manifest.get('format') == 'synthetic-match-v2'
        eligible = manifest.get('training_eligible') is True and manifest.get('supplied_match') is False
        if not (synthetic or eligible):
            raise ValueError('Training requires synthetic data or explicit other-match provenance')
        if not synthetic and not manifest.get('match_id'):
            raise ValueError('Other-match training requires match_id')
    n = manifest['n_frames']
    frame_start = manifest.get('frame_start', 0)
    if type(frame_start) is not int or frame_start < 0:
        raise ValueError('frame_start must be a nonnegative video frame')
    blocks = _blocks(manifest['blocks'], n)
    fps, pitch = manifest['fps'], manifest['pitch']
    if 'observations' in files:
        with np.load(files['observations'], allow_pickle=False) as data:
            players, ball, teams = data['players'], data['ball'], data['teams'].tolist()
            if not np.array_equal(data['frame_ids'], frame_start + np.arange(n)):
                raise ValueError('CV arrays must match the manifest video frame interval')
    else:
        if frame_start:
            raise ValueError('Nonzero frame_start requires a windowed observation NPZ')
        teams = team_of_each_track(files['atd_meta'])
        players, ball = tracking_arrays(files['atd_raw'], teams, n, pitch)
    if len(ball) != n:
        raise ValueError('Manifest and observation lengths differ')
    restarts = []
    if 'tagging' in files:
        if manifest.get('camera', 'tactical') != manifest.get('tagging_camera', 'tactical'):
            raise ValueError('Tagging and observations must share a camera timeline')
        if files['tagging'].suffix.lower() == '.xml':
            events = []
            for event in ET.parse(files['tagging']).iter('instance'):
                events.append(dict(code=event.findtext('code'),
                    start_frame=round(float(event.findtext('start'))*fps)-frame_start,
                    end_frame=round(float(event.findtext('end'))*fps)-frame_start,
                    tags={tag.findtext('group'): tag.findtext('text') for tag in event.findall('label')}))
            names = manifest.get('team_names') or team_names(files['atd_meta'])
            restarts = restart_events(events, names)
            restarts = [Restart(max(0, r.start), min(n, r.end), r.kind, r.team)
                        for r in restarts if r.start < n and r.end > 0]
        else:
            restarts = [Restart(**event) for event in json.loads(files['tagging'].read_text())]
        for event in restarts:
            if (type(event.start) is not int or type(event.end) is not int or
                    event.start < 0 or event.end <= event.start or event.end > n or
                    event.team not in ('H', 'A', None)):
                raise ValueError('Invalid restart interval or team')
        restarts.sort(key=lambda event: (event.start, event.end))
    x = feature_table(players, ball, teams, pitch, blocks, fps, restarts)
    return dict(x=x, blocks=blocks, files=files, manifest=manifest,
                manifest_path=str(path), fps=fps, frames=frame_start + np.arange(n))


def targets(table):
    data = pd.read_csv(table['files']['gt'], keep_default_na=False)
    if list(data.columns) != ['frame', 'label'] or not np.array_equal(data.frame, table.get('frames', np.arange(len(table['x'])))) or not data.label.isin(['H', 'A', 'D', 'X']).all():
        raise ValueError('Targets must be complete zero-based frame,label CSV with H/A/D/X')
    y = np.full(len(data), -1, dtype=int)
    for code, label in enumerate(LABELS):
        y[data.label == label] = code
    active = np.zeros(len(y), bool)
    for _, first, last in table['blocks']:
        active[first:last+1] = True
    y[~active] = -1
    return y


def fit(manifests, output, config=Config()):
    tables = [load_input(path, training=True) for path in manifests]
    return fit_tables(tables, output, config)


def fit_tables(tables, output, config=Config()):
    """Fit already-extracted tables whose provenance was validated by load_input."""
    import lightgbm as lgb
    output = private_path(output)
    if output.exists():
        raise ValueError('Use a new model directory')
    if not tables:
        raise ValueError('At least one training match is required')
    x = pd.concat([t['x'] for t in tables], ignore_index=True)
    y = np.concatenate([targets(t) for t in tables])
    selected = y >= 0
    if set(y[selected]) != {0, 1, 2}:
        raise ValueError('Training data must contain H, A and explicit D labels')
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3,
        n_estimators=config.trees, num_leaves=config.leaves,
        learning_rate=config.learning_rate, min_child_samples=config.min_leaf,
        random_state=config.seed, n_jobs=1, deterministic=True, force_col_wise=True,
        verbosity=-1)
    model.fit(x.loc[selected], y[selected])
    receipt = dict(schema=SCHEMA, config=asdict(config), feature_names=list(x.columns),
        labels=LABELS.tolist(), lightgbm_version=lgb.__version__,
        code_sha256=digest(__file__), training=[])
    for table in tables:
        used = {k: digest(v) for k, v in table['files'].items()
                if k in ('observations', 'atd_raw', 'atd_meta', 'tagging', 'gt')}
        receipt['training'].append(dict(manifest_sha256=digest(table['manifest_path']),
            match_id=table['manifest'].get('match_id'), seed=table['manifest'].get('seed'),
            files_sha256=used))
    output.mkdir(parents=True)
    model.booster_.save_model(str(output/'model.txt'))
    receipt['model_sha256'] = digest(output/'model.txt')
    (output/'model.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def predict(table, model_dir):
    import lightgbm as lgb
    folder = Path(model_dir)
    receipt = json.loads((folder/'model.json').read_text())
    if receipt['schema'] != SCHEMA or receipt['labels'] != LABELS.tolist() or receipt['feature_names'] != list(table['x'].columns):
        raise ValueError('Model feature or label schema mismatch')
    if digest(folder/'model.txt') != receipt['model_sha256']:
        raise ValueError('Model checksum mismatch')
    config = Config(**receipt['config'])
    model = lgb.Booster(model_file=str(folder/'model.txt'))
    probabilities = model.predict(table['x'], num_threads=1)
    if probabilities.shape != (len(table['x']), 3) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid model probabilities')
    # Scale emission evidence by elapsed time relative to the 25 fps reference.
    logits = np.log(np.clip(probabilities, 1e-12, 1))*25/table['fps']
    transitions = np.full((3, 3), -config.switch_cost)
    np.fill_diagonal(transitions, 0.)
    weights = dict(transition_logits=transitions, initial_logits=np.zeros(3))
    labels = predict_from_logits(logits, weights, table['blocks'])
    return probabilities, labels


def write_predictions(manifest, model, output):
    output = private_path(output)
    if output.exists():
        raise ValueError('Use a new prediction directory')
    table = load_input(manifest)
    probabilities, labels = predict(table, model)
    output.mkdir(parents=True)
    pd.DataFrame({'frame': table['frames'], 'label': labels}).to_csv(output/'possession.csv', index=False)
    np.savez_compressed(output/'probabilities.npz', probabilities=probabilities,
                        labels=LABELS, frame_ids=table['frames'])
    (output/'receipt.json').write_text(json.dumps(dict(
        manifest_sha256=digest(manifest), model_sha256=digest(Path(model)/'model.txt'),
        code_sha256=digest(__file__), frames=len(labels),
        frame_start=int(table['frames'][0]), frame_end=int(table['frames'][-1])), indent=2)+'\n')
    return dict(output=str(output), frames=len(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--manifests', nargs='+', required=True)
    train.add_argument('--out', required=True)
    train.add_argument('--config', help='JSON Config overrides; choose on development data')
    infer = commands.add_parser('predict')
    infer.add_argument('--manifest', required=True)
    infer.add_argument('--model', required=True)
    infer.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        config = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
        receipt = fit(args.manifests, args.out, config)
        print(json.dumps(dict(output=args.out, training_matches=len(receipt['training']))))
    else:
        print(json.dumps(write_predictions(args.manifest, args.model, args.out)))


if __name__ == '__main__':
    main()
