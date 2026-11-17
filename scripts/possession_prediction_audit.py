"""Label-blind validation and paired comparison of frozen possession outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage1_cv_baseline import ROOT, digest

INFERENCE_FILES = ('possession_gbdt.py', 'direct_weight_model.py', 'control_motion.py',
                   'sequence_decoder.py', 'tagging_evidence.py', 'cv_to_possession.py')


def validate(folder, frames, probability_atol=1e-8):
    folder = Path(folder)
    frames = np.asarray(frames)
    csv = pd.read_csv(folder/'possession.csv')
    if list(csv.columns) != ['frame', 'label'] or not np.array_equal(csv['frame'], frames):
        raise ValueError('CSV columns or exact ordered frame IDs differ')
    if csv['frame'].dtype.kind not in 'iu' or csv['frame'].duplicated().any():
        raise ValueError('Frame IDs must be unique integers')
    labels = csv['label'].to_numpy()
    if not np.isin(labels, ['H', 'A', 'D']).all():
        raise ValueError('Clip labels must be H/A/D only')
    with np.load(folder/'probabilities.npz', allow_pickle=False) as data:
        p = data['probabilities']
        if (not np.array_equal(data['frame_ids'], frames) or data['frame_ids'].dtype.kind not in 'iu'
                or not np.array_equal(data['labels'], ['H', 'A', 'D'])):
            raise ValueError('Probability class order or frame IDs differ')
        if (p.shape != (len(frames), 3) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(1), 1., atol=probability_atol, rtol=0)):
            raise ValueError('Probabilities must be finite normalized N x 3')
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    run_labels = labels[edges[:-1]]
    return dict(frames=len(frames), frame_start=int(frames[0]), frame_end=int(frames[-1]),
                label_counts={k: int((labels == k).sum()) for k in ('H', 'A', 'D')},
                label_transitions=int(len(edges)-2),
                direct_team_transitions=int(((labels[1:] != labels[:-1]) &
                                             np.isin(labels[1:], ['H', 'A']) &
                                             np.isin(labels[:-1], ['H', 'A'])).sum()),
                possession_spells=int(np.isin(run_labels, ['H', 'A']).sum()),
                dead_ball_runs=int((run_labels == 'D').sum()),
                class_order=['H', 'A', 'D'], probabilities_valid=True,
                output_sha256={p.name: digest(p) for p in (folder/'possession.csv', folder/'probabilities.npz')},
                accuracy_measured=False, full_match_submission=False)


def compare(first, second):
    """Descriptive comparison only; refuses model, decoder and clock drift.

Both inputs need the handoff validation receipt, including source observation
coverage. Nothing here reads truth or calls disagreement an accuracy delta.
"""
    first, second = Path(first), Path(second)
    reports = [json.loads((p/'validation.json').read_text()) for p in (first, second)]
    for key in ('model_sha256', 'model_receipt_sha256', 'decoder_configuration', 'inference_code_sha256',
                'feature_names', 'player_tracks_sha256', 'tagging_sha256', 'blocks', 'fps', 'class_order'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('Paired comparison differs in '+key)
    a, b = (pd.read_csv(p/'possession.csv') for p in (first, second))
    if not np.array_equal(a['frame'], b['frame']):
        raise ValueError('Paired prediction coverage differs')
    for path, report in zip((first, second), reports):
        checked = validate(path, a['frame'].to_numpy())
        if checked['output_sha256'] != report['output_sha256']:
            raise ValueError('Prediction changed since validation')
    return dict(frames=len(a), disagreeing_frames=int((a['label'] != b['label']).sum()),
                disagreement_fraction=float((a['label'] != b['label']).mean()),
                missing_observation_fractions=[r['missing_observation_fraction'] for r in reports],
                accuracy_measured=False, interpretation='Prediction disagreement and coverage differences are not accuracy loss.')


def attach_validation(manifest_path, model_folder, output):
    """Attach comparison metadata to a newly inferred observation-manifest clip.

This validates inference identities and files; historical CV source verification
is separately performed by the adapter/handoff. It never opens target labels.
"""
    manifest_path, model_folder, output = Path(manifest_path), Path(model_folder), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if set(manifest['files']) != {'observations', 'tagging'}:
        raise ValueError('Expected an observation-only adapter manifest')
    model = json.loads((model_folder/'model.json').read_text())
    receipt = json.loads((output/'receipt.json').read_text())
    current = {name: digest(ROOT/'scripts'/name) for name in INFERENCE_FILES}
    if (receipt['manifest_sha256'] != digest(manifest_path)
            or receipt['model_sha256'] != digest(model_folder/'model.txt')
            or model['model_sha256'] != receipt['model_sha256']
            or receipt['code_sha256'] != current['possession_gbdt.py']
            or model['code_sha256'] != current['possession_gbdt.py']
            or manifest['adapter_sha256'] != current['cv_to_possession.py']):
        raise ValueError('Manifest/model/implementation identity changed')
    observations = manifest_path.parent/manifest['files']['observations']
    tags = manifest_path.parent/manifest['files']['tagging']
    frames = manifest.get('frame_start', 0)+np.arange(manifest['n_frames'])
    with np.load(observations, allow_pickle=False) as data:
        if not np.array_equal(data['frame_ids'], frames):
            raise ValueError('Observation clock differs')
        observed = np.isfinite(data['ball']).all(1)
        if not np.array_equal(observed, data['ball_observed']):
            raise ValueError('Observation visibility differs from missing coordinates')
        players_hash = hashlib.sha256(data['players'].tobytes()+data['teams'].tobytes()).hexdigest()
    result = validate(output, frames)
    # The legacy prediction receipt pins model.txt but not model.json. Re-run
    # the current frozen forward/decoder to prove this configuration produces
    # these exact saved outputs, rather than infer provenance from file names.
    from possession_gbdt import load_input, predict
    probabilities, labels = predict(load_input(manifest_path), model_folder)
    with np.load(output/'probabilities.npz', allow_pickle=False) as data:
        if not np.allclose(data['probabilities'], probabilities, atol=1e-12, rtol=0):
            raise ValueError('Frozen forward probabilities do not reproduce')
    if not np.array_equal(pd.read_csv(output/'possession.csv')['label'], labels):
        raise ValueError('Frozen decoder does not reproduce saved labels')
    result.update(model_sha256=model['model_sha256'], model_receipt_sha256=digest(model_folder/'model.json'),
                  decoder_configuration=model['config'], feature_names=model['feature_names'],
                  inference_code_sha256=current, player_tracks_sha256=players_hash,
                  tagging_sha256=digest(tags), blocks=manifest['blocks'], fps=manifest['fps'],
                  observed_ball_frames=int(observed.sum()), missing_ball_frames=int((~observed).sum()),
                  missing_observation_fraction=float((~observed).mean()),
                  input_sha256={str(p):digest(p) for p in (manifest_path,observations,tags,model_folder/'model.json',model_folder/'model.txt')},
                  training_performed=False, public_labels_read=False, frozen_forward_verified=True)
    (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', nargs='?')
    parser.add_argument('second', nargs='?')
    parser.add_argument('--manifest')
    parser.add_argument('--model')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.manifest or args.model or args.output:
        if not all((args.manifest,args.model,args.output)) or args.first or args.second:
            parser.error('Use --manifest, --model and --output together without comparison arguments')
        result = attach_validation(args.manifest,args.model,args.output)
        print(json.dumps({k:result[k] for k in ('frames','label_counts','missing_observation_fraction')},indent=2))
    else:
        if not args.first or not args.second:
            parser.error('Provide two prediction directories')
        print(json.dumps(compare(args.first, args.second), indent=2))
