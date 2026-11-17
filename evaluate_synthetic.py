"""One locked frozen-model audit on fresh synthetic seeds; never model search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'challenge'), str(ROOT/'challenge/starter'), str(ROOT/'challenge/evaluation')]
import numpy as np
from control_motion import tracking_arrays
from decoder_experiments import (generate_bundles, load_scenarios, score, load_seed_ledger,
                                check_seed_ledger, update_seed_ledger)
from possession_gbdt import load_input, predict
from predict_frozen_tracking import predict_arrays
from sequence_decoder import team_of_each_track
from stage1_cv_baseline import digest, private_path


def run(out, ledger_path, seeds):
    out = private_path(out)
    if out.exists():
        raise ValueError('Fresh audit directory required')
    ledger = load_seed_ledger(ledger_path)
    check_seed_ledger(ledger, [], seeds)
    if set(seeds) & (set(ledger['used_dev']) | set(range(230001, 230111))):
        raise ValueError('Audit seeds must never have been development/training seeds')
    scenarios = load_scenarios(str(ROOT/'scripts/scenarios_default.json'), n_frames=12000)
    model_files = list((ROOT/'models').rglob('*'))
    pins = {str(p.relative_to(ROOT)): digest(p) for p in model_files if p.is_file()}
    code = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *(ROOT/'scripts').glob('*.py')]}
    out.mkdir(parents=True)
    protocol = dict(seeds=seeds, heldout=True, training=False, model_selection=False,
                    deployment_choice='Frozen cycle018 TCN/cycle005 tree chosen before audit; no retuning',
                    synthetic=True, scenarios={k: asdict(v) for k, v in scenarios.items()},
                    model_sha256=pins, source_sha256=code, ledger_before_sha256=digest(ledger_path))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    results = []
    for name, config in scenarios.items():
        for bundle in generate_bundles(seeds, config, str(out/'cases'/name)):
            teams = team_of_each_track(bundle.paths['atd_meta'])
            players, ball = tracking_arrays(bundle.paths['atd_raw'], teams, bundle.n_frames, bundle.pitch, dtype=np.float64)
            folder = ROOT/'models/ensemble'
            labels = dict(ensemble=predict_arrays(players, ball, teams, bundle.pitch, bundle.blocks, bundle.fps,
                           folder/'weights.json', folder/'tree.joblib', folder/'receipt.json'),
                          gbdt=predict(load_input(bundle.paths['manifest']), ROOT/'models/gbdt')[1],
                          baseline=bundle.baseline_labels(50))
            measured = {arm: {k: float(value[k]) for k in ('spell_f1_mean', 'macro_f1', 'accuracy')}
                        for arm, prediction in labels.items() for value in [score(bundle.gt_codes, prediction)]}
            results.append(dict(scenario=name, seed=bundle.seed, metrics=measured))
            np.savez_compressed(out/(name+'-'+str(bundle.seed)+'.npz'), target=bundle.gt_codes, **labels)
            (out/'partial.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(results[-1]), flush=True)
    for p, expected in {**pins, **code}.items():
        if digest(ROOT/p) != expected:
            raise ValueError('Locked model/source changed')
    aggregate = {arm: {metric: float(np.mean([r['metrics'][arm][metric] for r in results]))
                       for metric in ('spell_f1_mean', 'macro_f1', 'accuracy')}
                 for arm in ('ensemble', 'gbdt', 'baseline')}
    update_seed_ledger(ledger_path, [], seeds)
    report = dict(protocol=protocol, results=results, aggregate=aggregate,
                  runtime_seconds=time.monotonic()-started, ledger_after_sha256=digest(ledger_path),
                  limitation='Synthetic scores do not establish real-match accuracy; two seed clusters only')
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[450101, 450102])
    args = parser.parse_args()
    run(args.out, args.ledger, args.seeds)
