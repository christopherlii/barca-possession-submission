"""Locked v1/v2 observation-policy comparison on fresh synthetic match seeds."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'challenge'),str(ROOT/'challenge/starter'),str(ROOT/'challenge/evaluation')]
import numpy as np
from control_motion import tracking_arrays
from decoder_experiments import generate_bundles,load_scenarios,score,load_seed_ledger,check_seed_ledger,update_seed_ledger
from observation_gap_mask import gap_mask
from possession_observation_quality import QualityConfig,select_observations
from predict_frozen_tracking import predict_arrays
from sequence_decoder import team_of_each_track
from stage1_cv_baseline import private_path,digest


def run(out,ledger,development):
    out=private_path(out);development=Path(development)
    if out.exists():raise ValueError('Use a new output directory')
    dev=json.loads(development.read_text());candidate=QualityConfig(unknown_airborne_policy='allow_uncertain')
    if not dev['passed'] or dev['protocol']['config']!=asdict(candidate):raise ValueError('Development gate/config mismatch')
    seeds=[450201,450202];state=load_seed_ledger(ledger);check_seed_ledger(state,[],seeds)
    if set(seeds)&set(state['used_dev']):raise ValueError('Previously used development seeds')
    code={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*(ROOT/'scripts').glob('*.py')]}
    models={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'models/ensemble').iterdir() if p.is_file()}
    protocol=dict(seeds=seeds,config=asdict(candidate),development_sha256=digest(development),
        code_sha256=code,model_sha256=models,ledger_before_sha256=digest(ledger),training=False,synthetic=True,
        observation_seeds='match_seed+100 gaps, +200 noise, +300 noise placement',frames=12000)
    out.mkdir(parents=True);(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.monotonic();rows=[]
    for name,config in load_scenarios(str(ROOT/'scripts/scenarios_default.json'),n_frames=12000).items():
        for b in generate_bundles(seeds,config,str(out/'cases'/name)):
            teams=team_of_each_track(b.paths['atd_meta'])
            players,ball=tracking_arrays(b.paths['atd_raw'],teams,b.n_frames,b.pitch,dtype=np.float64)
            baseline=ball.copy();baseline[gap_mask(b.n_frames,b.blocks,250,b.seed+100)]=np.nan
            rng=np.random.default_rng(b.seed+200);cv=ball+rng.normal(0,.15,ball.shape)
            noisy=gap_mask(b.n_frames,b.blocks,5,b.seed+300,share=.12)
            cv[noisy]+=rng.normal(0,8.,(int(noisy.sum()),2))
            confidence=np.where(noisy,.1,.9);sigma=np.where(noisy,8.,.2);unknown=np.full(b.n_frames,np.nan)
            metrics={};predictions={};counts={}
            for arm,policy in [('incumbent',QualityConfig()),('candidate',candidate)]:
                points,q=select_observations(baseline,cv,confidence,sigma,unknown,b.blocks,policy)
                counts[arm]=int(q['cv_used'].sum());folder=ROOT/'models/ensemble'
                pred=predict_arrays(players,points,teams,b.pitch,b.blocks,b.fps,folder/'weights.json',folder/'tree.joblib',folder/'receipt.json')
                predictions[arm]=pred;m=score(b.gt_codes,pred)
                metrics[arm]={k:float(m[k]) for k in ('spell_f1_mean','macro_f1','accuracy')}
            row=dict(scenario=name,seed=b.seed,metrics=metrics,cv_fills=counts);rows.append(row)
            np.savez_compressed(out/(name+'-'+str(b.seed)+'.npz'),target=b.gt_codes,**predictions)
            (out/'partial.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    aggregate={a:{k:float(np.mean([r['metrics'][a][k] for r in rows])) for k in ('spell_f1_mean','macro_f1','accuracy')} for a in ('incumbent','candidate')}
    checks=dict(mean_spell_gain=aggregate['candidate']['spell_f1_mean']>=aggregate['incumbent']['spell_f1_mean']+.01,
        frame_metrics=all(aggregate['candidate'][k]>=aggregate['incumbent'][k]-.01 for k in ('macro_f1','accuracy')),
        scenarios=all(np.mean([r['metrics']['candidate']['spell_f1_mean']-r['metrics']['incumbent']['spell_f1_mean'] for r in rows if r['scenario']==s])>=-.02 for s in set(r['scenario'] for r in rows)))
    for p,h in {**code,**models}.items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen code or model changed')
    update_seed_ledger(ledger,[],seeds)
    result=dict(protocol=protocol,results=rows,aggregate=aggregate,gate={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),runtime_seconds=time.monotonic()-start,ledger_after_sha256=digest(ledger))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(aggregate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--ledger',required=True);p.add_argument('--development',required=True)
    a=p.parse_args();run(a.out,a.ledger,a.development)
