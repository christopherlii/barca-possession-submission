"""Observed-input team-handover head and structured H/A transitions.

The auxiliary training target is a change between H and A, never a test input.
Dead-ball transitions keep the fixed prior. Correct weighted-BCE logits for the
training positive weight before interpreting them as switch probabilities.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from direct_weight_model import _blocks
from path_guard import check_paths
from possession_tcn import HIDDEN,RADIUS,numpy_logits,validate_weights

POSITIVE_WEIGHT=30.


def team_switch_targets(labels):
    y=np.asarray(labels)
    if y.ndim!=2 or not np.isin(y,[0,1,2,3]).all():raise ValueError('invalid training labels')
    target=np.zeros(y.shape,np.float32);valid=np.zeros(y.shape,bool)
    valid[:,1:]=(y[:,1:]<3)&(y[:,:-1]<3)
    target[:,1:]=(y[:,1:]<2)&(y[:,:-1]<2)&(y[:,1:]!=y[:,:-1])
    return target,valid


def validate_head(head):
    if set(head)!= {'weight','bias'}:raise ValueError('invalid boundary tensor names')
    w,b=np.asarray(head['weight'],np.float32),np.asarray(head['bias'],np.float32)
    if w.shape!=(HIDDEN,) or b.shape!=(1,) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('invalid boundary tensors')
    return dict(weight=w,bias=b)


def save_head(path,head,controller_path):
    check_paths([path,controller_path]);head=validate_head(head)
    payload=dict(schema=1,target='H-A-adjacent-change-v1',parameter_count=HIDDEN+1,
                 controller_sha256=hashlib.sha256(Path(controller_path).read_bytes()).hexdigest(),
                 positive_weight=POSITIVE_WEIGHT,tensors={k:v.tolist() for k,v in head.items()})
    Path(path).write_text(json.dumps(payload,indent=2)+'\n')


def load_head(path,controller_path):
    check_paths([path,controller_path])
    if Path(path).stat().st_size>8192:raise ValueError('oversized boundary head')
    data=json.loads(Path(path).read_text())
    if (data.get('schema')!=1 or data.get('target')!='H-A-adjacent-change-v1' or
        data.get('parameter_count')!=HIDDEN+1 or data.get('positive_weight')!=POSITIVE_WEIGHT or
        data.get('controller_sha256')!=hashlib.sha256(Path(controller_path).read_bytes()).hexdigest()):
        raise ValueError('boundary declaration differs')
    return validate_head(data['tensors'])


def boundary_predictions(features,weights,head,blocks,chunk=4096):
    weights=validate_weights(weights);head=validate_head(head);x=np.asarray(features)
    if chunk<1:raise ValueError('invalid chunk')
    probabilities=np.full((len(x),3),1/3.,np.float32)
    switches=np.zeros(len(x),np.float32)
    for _,first,last in _blocks(blocks,len(x)):
        for start in range(first,last+1,chunk):
            end=min(last+1,start+chunk);left=max(first,start-RADIUS);right=min(last+1,end+RADIUS)
            hidden=numpy_logits(x[left:right],weights,return_hidden=True)[start-left:end-left]
            with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                logits=hidden@weights['head.weight'][:,:,0].T+weights['head.bias']
                switch=hidden@head['weight']+head['bias'][0]-np.log(POSITIVE_WEIGHT)
            if not np.isfinite(logits).all() or not np.isfinite(switch).all():
                raise ValueError('non-finite boundary activation')
            p=np.exp(logits-logits.max(1,keepdims=True))
            probabilities[start:end]=p/p.sum(1,keepdims=True)
            switches[start:end]=np.exp(-np.logaddexp(0.,-switch))
    return probabilities,switches


def decode_team_switch(probabilities,switches,blocks,team_scale=2.,dead_cost=16.,team_cost_floor=None):
    p=np.asarray(probabilities,float);q=np.asarray(switches,float)
    if (p.ndim!=2 or p.shape[1]!=3 or q.shape!=(len(p),) or not np.isfinite(p).all()
        or not np.isfinite(q).all() or (p<0).any() or not np.allclose(p.sum(1),1.) or
        (q<0).any() or (q>1).any() or not np.isfinite(team_scale) or team_scale<=0 or
        not np.isfinite(dead_cost) or dead_cost<0 or
        (team_cost_floor is not None and (not np.isfinite(team_cost_floor) or team_cost_floor<0))):
        raise ValueError('invalid boundary decoding inputs')
    q=np.clip(q,1e-6,1-1e-6)
    costs=team_scale*(np.log(2.)+np.log1p(-q)-np.log(q))
    if team_cost_floor is not None:costs=np.maximum(costs,float(team_cost_floor))
    emissions=np.log(np.maximum(p,1e-12));output=np.full(len(p),'X',dtype='<U1')
    for _,first,last in _blocks(blocks,len(p)):
        values=emissions[first].copy();back=[]
        for t in range(first+1,last+1):
            c=costs[t]
            transitions=np.array([[0.,-c,-dead_cost],[-c,0.,-dead_cost],[-dead_cost,-dead_cost,0.]])
            candidates=values[:,None]+transitions
            indices=candidates.argmax(0);back.append(indices)
            values=candidates.max(0)+emissions[t]
        state=int(values.argmax())
        for t in range(last,first-1,-1):
            output[t]=('H','A','D')[state]
            if t>first:state=int(back[t-first-1][state])
    return output
