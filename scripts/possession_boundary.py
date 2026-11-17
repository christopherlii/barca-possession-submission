"""Frozen submission inference: possession boundary."""
import numpy as np


from direct_weight_model import _blocks


POSITIVE_WEIGHT=30.


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

