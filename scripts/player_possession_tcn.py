"""Stable-player temporal encoding, invariant team pooling, global possession TCN."""
import json
from pathlib import Path

import numpy as np

from direct_weight_model import _blocks
from path_guard import check_paths
from player_track_features import ACTOR_RADIUS
from possession_tcn import (shapes as global_shapes,torch_model as global_model,
                            numpy_logits,RADIUS as GLOBAL_RADIUS)
from possession_boundary import POSITIVE_WEIGHT

WIDTH=16
DILATIONS=(1,5,25)
RADIUS=GLOBAL_RADIUS+ACTOR_RADIUS
assert sum(DILATIONS)==ACTOR_RADIUS


def shapes():
    result={'actor.proj.weight':(WIDTH,20,1),'actor.proj.bias':(WIDTH,),
            'boundary.weight':(1,32,1),'boundary.bias':(1,)}
    for i in range(3):
        prefix=f'actor.layers.{i}.'
        result.update({prefix+'dw.weight':(WIDTH,1,3),prefix+'dw.bias':(WIDTH,),
                       prefix+'pw.weight':(WIDTH,WIDTH,1),prefix+'pw.bias':(WIDTH,)})
    result.update({'controller.'+k:v for k,v in global_shapes(120).items()})
    return result


def torch_model():
    import torch
    from torch import nn

    class Separable(nn.Module):
        def __init__(self,d):
            super().__init__();self.dw=nn.Conv1d(WIDTH,WIDTH,3,padding=d,dilation=d,groups=WIDTH)
            self.pw=nn.Conv1d(WIDTH,WIDTH,1)

        def forward(self,x):return torch.relu(self.pw(self.dw(x)))

    class Actor(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Conv1d(20,WIDTH,1)
            self.layers=nn.ModuleList([Separable(d) for d in DILATIONS])

        def forward(self,x):
            x=torch.relu(self.proj(x))
            for layer in self.layers:x=x+.25*layer(x)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.actor=Actor();self.controller=global_model(120)
            self.boundary=nn.Conv1d(32,1,1)
            nn.init.zeros_(self.boundary.weight);nn.init.constant_(self.boundary.bias,-2.)

        def forward(self,global_features,actors,presence,teams):
            batch,slots,_,frames=actors.shape
            encoded=self.actor(actors.reshape(batch*slots,20,frames)).reshape(batch,slots,WIDTH,frames)
            pooled=[]
            for team in (0,1):
                valid=presence&(teams[:,:,None]==team)
                count=valid.sum(1)[:,None,:]
                mean=(encoded*valid[:,:,None,:]).sum(1)/count.clamp_min(1)
                maximum=encoded.masked_fill(~valid[:,:,None,:],-1e9).amax(1)
                maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
                pooled.extend([mean,maximum])
            hidden=self.controller.encode(torch.cat([global_features]+pooled,dim=1))
            return self.controller.head(hidden),self.boundary(hidden)[:,0]

    return Model()


def validate(weights):
    expected=shapes()
    if set(weights)!=set(expected):raise ValueError('player model tensor names differ')
    result={}
    for key,shape in expected.items():
        a=np.asarray(weights[key],np.float32)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid player tensor '+key)
        result[key]=a
    return result


def save_weights(path,model):
    check_paths([path]);w=validate({k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    data=dict(schema=1,kind='stable-player-tcn-v1',actor_layout='player20-float16-v1',
        global_columns=56,parameter_count=sum(a.size for a in w.values()),receptive_radius=RADIUS,
        positive_weight=POSITIVE_WEIGHT,tensors={k:v.tolist() for k,v in w.items()})
    Path(path).write_text(json.dumps(data)+'\n')


def load_weights(path):
    check_paths([path])
    if Path(path).stat().st_size>2*1024*1024:raise ValueError('oversized player model')
    data=json.loads(Path(path).read_text())
    if (data.get('schema')!=1 or data.get('kind')!='stable-player-tcn-v1' or
        data.get('actor_layout')!='player20-float16-v1' or data.get('global_columns')!=56 or
        data.get('receptive_radius')!=RADIUS or data.get('positive_weight')!=POSITIVE_WEIGHT):
        raise ValueError('invalid player model declaration')
    w=validate(data['tensors'])
    if data.get('parameter_count')!=sum(a.size for a in w.values()):raise ValueError('player parameter count differs')
    return w


def raw_logits(global_features,actors,presence,teams,weights):
    w=validate(weights);g=np.asarray(global_features,np.float32)
    # The same float16 input representation is used for raw and cached inference.
    a=np.asarray(actors,np.float16).astype(np.float32);m=np.asarray(presence,bool);t=np.asarray(teams)
    if (a.ndim!=3 or a.shape[1]<1 or a.shape[2]!=20 or g.shape!=(len(a),56) or m.shape!=a.shape[:2]
        or t.shape!=(a.shape[1],) or not np.isin(t,[-1,0,1]).all() or
        not np.isfinite(g).all() or not np.isfinite(a).all()):raise ValueError('invalid player model inputs')
    def pointwise(x,prefix):
        with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
            result=x@w[prefix+'.weight'][:,:,0].T+w[prefix+'.bias']
        if not np.isfinite(result).all():raise ValueError('non-finite player activation')
        return result
    x=np.maximum(pointwise(a,'actor.proj'),0.)
    for i,d in enumerate(DILATIONS):
        prefix=f'actor.layers.{i}';dw=w[prefix+'.dw.weight'][:,0,:]
        z=x*dw[:,1]+w[prefix+'.dw.bias']
        if d<len(x):z[d:]+=x[:-d]*dw[:,0];z[:-d]+=x[d:]*dw[:,2]
        x=x+.25*np.maximum(pointwise(z,prefix+'.pw'),0.)
    pooled=[]
    for team in (0,1):
        valid=m&(t[None,:]==team);count=valid.sum(1)[:,None]
        mean=(x*valid[:,:,None]).sum(1)/np.maximum(count,1)
        maximum=np.where(valid[:,:,None],x,-1e9).max(1)
        maximum=np.where(count>0,maximum,0.)
        pooled.extend([mean.astype(np.float32),maximum])
    combined=np.column_stack([g]+pooled)
    cw={k.removeprefix('controller.'):v for k,v in w.items() if k.startswith('controller.')}
    hidden=numpy_logits(combined,cw,return_hidden=True)
    return np.column_stack((pointwise(hidden,'controller.head'),pointwise(hidden,'boundary')))


def probabilities(global_features,actors,presence,teams,weights,blocks,chunk=2048):
    if chunk<1:raise ValueError('invalid chunk size')
    n=len(global_features);p=np.full((n,3),1/3.,np.float32);q=np.zeros(n,np.float32)
    for _,first,last in _blocks(blocks,n):
        for start in range(first,last+1,chunk):
            end=min(last+1,start+chunk);left=max(first,start-RADIUS);right=min(last+1,end+RADIUS)
            logits=raw_logits(global_features[left:right],actors[left:right],presence[left:right],teams,weights)
            logits=logits[start-left:end-left]
            state=np.exp(logits[:,:3]-logits[:,:3].max(1,keepdims=True))
            p[start:end]=state/state.sum(1,keepdims=True)
            q[start:end]=np.exp(-np.logaddexp(0.,-(logits[:,3]-np.log(POSITIVE_WEIGHT))))
    return p,q
