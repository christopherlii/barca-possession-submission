"""Small temporal controller: optional Torch fitting, NumPy inference.

Forty observed tracking/motion inputs (20,355 parameters), optionally sixteen
additional ball-direction features (20,867), or 40 more relative-actor inputs
(22,147). Receptive field is 1,021 frames.
No label or decoder-prediction input.
"""
import json
from pathlib import Path

import numpy as np

from direct_weight_model import _blocks
from path_guard import check_paths

INPUTS=40
HIDDEN=32
DILATIONS=(1,2,4,8,16,32,64,128)
RADIUS=2*sum(DILATIONS)


def shapes(inputs=INPUTS):
    if inputs not in (40,56,96,120):raise ValueError('unsupported TCN feature count')
    result={'proj.weight':(HIDDEN,inputs,1),'proj.bias':(HIDDEN,),
            'head.weight':(3,HIDDEN,1),'head.bias':(3,)}
    for block in range(len(DILATIONS)):
        for layer in range(2):
            prefix=f'blocks.{block}.{layer}.'
            result.update({prefix+'dw.weight':(HIDDEN,1,3),prefix+'dw.bias':(HIDDEN,),
                prefix+'pw.weight':(HIDDEN,HIDDEN,1),prefix+'pw.bias':(HIDDEN,)})
    return result


def torch_model(inputs=INPUTS):
    """Import training backend only when explicitly requested by a coordinator."""
    import torch
    from torch import nn
    shapes(inputs)

    class Separable(nn.Module):
        def __init__(self,dilation):
            super().__init__()
            self.dw=nn.Conv1d(HIDDEN,HIDDEN,3,padding=dilation,dilation=dilation,groups=HIDDEN)
            self.pw=nn.Conv1d(HIDDEN,HIDDEN,1)

        def forward(self,x):
            return torch.relu(self.pw(self.dw(x)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj=nn.Conv1d(inputs,HIDDEN,1)
            self.blocks=nn.ModuleList([nn.Sequential(Separable(d),Separable(d)) for d in DILATIONS])
            self.head=nn.Conv1d(HIDDEN,3,1)

        def encode(self,x):
            x=torch.relu(self.proj(x))
            for block in self.blocks:x=x+.25*block(x)
            return x

        def forward(self,x):
            return self.head(self.encode(x))

    return Model()


def validate_weights(weights):
    if 'proj.weight' not in weights:raise ValueError('missing TCN projection')
    projection=np.asarray(weights['proj.weight'])
    if projection.ndim!=3 or projection.shape[0]!=HIDDEN or projection.shape[2]!=1:
        raise ValueError('invalid TCN projection shape')
    expected=shapes(projection.shape[1])
    if set(weights)!=set(expected):raise ValueError('TCN tensor names differ')
    result={}
    for key,shape in expected.items():
        array=np.asarray(weights[key],dtype=np.float32)
        if array.shape!=shape or not np.isfinite(array).all():
            raise ValueError('invalid TCN tensor '+key)
        result[key]=array
    return result


def save_weights(path,model):
    check_paths([path])
    weights=validate_weights({key:value.detach().cpu().numpy() for key,value in model.state_dict().items()})
    inputs=weights['proj.weight'].shape[1]
    Path(path).write_text(json.dumps({'schema':1,'input_count':inputs,
        'feature_layout':{40:'tracking16-motion24-v1',56:'tracking16-motion24-touch16-v1',
                          96:'tracking16-motion24-touch16-actor40-v1',
                          120:'global56-playerpool64-v1'}[inputs],
        'parameter_count':sum(a.size for a in weights.values()),
        'tensors':{key:value.tolist() for key,value in weights.items()}})+'\n')


def load_weights(path):
    check_paths([path])
    if Path(path).stat().st_size>2*1024*1024:raise ValueError('oversized TCN file')
    data=json.loads(Path(path).read_text())
    if data.get('schema')!=1 or data.get('input_count') not in (40,56,96,120):
        raise ValueError('invalid TCN schema')
    if data['input_count']==56 and data.get('feature_layout')!='tracking16-motion24-touch16-v1':
        raise ValueError('unknown TCN touch feature layout')
    if data['input_count']==96 and data.get('feature_layout')!='tracking16-motion24-touch16-actor40-v1':
        raise ValueError('unknown TCN actor feature layout')
    if data['input_count']==120 and data.get('feature_layout')!='global56-playerpool64-v1':
        raise ValueError('unknown TCN player-pool feature layout')
    weights=validate_weights(data['tensors'])
    if weights['proj.weight'].shape[1]!=data['input_count']:
        raise ValueError('TCN input declaration differs from projection')
    if data.get('parameter_count')!=sum(a.size for a in weights.values()):
        raise ValueError('TCN parameter count differs')
    return weights


def numpy_logits(features,weights,return_hidden=False):
    """Exact same-padded depthwise/pointwise convolution graph as Torch model."""
    weights=validate_weights(weights)
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[1]!=weights['proj.weight'].shape[1] or not np.isfinite(features).all():
        raise ValueError('invalid TCN observed features')
    def pointwise(x,prefix):
        # Check actual values explicitly, including when a BLAS backend raises
        # floating-point flags despite producing finite results.
        with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
            result=x@weights[prefix+'.weight'][:,:,0].T+weights[prefix+'.bias']
        if not np.isfinite(result).all():raise ValueError('non-finite TCN activation')
        return result
    x=np.maximum(pointwise(features,'proj'),0.)
    for block,dilation in enumerate(DILATIONS):
        z=x
        for layer in range(2):
            prefix=f'blocks.{block}.{layer}'
            w=weights[prefix+'.dw.weight'][:,0,:]
            depth=z*w[:,1]+weights[prefix+'.dw.bias']
            if dilation<len(z):
                depth[dilation:]+=z[:-dilation]*w[:,0]
                depth[:-dilation]+=z[dilation:]*w[:,2]
            z=np.maximum(pointwise(depth,prefix+'.pw'),0.)
        x=x+.25*z
    return x if return_hidden else pointwise(x,'head')


def probabilities(features,weights,blocks,chunk=4096):
    """Half-isolated chunked inference, preserving complete receptive context."""
    if chunk<1:raise ValueError('invalid inference chunk')
    x=np.asarray(features)
    output=np.full((len(x),3),1/3.,np.float32)
    for _,first,last in _blocks(blocks,len(x)):
        for start in range(first,last+1,chunk):
            end=min(last+1,start+chunk)
            left=max(first,start-RADIUS);right=min(last+1,end+RADIUS)
            logits=numpy_logits(x[left:right],weights)[start-left:end-left]
            logits-=logits.max(axis=1,keepdims=True)
            p=np.exp(logits)
            output[start:end]=p/p.sum(axis=1,keepdims=True)
    return output
