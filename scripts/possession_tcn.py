"""Frozen submission inference: possession tcn."""
import numpy as np


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

