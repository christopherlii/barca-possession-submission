"""Observed-only stable-player trajectories; no team/carrier/target label input.

20 columns per track, quantized identically to float16 in fitting and inference.
Presence masks use observations in the actor encoder's 31-frame radius only.
"""
import numpy as np

from direct_weight_model import _blocks

ACTOR_RADIUS=31
VECTOR_COLUMNS=(4,5,6,7,8,9,10,11)


def player_track_features(players,ball,pitch,blocks,fps=25.):
    p=np.asarray(players,np.float32);b=np.asarray(ball,np.float32)
    if (p.ndim!=3 or p.shape[2]!=2 or b.shape!=(len(p),2) or
        np.isinf(p).any() or np.isinf(b).any() or not np.isfinite(fps) or fps<=0 or
        len(pitch)!=2 or not np.isfinite(pitch).all() or min(pitch)<=0):
        raise ValueError('invalid observed player trajectories')
    features=np.zeros((len(p),p.shape[1],20),np.float16)
    presence=np.zeros(p.shape[:2],bool);lag=max(1,round(.2*fps))
    for _,first,last in _blocks(blocks,len(p)):
        a,ball=p[first:last+1],b[first:last+1];n=len(a)
        pk=np.isfinite(a).all(2);bk=np.isfinite(ball).all(1)
        clean=np.where(pk[:,:,None],a,0.)
        vp0=np.zeros_like(a);vp1=np.zeros_like(a);vb0=np.zeros_like(ball);vb1=np.zeros_like(ball)
        kp0=np.zeros(pk.shape,bool);kp1=kp0.copy();kb0=np.zeros(n,bool);kb1=kb0.copy()
        if lag<n:
            kp=pk[lag:]&pk[:-lag];kb=bk[lag:]&bk[:-lag]
            dv=np.where(kp[:,:,None],(a[lag:]-a[:-lag])*fps/lag,0.)
            db=np.where(kb[:,None],(ball[lag:]-ball[:-lag])*fps/lag,0.)
            vp0[lag:]=dv;vp1[:-lag]=dv;kp0[lag:]=kp;kp1[:-lag]=kp
            vb0[lag:]=db;vb1[:-lag]=db;kb0[lag:]=kb;kb1[:-lag]=kb
        out=np.zeros((n,a.shape[1],20),np.float32)
        out[:,:,0]=pk;out[:,:,1]=bk[:,None];out[:,:,2]=kp0;out[:,:,3]=kp1
        out[:,:,4:6]=np.where(pk[:,:,None],np.clip(2*clean/np.asarray(pitch)-1.,-2.,2.),0.)
        out[:,:,6:8]=np.clip(vp0/12.,-2.,2.);out[:,:,8:10]=np.clip(vp1/12.,-2.,2.)
        valid=pk&bk[:,None];relative=np.where(valid[:,:,None],ball[:,None,:]-a,0.)
        distance=np.linalg.norm(relative,axis=2);denom=np.maximum(distance,.1)
        out[:,:,10:12]=np.clip(relative/8.,-2.,2.)
        out[:,:,12]=np.where(valid,np.clip(1-distance/10.,0.,1.),0.)
        for j,(vp,vb,kp,kb) in enumerate(((vp0,vb0,kp0,kb0),(vp1,vb1,kp1,kb1))):
            known=valid&kp&kb[:,None];error=vb[:,None,:]-vp
            out[:,:,13+j]=np.where(known,np.clip((error*relative).sum(2)/denom/40.,-2.,2.),0.)
            out[:,:,15+j]=np.where(known,np.minimum(np.linalg.norm(error,axis=2)/40.,2.),0.)
            out[:,:,18+j]=known
        heading=(vp0+vp1)/np.maximum(kp0.astype(np.float32)+kp1,1.)[:,:,None]
        speed=np.linalg.norm(heading,axis=2);alignment=np.zeros(pk.shape,np.float32)
        np.divide((heading*relative).sum(2),speed*denom,out=alignment,where=valid&(speed>.5))
        out[:,:,17]=np.clip(alignment,-1.,1.)
        if not np.isfinite(out).all():raise ValueError('non-finite player features')
        features[first:last+1]=out.astype(np.float16)
        summed=np.concatenate([np.zeros((1,a.shape[1]),np.int32),np.cumsum(pk,axis=0,dtype=np.int32)])
        t=np.arange(n);left=np.maximum(0,t-ACTOR_RADIUS);right=np.minimum(n,t+ACTOR_RADIUS+1)
        presence[first:last+1]=(summed[right]-summed[left])>0
    return features,presence


def swap_player_features(features):
    x=np.asarray(features)
    if x.ndim!=3 or x.shape[2]!=20 or not np.isfinite(x).all():raise ValueError('invalid player feature layout')
    result=x.copy();result[:,:,VECTOR_COLUMNS]*=-1
    return result
