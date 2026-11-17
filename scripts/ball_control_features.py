"""Frozen submission inference: ball control features."""
import numpy as np


from direct_weight_model import _blocks


NAMES=('backward_velocity_known','forward_velocity_known','backward_vx','backward_vy',
       'forward_vx','forward_vy','velocity_change','turning_angle')


def ball_control_features(ball,blocks,fps=25.,span_seconds=.2):
    ball=np.asarray(ball,dtype=float)
    if (ball.ndim!=2 or ball.shape[1]!=2 or np.isinf(ball).any() or
            not np.isfinite(fps) or fps<=0 or not np.isfinite(span_seconds) or span_seconds<=0):
        raise ValueError('invalid observed ball trajectory or frame rate')
    output=np.zeros((len(ball),len(NAMES)),np.float32)
    lag=max(1,round(span_seconds*fps))
    for _,first,last in _blocks(blocks,len(ball)):
        points=ball[first:last+1]
        known=np.isfinite(points).all(axis=1)
        before=np.zeros(points.shape);after=np.zeros(points.shape)
        back=np.zeros(len(points),bool);ahead=back.copy()
        if lag<len(points):
            valid=known[lag:]&known[:-lag]
            velocity=np.where(valid[:,None],(points[lag:]-points[:-lag])*fps/lag,0.)
            before[lag:]=velocity;after[:-lag]=velocity
            back[lag:]=valid;ahead[:-lag]=valid
        out=output[first:last+1]
        out[:,0]=back;out[:,1]=ahead
        out[:,2:4]=np.clip(before/40.,-1,1)
        out[:,4:6]=np.clip(after/40.,-1,1)
        both=back&ahead
        out[:,6]=np.where(both,np.minimum(np.linalg.norm(after-before,axis=1)/40.,1),0.)
        speed_before=np.linalg.norm(before,axis=1);speed_after=np.linalg.norm(after,axis=1)
        turning=both&(speed_before>1.)&(speed_after>1.)
        cosine=np.zeros(len(points))
        np.divide((before*after).sum(axis=1),speed_before*speed_after,out=cosine,where=turning)
        out[:,7]=np.where(turning,(1-np.clip(cosine,-1,1))/2,0.)
    if not np.isfinite(output).all():raise ValueError('non-finite ball-control features')
    return output

