"""Short image-space motion estimates; predictions are never detections.

Constant-velocity fits reset at cuts, abrupt direction changes and long gaps.
They cannot know a future kick, deflection or camera move. Offline smoothing
uses only short, bracketed gaps whose observed endpoints support the motion.
"""
import numpy as np


class BallMotion:
    def __init__(self, max_gap=5, window=5, gate=40.):
        self.max_gap, self.window, self.gate = max_gap, window, gate
        self.history = []

    def forecast(self, frame):
        if len(self.history)<2 or not 0<frame-self.history[-1][0]<=self.max_gap:
            return None
        times=np.array([r[0] for r in self.history],float)
        points=np.array([r[1] for r in self.history],float)
        dt=times-times[-1]
        fit=np.linalg.lstsq(np.c_[np.ones(len(dt)),dt],points,rcond=None)[0]
        residual=float(np.sqrt(np.mean((np.c_[np.ones(len(dt)),dt]@fit-points)**2)))
        if residual>self.gate/2 or np.linalg.norm(fit[1])>80:
            return None
        return fit[0]+fit[1]*(frame-times[-1])

    def update(self, candidates, guide, frame, cut=False):
        if cut or (self.history and frame-self.history[-1][0]>self.max_gap):
            self.history=[]
        expected=self.forecast(frame)
        points=np.asarray(candidates,float).reshape(-1,3)
        chosen=None
        if len(points):
            reference=expected if expected is not None else guide
            if reference is None or not np.isfinite(reference).all():
                chosen=points[np.argmax(points[:,2])]
            else:
                distances=np.linalg.norm(points[:,:2]-reference,axis=1)
                j=np.argmin(distances)
                if expected is None or distances[j]<=self.gate:
                    chosen=points[j]
                elif guide is not None and np.isfinite(guide).all():
                    # Abrupt redirection supported by a new image observation
                    # close to current tracking: reacquire, do not smooth a kick.
                    j=np.argmin(np.linalg.norm(points[:,:2]-guide,axis=1))
                    if np.linalg.norm(points[j,:2]-guide)<=self.gate/2:
                        self.history=[];chosen=points[j]
        if chosen is not None:
            self.history.append((frame,chosen[:2].copy()))
            self.history=self.history[-self.window:]
        return chosen.tolist() if chosen is not None else None, expected.tolist() if expected is not None else None


def smooth_path(rows, segments, max_gap=5):
    """Return short-gap interpolation only; retain direct detection coordinates."""
    result=[dict(r) for r in rows]
    for i,r in enumerate(result):
        r['context_position']=r['selected'][:2] if r['selected'] else r.get('motion_prediction')
        r['context_source']='detected' if r['selected'] else 'predicted' if r.get('motion_prediction') is not None else 'missing'
    for i in range(1,len(rows)):
        if rows[i]['selected'] or not rows[i-1]['selected']:continue
        j=i
        while j<len(rows) and not rows[j]['selected'] and segments[j]==segments[i-1]:j+=1
        if j==len(rows) or j-i>max_gap or segments[j]!=segments[i-1]:continue
        left=np.array(rows[i-1]['selected'][:2]);right=np.array(rows[j]['selected'][:2])
        if np.linalg.norm(right-left)/(j-i+1)>80:continue
        pred=rows[j].get('motion_prediction')
        if pred is None or np.linalg.norm(np.array(pred)-right)>40:continue
        for k in range(i,j):
            result[k]['context_position']=(left+(right-left)*(k-i+1)/(j-i+1)).tolist()
            result[k]['context_source']='interpolated'
    return result
