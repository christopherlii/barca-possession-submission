"""Bound candidate reacquisition by the last trusted image-space observation.

The anchor survives Kalman expiry. It moves only with measured camera motion
until a plausible detection updates it. After 50 missing frames it expires;
reacquisition then starts a separate track. Missing detections cannot relocate it.
"""
import numpy as np
from image_ball_gate import affine_points

CONFIG=dict(base_radius_px=18.,speed_margin_px_frame=8.,max_speed_px_frame=48.,
            max_reacquire_distance_px=160.,max_innovation_px=80.,anchor_expiry_frames=50)


class ContinuityGuard:
    def __init__(self,config=None):
        self.config=dict(CONFIG,**(config or {}));self.reset()

    def reset(self):
        self.anchor=None;self.velocity=np.zeros(2);self.elapsed=0

    def filter(self,candidates,affine,cut=False):
        if cut:self.reset()
        if self.anchor is not None and affine is not None:
            self.anchor=affine_points(self.anchor,affine)
            self.velocity=affine[:,:2]@self.velocity
        self.elapsed+=1
        if self.elapsed>self.config['anchor_expiry_frames']:
            self.reset()
        points=np.asarray(candidates,float).reshape(-1,3)
        if self.anchor is None:return points,0
        c=self.config;dt=self.elapsed;speed=np.linalg.norm(self.velocity)
        reach=min(c['max_reacquire_distance_px'],c['base_radius_px']+(min(speed+c['speed_margin_px_frame'],c['max_speed_px_frame']))*dt)
        innovation=min(c['max_innovation_px'],c['base_radius_px']+8*np.sqrt(dt))
        expected=self.anchor+self.velocity*dt
        keep=(np.linalg.norm(points[:,:2]-self.anchor,axis=1)<=reach)&(np.linalg.norm(points[:,:2]-expected,axis=1)<=innovation)
        return points[keep],int((~keep).sum())

    def observe(self,point):
        if point is None:return
        p=np.asarray(point[:2],float)
        if self.anchor is not None:
            measured=(p-self.anchor)/max(1,self.elapsed)
            self.velocity=.5*self.velocity+.5*measured
        self.anchor=p.copy();self.elapsed=0
