"""Pixel-space ball prediction with image-estimated pan/zoom compensation.

No ball-to-pitch projection. Affine camera motion is estimated from background
optical flow. The circular gate uses innovation covariance, bounded to a fixed
pixel range. Predicted points are separate from observations and expire quickly.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from stage1_cv_baseline import ROOT, private_path, digest
from local_cv_possession import atomic

CONFIG=dict(min_radius_px=15.,max_radius_px=90.,sigma_gate=3.,measurement_sigma_px=5.,
    process_sigma_px=2.,max_velocity_px_frame=80.,max_acceleration_px_frame2=10.,
    prediction_frames=5,expiry_frames=12,reacquire_frames=3)


def affine_points(points, affine):
    return np.asarray(points)@affine[:,:2].T+affine[:,2]


class PixelGate:
    def __init__(self,config=None):
        self.config=dict(CONFIG if config is None else config)
        self.x=None;self.p=None;self.missed=0;self.pending=[];self.track=0

    def reset(self):
        self.x=self.p=None;self.missed=0;self.pending=[];self.adaptive_noise=1.

    def update(self,candidates,affine=None,camera_error=0.,cut=False):
        if cut:self.reset()
        a=np.array([[1.,0,0],[0,1.,0]]) if affine is None else affine
        points=np.asarray(candidates,float).reshape(-1,3)
        points=points[np.isfinite(points).all(1)]
        c=self.config
        self.pending=[affine_points(p,a) for p in self.pending]
        center=None;radius=None;accepted=None;forecast=None;ellipse=None
        r=np.eye(2)*c['measurement_sigma_px']**2
        if self.x is not None:
            linear=a[:,:2]
            self.x[:2]=affine_points(self.x[:2],a)
            self.x[2:]=linear@self.x[2:]
            transport=np.zeros((4,4));transport[:2,:2]=linear;transport[2:,2:]=linear
            self.p=transport@self.p@transport.T
            f=np.eye(4);f[0,2]=f[1,3]=1.
            self.x=f@self.x
            q=c['process_sigma_px']**2*getattr(self,'adaptive_noise',1.)
            noise=np.diag([q+camera_error**2,q+camera_error**2,q,q])
            if c.get('adaptive_process',False) and np.linalg.norm(self.x[2:])>1:
                direction=self.x[2:]/np.linalg.norm(self.x[2:])
                along=q*np.outer(direction,direction)
                noise[:2,:2]+=along;noise[2:,2:]+=along
            self.p=f@self.p@f.T+noise
            center=self.x[:2].copy()
            innovation_cov=self.p[:2,:2]+r
            eigenvalues,eigenvectors=np.linalg.eigh(innovation_cov)
            axes=np.clip(c['sigma_gate']*np.sqrt(np.maximum(eigenvalues,1e-6)),c['min_radius_px'],c['max_radius_px'])
            radius=float(axes.max())
            if c.get('ellipse',False):ellipse=(axes,eigenvectors)
            if len(points):
                delta=points[:,:2]-center
                errors=(np.linalg.norm((delta@eigenvectors)/axes,axis=1)
                        if ellipse is not None else np.linalg.norm(delta,axis=1)/radius)
                j=int(np.argmin(errors))
                if errors[j]<=1.:
                    accepted=points[j].copy()
                    innovation=accepted[:2]-self.x[:2]
                    if c.get('adaptive_process',False):
                        nis=float(innovation@np.linalg.solve(innovation_cov,innovation))
                        self.adaptive_noise=float(np.clip(.75*getattr(self,'adaptive_noise',1.)+.25*max(1.,nis/2),1.,6.))
                    # A robust isotropic measurement update remains inside the
                    # circular test above; an outlier never enlarges its own gate.
                    k=self.p[:,:2]@np.linalg.inv(self.p[:2,:2]+r)
                    old_velocity=self.x[2:].copy()
                    self.x+=k@innovation
                    dv=self.x[2:]-old_velocity
                    n=np.linalg.norm(dv)
                    if n>c['max_acceleration_px_frame2']:
                        self.x[2:]=old_velocity+dv*c['max_acceleration_px_frame2']/n
                    n=np.linalg.norm(self.x[2:])
                    if n>c['max_velocity_px_frame']:self.x[2:]*=c['max_velocity_px_frame']/n
                    ident=np.eye(4);ident[:,:2]-=k
                    self.p=ident@self.p@ident.T+k@r@k.T
                    self.missed=0
            if accepted is None:self.missed+=1
            if self.missed>c['expiry_frames']:self.reset()
        if self.x is None:
            if len(points):
                if self.pending:
                    distance=np.linalg.norm(points[:,:2]-self.pending[-1],axis=1)
                    j=int(np.argmin(distance))
                    if distance[j]>c['max_velocity_px_frame']:
                        self.pending=[]
                else:j=int(np.argmax(points[:,2]))
                self.pending.append(points[j,:2].copy())
                self.pending=self.pending[-c['reacquire_frames']:]
                if len(self.pending)==c['reacquire_frames']:
                    v=(self.pending[-1]-self.pending[0])/(len(self.pending)-1)
                    self.x=np.r_[self.pending[-1],v]
                    self.p=np.diag([25.,25.,16.,16.]);self.track+=1;self.missed=0
                    accepted=points[j].copy();self.pending=[]
            else:self.pending=[]
        position=None
        if self.x is not None and self.missed<=c['prediction_frames']:
            position=self.x[:2].copy();forecast=self.x[:2]+self.x[2:]*3
        return dict(position=position,accepted=accepted,center=center,radius=radius,
            ellipse=ellipse,
            forecast=forecast,source='gated_detection' if accepted is not None else 'predicted' if position is not None else 'missing',
            track=self.track,rejected=len(points)-int(accepted is not None))


def background_motion(previous,current,boxes,scale):
    import cv2
    mask=np.full(previous.shape,255,np.uint8)
    for box in boxes:
        x,y,xx,yy=np.rint(np.asarray(box)*scale).astype(int)
        cv2.rectangle(mask,(x-3,y-3),(xx+3,yy+3),0,-1)
    p=cv2.goodFeaturesToTrack(previous,maxCorners=180,qualityLevel=.02,minDistance=7,mask=mask)
    if p is None or len(p)<12:return None,None
    q,ok,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,winSize=(21,21),maxLevel=3)
    if q is None:return None,None
    back,ok2,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3)
    if back is None:return None,None
    valid=ok[:,0].astype(bool)&ok2[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)
    if valid.sum()<12:return None,None
    matrix,inliers=cv2.estimateAffinePartial2D(p[valid],q[valid],method=cv2.RANSAC,ransacReprojThreshold=2.)
    if matrix is None or inliers.sum()<10:return None,None
    zoom=np.sqrt(abs(np.linalg.det(matrix[:,:2])))
    if not .9<zoom<1.1:return None,None
    residual=np.linalg.norm(affine_points(p[valid,0],matrix)-q[valid,0],axis=1)
    err=float(np.median(residual[inliers[:,0].astype(bool)])/scale)
    matrix[:,2]/=scale
    return matrix,err


def run(cache,out):
    import cv2
    cv2.setNumThreads(2);cv2.setRNGSeed(109)
    cache,out=private_path(cache),private_path(out)
    if out.exists():raise ValueError('Use a new output directory')
    config=json.loads((cache/'config.json').read_text())
    rows=[json.loads(r) for r in (cache/'detections.jsonl').read_text().splitlines()]
    features=ROOT/'outputs/two-stage-cv-view/tactical-fused-v1.npz'
    if digest(features)!=config['features_sha256']:raise ValueError('Feature identity changed')
    with np.load(features,allow_pickle=False) as z:
        frames=z['frame_ids'];segments=z['segments'];offsets=z['player_offsets'];boxes=z['player_boxes'];meta=json.loads(str(z['metadata']))
    if [r['frame'] for r in rows]!=frames.tolist():raise ValueError('Source cache not complete')
    out.mkdir(parents=True)
    config.update(image_gate=CONFIG,context_precomputed=True,temporal=True,
        coordinate_space='image pixels',camera_compensation='background LK flow + RANSAC partial affine',
        homography_used_for_ball=False,gate_code_sha256=digest(__file__),
        parent_cache_sha256=digest(cache/'detections.jsonl'),new_detector_inference=False)
    atomic(out/'config.json',config)
    cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    tracker=PixelGate();previous=None;result=[];camera_good=0;scale=480/meta['width']
    try:
        for i,row in enumerate(rows):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Video decode failed')
            gray=cv2.cvtColor(cv2.resize(bgr,(480,round(meta['height']*scale))),cv2.COLOR_BGR2GRAY)
            cut=i==0 or segments[i]!=segments[i-1]
            a,error=(None,None) if cut else background_motion(previous,gray,boxes[offsets[i-1]:offsets[i]],scale)
            previous=gray
            if a is not None:camera_good+=1
            # Failure to estimate camera motion resets, rather than interpreting
            # unknown camera displacement as an enormous ball velocity.
            state=tracker.update(row['candidates'],a,error or 0.,cut or a is None)
            polygon=None
            if state['center'] is not None:
                angles=np.linspace(0,2*np.pi,49)
                polygon=(state['center']+state['radius']*np.c_[np.cos(angles),np.sin(angles)]).tolist()
            def serial(v):return v.tolist() if v is not None else None
            new=dict(row,raw_selected=row['selected'],selected=serial(state['accepted']),
                context_position=serial(state['position']),context_source=state['source'],
                forecast_3_frames=serial(state['forecast']),reachable_polygon=polygon,
                gate_track=state['track'],gate_rejected=state['rejected'],
                camera_motion_valid=a is not None,gate_radius_px=state['radius'])
            result.append(new)
            if (i+1)%250==0:print(f'{i+1}/{len(rows)} image-space frames',flush=True)
    finally:cap.release()
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    atomic(out/'status.json',dict(state='complete',completed=len(rows),total=len(rows)))
    radii=[r['gate_radius_px'] for r in result if r['gate_radius_px'] is not None]
    summary=dict(frames=len(rows),sources=dict(Counter(r['context_source'] for r in result)),
        valid_camera_motion_frames=camera_good,radius_range_px=[min(radii),max(radii)] if radii else None,
        rejected_candidates=sum(r['gate_rejected'] for r in result),accuracy_measured=False)
    atomic(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.out)
