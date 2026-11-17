"""Image-projected pitch geometry and persistent static-candidate evidence.

Homography projects known field locations to pixels; ball motion stays in the
image. Geometry is evidence, not a pitch mask: no candidate is excluded merely
because it is near a marking or outside the field. Nearby players exempt a
stationary candidate, preserving plausible placed balls. No identity/labels.
"""
import numpy as np
from fuse_tactical_cv import transform

CONFIG=dict(marking_distance_px=10.,stationary_residual_px=3.,
            persistence_frames=6,player_distance_px=40.)


def project_geometry(h,pitch=(105.3,68.)):
    if h is None or not np.isfinite(h).all():return []
    length,width=pitch
    shapes=[]
    def line(name,xy):
        pixels=transform(np.asarray(xy)/pitch,h)
        if np.isfinite(pixels).all():shapes.append(dict(name=name,points=pixels))
    line('touchlines',[(0,0),(length,0),(length,width),(0,width),(0,0)])
    line('halfway',[(length/2,0),(length/2,width)])
    for side,x in [('left',0.),('right',length)]:
        direction=1 if x==0 else -1
        for distance,halfwidth,label in [(16.5,20.16,'penalty_area'),(5.5,9.16,'goal_area')]:
            line(side+'_'+label,[(x,width/2-halfwidth),(x+direction*distance,width/2-halfwidth),
                 (x+direction*distance,width/2+halfwidth),(x,width/2+halfwidth)])
        line(side+'_penalty_spot',[(x+direction*11,width/2)])
    line('centre_spot',[(length/2,width/2)])
    angles=np.linspace(0,2*np.pi,73)
    line('centre_circle',np.c_[length/2+9.15*np.cos(angles),width/2+9.15*np.sin(angles)])
    # Only the part of each penalty arc outside the penalty area is painted.
    arc=np.linspace(-np.arccos(5.5/9.15),np.arccos(5.5/9.15),37)
    line('left_penalty_arc',np.c_[11+9.15*np.cos(arc),width/2+9.15*np.sin(arc)])
    line('right_penalty_arc',np.c_[length-11-9.15*np.cos(arc),width/2+9.15*np.sin(arc)])
    for x in [0,length]:
        for y in [0,width]:
            line('corner_'+str(x)+'_'+str(y),[(x,y)])
            theta=np.linspace(0,np.pi/2,19)
            line('arc_corner_'+str(x)+'_'+str(y),np.c_[x+(1 if x==0 else -1)*np.cos(theta),y+(1 if y==0 else -1)*np.sin(theta)])
    return shapes


def is_field_boundary(name):
    return name == 'touchlines' or name.startswith(('corner_', 'arc_corner_'))


def closest_marking(point,shapes,ignore_field_boundary=False):
    best=(float('inf'),None)
    for shape in shapes:
        if ignore_field_boundary and is_field_boundary(shape['name']):
            continue
        p=shape['points']
        if len(p)==1:distance=float(np.linalg.norm(point-p[0]))
        else:
            a,b=p[:-1],p[1:];v=b-a
            t=np.clip(np.sum((point-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-9),0,1)
            distance=float(np.linalg.norm(point-(a+t[:,None]*v),axis=1).min())
        if distance<best[0]:best=(distance,shape['name'])
    return best


class StaticMarkingEvidence:
    def __init__(self,config=None):
        self.config=dict(CONFIG if config is None else config);self.previous=[]

    def update(self,candidates,geometry,affine,player_boxes,cut=False):
        from image_ball_gate import affine_points
        if cut or affine is None:self.previous=[]
        points=np.asarray(candidates,float).reshape(-1,3);details=[];current=[];used=set()
        boxes=np.asarray(player_boxes,float).reshape(-1,4)
        feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
        for point in points:
            distance,mark=closest_marking(point[:2],geometry,self.config.get('ignore_field_boundary',False))
            near=distance<=self.config['marking_distance_px']
            player_near=bool(len(feet) and np.linalg.norm(feet-point[:2],axis=1).min()<=self.config['player_distance_px'])
            streak=1
            if near and affine is not None:
                matches=[]
                for j,old in enumerate(self.previous):
                    if j not in used and old['mark']==mark:
                        residual=np.linalg.norm(point[:2]-affine_points(old['point'],affine))
                        if residual<=self.config['stationary_residual_px']:matches.append((residual,j))
                if matches:
                    _,j=min(matches);used.add(j);streak=self.previous[j]['streak']+1
            suspicious=bool(near and streak>=self.config['persistence_frames'] and not player_near)
            if near:current.append(dict(point=point[:2].copy(),mark=mark,streak=streak))
            details.append(dict(marking=mark if near else None,distance_px=distance if mark else None,
                stationary_frames=streak if near else 0,near_player=player_near,rejected=suspicious))
        self.previous=current
        keep=np.array([not r['rejected'] for r in details],bool)
        return points[keep],details


PERSISTENT_CONFIG=dict(CONFIG,memory_frames=25,contact_distance_px=12.,
                       contact_grace_frames=12,arrival_grace_frames=50,
                       association_distance_px=12.)


class PersistentMarkingEvidence:
    """Remember stationary distractors across detector flicker, with bounded grace.

    Player proximity alone never provides an indefinite exemption. A short grace
    covers close contact; longer grace needs observed movement before stopping.
    This can reject a genuinely stationary ball without enough movement evidence.
    """
    def __init__(self,config=None):
        self.config=dict(PERSISTENT_CONFIG,**(config or {}));self.memory=[]

    def update(self,candidates,geometry,affine,player_boxes,cut=False):
        from image_ball_gate import affine_points
        c=self.config
        if cut or affine is None:self.memory=[]
        for old in self.memory:
            old['point']=affine_points(old['point'],affine)
            old['age']+=1;old['grace']=max(0,old['grace']-1)
        self.memory=[r for r in self.memory if r['age']<=c['memory_frames']]
        points=np.asarray(candidates,float).reshape(-1,3)
        boxes=np.asarray(player_boxes,float).reshape(-1,4)
        feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
        details=[];used=set()
        for point in points:
            distance,mark=closest_marking(point[:2],geometry,c.get('ignore_field_boundary',False))
            near=distance<=c['marking_distance_px']
            player_distance=float(np.linalg.norm(feet-point[:2],axis=1).min()) if len(feet) else float('inf')
            close=player_distance<=c['contact_distance_px']
            matches=[(float(np.linalg.norm(point[:2]-r['point'])),j) for j,r in enumerate(self.memory)
                     if j not in used]
            match=min(matches) if matches else (float('inf'),None)
            if match[0]<=c['association_distance_px']:
                residual,j=match;used.add(j);record=self.memory[j]
                moving=residual>c['stationary_residual_px']
                if moving:
                    record['hits']=1
                    if close:record['grace']=c['arrival_grace_frames']
                elif near:record['hits']+=1
                record.update(point=point[:2].copy(),age=0)
            else:
                record=dict(point=point[:2].copy(),age=0,hits=1,
                            grace=c['contact_grace_frames'] if close else 0)
                self.memory.append(record);used.add(len(self.memory)-1)
            rejected=bool(near and record['hits']>=c['persistence_frames'] and record['grace']==0)
            details.append(dict(marking=mark if near else None,distance_px=distance if mark else None,
                stationary_frames=record['hits'] if near else 0,near_player=player_distance<=c['player_distance_px'],
                contact_grace_frames=record['grace'],rejected=rejected,
                reason='persistent_static_marking' if rejected else None))
        return points[np.array([not d['rejected'] for d in details],bool)],details
