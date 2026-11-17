"""Extra YOLO searches from camera-compensated motion and nearby player feet.

Search proposals use algorithmic inputs only. They are not observations: all
returned points must pass the caller's marking, motion and continuity checks.
"""
import numpy as np

CONFIG=dict(max_crops=3,nearby_player_radius_px=260.,context_radius_px=128.,
            detail_radius_px=72.,foot_radius_px=88.,max_sweep_px=240.,imgsz=640)


def proposals(width,height,center=None,velocity=None,player_boxes=(),guide=None,config=None):
    c=dict(CONFIG,**(config or {}));plans=[]
    center=np.asarray(center if center is not None else guide,float) if center is not None or guide is not None else None
    if center is None or not np.isfinite(center).all():return []
    velocity=np.zeros(2) if velocity is None else np.asarray(velocity,float)
    end=center+np.clip(velocity*3,-c['max_sweep_px'],c['max_sweep_px'])

    def add(kind,lo,hi):
        lo=np.maximum([0,0],np.floor(lo)).astype(int);hi=np.minimum([width,height],np.ceil(hi)).astype(int)
        box=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
        if min(hi-lo)>=32 and not any(p['box']==box for p in plans):plans.append(dict(kind=kind,box=box))

    r=c['context_radius_px'];add('motion_sweep',np.minimum(center,end)-r,np.maximum(center,end)+r)
    boxes=np.asarray(player_boxes,float).reshape(-1,4)
    feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
    distances=np.linalg.norm(feet-center,axis=1)
    for j in np.argsort(distances)[:2]:
        if distances[j]>c['nearby_player_radius_px']:continue
        r=c['foot_radius_px'];add('player_feet',feet[j]-[r,r],feet[j]+[r,r*.4])
    if len(plans)<c['max_crops']:
        r=c['detail_radius_px'];add('motion_detail',center-r,center+r)
    return plans[:c['max_crops']]


def detect_recovery(model,image,plans,device):
    """Batch crops at a fixed network size; restore original pixel coordinates."""
    from local_cv_tiling import merge_points
    if not plans:return np.empty((0,3))
    crops=[image[p['box'][1]:p['box'][3],p['box'][0]:p['box'][2]] for p in plans]
    results=model.predict(crops,classes=[32],conf=.1,iou=.7,imgsz=CONFIG['imgsz'],
                          device=device,verbose=False)
    points=[]
    for result,plan in zip(results,plans):
        xy=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
        offset=np.array(plan['box'][:2])
        points.append(np.c_[(xy[:,:2]+xy[:,2:])/2+offset,scores])
    return merge_points(points)
