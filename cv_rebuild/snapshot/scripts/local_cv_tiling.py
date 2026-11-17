"""Fixed overlapping crops for tiny-ball proposals; no fitting or labels."""
import numpy as np


def tile_windows(width, height):
    if min(width,height)<32:
        raise ValueError('Image too small')
    tw,th=round(width*.6),round(height*.6)
    return [(x,y,x+tw,y+th) for y in (0,height-th) for x in (0,width-tw)]


def merge_points(groups, radius=8.):
    # Greedy score-ordered spatial NMS in original native pixel coordinates.
    points=sorted([list(map(float,p)) for group in groups for p in group],key=lambda p:-p[2])
    chosen=[]
    for point in points:
        if not all(np.isfinite(point)) or not 0<=point[2]<=1:
            raise ValueError('Invalid point')
        if not any(np.linalg.norm(np.array(point[:2])-q[:2])<=radius for q in chosen):
            chosen.append(point)
    return np.asarray(chosen[:100]).reshape(-1,3)


def infer(model,rgb,windows):
    import torch
    import yolov8_ball as yolo
    from ultralytics.utils import ops
    device=next(model.parameters()).device
    result=[]
    for x,y,xx,yy in windows:
        crop=rgb[y:yy,x:xx]
        tensor=yolo.input_tensor(crop).to(device)
        with torch.inference_mode():
            boxes=ops.non_max_suppression(model(tensor),conf_thres=.25,iou_thres=.7,
                classes=[32],max_det=100)[0]
            boxes[:,:4]=ops.scale_boxes(tensor.shape[2:],boxes[:,:4],crop.shape)
        points=[]
        for a,b,c,d,score,_ in boxes.cpu().numpy():
            points.append([float((a+c)/2+x),float((b+d)/2+y),float(score)])
        result.append(points)
    return merge_points(result)
