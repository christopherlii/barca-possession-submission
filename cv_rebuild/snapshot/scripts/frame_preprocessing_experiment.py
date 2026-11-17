"""Fixed per-frame preprocessing screen on already-inspected H250 development.

No fitting, held-out access, supplied labels or promotion. Freeze every prediction
before opening development targets. A separate supplied-input probe has no scores.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_tiling import merge_points, tile_windows
from local_cv_possession import atomic
from path_guard import check_paths

REST = ROOT/'outputs/preprocessing-source'
PINS = {'restormer_arch.py':'3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a',
        'motion_deblurring.pth':'194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31',
        'LICENSE':'2b93776512924bc095ec5d97a79d76cf1ab401ae641d1ae1f46e94f1c53a2e59'}
ARMS = ['baseline640', 'resolution1280', 'resolution1920', 'tiles',
        'clahe', 'gamma', 'sharpen', 'bilateral', 'pitch_crop', 'deblur']


def preprocess(image, mode):
    import cv2
    if mode == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8,8)).apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if mode == 'gamma':
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mean = max(.01, float(y[:,:,0].mean())/255)
        exponent = float(np.clip(np.log(.5)/np.log(min(mean,.99)), .75, 1.25))
        lut = np.rint(255*(np.arange(256)/255.)**exponent).astype(np.uint8)
        y[:,:,0] = cv2.LUT(y[:,:,0], lut)
        return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
    if mode == 'sharpen':
        return cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (0,0), 1.), -.5, 0)
    if mode == 'bilateral':
        return cv2.bilateralFilter(image, 5, 20, 20)
    return image.copy()


def guided_box(point, width, height, radius=160):
    if point is None or not np.isfinite(point).all(): return None
    x, y = map(float, point)
    if not (0<=x<width and 0<=y<height): return None
    return max(0,int(x-radius)),max(0,int(y-radius)),min(width,int(x+radius)),min(height,int(y+radius))


def pitch_box(image):
    """Image-only green-support bbox with margin, not a calibrated pitch polygon."""
    import cv2
    small = cv2.resize(image, (320, max(1, round(320*image.shape[0]/image.shape[1]))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:,:,0]>=25)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=35)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    h,w = image.shape[:2]
    if n<=1: return (0,0,w,h)
    k = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    x,y,ww,hh,area = stats[k]
    if area<mask.size*.1: return (0,0,w,h)
    sx,sy = w/mask.shape[1],h/mask.shape[0]
    return max(0,int(x*sx-.1*w)),max(0,int(y*sy-.1*h)),min(w,int((x+ww)*sx+.1*w)),min(h,int((y+hh)*sy+.1*h))


class Deblurrer:
    def __init__(self, device):
        import torch
        for name,pin in PINS.items():
            if digest(REST/name)!=pin: raise ValueError('Restormer source/weight pin mismatch')
        spec = importlib.util.spec_from_file_location('pinned_restormer', REST/'restormer_arch.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.model = module.Restormer().to(device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(REST/'motion_deblurring.pth',map_location='cpu',weights_only=True)['params'],strict=True)
        self.device=device

    def __call__(self, image):
        import torch
        from torch.nn import functional as F
        # Full native coverage in bounded tiles. Overlap averaging limits seams.
        h,w = image.shape[:2]; size=384; overlap=32
        def starts(n):
            return sorted(set(list(range(0,max(1,n-size+1),size-overlap))+[max(0,n-size)]))
        total=np.zeros_like(image,dtype=np.float32); count=np.zeros((h,w,1),np.float32)
        for y in starts(h):
            for x in starts(w):
                crop=image[y:y+size,x:x+size]
                hh,ww=crop.shape[:2]
                t=torch.from_numpy(crop[:,:,::-1].copy().transpose(2,0,1)[None]).to(self.device).float()/255
                t=F.pad(t,(0,(-ww)%8,0,(-hh)%8),mode='replicate')
                with torch.inference_mode(): result=self.model(t)[0,:,:hh,:ww].clamp(0,1).cpu().numpy()
                total[y:y+hh,x:x+ww]+=result.transpose(1,2,0)[:,:,::-1]*255
                count[y:y+hh,x:x+ww]+=1
        return np.rint(total/count).clip(0,255).astype(np.uint8)


def detect(model, image, device, imgsz=640, box=None):
    offset=np.array([0.,0.])
    if box:
        x,y,xx,yy=box; image=image[y:yy,x:xx]; offset=np.array([x,y])
    result=model.predict(image,classes=[32],conf=.1,iou=.7,imgsz=imgsz,
                         device=device,verbose=False)[0].boxes
    xy=result.xyxy.cpu().numpy(); scores=result.conf.cpu().numpy()
    points=np.c_[(xy[:,:2]+xy[:,2:])/2+offset,scores]
    return merge_points([points])


def score(predictions, selected, pins):
    from visual_ball_metrics import score_image, aggregate
    methods={name:[] for name in ARMS}; groups={}
    base=ROOT/'outputs/visual-ball-h250-v1/h250-development-source'
    for row,(group,name) in zip(predictions, selected):
        path=base/'labels'/(Path(name).stem+'.txt'); check_paths([path])
        if digest(path)!=pins[name]['label_sha256']: raise ValueError('Development labels changed')
        truth=np.array([[float(v[1])*row['width'],float(v[2])*row['height']]
            for line in path.read_text().splitlines() if (v:=line.split()) and v[0]=='0']).reshape(-1,2)
        for method in ARMS:
            c=score_image(np.asarray(row['predictions'][method]).reshape(-1,3),truth,10.)
            methods[method].append(c); groups.setdefault(group,{m:[] for m in ARMS})[method].append(c)
    totals={m:aggregate(v,uncertainty=False) for m,v in methods.items()}
    by_match={g:{m:aggregate(v,uncertainty=False) for m,v in mm.items()} for g,mm in groups.items()}
    paired={}
    for m in ARMS[1:]:
        counts=np.array([[[v[k][f] for f in ['true_positive','predictions','targets']]
                         for k in ['baseline640',m]] for v in by_match.values()])
        rng=np.random.default_rng(109)
        b=counts[rng.integers(len(counts),size=(2000,len(counts)))].sum(1)
        den=b[:,:,1]+b[:,:,2]; f=np.divide(2*b[:,:,0],den,out=np.zeros_like(den,dtype=float),where=den>0)
        paired[m]=dict(delta_f1=totals[m]['point_f1']-totals['baseline640']['point_f1'],
            match_bootstrap_delta95=np.percentile(f[:,1]-f[:,0],[2.5,97.5]).tolist(),
            regressed_matches=sum(v[m]['point_f1']<v['baseline640']['point_f1'] for v in by_match.values()))
    return totals,by_match,paired


def run(args):
    import torch,cv2
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings')
    os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    out=private_path(args.out);check_paths([out])
    if out.exists(): raise ValueError('New experiment directory required')
    selection_path=ROOT/'outputs/h250-training-v1/private-development-selection.json'
    pins_path=ROOT/'outputs/visual-ball-h250-v1/h250-development/private-inputs.json'
    selected=json.loads(selection_path.read_text()); counts={}; subset=[]
    for group,name in selected:
        if counts.get(group,0)<args.per_match: subset.append((group,name));counts[group]=counts.get(group,0)+1
    pins={p['source_image']:p for p in json.loads(pins_path.read_text())}
    weights=ROOT/'outputs/two-stage-cv-models/yolo11x.pt'
    plan=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=digest(__file__),model_sha256=digest(weights),restormer=PINS,
        restormer_revision='68dc6ac472db26f16361150cb7a96a1bc87da93f',
        selection_sha256=digest(selection_path),pins_sha256=digest(pins_path),
        images=len(subset),per_match=args.per_match,groups=len(counts),arms=ARMS,
        settings=dict(confidence=.1,iou=.7,point_nms=8.,matching_radius=10.,
            clahe_clip=2.,clahe_grid=[8,8],gamma_range=[.75,1.25],unsharp_amount=.5,
            bilateral=[5,20,20],tile_fraction=.6,restormer_tile=384,restormer_overlap=32),
        hypothesis='Native resolution and per-frame transformations may recover missed tiny balls without excess false positives',
        development_only=True,heldout_seeds=[],development_seeds=[],heldout_matches=[],
        training=False,promotion_allowed=False,supplied_match_scored=False,
        selection_rule='First fixed N images per already-inspected development match, before target access')
    out.mkdir(parents=True);atomic(out/'plan.json',plan)
    torch.set_num_threads(2)
    model=YOLO(weights);deblur=Deblurrer(args.device)
    timing={m:0. for m in ARMS};predictions=[]
    for i,(group,name) in enumerate(subset):
        path=ROOT/'outputs/visual-ball-h250-v1/h250-development-source/images'/Path(name).name
        check_paths([path])
        if digest(path)!=pins[name]['image_sha256']: raise ValueError('Development image changed')
        image=cv2.imread(str(path));h,w=image.shape[:2];methods={}
        for arm in ARMS:
            start=time.perf_counter()
            if arm.startswith('resolution'):
                points=detect(model,image,args.device,int(arm.removeprefix('resolution')))
            elif arm=='tiles':
                points=merge_points([methods['baseline640']]+[
                    detect(model,image,args.device,640,box) for box in tile_windows(w,h)])
            elif arm=='pitch_crop':
                points=detect(model,image,args.device,640,pitch_box(image))
            elif arm=='deblur': points=detect(model,deblur(image),args.device)
            else: points=detect(model,preprocess(image,arm),args.device)
            timing[arm]+=time.perf_counter()-start;methods[arm]=points.tolist()
        predictions.append(dict(name=name,match=group,width=w,height=h,predictions=methods))
        atomic(out/'status.json',dict(state='running',completed=i+1,total=len(subset)))
        print(f'{i+1}/{len(subset)} all preprocessing arms',flush=True)
    atomic(out/'private-predictions.json',predictions)
    totals,groups,paired=score(predictions,subset,pins)
    timing['tiles']+=timing['baseline640']
    result=dict(plan=plan,metrics=totals,by_match=groups,paired=paired,
        seconds_per_image={m:v/len(subset) for m,v in timing.items()},
        decision='Development screen only; no promotion',limitations=[
        '32-image default screen is small, previously inspected and correlated within matches',
        'Image-only green-support pitch crop is not homography; tracking-guided crops require separate inputs',
        'Restormer uses native 384px tiles with overlap; no restored pixels count as ground truth',
        'No continuous possession truth; spell F1, macro F1 and frame accuracy unavailable'])
    atomic(out/'results.json',result)
    atomic(out/'status.json',dict(state='complete',completed=len(subset),total=len(subset)))
    print(json.dumps(dict(metrics=totals,paired=paired,seconds_per_image=result['seconds_per_image'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-match',type=int,default=4);p.add_argument('--device',default='mps')
    args=p.parse_args()
    if args.per_match<1:p.error('Positive per-match count required')
    run(args)
