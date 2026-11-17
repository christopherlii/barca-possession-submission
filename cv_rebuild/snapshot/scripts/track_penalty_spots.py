"""Track the two penalty-spot paint marks as separate static image objects.

Homography provides search locations, camera motion transports the previous
locations, and compact bright components plus patch matching supply image
measurements. Repeated field-stationary measurements are required for a lock.
Moving-ball candidates cannot update a locked spot when their residual motion
exceeds the stationary threshold. Scores are heuristic, not probabilities.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stage1_cv_baseline import digest, private_path
from local_cv_possession import atomic
from sam_confirmation_recovery import read_rows


CONFIG = dict(search_radius_px=24, stationary_residual_px=2.5, confirmations=3,
              maximum_projection_offset_px=8.,
              template_size=15, minimum_correlation=.65, maximum_misses=8,
              minimum_area=2, maximum_area=120, brightness=160, saturation=90,
              minimum_contrast=30., veto_padding_px=4.)
NAMES = ('left_penalty_spot', 'right_penalty_spot')


class PenaltySpotTracker:
    def __init__(self):
        self.center = None
        self.template = None
        self.streak = 0
        self.misses = 0
        self.confirmed = False
        self.radius = 2.
        self.locked_offset = None

    def reset(self):
        self.__init__()

    def update(self, gray, hsv, projection, affine, cut=False):
        import cv2
        if cut:
            self.reset()
        if projection is None or not np.isfinite(projection).all():
            self.reset()
            return dict(center=None, state='unavailable', confirmed=False)
        projection = np.asarray(projection, float)
        height, width = gray.shape
        if not (0 <= projection[0] < width and 0 <= projection[1] < height):
            self.reset()
            return dict(center=projection.tolist(), state='offscreen', confirmed=False)
        expected = projection.copy()
        if self.center is not None and affine is not None:
            expected = np.asarray(affine)[:, :2]@self.center + np.asarray(affine)[:, 2]
            if np.linalg.norm(expected-projection) > CONFIG['search_radius_px']:
                self.reset()
                expected = projection.copy()
        if self.confirmed and self.locked_offset is not None:
            if affine is not None:
                self.locked_offset = np.asarray(affine)[:, :2]@self.locked_offset
            expected = projection+self.locked_offset
        radius = CONFIG['search_radius_px']
        x, y = np.maximum(0, np.floor(expected-radius)).astype(int)
        xx, yy = np.minimum([width,height], np.ceil(expected+radius+1)).astype(int)
        local = hsv[y:yy,x:xx]
        luminance = gray[y:yy,x:xx]
        floor = max(CONFIG['brightness'], float(np.median(luminance))+CONFIG['minimum_contrast'])
        mask = ((local[:,:,1] <= CONFIG['saturation']) & (local[:,:,2] >= floor)).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for index in range(1,count):
            area = int(stats[index,cv2.CC_STAT_AREA])
            bw, bh = stats[index,cv2.CC_STAT_WIDTH:cv2.CC_STAT_HEIGHT+1]
            if not CONFIG['minimum_area'] <= area <= CONFIG['maximum_area'] or max(bw,bh) > 2.5*min(bw,bh):
                continue
            center = centers[index]+[x,y]
            if np.linalg.norm(center-projection) > CONFIG['maximum_projection_offset_px']:
                continue
            patch = cv2.getRectSubPix(gray,(CONFIG['template_size'],CONFIG['template_size']),tuple(center.astype(float)))
            correlation = 1.
            if self.template is not None:
                correlation = float(cv2.matchTemplate(patch,self.template,cv2.TM_CCOEFF_NORMED)[0,0])
                if not np.isfinite(correlation) or correlation < CONFIG['minimum_correlation']:
                    continue
            residual = float(np.linalg.norm(center-expected))
            if self.confirmed and residual > CONFIG['stationary_residual_px']:
                continue
            candidates.append((residual,center,area,patch,correlation))
        if candidates:
            residual,center,area,patch,correlation = min(candidates,key=lambda item:item[0])
            self.streak = self.streak+1 if residual <= CONFIG['stationary_residual_px'] or self.center is None else 1
            self.center = center
            self.radius = float(np.sqrt(area/np.pi))
            self.misses = 0
            self.confirmed = self.confirmed or self.streak >= CONFIG['confirmations']
            if self.confirmed and self.template is None:
                self.template = patch.copy()
                self.locked_offset = center-projection
            state = 'image_tracked' if self.confirmed else 'confirming'
            return dict(center=center.tolist(), state=state, confirmed=self.confirmed,
                        stationary_frames=self.streak, image_score=correlation,
                        paint_radius_px=self.radius, veto_radius_px=max(6.,self.radius+CONFIG['veto_padding_px']))
        self.center = expected
        self.misses += 1
        if self.misses > CONFIG['maximum_misses']:
            self.reset()
            self.center = projection
        return dict(center=self.center.tolist(),state='camera_predicted' if self.confirmed else 'projected',
                    confirmed=self.confirmed, missed_frames=self.misses,
                    paint_radius_px=self.radius,veto_radius_px=max(6.,self.radius+CONFIG['veto_padding_px']))


def run(args):
    import cv2
    output = private_path(args.out)
    if output.exists():
        raise ValueError('Use a new output directory')
    rows = read_rows(args.cache/'detections.jsonl')
    config = json.loads((args.cache/'config.json').read_text())
    cap = cv2.VideoCapture(config['source'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, rows[0]['frame'])
    cv2.setNumThreads(2)
    output.mkdir(parents=True)
    trackers = {name:PenaltySpotTracker() for name in NAMES}
    started = time.monotonic()
    counts = {name:Counter() for name in NAMES}
    try:
        for index,row in enumerate(rows):
            ok,image = cap.read()
            if not ok:
                raise ValueError('Video decode failed')
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            projections = {shape['name']:shape['points'][0] for shape in row.get('field_geometry',[])
                           if shape['name'] in NAMES and len(shape['points'])==1}
            cut = index==0 or row.get('camera_affine') is None or row['frame']!=rows[index-1]['frame']+1
            tracked = {}
            for name,tracker in trackers.items():
                tracked[name] = tracker.update(gray,hsv,projections.get(name),row.get('camera_affine'),cut)
                counts[name][tracked[name]['state']] += 1
            row['active_penalty_spots'] = tracked
            if (index+1)%500==0:
                status=dict(state='running',completed=index+1,total=len(rows),elapsed_seconds=time.monotonic()-started)
                atomic(output/'status.json',status)
                print(json.dumps(status),flush=True)
    finally:
        cap.release()
    config['active_penalty_spots']=dict(CONFIG,source_sha256=digest(args.cache/'detections.jsonl'),
                                        code_sha256=digest(__file__),strict_confirmed_paint_veto=True)
    summary=dict(frames=len(rows),states={name:dict(value) for name,value in counts.items()},
                 elapsed_seconds=time.monotonic()-started,accuracy_measured=False)
    atomic(output/'config.json',config)
    atomic(output/'summary.json',summary)
    (output/'detections.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    atomic(output/'status.json',dict(state='complete',**summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
