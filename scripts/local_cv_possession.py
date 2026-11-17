"""Local-only OTHER-match detector/sequence prototype, isolated from JEV.

Run/serve subcommands; detector observations cached for deterministic offline
replay. Never reads gold labels or writes reviewed annotations. No API client.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from path_guard import check_paths
from local_cv_core import BallTrack, PlayerTracks, ControlSequence, nearest_baseline

DEFAULT_VIDEO = ROOT/'outputs/ai-annotation-psg-barcelona/full-match/source.mp4'
RUNTIME = ROOT/'outputs/yolov8-runtime/lib/python3.12/site-packages'


def atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def kit(rgb, box):
    # Explicit OTHER-match kits. Unknown shirts (including keepers/referees)
    # are not forced into either side. This is not a general kit classifier.
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    patch = rgb[max(0, int(y1+.2*h)):int(y1+.52*h),
                max(0, int(x1+.2*w)):int(x2-.2*w)].astype(float)
    if patch.size < 18:
        return 'unknown', 0.
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    blue = float(((b-r > 12) & (b > g*.95)).mean())
    pink = float(((r-g > 20) & (r > b*1.08)).mean())
    if max(blue, pink) < .25 or abs(blue-pink) < .18:
        return 'unknown', max(blue, pink)
    return ('H' if blue > pink else 'A'), max(blue, pink)


def grass_support(rgb, point):
    x, y = map(int, point[:2])
    h, w = rgb.shape[:2]
    if not (5 < x < w-5 and h*.10 < y < h-8):
        return False
    patch = rgb[max(0,y-25):min(h,y+26), max(0,x-25):min(w,x+26)].astype(float)
    r, g, b = patch[...,0], patch[...,1], patch[...,2]
    return float(((g > r*1.12) & (g > b*1.08) & (g > 40)).mean()) > .25


def replay(rows):
    people, ball, sequence = PlayerTracks(), BallTrack(), ControlSequence()
    result = []
    for row in rows:
        p = people.update(row['players'], row['seconds'], row['cut'])
        b = ball.update(row['ball_candidates'], row['seconds'], row['cut'])
        evidence = sequence.update(b, p, row['seconds'], row['cut'])
        result.append(dict(frame=row['frame'], seconds=row['seconds'], players=p,
                           ball=b, cut=row['cut'], baseline=nearest_baseline(b,p), **evidence))
    return result


def summarize(rows, observations, elapsed):
    counts = Counter(r['label'] or 'unresolved' for r in rows)
    def switches(key):
        # Only adjacent observed assignments. Unknown gaps aren't erased.
        return sum(a[key] is not None and b[key] is not None and a[key] != b[key]
                   for a,b in zip(rows, rows[1:]))
    return dict(samples=len(rows), recording_seconds=rows[-1]['seconds'] if rows else 0,
        ball_observed_samples=sum(r['ball'] is not None for r in rows),
        repeated_control_samples=sum(r['control'] is not None for r in rows),
        assignments=dict(counts), adjacent_switches=switches('label'),
        nearest_baseline_adjacent_switches=switches('baseline'),
        foot_ball_candidate_samples=sum(bool(r['foot_balls']) for r in observations),
        yolo_ball_candidate_samples=sum(bool(r['yolo_balls']) for r in observations),
        elapsed_seconds=elapsed, external_api_calls=0, external_api_cost_usd=0,
        accuracy=None, spell_f1=None, macro_f1=None,
        limitations=['Uncalibrated image-space contact evidence; no possession accuracy measured',
          '10 Hz default sampling is not native-frame timing',
          'Whistles/restarts and goalkeeper team identity unresolved; no automatic D labels',
          'Camera cuts, missing balls and unclear teams leave gaps; no global HMM',
          'Visible co-motion is a heuristic, not proof of touch or control'])


def run(args):
    check_paths([args.video, args.out])
    args.out.mkdir(parents=True, exist_ok=True)
    # This version intentionally only supports the declared development source.
    # A new match requires a separate manifest/kit mapping, not a copied label.
    if args.video.resolve() != DEFAULT_VIDEO.resolve():
        raise ValueError('This prototype supports only the declared PSG development recording')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.stride < 1:
        raise ValueError('Positive duration and stride required')
    sys.path.append(str(RUNTIME))
    import torch
    import cv2
    import yolov8_ball as yolo
    import visual_ball as foot
    torch.set_num_threads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ym, _, ymeta = yolo.load_model()
    from ultralytics.utils import ops
    ym = ym.to(device)
    fp = ROOT/'outputs/footandball-source-intake/source/models/model_20201019_1416_final.pth'
    fm = foot.load_model(fp, device)
    adapted = {}
    if args.adapted:
        for name, model, experiment, checkpoint in [
            ('yolo', ym, 'yolov8-adapt-v1', 'yolov8-adapt-v1'),
            ('foot', fm, 'visual-ball-encoder-v1', 'visual-ball-encoder-v1')]:
            path = ROOT/'outputs'/checkpoint/'candidate.pth'
            receipt = json.loads((ROOT/'experiments'/f'{experiment}-results.json').read_text())
            expected_hash = receipt['fitting']['checkpoint_sha256']
            if sha(path) != expected_hash:
                raise ValueError(f'{name} adaptation checkpoint does not match experiment receipt')
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            adapted[name] = dict(path=str(path),sha256=expected_hash,
                role='Existing other-match H250 research model; failed promotion gate, not validated possession')
    stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(args.video)]))['streams'][0]
    fps = float(Fraction(stream['r_frame_rate']))
    end = min(int(stream['nb_frames']), math.ceil(args.seconds*fps))
    config = dict(schema=1, source=str(args.video.resolve()), source_sha256=sha(args.video),
        source_url='https://www.youtube.com/watch?v=LoGswckZCJE',
        source_license='Reuse rights not established; local private development only',
        width=stream['width'], height=stream['height'], fps=fps, end_frame_exclusive=end,
        stride=args.stride, device=device, yolo=ymeta, foot_weights_sha256=sha(fp), adaptations=adapted,
        scripts={p:sha(ROOT/'scripts'/p) for p in ['local_cv_possession.py','local_cv_core.py','local_cv_viewer.html']},
        teams={'H':'PSG / dark blue','A':'Barcelona / salmon pink'},
        model_role='Fresh detector evidence, not previous rejected annotation outputs',
        supplied_match_read=False, answers_read=False, training=False, training_eligible=False,
        ball_threshold=.7, player_threshold=.35, control=dict(dwell=.2,radius=.45,relative_speed=2.,hold=2.))
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output has different locked configuration; use a new output directory')
    atomic(config_path, config)
    raw = args.out/'observations.jsonl'
    observations = [json.loads(x) for x in raw.read_text().splitlines() if x.strip()] if raw.exists() else []
    expected = list(range(0, end, args.stride))
    if [r['frame'] for r in observations] != expected[:len(observations)]:
        raise ValueError('Noncontiguous detection cache')
    start_index = len(observations)
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index*args.stride)
    previous = None
    if start_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, expected[start_index-1])
        ok, old = cap.read()
        if not ok: raise ValueError('Cannot decode resume context')
        previous = cv2.resize(old,(64,36)).astype(float)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index*args.stride)
    started = time.monotonic()
    def publish(state):
        rows = replay(observations)
        atomic(args.out/'results.json',dict(config=config, rows=rows))
        atomic(args.out/'status.json',dict(state=state, completed=len(rows), total=len(expected),
            updated_at=datetime.now(timezone.utc).isoformat(),
            summary=summarize(rows,observations,time.monotonic()-started)))
    publish('running')
    try:
        with raw.open('a') as out:
            for index in range(start_index,len(expected)):
                frame = expected[index]
                ok,bgr = cap.read()
                if not ok: raise ValueError(f'Decode failed at zero-based frame {frame}')
                for _ in range(args.stride-1): cap.grab()
                rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                thumbnail = cv2.resize(bgr,(64,36)).astype(float)
                cut = previous is not None and float(np.mean(np.abs(thumbnail-previous)))/255 > .22
                previous = thumbnail
                tensor = yolo.input_tensor(rgb).to(device)
                with torch.inference_mode():
                    boxes = ops.non_max_suppression(ym(tensor),conf_thres=.25,iou_thres=.7,
                        classes=[0,32], max_det=100)[0]
                    boxes[:,:4] = ops.scale_boxes(tensor.shape[2:],boxes[:,:4],rgb.shape)
                    fb = foot.detect(fm,rgb,.7).tolist()
                boxes = boxes.cpu().numpy()
                players = []
                yb = []
                for x1,y1,x2,y2,score,cls in boxes:
                    box = list(map(float,[x1,y1,x2,y2]))
                    if int(cls)==0 and score >= .35:
                        team,color = kit(rgb,box)
                        players.append(dict(box=box,score=float(score),team=team,kit_score=color))
                    elif int(cls)==32:
                        yb.append([float((x1+x2)/2),float((y1+y2)/2),float(score)])
                # FootAndBall is the fixed ball signal; YOLO balls retained as
                # an independent inspectable comparison, never labelled truth.
                candidates = [b for b in fb if grass_support(rgb,b)]
                row = dict(frame=frame,seconds=frame/fps,players=players,cut=cut,
                    ball_candidates=candidates, foot_balls=fb, yolo_balls=yb)
                out.write(json.dumps(row,allow_nan=False)+'\n'); out.flush()
                observations.append(row)
                if index % 100 == 0 or index==len(expected)-1:
                    publish('running')
                    print(f'{len(observations)}/{len(expected)} samples; {frame/fps:.1f}s source',flush=True)
                if index % 300 == 0:
                    preview = bgr.copy()
                    for b in candidates: cv2.circle(preview,tuple(map(int,b[:2])),9,(0,255,255),2)
                    cv2.imwrite(str(args.out/f'preview-{frame:07d}.jpg'),preview)
        publish('complete')
    except BaseException:
        publish('interrupted')
        raise
    finally:
        cap.release()


def serve(args):
    check_paths([args.out])
    config = json.loads((args.out/'config.json').read_text())
    source = Path(config['source'])
    check_paths([source])
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?')[0]
            if route == '/video':
                size = source.stat().st_size
                span = self.headers.get('Range')
                start,end = 0,size-1
                if span:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)',span)
                    if not match or not any(match.groups()):
                        self.send_error(416); return
                    a,b = match.groups()
                    if a: start,end = int(a),min(int(b) if b else size-1,size-1)
                    else: start,end = max(0,size-int(b)),size-1
                    if start > end or start >= size:
                        self.send_response(416); self.send_header('Content-Range',f'bytes */{size}'); self.end_headers(); return
                self.send_response(206 if span else 200)
                self.send_header('Content-Type','video/mp4'); self.send_header('Accept-Ranges','bytes')
                self.send_header('Content-Length',str(end-start+1))
                if span: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with source.open('rb') as stream:
                        stream.seek(start); left=end-start+1
                        while left:
                            data=stream.read(min(left,1024*1024))
                            if not data: break
                            self.wfile.write(data); left-=len(data)
                except (BrokenPipeError,ConnectionResetError): pass
                return
            paths={'/':ROOT/'scripts/local_cv_viewer.html', '/results.json':args.out/'results.json',
                   '/status.json':args.out/'status.json'}
            if route not in paths: self.send_error(404); return
            data=paths[route].read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8' if route=='/' else 'application/json')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def log_message(self,*args): pass
    print(f'Local CV viewer: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    runp=sub.add_parser('run'); runp.add_argument('--video',type=Path,default=DEFAULT_VIDEO)
    runp.add_argument('--out',type=Path,required=True); runp.add_argument('--seconds',type=float,default=300)
    runp.add_argument('--stride',type=int,default=5)
    runp.add_argument('--adapted',action='store_true',help='Use existing H250 research adaptations (not promoted models)')
    viewer=sub.add_parser('serve'); viewer.add_argument('--out',type=Path,required=True)
    viewer.add_argument('--port',type=int,default=8799)
    args=parser.parse_args()
    (run if args.command=='run' else serve)(args)
