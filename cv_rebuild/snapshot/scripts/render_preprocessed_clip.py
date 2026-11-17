"""Render an aligned, local-only CLAHE display copy; never modify source video."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from stage1_cv_baseline import private_path, digest
from frame_preprocessing_experiment import preprocess
from local_cv_possession import atomic
from cv_pixel_preprocessing import detector_image


def render(features, out, mode='clahe', clip=2.):
    import cv2
    cv2.setNumThreads(2)
    out = private_path(out)
    if out.exists(): raise ValueError('Use a new output path')
    with np.load(features, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        frames = data['frame_ids']
    if not len(frames) or not np.all(np.diff(frames) == 1):
        raise ValueError('Expected consecutive frame indices')
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = out.with_name(out.stem+'.pending.mp4')
    cap = cv2.VideoCapture(meta['source'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    w,h,fps = meta['width'],meta['height'],meta['fps']
    process = subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y',
        '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}',
        '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2',
        '-movflags','+faststart',str(pending)], stdin=subprocess.PIPE)
    try:
        for i in range(len(frames)):
            ok,image = cap.read()
            if not ok: raise ValueError('Source decode ended early')
            if mode == 'zero-green':changed=detector_image(image,mode)
            else:
                lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                lab[:,:,0]=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(lab[:,:,0])
                changed=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            process.stdin.write(changed.tobytes())
            if (i+1)%500==0: print(f'{i+1}/{len(frames)} {mode} display frames',flush=True)
        process.stdin.close()
        if process.wait(): raise ValueError('Encoding failed')
        info = json.loads(subprocess.check_output(['ffprobe','-v','error',
            '-select_streams','v:0','-count_frames','-show_entries',
            'stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(pending)]))['streams'][0]
        from fractions import Fraction
        if (int(info['nb_read_frames'])!=len(frames) or float(Fraction(info['r_frame_rate']))!=fps
            or int(info['width'])!=w or int(info['height'])!=h):
            raise ValueError('Rendered clip alignment mismatch')
        pending.replace(out)
        atomic(out.with_suffix('.json'), dict(source=meta['source'],source_sha256=meta['source_sha256'],
            start_frame=int(frames[0]),frames=len(frames),fps=fps,width=w,height=h,
            method='Green channel zeroed' if mode=='zero-green' else 'LAB luminance CLAHE',
            preprocessing_mode=mode,clip_limit=clip if mode=='clahe' else None,
            grid=[8,8] if mode=='clahe' else None,
            display_only=True,detections_rerun=meta.get('detector_preprocessing')==mode,
            output_sha256=digest(out)))
        print('Preprocessed display clip verified and ready.',flush=True)
    finally:
        cap.release()
        if process.poll() is None: process.kill(); process.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['clahe','zero-green'],default='clahe')
    p.add_argument('--clahe-clip',type=float,default=2.)
    args=p.parse_args();render(args.features,args.out,args.mode,args.clahe_clip)
