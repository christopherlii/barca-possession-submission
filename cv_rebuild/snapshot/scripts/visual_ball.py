"""Local frozen FootAndBall ball-point inference; no tracking/labels accepted."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from path_guard import check_paths

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def verify_vendor():
    """Reverse only the three declared adaptations; recover pinned source bytes."""
    pins = json.loads((ROOT/'experiments/footandball-source-intake.json').read_text())
    expected = {r['path']: r['sha256'] for r in pins['files']}
    folder = ROOT/'third_party/footandball'
    text = (folder/'footandball.py').read_text()
    for after, before in [('from . import fpn', 'import network.fpn as fpn'),
            ('from . import point_nms as nms', 'import network.nms as nms'),
            ('BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE = 1, 2, 40',
             'from data.augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE')]:
        if text.count(after) != 1:
            raise ValueError('Unrecorded third-party source adaptation')
        text = text.replace(after, before)
    for name, data in [('network/footandball.py', text.encode()), ('network/fpn.py', (folder/'fpn.py').read_bytes()),
                       ('LICENSE', (folder/'LICENSE').read_bytes())]:
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Pinned vendor source differs: '+name)
    return pins


def load_model(path, device='cpu'):
    import torch
    pins = verify_vendor()
    from third_party.footandball.footandball import model_factory
    path = Path(path)
    check_paths([path])
    row = next(r for r in pins['files'] if r['path'] == 'models/model_20201019_1416_final.pth')
    if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Pretrained visual checkpoint differs')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    model = model_factory('fb1', 'train')
    model.load_state_dict(weights, strict=True)
    if len(weights) != 74 or sum(v.numel() for v in weights.values()) != 199649:
        raise ValueError('Unexpected visual checkpoint tensors')
    model.eval().requires_grad_(False)
    return model.to(device)


def input_tensor(image, device='cpu'):
    import torch
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32 or image.dtype != np.uint8:
        raise ValueError('Expected native-size RGB8 image at least 32x32')
    normalized = (image.astype(np.float32)/255-MEAN)/STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)[None].copy()).to(device)


def ball_map(model, tensor):
    import torch
    with torch.no_grad():
        return torch.softmax(model.ball_classifier(model.base_network(tensor)[0]), dim=1)[:, 1]


def points_from_map(confidence, threshold=.7, maximum=100):
    import torch
    from third_party.footandball.point_nms import NonMaximaSuppression2d
    if (not 0 <= threshold <= 1 or type(maximum) is not int or maximum < 1 or
            confidence.ndim != 2 or not torch.isfinite(confidence).all() or
            (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError('Invalid visual confidence map/configuration')
    nms = NonMaximaSuppression2d()(confidence[None, None])[0, 0]
    flat = nms.flatten()
    # Stable index tie breaking, with strictly suppressed plateaus.
    order = torch.argsort(flat, descending=True, stable=True)
    order = order[(flat[order] >= threshold) & (flat[order] > 0)][:maximum]
    width = nms.shape[1]
    return np.column_stack([(order % width).cpu().numpy()*4+1.5,
        (order // width).cpu().numpy()*4+1.5, flat[order].cpu().numpy()])


def detect(model, image, threshold=.7):
    device = next(model.parameters()).device
    tensor = input_tensor(image, device)
    return points_from_map(ball_map(model, tensor)[0], threshold)


def decode_image(path, width, height):
    path = Path(path)
    check_paths([path])
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(path),
        '-map', '0:v:0', '-frames:v', '1', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True)
    if p.returncode or len(p.stdout) != width*height*3:
        raise ValueError('Image decode dimensions differ')
    return np.frombuffer(p.stdout, np.uint8).reshape(height, width, 3)
