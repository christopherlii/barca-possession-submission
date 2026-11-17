"""Optional pinned YOLOv8n ball points; class-restricted safe model loading."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch

from download_yolov8 import ROOT, OUT, ARTIFACTS, WEIGHTS_SHA
from path_guard import check_paths

# Legacy pickle aliases differ from current class.__module__, so name every
# allowed constructor explicitly. Never derive an arbitrary import from pickle.
ALLOWED = {
    'torch.nn.modules.container.Sequential': ('torch.nn', 'Sequential'),
    'torch.nn.modules.container.ModuleList': ('torch.nn', 'ModuleList'),
    'torch.nn.modules.activation.SiLU': ('torch.nn', 'SiLU'),
    'torch.nn.modules.batchnorm.BatchNorm2d': ('torch.nn', 'BatchNorm2d'),
    'torch.nn.modules.conv.Conv2d': ('torch.nn', 'Conv2d'),
    'torch.nn.modules.pooling.MaxPool2d': ('torch.nn', 'MaxPool2d'),
    'torch.nn.modules.upsampling.Upsample': ('torch.nn', 'Upsample'),
    **{'ultralytics.nn.modules.'+n: ('ultralytics.nn.modules', n) for n in
        ('Bottleneck', 'C2f', 'Concat', 'Conv', 'DFL', 'Detect', 'SPPF')},
    'ultralytics.nn.tasks.DetectionModel': ('ultralytics.nn.tasks', 'DetectionModel'),
}


def configure_runtime():
    directory = ROOT/'outputs/yolov8-runtime-settings'
    check_paths([directory])
    directory.mkdir(parents=True, exist_ok=True)
    # Key order/types mirror the pinned SettingsManager; no user settings touched.
    settings = dict(settings_version='0.0.4', datasets_dir=str(directory/'datasets'),
        weights_dir=str(OUT), runs_dir=str(directory/'runs'), uuid='local-possession-research',
        sync=False, api_key='', openai_api_key='', clearml=False, comet=False, dvc=False,
        hub=False, mlflow=False, neptune=False, raytune=False, tensorboard=False, wandb=False)
    (directory/'settings.yaml').write_text(json.dumps(settings, indent=2)+'\n')
    os.environ['YOLO_CONFIG_DIR'] = str(directory)
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    os.environ['YOLO_VERBOSE'] = 'False'
    return directory


def load_model():
    configure_runtime()
    for name, (_, size, digest) in ARTIFACTS.items():
        p = OUT/name
        if p.stat().st_size != size or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned model/runtime source artifact differs')
    spec = importlib.util.find_spec('ultralytics')
    if spec is None:
        raise ValueError('Optional runtime not installed')
    installed = Path(spec.origin).parent.parent
    with zipfile.ZipFile(OUT/'ultralytics-8.1.0-py3-none-any.whl') as wheel:
        for name in wheel.namelist():
            if name.endswith('.py') and (installed/name).read_bytes() != wheel.read(name):
                raise ValueError('Installed upstream Python source differs')
    # Python 3.12 removed stdlib distutils. The pinned Hub SDK imports it;
    # activate the already-pinned setuptools compatibility before that import.
    import setuptools  # noqa: F401
    import ultralytics
    from ultralytics.utils import SETTINGS, AUTOINSTALL
    from ultralytics.nn.tasks import DetectionModel
    if ultralytics.__version__ != '8.1.0' or SETTINGS['sync'] or SETTINGS['hub'] or AUTOINSTALL:
        raise ValueError('Pinned local runtime settings differ')
    path = OUT/'yolov8n.pt'
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unsafe != set(ALLOWED):
        raise ValueError('Checkpoint class allowlist differs')
    allowed = [(getattr(importlib.import_module(module), name), alias) for alias, (module, name) in ALLOWED.items()]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    saved = checkpoint['model'].float().eval().requires_grad_(False)
    if type(saved) is not DetectionModel or saved.yaml['nc'] != 80 or saved.names[32] != 'sports ball':
        raise ValueError('Unexpected pretrained architecture/class meaning')
    model = DetectionModel(deepcopy(saved.yaml), ch=3, nc=80, verbose=False)
    model.load_state_dict(saved.state_dict(), strict=True)
    model.names = dict(saved.names)
    model.eval().requires_grad_(False)
    metadata = dict(checkpoint_date=checkpoint.get('date'), checkpoint_version=checkpoint.get('version'),
        training_data=checkpoint.get('train_args', {}).get('data'),
        training_model=checkpoint.get('train_args', {}).get('model'),
        parameter_count=sum(p.numel() for p in model.parameters()), tensor_count=len(model.state_dict()),
        class_32=model.names[32], weights_sha256=WEIGHTS_SHA, restricted_load=True,
        strict_architecture_rebuild=True, code_license='AGPL-3.0')
    return model, saved, metadata


def input_tensor(rgb):
    from ultralytics.data.augment import LetterBox
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError('Expected RGB8 image')
    padded = LetterBox(new_shape=(1280, 1280), auto=False, stride=32)(image=image)
    return torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)[None])).float()/255.


def detect(model, rgb):
    from ultralytics.utils import ops
    tensor = input_tensor(rgb)
    with torch.inference_mode():
        prediction = model(tensor)
        detections = ops.non_max_suppression(prediction, conf_thres=.25, iou_thres=.70,
            classes=[32], max_det=100)[0]
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], rgb.shape)
        boxes = detections.cpu().numpy()
    return np.column_stack([(boxes[:, 0]+boxes[:, 2])/2, (boxes[:, 1]+boxes[:, 3])/2, boxes[:, 4]])


def parity(model, saved):
    from types import SimpleNamespace
    from ultralytics.engine.predictor import BasePredictor
    torch.set_num_threads(2)
    rng = np.random.default_rng(109)
    graph, preprocessing = [], []
    for height, width in ((64, 96), (96, 160), (640, 640), (1280, 1280)):
        tensor = torch.from_numpy(rng.random((1, 3, height, width), dtype=np.float32))
        with torch.inference_mode():
            a, b = model(tensor)[0], saved(tensor)[0]
        error = float((a-b).abs().max())
        if error != 0:
            raise ValueError('Rebuilt/saved YOLO forward differs')
        graph.append(dict(shape=[height, width], maximum_abs_error=error))
    predictor = BasePredictor(overrides=dict(imgsz=1280, save=False, verbose=False))
    predictor.imgsz, predictor.device = (1280, 1280), torch.device('cpu')
    predictor.model = SimpleNamespace(pt=False, stride=32, fp16=False)
    for height, width in ((720, 1280), (1080, 1920), (480, 854)):
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        original = predictor.preprocess([rgb[:, :, ::-1].copy()])
        actual = input_tensor(rgb)
        if not torch.equal(actual, original):
            raise ValueError('RGB path differs from upstream BGR preprocessing')
        preprocessing.append(dict(shape=[height, width], tensor_exact=True))
    smoke = detect(model, np.zeros((64, 96, 3), np.uint8))
    if (smoke.ndim != 2 or smoke.shape[1] != 3 or not np.isfinite(smoke).all()
            or (smoke[:, :2] < 0).any() or (smoke[:, :2] > [96, 64]).any()
            or (smoke[:, 2] < .25).any() or (smoke[:, 2] > 1).any()):
        raise ValueError('End-to-end synthetic detector sanity failed')
    return dict(forward=graph, rgb_vs_upstream_bgr=preprocessing,
        full_detection_smoke=dict(input_shape=[64, 96], predictions=len(smoke), finite_original_pixel_points=True))
