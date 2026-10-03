"""Export a trained checkpoint to ONNX for the browser (models/policy.onnx + models/policy.json).

    .venv/bin/python train/export.py models/policy.pt

Checks that ONNX Runtime reproduces the PyTorch outputs before writing.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import COMMANDS, LIGHTS, PAST_DT  # noqa: E402
from model import IMAGE_SIZE, N_WP, Policy  # noqa: E402

OUTPUTS = ['waypoints', 'v_target', 'seg', 'depth', 'light', 'stop_dist', 'lead', 'attention']
WP_DIST = [2, 4, 6, 8, 11, 14, 18, 23]  # must match src/labels.js
CLASSES = ['sky', 'road', 'marking', 'sidewalk', 'building', 'vegetation', 'pole', 'traffic light', 'vehicle', 'pedestrian', 'terrain']


def export(model, base, info=None):
    model = model.eval().cpu()
    h, w = IMAGE_SIZE
    image = torch.rand(1, 3, h, w)
    tele = torch.rand(1, 3, h, w)
    speed = torch.tensor([[5.0]])
    past = torch.rand(1, 3, h, w)
    onnx_path = base + '.onnx'
    torch.onnx.export(
        model, (image, tele, speed, past), onnx_path, input_names=['image', 'tele', 'speed', 'past'], output_names=OUTPUTS,
        dynamic_axes={n: {0: 'batch'} for n in ['image', 'tele', 'speed', 'past', *OUTPUTS]}, opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    got = sess.run(None, {'image': image.numpy(), 'tele': tele.numpy(), 'speed': speed.numpy(), 'past': past.numpy()})
    with torch.no_grad():
        want = [t.numpy() for t in model(image, tele, speed, past)]
    worst = max(float(np.abs(a - b).max()) for a, b in zip(got, want))
    if worst > 1e-3:
        raise SystemExit(f'ONNX output differs from PyTorch by {worst}')

    meta = {
        'inputs': {'image': [1, 3, h, w], 'tele': [1, 3, h, w], 'speed': [1, 1], 'past': [1, 3, h, w]},
        'past_dt': PAST_DT,
        'image': 'RGB, float in [0, 1], rows top to bottom (the roof camera, see src/sensor.js)',
        'tele': 'the traffic-light camera, same format (src/sensor.js TELE)',
        'lights': LIGHTS,
        'speed_units': 'm/s',
        'outputs': OUTPUTS,
        'commands': COMMANDS,
        'waypoint_distances': WP_DIST,
        'waypoint_frame': 'x forward, y left, meters, car center',
        'n_waypoints': N_WP,
        'classes': CLASSES,
        'depth_range': 100,
        'onnx_check_max_abs_diff': worst,
        **(info or {}),
    }
    with open(base + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    size = os.path.getsize(onnx_path) / 1e6
    print(f'wrote {onnx_path} ({size:.1f} MB) and {base}.json; ONNX matches PyTorch to {worst:.1e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out', default=None, help='output base path (default: checkpoint without .pt)')
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu')
    model = Policy()
    model.load_state_dict(ck['model'])
    export(model, a.out or a.checkpoint.removesuffix('.pt'), ck.get('info'))
