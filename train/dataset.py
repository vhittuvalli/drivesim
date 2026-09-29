"""Reads the runs collected by the browser (data/<run>/, see src/collect.js).

Each sample: roof-camera image, current speed, and targets:
  wp       (3, 8, 2)  waypoints (x forward, y left, meters) for each command branch
  wp_mask  (3,)       which branches have a label (all of them away from intersections)
  cmd      ()         index of the command actually taken (the branch the speed label is for)
  v_target ()         speed the expert will have in 1 s
  seg      (32, 64)   semantic class per pixel (auxiliary target)
  depth    (32, 64)   depth / 100 m (auxiliary target)
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

COMMANDS = ['left', 'straight', 'right']
N_WP = 8
AUX_STRIDE = 4  # label images are 256x128 -> 64x32 auxiliary maps


def load_runs(root):
    """All runs under root as {run: [sample rows]}, keeping only rows whose files exist."""
    runs = {}
    for run in sorted(os.listdir(root)):
        path = os.path.join(root, run, 'samples.jsonl')
        if not os.path.isfile(path):
            continue
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                name = f"{r['frame']:06d}"
                if os.path.isfile(os.path.join(root, run, 'frames', name + '.jpg')) and os.path.isfile(os.path.join(root, run, 'labels', name + '.png')):
                    rows.append(r)
        if rows:
            runs[run] = rows
    return runs


def split(runs, val_frac=0.1, block=200):
    """Train/val split. By run when there are enough runs (different cities and conditions), else by
    contiguous blocks of frames so neighbouring, near-identical frames don't leak across."""
    names = sorted(runs)
    if len(names) >= 5:
        n_val = max(1, round(len(names) * val_frac))
        val = set(names[::max(1, len(names) // n_val)][:n_val])
        pick = lambda run, i: run in val
    else:
        every = max(2, round(1 / val_frac))
        # Blocks shrink for short runs so each run still contributes some validation frames.
        size = {run: max(10, min(block, len(runs[run]) // (every * 2))) for run in names}
        pick = lambda run, i: (i // size[run]) % every == every - 1
    train, val_items = [], []
    for run in names:
        for i, r in enumerate(runs[run]):
            (val_items if pick(run, i) else train).append((run, r))
    return train, val_items


def sample_weight(r):
    """Oversample the informative moments: turns, speed changes. Undersample waiting at lights."""
    w = 1.0
    if r['command'] != 'straight' and r['cmd_dist'] is not None and r['cmd_dist'] < 30:
        w *= 2.5
    if abs(r['v_target'] - r['v']) > 1:
        w *= 2
    if r['v'] < 0.3 and r['v_target'] < 0.3:
        w *= 0.4
    if r.get('overtaking'):
        w *= 2
    return w


class DriveDataset(Dataset):
    def __init__(self, root, items, speed_noise=0.0):
        self.root, self.items, self.speed_noise = root, items, speed_noise

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        run, r = self.items[i]
        name = f"{r['frame']:06d}"
        img = np.array(Image.open(os.path.join(self.root, run, 'frames', name + '.jpg')).convert('RGB'), dtype=np.uint8)
        lab = np.array(Image.open(os.path.join(self.root, run, 'labels', name + '.png')).convert('RGB'), dtype=np.uint8)
        s = AUX_STRIDE
        seg = lab[s // 2::s, s // 2::s, 0].astype(np.int64)
        h, w = lab.shape[0] // s, lab.shape[1] // s
        depth = lab[: h * s, : w * s, 1].reshape(h, s, w, s).mean(axis=(1, 3)) / 255.0
        depth[seg == 0] = 1.0  # sky / nothing: far away

        wp = np.zeros((3, N_WP, 2), np.float32)
        mask = np.zeros(3, np.float32)
        for k, c in enumerate(COMMANDS):
            pts = r['wp'].get(c)
            if pts:
                wp[k] = pts
                mask[k] = 1
        v = r['v'] + (np.random.randn() * self.speed_noise if self.speed_noise else 0.0)
        return {
            'image': torch.from_numpy(img).permute(2, 0, 1),  # uint8 CHW
            'speed': torch.tensor([max(0.0, v)], dtype=torch.float32),
            'wp': torch.from_numpy(wp),
            'wp_mask': torch.from_numpy(mask),
            'cmd': torch.tensor(COMMANDS.index(r['command'])),
            'v_target': torch.tensor(r['v_target'], dtype=torch.float32),
            'seg': torch.from_numpy(seg),
            'depth': torch.from_numpy(depth.astype(np.float32)),
        }
