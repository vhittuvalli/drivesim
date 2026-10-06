"""Reads the runs collected by the browser (data/<run>/, see src/collect.js).

Each sample: roof-camera image, current speed, and targets:
  tele     traffic-light camera image (runs from label version 3 on; older runs are skipped)
  past     the main camera PAST_DT seconds earlier (the same frame when there is none)
  past2    the main camera PAST2_DT seconds earlier (likewise)
  light    ()         signal ahead: index into LIGHTS
  wp       (3, 8, 2)  waypoints (x forward, y left, meters) for each command branch
  wp_mask  (3,)       which branches have a label (all of them away from intersections)
  cmd      ()         index of the command actually taken (the branch the speed label is for)
  v_target ()         speed the expert will have in 1 s
  seg      (32, 64)   semantic class per pixel (auxiliary target)
  depth    (32, 64)   depth / 100 m (auxiliary target)
"""
import json
import os
import re
import zlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

COMMANDS = ['left', 'straight', 'right', 'overtake']  # must match src/labels.js


def overtake_command(r):
    """Rows from before the overtake command: an expert overtake (labeled with the next turn's
    command and the offset path) becomes the 'overtake' command with that path."""
    if r.get('overtaking') and r['command'] != 'overtake' and r.get('road') != 'highway':
        path = r['wp'].get(r['command'])
        r['wp'] = {'overtake': path}
        r['command'] = 'overtake'
    return r
LIGHTS = ['none', 'red', 'yellow', 'green']  # must match src/labels.js
# Beyond this distance a lamp is a pixel or two even in the traffic-light camera, so the light
# label isn't learnable there: it is left out of the light loss (driving labels are unaffected).
LIGHT_READABLE = 50.0  # m to the intersection
IGNORE = -100


HALF_LEN = 2.35  # car center to front bumper (src/planner.js)


STOP_FAR = 60.0  # stop-line distances beyond this are labeled as this: "far away"


def stop_target(r):
    """Front bumper to the stop line of the signal ahead (cmd_dist is from the car center), capped
    at STOP_FAR, and whether it is labeled. Far signals are labeled too (as STOP_FAR): unlabeled,
    the head answered anything there, including "right here" for a light 80 m away."""
    d = r.get('cmd_dist')
    if r['light'] == 'none' or d is None or r.get('road') == 'highway':
        return 0.0, 0.0
    return min(max(0.0, d - HALF_LEN), STOP_FAR), 1.0


def lead_target(r):
    """(gap, speed) of the lead obstacle and whether it is labeled (label version 4 on)."""
    if 'lead_gap' not in r:
        return (80.0, r['v']), 0.0
    return (r['lead_gap'], r['lead_v']), 1.0


TTC_INV_MAX = 2.0  # 1/s


def ttc_target(r):
    """1 / time to collision with the lead obstacle (0 when not closing or nothing ahead), and
    whether it is labeled (label version 4 on). Scale-free, so two frames can show it as looming."""
    if 'lead_gap' not in r:
        return 0.0, 0.0
    if r['lead_gap'] >= 80:
        return 0.0, 1.0
    return min(TTC_INV_MAX, max(0.0, r['v'] - r['lead_v']) / max(r['lead_gap'], 1.0)), 1.0


def light_target(r):
    if r['light'] != 'none' and r.get('cmd_dist') is not None and r['cmd_dist'] > LIGHT_READABLE:
        return IGNORE
    return LIGHTS.index(r['light'])
N_WP = 8
AUX_STRIDE = 4  # label images are 256x128 -> 64x32 auxiliary maps
PAST_DT = 0.3  # s: the earlier main-camera frame the network also sees (exported to policy.json)
PAST2_DT = 1.0  # s: and an older one, long enough for a closing car to visibly grow


def link_past(rows, dt=PAST_DT, tol=0.06, key='past'):
    """Give each row the frame number of the row about dt seconds earlier in the same run (itself
    when there is none, e.g. at the start of an episode or across an upload gap)."""
    rows = sorted(rows, key=lambda r: r['t'])
    j = 0
    for r in rows:
        want = r['t'] - dt
        while j + 1 < len(rows) and rows[j + 1]['t'] <= want + tol:
            j += 1
        r[key] = rows[j]['frame'] if abs(rows[j]['t'] - want) <= tol else r['frame']
    return rows
# Label version 2 (src/labels.js) spreads waypoints out with speed above this; version-1 rows
# faster than it have fixed spacing, which means something else, so they are left out.
WP_SCALE_SPEED = 12.0


def usable(r):
    return r.get('labels', 1) >= 2 or r['v'] <= WP_SCALE_SPEED


def load_runs(root):
    """All runs under root as {run: [sample rows]}, keeping only rows whose files exist and whose
    labels mean the same as the current version."""
    runs = {}
    for run in sorted(os.listdir(root)):
        path = os.path.join(root, run, 'samples.jsonl')
        if not os.path.isfile(path):
            continue
        rows, seen = [], set()
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r['frame'] in seen:  # a retried upload that had landed the first time
                    continue
                seen.add(r['frame'])
                name = f"{r['frame']:06d}"
                files = [os.path.join(root, run, d, name + ext) for d, ext in (('frames', '.jpg'), ('tele', '.jpg'), ('labels', '.png'))]
                if usable(r) and 'light' in r and all(os.path.isfile(p) for p in files):
                    rows.append(overtake_command(r))
        if rows:
            runs[run] = link_past(link_past(rows), PAST2_DT, key='past2')
    return runs


def city_of(run):
    """The city seed in a run name (data/<stamp>-s<seed>-<driver>-e<episode>), or the run itself."""
    m = re.search(r'-s(\d+)-', run)
    return m.group(1) if m else run


def split(runs, val_frac=0.1, block=200):
    """Train/val split. By city when there are enough runs: a city goes to validation by a stable
    hash of its seed, so validation is always cities the network never trained on, and adding data
    never moves a city from one side to the other (the old every-Nth-run split did, and then scored
    fine-tuned models on runs an earlier stage had trained on). Few runs: contiguous blocks of
    frames, so neighbouring, near-identical frames don't leak across."""
    names = sorted(runs)
    if len(names) >= 5:
        val = {run for run in names if zlib.crc32(city_of(run).encode()) % round(1 / val_frac) == 0}
        if not val or len(val) == len(names):  # degenerate hash outcome on tiny datasets
            val = set(names[:: max(2, round(1 / val_frac))])
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
    """Oversample the informative moments: turns, lane changes, speed changes, pulling away from a
    stop and hard braking. Undersample waiting at lights."""
    w = 1.0
    if r['command'] != 'straight' and r['cmd_dist'] is not None and r['cmd_dist'] < 30:
        w *= 2.5
    if r['command'] != 'straight' and r.get('road') == 'highway':  # a lane change
        w *= 2.5
    if abs(r['v_target'] - r['v']) > 1:
        w *= 2
    if r['v'] < 0.3 and r['v_target'] < 0.3:
        # Waiting at a red light is the counterpart of pulling away at a green one: keep it.
        w *= 1.5 if r.get('light') in ('red', 'yellow') else 0.4
    if r['v'] < 1 and r['v_target'] > r['v'] + 0.5:  # pulling away: rare, and the network stalls without it
        w *= 4
    if r['v_target'] < r['v'] - 2:  # braking hard
        w *= 1.5
    if r.get('lead_gap', 80) < 30:  # an obstacle close ahead: the lead output's important cases
        w *= 2
    if r.get('overtaking'):
        w *= 2
    if ttc_target(r)[0] > 0.4:  # closing fast on something: the braking decisions
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
        tele = np.array(Image.open(os.path.join(self.root, run, 'tele', name + '.jpg')).convert('RGB'), dtype=np.uint8)
        past_name = f"{r.get('past', r['frame']):06d}"
        past = np.array(Image.open(os.path.join(self.root, run, 'frames', past_name + '.jpg')).convert('RGB'), dtype=np.uint8)
        past2_name = f"{r.get('past2', r['frame']):06d}"
        past2 = np.array(Image.open(os.path.join(self.root, run, 'frames', past2_name + '.jpg')).convert('RGB'), dtype=np.uint8)
        lab = np.array(Image.open(os.path.join(self.root, run, 'labels', name + '.png')).convert('RGB'), dtype=np.uint8)
        s = AUX_STRIDE
        seg = lab[s // 2::s, s // 2::s, 0].astype(np.int64)
        h, w = lab.shape[0] // s, lab.shape[1] // s
        depth = lab[: h * s, : w * s, 1].reshape(h, s, w, s).mean(axis=(1, 3)) / 255.0
        depth[seg == 0] = 1.0  # sky / nothing: far away

        wp = np.zeros((len(COMMANDS), N_WP, 2), np.float32)
        mask = np.zeros(len(COMMANDS), np.float32)
        for k, c in enumerate(COMMANDS):
            pts = r['wp'].get(c)
            if pts:
                wp[k] = pts
                mask[k] = 1
        v = r['v'] + (np.random.randn() * self.speed_noise if self.speed_noise else 0.0)
        return {
            'image': torch.from_numpy(img).permute(2, 0, 1),  # uint8 CHW
            'tele': torch.from_numpy(tele).permute(2, 0, 1),
            'past': torch.from_numpy(past).permute(2, 0, 1),
            'past2': torch.from_numpy(past2).permute(2, 0, 1),
            'light': torch.tensor(light_target(r)),
            'stop': torch.tensor(stop_target(r)[0], dtype=torch.float32),
            'stop_mask': torch.tensor(stop_target(r)[1], dtype=torch.float32),
            'lead': torch.tensor(lead_target(r)[0], dtype=torch.float32),
            'lead_mask': torch.tensor(lead_target(r)[1], dtype=torch.float32),
            'ttc': torch.tensor(ttc_target(r)[0], dtype=torch.float32),
            'ttc_mask': torch.tensor(ttc_target(r)[1], dtype=torch.float32),
            'speed': torch.tensor([max(0.0, v)], dtype=torch.float32),
            'wp': torch.from_numpy(wp),
            'wp_mask': torch.from_numpy(mask),
            'cmd': torch.tensor(COMMANDS.index(r['command'])),
            'v_target': torch.tensor(r['v_target'], dtype=torch.float32),
            'seg': torch.from_numpy(seg),
            'depth': torch.from_numpy(depth.astype(np.float32)),
        }
