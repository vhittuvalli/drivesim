"""Train the driving policy on data collected in the browser, then export it for the app.

    .venv/bin/python train/train.py                      # all runs in data/, writes models/policy.*
    .venv/bin/python train/train.py --init models/policy.pt --epochs 6    # DAgger: fine-tune

Uses the Apple GPU (MPS) or CUDA when available.
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DriveDataset, load_runs, sample_weight, split  # noqa: E402
from export import export  # noqa: E402
from model import Policy, losses  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def augment(img):
    """Photometric jitter on a uint8 batch -> float [0, 1]. No geometric changes: they'd break the
    relationship between image and waypoints."""
    x = img.float() / 255
    b = x.shape[0]
    rnd = lambda lo, hi: torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * rnd(0.6, 1.4)  # saturation
    x = (x - x.mean((1, 2, 3), keepdim=True)) * rnd(0.7, 1.3) + x.mean((1, 2, 3), keepdim=True)  # contrast
    x = x * rnd(0.65, 1.35)  # brightness
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums, n = {}, 0
    counts = {'light': [0, 0], 'signal': [0, 0]}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch['image'].float() / 255, batch['tele'].float() / 255, batch['speed'])
        _, parts = losses(pred, batch)
        wp = pred[0]
        taken = wp[torch.arange(len(wp)), batch['cmd']]
        target = batch['wp'][torch.arange(len(wp)), batch['cmd']]
        parts['lateral_at_8m'] = (taken[:, 3, 1] - target[:, 3, 1]).abs().mean().item()
        parts['seg_acc'] = (pred[2].argmax(1) == batch['seg']).float().mean().item()
        # Light accuracy as counts: per-batch means would score batches with no signal in view
        # (a highway stretch) as 0%.
        hit = pred[4].argmax(1) == batch['light']
        graded, lit = batch['light'] >= 0, batch['light'] > 0
        counts['light'] = [counts['light'][0] + int(hit[graded].sum()), counts['light'][1] + int(graded.sum())]
        counts['signal'] = [counts['signal'][0] + int(hit[lit].sum()), counts['signal'][1] + int(lit.sum())]
        k = len(wp)
        for key, val in parts.items():
            sums[key] = sums.get(key, 0) + val * k
        n += k
    model.train()
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out['light_acc'], out['signal_acc'] = (c / max(t, 1) for c, t in (counts['light'], counts['signal']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'policy'))
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--init', default=None, help='checkpoint to start from (DAgger fine-tuning)')
    a = ap.parse_args()

    runs = load_runs(a.data)
    if not runs:
        raise SystemExit(f'no data in {a.data}: collect some first (Collect button in the app)')
    train_items, val_items = split(runs)
    if not val_items or len(train_items) < a.batch:
        raise SystemExit(f'not enough data: {len(train_items)} train / {len(val_items)} val frames')
    by_driver = {}
    for _, r in train_items:
        by_driver[r.get('driver', 'expert')] = by_driver.get(r.get('driver', 'expert'), 0) + 1
    print(f'{len(runs)} runs, {len(train_items)} train / {len(val_items)} val frames, drivers {by_driver}')

    device = pick_device(a.device)
    train_ds = DriveDataset(a.data, train_items, speed_noise=0.3)
    sampler = WeightedRandomSampler([sample_weight(r) for _, r in train_items], num_samples=len(train_items), replacement=True)
    kw = dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(DriveDataset(a.data, val_items), shuffle=False, **kw)

    model = Policy().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu')['model'])
        print(f'fine-tuning from {a.init}')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    print(f'training on {device}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {steps} steps')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best, best_metrics = float('inf'), None
    for epoch in range(1, a.epochs + 1):
        t0, run_loss = time.time(), 0.0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(augment(batch['image']), augment(batch['tele']), batch['speed'])
            loss, _ = losses(pred, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
        m = evaluate(model, val_dl, device)
        score = m['wp'] + 0.5 * m['speed']
        tag = ''
        if score < best:
            best, best_metrics, tag = score, m, '  * best'
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': m}, a.out + '.pt')
        print(
            f'epoch {epoch:2d}  train {run_loss / len(train_dl):.3f}  val wp {m["wp"]:.3f} m  lateral@8m {m["lateral_at_8m"]:.2f} m  '
            f'speed {m["speed"]:.2f} m/s  light acc {m["light_acc"]:.1%} (at signals within 50 m {m["signal_acc"]:.1%})  seg acc {m["seg_acc"]:.1%}  depth {m["depth"]:.3f}  ({time.time() - t0:.0f}s){tag}'
        )

    ck = torch.load(a.out + '.pt', map_location='cpu')
    info = {
        'trained': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'frames': len(train_items), 'runs': len(runs), 'drivers': by_driver, 'epoch': ck['epoch'],
        'val': {k: round(v, 4) for k, v in best_metrics.items()}, 'init': a.init,
    }
    ck['info'] = info
    torch.save(ck, a.out + '.pt')
    model = Policy()
    model.load_state_dict(ck['model'])
    export(model, a.out, info)


if __name__ == '__main__':
    main()
