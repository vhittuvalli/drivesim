"""Train the driving policy on data collected in the browser, then export it for the app.

    .venv/bin/python train/train.py                      # all runs in data/, writes models/policy.*
    .venv/bin/python train/train.py --init models/policy.pt --epochs 6    # DAgger: fine-tune

Uses the Apple GPU (MPS) or CUDA when available.
"""
import argparse
import copy
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


def augment(*imgs):
    """Photometric jitter on uint8 batches -> float [0, 1], the same jitter for every image of a sample
    (so the change between the current and past frame is motion, not augmentation). No geometric
    changes: they'd break the relationship between image and waypoints."""
    b, dev = imgs[0].shape[0], imgs[0].device
    rnd = lambda lo, hi: torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)
    sat, con, bri = rnd(0.6, 1.4), rnd(0.7, 1.3), rnd(0.65, 1.35)
    out = []
    for img in imgs:
        x = img.float() / 255
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * sat  # saturation
        x = (x - x.mean((1, 2, 3), keepdim=True)) * con + x.mean((1, 2, 3), keepdim=True)  # contrast
        x = x * bri  # brightness
        x = x + torch.randn_like(x) * 0.02
        out.append(x.clamp(0, 1))
    return out if len(out) > 1 else out[0]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums, n = {}, 0
    counts = {'light': [0, 0], 'signal': [0, 0], 'stop': [0, 0], 'gap': [0, 0], 'veh': [0, 0], 'ttc_r': [0, 0], 'ttc_p': [0, 0]}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch['image'].float() / 255, batch['tele'].float() / 255, batch['speed'], batch['past'].float() / 255, batch['past2'].float() / 255)
        _, parts = losses(pred, batch)
        wp = pred[0]
        taken = wp[torch.arange(len(wp)), batch['cmd']]
        target = batch['wp'][torch.arange(len(wp)), batch['cmd']]
        parts['lateral_at_8m'] = (taken[:, 3, 1] - target[:, 3, 1]).abs().mean().item()
        parts['seg_acc'] = (pred[2].argmax(1) == batch['seg']).float().mean().item()
        veh = batch['seg'] == 8  # recall on vehicle pixels: what the braking depends on
        counts['veh'] = [counts['veh'][0] + int((pred[2].argmax(1)[veh] == 8).sum()), counts['veh'][1] + int(veh.sum())]
        sm = batch['stop_mask'] > 0
        # Time to collision: does it flag the dangerous closings (1/TTC > 0.4), and only those?
        tm = batch['ttc_mask'] > 0
        truth, flag = (batch['ttc'] > 0.4) & tm, (pred[8] > 0.4) & tm
        counts['ttc_r'] = [counts['ttc_r'][0] + int((flag & truth).sum()), counts['ttc_r'][1] + int(truth.sum())]
        counts['ttc_p'] = [counts['ttc_p'][0] + int((flag & truth).sum()), counts['ttc_p'][1] + int(flag.sum())]
        lm = (batch['lead_mask'] > 0) & (batch['lead'][:, 0] < 30)  # gap error where it matters: obstacles within 30 m
        counts['gap'] = [counts['gap'][0] + float((pred[6][lm, 0] - batch['lead'][lm, 0]).abs().sum()), counts['gap'][1] + int(lm.sum())]
        counts['stop'] = [counts['stop'][0] + float((pred[5][sm] - batch['stop'][sm]).abs().sum()), counts['stop'][1] + int(sm.sum())]
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
    out['light_acc'], out['signal_acc'], out['stop_err'], out['gap_err'], out['veh_recall'] = (c / max(t, 1) for c, t in (counts['light'], counts['signal'], counts['stop'], counts['gap'], counts['veh']))
    out['ttc_recall'], out['ttc_precision'] = (c / max(t, 1) for c, t in (counts['ttc_r'], counts['ttc_p']))
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
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run from <out>.last.pt')
    ap.add_argument('--retries', type=int, default=3, help='restarts of an epoch after a data-loader failure (e.g. after sleep)')
    ap.add_argument('--speed-dropout', type=float, default=0.5, help='share of training samples whose speed is hidden from the target-speed heads')
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
    val_ds = DriveDataset(a.data, val_items)

    def loaders():
        # Rebuilt after a failure: worker processes and their shared memory don't survive sleep well.
        return DataLoader(train_ds, sampler=sampler, drop_last=True, **kw), DataLoader(val_ds, shuffle=False, **kw)

    train_dl, val_dl = loaders()

    model = Policy().to(device)
    if a.init:
        own = model.state_dict()
        ck = torch.load(a.init, map_location='cpu')['model']
        init = {k: v for k, v in ck.items() if k in own and own[k].shape == v.shape}
        # A checkpoint from before the older past frame: its first fc layer takes [current, change
        # since 0.3 s]. Keep those weights and start the new change-since-1 s inputs at zero, so the
        # network begins exactly where the checkpoint was and learns to use them.
        fw = ck.get('fc.1.weight')
        if fw is not None and fw.shape != own['fc.1.weight'].shape and fw.shape[0] == own['fc.1.weight'].shape[0]:
            w = torch.zeros_like(own['fc.1.weight'])
            w[:, : fw.shape[1]] = fw
            init['fc.1.weight'] = w
            print('older past frame: new fc inputs start at zero')
        missing, _ = model.load_state_dict(init, strict=False)
        if missing:
            print(f'new layers (not in {a.init}): {sorted({k.split(".")[0] for k in missing})}')
        # Fill in only the heads the checkpoint doesn't have: target-speed heads (split off the
        # combined heads) start from the combined heads' speed output.
        gone = lambda prefix: any(k.startswith(prefix) for k in missing)
        with torch.no_grad():
            for i, (h, sh) in enumerate(zip(model.heads, model.speed_heads)):
                if gone(f'speed_heads.{i}.'):
                    sh[0].load_state_dict(h[0].state_dict())
                    sh[2].weight.copy_(h[2].weight[-1:])
                    sh[2].bias.copy_(h[2].bias[-1:])
                    print(f'target-speed head {i} initialized from the combined head')
        print(f'fine-tuning from {a.init}')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    print(f'training on {device}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {steps} steps')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best, best_metrics, start = float('inf'), None, 1
    last_path = a.out + '.last.pt'
    if a.resume:
        if not os.path.isfile(last_path):
            raise SystemExit(f'nothing to resume: {last_path} not found')
        st = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        best, best_metrics, start = st['best'], st['best_metrics'], st['epoch'] + 1
        print(f'resuming after epoch {st["epoch"]} (best score {best:.3f})')

    def train_epoch():
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            image, past, past2 = augment(batch['image'], batch['past'], batch['past2'])
            # Hide the speed reading in a share of samples, so the speed head can't just copy it and
            # has to read motion from the two frames (the targets keep the true speed).
            batch['true_speed'] = batch['speed']
            hide = (torch.rand(len(image), 1, device=image.device) < a.speed_dropout).float()
            pred = model(image, augment(batch['tele']), batch['speed'], past, past2, speed_v=batch['speed'] * (1 - hide))
            loss, _ = losses(pred, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
        return total

    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        # A data-loader failure (typically "Shared memory manager connection has timed out" after
        # the machine slept) restarts the epoch from its starting weights instead of ending the run.
        snap = ({k: v.detach().clone() for k, v in model.state_dict().items()}, copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()))
        for attempt in range(a.retries + 1):
            try:
                run_loss = train_epoch()
                m = evaluate(model, val_dl, device)
                break
            except RuntimeError as e:
                if attempt == a.retries:
                    raise
                print(f'epoch {epoch}: data loader failed ({str(e).splitlines()[0]}); restarting the epoch')
                model.load_state_dict(snap[0])
                opt.load_state_dict(snap[1])
                sched.load_state_dict(snap[2])
                train_dl, val_dl = loaders()
        score = m['wp'] + 0.5 * m['speed']
        tag = ''
        if score < best:
            best, best_metrics, tag = score, m, '  * best'
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': m}, a.out + '.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch,
                    'best': best, 'best_metrics': best_metrics}, last_path)
        print(
            f'epoch {epoch:2d}  train {run_loss / len(train_dl):.3f}  val wp {m["wp"]:.3f} m  lateral@8m {m["lateral_at_8m"]:.2f} m  '
            f'speed {m["speed"]:.2f} m/s  light acc {m["light_acc"]:.1%} (at signals within 50 m {m["signal_acc"]:.1%})  stop line ±{m["stop_err"]:.1f} m  lead gap (<30 m) ±{m["gap_err"]:.1f} m  vehicle px recall {m["veh_recall"]:.1%}  speed from vision ±{m["ego_speed"]:.2f} m/s  danger (TTC<2.5 s) recall {m["ttc_recall"]:.0%} precision {m["ttc_precision"]:.0%}  seg acc {m["seg_acc"]:.1%}  depth {m["depth"]:.3f}  ({time.time() - t0:.0f}s){tag}'
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
