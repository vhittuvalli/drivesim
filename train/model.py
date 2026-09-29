"""Driving policy: camera image + speed -> waypoints and target speed for each command.

Conditional imitation learning (Codevilla et al. 2018): a shared convolutional trunk with one
output head per navigation command (left / straight / right at the next intersection). All
heads are trained on every frame where the expert can label them (see src/labels.js); at run
time the route's command picks the head.

Auxiliary outputs, used for training signal and visualization:
  seg, depth   coarse semantic segmentation and depth (64x32) from a small decoder
  attention    VisualBackProp-style map of which image regions the trunk responds to
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CMD, N_WP, N_CLASSES = 3, 8, 11
IMAGE_SIZE = (128, 256)  # H, W
SPEED_SCALE = 15.0  # m/s
WP_SCALE = 10.0  # meters per output unit


def block(cin, cout, k, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))
        self.c1 = block(3, 24, 5, 2)  # 64x128
        self.c2 = block(24, 36, 3, 2)  # 32x64
        self.c3 = block(36, 48, 3, 2)  # 16x32
        self.c4 = block(48, 64, 3, 1)  # 16x32
        self.c5 = block(64, 96, 3, 2)  # 8x16
        self.c6 = block(96, 64, 3, 2)  # 4x8
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.speed_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(256 + 32, 128), nn.ReLU(inplace=True), nn.Linear(128, N_WP * 2 + 1)) for _ in range(N_CMD)
        )
        self.dec = block(64, 24, 3, 1)
        self.fuse = block(24 + 36, 24, 1, 1)
        self.seg = nn.Conv2d(24, N_CLASSES, 1)
        self.depth = nn.Conv2d(24, 1, 1)

    def forward(self, image, speed):
        """image: (B, 3, 128, 256) float in [0, 1]; speed: (B, 1) m/s.
        Returns waypoints (B, 3, 8, 2) meters, target speed (B, 3) m/s, seg logits (B, 11, 32, 64),
        depth (B, 1, 32, 64) in [0, 1], attention (B, 1, 32, 64) in [0, 1]."""
        x = (image - self.mean) / self.std
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        f4 = self.c4(f3)
        f5 = self.c5(f4)
        f6 = self.c6(f5)

        z = torch.cat([self.fc(f6), self.speed_fc(speed / SPEED_SCALE)], 1)
        out = torch.stack([h(z) for h in self.heads], 1)  # (B, 3, 17)
        wp = out[..., : N_WP * 2].reshape(-1, N_CMD, N_WP, 2) * WP_SCALE
        v_target = F.softplus(out[..., N_WP * 2]) * 4.0

        d = F.interpolate(self.dec(f4), size=f2.shape[-2:], mode='nearest')
        fused = self.fuse(torch.cat([d, f2], 1))
        seg = self.seg(fused)
        depth = torch.sigmoid(self.depth(fused))

        # VisualBackProp (Bojarski et al. 2016), simplified: multiply channel-averaged activations
        # from deep to shallow, upsampling as we go.
        m = f6.mean(1, keepdim=True)
        for f in (f5, f4, f3, f2):
            m = F.interpolate(m, size=f.shape[-2:], mode='nearest') * f.mean(1, keepdim=True)
        lo = m.amin(dim=(2, 3), keepdim=True)
        hi = m.amax(dim=(2, 3), keepdim=True)
        attention = (m - lo) / (hi - lo + 1e-6)
        return wp, v_target, seg, depth, attention


def losses(pred, batch, w_speed=0.5, w_seg=0.2, w_depth=2.0):
    """Masked L1 on every labelled branch, L1 on the taken branch's target speed, and the aux terms."""
    wp, v_target, seg, depth, _ = pred
    mask = batch['wp_mask'][:, :, None, None]
    # Nearer waypoints matter most for control: weight them up.
    near = torch.linspace(1.5, 0.7, N_WP, device=wp.device)[None, None, :, None]
    l_wp = ((wp - batch['wp']).abs() * near * mask).sum() / (mask.sum() * N_WP * 2).clamp(min=1)
    v = v_target.gather(1, batch['cmd'][:, None])[:, 0]
    l_speed = (v - batch['v_target']).abs().mean()
    l_seg = F.cross_entropy(seg, batch['seg'])
    l_depth = (depth[:, 0] - batch['depth']).abs().mean()
    total = l_wp + w_speed * l_speed + w_seg * l_seg + w_depth * l_depth
    return total, {'wp': l_wp.item(), 'speed': l_speed.item(), 'seg': l_seg.item(), 'depth': l_depth.item()}
