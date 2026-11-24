"""Level 3 (shared style): style transforms, style evaluators, paths.

A style s in {0,1,2,3}: thick = s // 2, dark = s % 2. Both cells of an image share one style.
  brightness: bright = pixels x 1.0, dark = pixels x DARK_FACTOR   (applied in [0,1] space)
  thickness : thin = original MNIST, thick = grey-level dilation (max-pool, kernel DILATE_K)
Final parameters are recorded in level3/checkpoints/style_params.json.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, NUM_CLASSES, ROOT, load_mnist04

L3_DIR = os.path.join(ROOT, "level3")
L3_CKPT = os.path.join(L3_DIR, "checkpoints")
L3_CACHE = os.path.join(L3_DIR, "cache")     # large caches: deleted after everything is done
L3_RES = os.path.join(L3_DIR, "results")
L3_LOGS = os.path.join(L3_DIR, "logs")
for _d in (L3_DIR, L3_CKPT, L3_CACHE, L3_RES, L3_LOGS):
    os.makedirs(_d, exist_ok=True)

DARK_FACTOR = 0.5
DILATE_K = 5   # 3x3 dilation gave only 94.6% thickness accuracy on thin digits (< 98%); widened to 5x5
TOPK = 20                 # brightness feature = mean of the TOPK brightest pixels of a cell
BRIGHT_THRESH = 0.75      # feature >= thresh -> bright
BRIGHT_CLEAR = (0.65, 0.85)  # "clearly judged": feature < lo (dark) or > hi (bright)
THICK_CLEAR = 0.9         # thickness classifier: max prob >= this counts as clearly judged
STYLE_NAMES = ["bright-thin", "dark-thin", "bright-thick", "dark-thick"]


def style_parts(s):
    return s // 2, s % 2   # thick, dark


def apply_style(x, s):
    """x (B,1,28,28) in [-1,1], s (B,) style ids -> styled cells in [-1,1]."""
    u = (x + 1) / 2
    thick, dark = style_parts(s)
    if DILATE_K > 1:
        dil = F.max_pool2d(u, DILATE_K, 1, DILATE_K // 2)
        u = torch.where(thick.bool().view(-1, 1, 1, 1), dil, u)
    u = u * torch.where(dark.bool(), DARK_FACTOR, 1.0).view(-1, 1, 1, 1)
    return u * 2 - 1


def apply_style_pair(img, s):
    """img (B,1,28,56): same style on both cells (per cell, so dilation never crosses the midline)."""
    return torch.cat([apply_style(img[..., :28], s), apply_style(img[..., 28:], s)], 3)


class StylePairSampler:
    """Left / right digits independent uniform (kappa = 1); one style per image, uniform over 4."""

    def __init__(self, device=DEVICE):
        from diffusion import PairSampler
        self.base = PairSampler(1.0, device=device)
        self.device = device

    def sample(self, n, gen=None):
        img, l, r = self.base.sample(n, gen)
        s = torch.randint(0, 4, (n,), device=self.device, generator=gen)
        return apply_style_pair(img, s), l, r, s


# ----------------------------------------------------------------------------- style evaluators
def brightness_feature(cell):
    """cell (B,1,28,28) in [-1,1] -> (B,) mean of the TOPK brightest pixels, in [0,1]."""
    u = ((cell + 1) / 2).flatten(1)
    return u.topk(TOPK, dim=1).values.mean(1)


def judge_brightness(cell):
    """-> (is_dark (B,) long: 1 = dark, clear (B,) bool)."""
    f = brightness_feature(cell)
    return (f < BRIGHT_THRESH).long(), (f < BRIGHT_CLEAR[0]) | (f > BRIGHT_CLEAR[1])


def normalize_brightness(cell):
    """Divide by the brightness feature so the thickness classifier is brightness-invariant."""
    u = (cell + 1) / 2
    f = brightness_feature(cell).clamp_min(0.05).view(-1, 1, 1, 1)
    return (u / f).clamp(0, 1) * 2 - 1


class ThickCNN(nn.Module):
    """Small thin/thick classifier on brightness-normalised cells."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = normalize_brightness(x)
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x)).mean(dim=(2, 3))
        return self.fc(x)


def _load(cls, name, device=DEVICE):
    m = cls().to(device)
    m.load_state_dict(torch.load(os.path.join(L3_CKPT, f"{name}.pt"), map_location=device))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_l3_classifiers(device=DEVICE):
    from common import EvalResNet, JudgeCNN
    return (_load(JudgeCNN, "digit_judge", device), _load(EvalResNet, "digit_evaluator", device),
            _load(ThickCNN, "thick_clf", device))


@torch.no_grad()
def cell_attrs(digit_clf, thick_clf, x, chunk=2048):
    """x (P,1,28,56) -> dict of per-cell tensors for left (L) and right (R):
    digit class / max prob, dark (0/1) / clear, thick (0/1) / thick max prob."""
    out = {}
    for side, cell in (("L", x[..., :28]), ("R", x[..., 28:])):
        lp = torch.cat([F.softmax(digit_clf(cell[i:i + chunk]).float(), -1) for i in range(0, len(cell), chunk)])
        tp = torch.cat([F.softmax(thick_clf(cell[i:i + chunk]).float(), -1) for i in range(0, len(cell), chunk)])
        d, dc = judge_brightness(cell)
        m, tm = lp.max(-1), tp.max(-1)
        out[side] = {"digit": m.indices, "dprob": m.values, "dark": d, "dark_clear": dc,
                     "thick": tm.indices, "tprob": tm.values}
    return out


def load_l3_model(device=DEVICE, step=None):
    """EMA weights of the level-3 DDPM (final, or an intermediate training state)."""
    from diffusion import build_unet
    p = os.path.join(L3_CKPT, "ddpm_l3.pt" if step is None else f"ddpm_l3_step{step}.pt")
    unet = build_unet().to(device)
    unet.load_state_dict(torch.load(p, map_location=device)["ema"])
    unet.eval()
    for q in unet.parameters():
        q.requires_grad_(False)
    return unet


def save_grid(x, nrow, path, pad=2):
    """x (B,1,H,W) in [-1,1] -> PNG grid (white separators so dark styles stay visible)."""
    import torchvision
    torchvision.utils.save_image(x.add(1).div(2).clamp(0, 1), path, nrow=nrow, padding=pad, pad_value=0.35)


def save_style_params(extra=None):
    p = {"DARK_FACTOR": DARK_FACTOR, "DILATE_K": DILATE_K, "TOPK": TOPK, "BRIGHT_THRESH": BRIGHT_THRESH,
         "BRIGHT_CLEAR": BRIGHT_CLEAR, "THICK_CLEAR": THICK_CLEAR,
         "thin": "original MNIST (no erosion)", "thick": f"grey dilation, max-pool {DILATE_K}x{DILATE_K}"}
    p.update(extra or {})
    json.dump(p, open(os.path.join(L3_CKPT, "style_params.json"), "w"), indent=2)


def load_digits(train):
    x, y = load_mnist04(train=train)
    assert y.max() < NUM_CLASSES
    return x, y
