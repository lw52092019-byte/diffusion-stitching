"""Unconditional DDPM on 28x56 two-cell MNIST images (digits 0-4).

Data: left digit ~ Uniform{0..4}; right digit = left with prob 1-kappa, else Uniform{0..4}.
Pixel range [-1, 1]. Model: diffusers UNet2DModel. Sampler: DDPM ancestral, 100 steps.
"""
import copy
import os

import torch
from diffusers import DDPMScheduler

from common import CKPT_DIR, DEVICE, NUM_CLASSES, load_mnist04
from unet import SmallUNet

H, W = 28, 56
NUM_TRAIN_T = 1000
NUM_SAMPLE_STEPS = 100


UNET_CFG = dict(ch=(32, 64, 128), blocks=2, attn_lowest=True)


def build_unet():
    # Hand-written UNet (diffusers' UNet2DModel was too slow on this machine for the budget).
    return SmallUNet(**UNET_CFG)


def build_scheduler():
    # Linear-beta DDPM; "trailing" spacing so 100-step sampling starts at t = 999.
    return DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_T,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=True,
        variance_type="fixed_small",
        timestep_spacing="trailing",
    )


def ckpt_path(kappa, step=None):
    suffix = "" if step is None else f"_step{step}"
    return os.path.join(CKPT_DIR, f"ddpm_kappa{kappa:g}{suffix}.pt")


class PairSampler:
    """Draws two-cell images on the fly from MNIST-train digits 0-4."""

    def __init__(self, kappa, device=DEVICE):
        x, y = load_mnist04(train=True, device=device)
        order = torch.argsort(y)
        self.x = x[order]
        counts = torch.bincount(y, minlength=NUM_CLASSES)
        self.counts = counts.to(device)
        self.offsets = torch.cumsum(counts, 0).sub(counts).to(device)
        self.kappa = kappa
        self.device = device

    def labels(self, n, gen=None):
        left = torch.randint(0, NUM_CLASSES, (n,), device=self.device, generator=gen)
        indep = torch.randint(0, NUM_CLASSES, (n,), device=self.device, generator=gen)
        use_indep = torch.rand(n, device=self.device, generator=gen) < self.kappa
        right = torch.where(use_indep, indep, left)
        return left, right

    def _pick(self, cls, gen=None):
        r = torch.rand(len(cls), device=self.device, generator=gen)
        idx = self.offsets[cls] + (r * self.counts[cls]).long().clamp_max(self.counts[cls] - 1)
        return self.x[idx]

    def sample(self, n, gen=None):
        left, right = self.labels(n, gen)
        img = torch.cat([self._pick(left, gen), self._pick(right, gen)], dim=3)  # (n,1,28,56)
        return img, left, right


def load_model(kappa, device=DEVICE, step=None):
    """Load EMA weights; `step` selects an intermediate checkpoint."""
    unet = build_unet().to(device)
    state = torch.load(ckpt_path(kappa, step), map_location=device)
    unet.load_state_dict(state["ema"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, decay=None):
        d = self.decay if decay is None else decay
        torch._foreach_lerp_(list(self.model.parameters()), list(model.parameters()), 1 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)


@torch.no_grad()
def eps_model(unet, x, t):
    """Noise prediction in bf16 autocast; returns float32."""
    tt = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = unet(x, tt)
    return out.float()


@torch.no_grad()
def sample_plain(unet, n, gen, steps=NUM_SAMPLE_STEPS, chunk=1024):
    """Independent DDPM ancestral sampling (no resampling). Returns x0 in [-1,1]."""
    sched = build_scheduler()
    sched.set_timesteps(steps, device=DEVICE)
    outs = []
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        x = torch.randn(m, 1, H, W, device=DEVICE, generator=gen)
        for t in sched.timesteps:
            x = sched.step(eps_model(unet, x, t), t, x, generator=gen).prev_sample
        outs.append(x.clamp(-1, 1))
    return torch.cat(outs)
