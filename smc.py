"""Milestone 4: three ways for a batch of N samples to use each other.

  A  independent DDPM ancestral sampling
  B  whole-image resampling (SMC / Feynman-Kac steering)
  C  stitching: left and right cells resampled independently, then hard-joined at the midline

Many batches (groups of N particles) are simulated in parallel as one big tensor; resampling
only happens within a group. Every method does exactly one UNet forward per particle per step
(100 steps), so model-forward counts are identical.

Resampling schedule (shared by B and C): at iteration i (0-based, 100 iterations) we take the
DDPM step x_t -> x_{t-1}, which also gives x0_hat(x_t) (scheduler's pred_original_sample).
If i is a resampling iteration, the judge scores x0_hat(x_t) and the *post-step* particles
x_{t-1} are resampled with those weights. Because the step is independent per particle given
x_t, this is distributionally the same as resampling x_t and then stepping, but needs no extra
model forward (and stitched images get their first forward at the next step).
Resampling iterations: i = 10, 20, ..., 90 intersected with the window
  full : all of them            early : i < 45  -> 10,20,30,40       late : i >= 45 -> 50,...,90
Baseline "previous score" for the difference potential = judge score at i = 0.
"""
import math
import time

import torch
import torch.nn.functional as F

from common import DEVICE, load_mnist04
from diffusion import H, NUM_SAMPLE_STEPS, W, build_scheduler, eps_model

HALF = W // 2
WINDOWS = {"full": (0, 1.0), "early": (0, 0.45), "late": (0.45, 1.0)}


def resample_iters(window, interval=10, steps=NUM_SAMPLE_STEPS):
    lo, hi = WINDOWS[window]
    return {i for i in range(interval, steps, interval) if lo * steps <= i < hi * steps}


@torch.no_grad()
def half_logprobs(clf, x):
    """x (P,1,28,56) -> log-softmax for left and right cells, each (P,5)."""
    both = torch.cat([x[..., :HALF], x[..., HALF:]], 0)
    # chunk 1024: an 8192 chunk added ~3 GB peak and pushed the process past VRAM (spill to
    # shared memory made B/C ~6x slower than A); chunking does not change results.
    lp = torch.cat([F.log_softmax(clf(both[i:i + 1024]).float(), -1) for i in range(0, len(both), 1024)])
    return lp[: len(x)], lp[len(x):]


def _ancestors(logw, n_groups, n, gen):
    """logw (P,) -> global ancestor indices (P,), multinomial with replacement within each group."""
    w = torch.softmax(logw.view(n_groups, n), dim=1)
    idx = torch.multinomial(w, n, replacement=True, generator=gen)
    return (idx + torch.arange(n_groups, device=logw.device)[:, None] * n).view(-1)


@torch.no_grad()
def sample(unet, judge, method, n_groups, n, lam=0.0, window="full", target=(1, 3), gen=None,
           interval=10, steps=NUM_SAMPLE_STEPS, chunk=2048, iters=None):
    """Returns final samples (n_groups*n, 1, 28, 56) in [-1,1], grouped consecutively, plus info.

    `iters` (optional): explicit set of resampling iterations, overriding `window`
    (used by timing.py; default behaviour unchanged)."""
    assert method in ("A", "B", "C")
    P = n_groups * n
    sched = build_scheduler()
    sched.set_timesteps(steps, device=DEVICE)
    its = (set(iters) if iters is not None else resample_iters(window, interval, steps)) \
        if method != "A" else set()
    x = torch.randn(P, 1, H, W, device=DEVICE, generator=gen)
    prevL = prevR = None
    n_resample, ess_log = 0, []
    for i, t in enumerate(sched.timesteps):
        eps = torch.cat([eps_model(unet, x[j:j + chunk], t) for j in range(0, P, chunk)])
        out = sched.step(eps, t, x, generator=gen)
        x = out.prev_sample
        if method == "A" or (i != 0 and i not in its):
            continue
        lpL, lpR = half_logprobs(judge, out.pred_original_sample)
        rL, rR = lpL[:, target[0]], lpR[:, target[1]]
        if i == 0:                       # baseline scores for the difference potentials
            prevL, prevR = rL, rR
            continue
        n_resample += 1
        if method == "B":
            r, prev = rL + rR, prevL + prevR
            logw = lam * (r - prev)
            a = _ancestors(logw, n_groups, n, gen)
            x, prevL, prevR = x[a], rL[a], rR[a]
            ess_log.append(_ess(logw, n_groups, n))
        else:  # C: independent left/right resampling, hard stitch at the midline
            logwL, logwR = lam * (rL - prevL), lam * (rR - prevR)
            aL = _ancestors(logwL, n_groups, n, gen)
            aR = _ancestors(logwR, n_groups, n, gen)
            x = stitch(x, aL, aR)
            prevL, prevR = rL[aL], rR[aR]
            ess_log.append(0.5 * (_ess(logwL, n_groups, n) + _ess(logwR, n_groups, n)))
    return x.clamp(-1, 1), {"n_resample": n_resample,
                            "mean_ess_frac": (sum(ess_log) / len(ess_log)) if ess_log else None}


def stitch(x, aL, aR):
    """New particle k = left half of particle aL[k] | right half of particle aR[k]."""
    return torch.cat([x[aL][..., :HALF], x[aR][..., HALF:]], dim=3)


def _ess(logw, n_groups, n):
    w = torch.softmax(logw.view(n_groups, n), dim=1)
    return (1.0 / (w ** 2).sum(1) / n).mean().item()


# --------------------------------------------------------------------------- metrics

@torch.no_grad()
def metrics(evaluator, x, n_groups, n, target=(1, 3)):
    """All metrics use the evaluator. Per-batch values -> mean and standard error over batches."""
    pL, pR = (lp.exp() for lp in half_logprobs(evaluator, x))
    mL, mR = pL.max(-1), pR.max(-1)
    okL, okR = mL.values >= 0.5, mR.values >= 0.5
    hitL = okL & (mL.indices == target[0])
    hitR = okR & (mR.indices == target[1])
    succ = hitL & hitR
    waste = ~(okL & okR)
    img = x.add(1).div(2)                                      # [0,1] for seam / diversity
    # Extra (not in TASK.md): the classifiers call a blank cell "1" with high confidence, so we
    # also report versions that require the cell's mean ink to lie in the real-digit range
    # ([0.5%, 99.5%] of MNIST-test digits of the predicted class).
    rng = _ink_ranges()
    inkL, inkR = img[..., :HALF].mean(dim=(1, 2, 3)), img[..., HALF:].mean(dim=(1, 2, 3))
    inkokL = (inkL >= rng[mL.indices, 0]) & (inkL <= rng[mL.indices, 1])
    inkokR = (inkR >= rng[mR.indices, 0]) & (inkR <= rng[mR.indices, 1])
    hitL_i, hitR_i = hitL & inkokL, hitR & inkokR
    waste_i = ~(okL & okR & inkokL & inkokR)
    seam = (img[..., HALF - 1] - img[..., HALF]).abs().mean(dim=(1, 2))

    g = lambda v: v.float().view(n_groups, n)
    per_batch = {
        "success": g(succ).mean(1),
        "any_success": g(succ).amax(1),
        "hit_left": g(hitL).mean(1),
        "hit_right": g(hitR).mean(1),
        "waste": g(waste).mean(1),
        "seam": g(seam).mean(1),
        "success_ink": g(hitL_i & hitR_i).mean(1),
        "any_success_ink": g(hitL_i & hitR_i).amax(1),
        "hit_left_ink": g(hitL_i).mean(1),
        "hit_right_ink": g(hitR_i).mean(1),
        "waste_ink": g(waste_i).mean(1),
        "left_near_blank": g(inkL < 0.02).mean(1),
    }
    res = {k: _mean_se(v) for k, v in per_batch.items()}

    # diversity: mean pairwise L2 among successful samples within a batch (batches with >= 2)
    flat = img.view(n_groups, n, -1)
    sm = succ.view(n_groups, n)
    d = torch.cdist(flat, flat)                                # (G,n,n)
    pair = sm[:, :, None] & sm[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)
    npair = pair.sum((1, 2))
    has = npair > 0
    div = (d * pair).sum((1, 2))[has] / npair[has]
    res["diversity"] = _mean_se(div) if has.any() else {"mean": None, "se": None}
    res["diversity"]["n_batches"] = int(has.sum())
    # fraction of exact duplicates among successful pairs (collapsed copies)
    res["dup_frac_success_pairs"] = ((d < 1e-3) & pair).sum().item() / max(1, pair.sum().item())
    return res


_INK = None


def _ink_ranges():
    global _INK
    if _INK is None:
        x, y = load_mnist04(train=False)
        ink = x.add(1).div(2).mean(dim=(1, 2, 3))
        q = torch.tensor([0.005, 0.995], device=ink.device)
        _INK = torch.stack([torch.quantile(ink[y == c], q) for c in range(5)])
    return _INK


def _mean_se(v):
    v = v.float()
    return {"mean": v.mean().item(), "se": (v.std() / math.sqrt(len(v))).item() if len(v) > 1 else None}


@torch.no_grad()
def select_D(judge, x, n_groups, n, target=(1, 3)):
    """Post-hoc selection on finished independent samples (method D, reuses A's samples).

    D-whole : per batch, the single sample with the highest judge total r_L + r_R.
    D-stitch: per batch, left half of the sample with the highest r_L, right half of the sample
              with the highest r_R, hard-joined at the midline.
    Returns (x_whole, x_stitch), each (n_groups, 1, 28, 56) -> evaluate with n = 1.
    """
    lpL, lpR = half_logprobs(judge, x)
    rL, rR = lpL[:, target[0]].view(n_groups, n), lpR[:, target[1]].view(n_groups, n)
    base = torch.arange(n_groups, device=x.device) * n
    whole = x[base + (rL + rR).argmax(1)]
    stitched = stitch(x, base + rL.argmax(1), base + rR.argmax(1))
    return whole, stitched


def run_config(unet, judge, evaluator, method, n_groups, n, lam=0.0, window="full", seed=0, **kw):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    x, info = sample(unet, judge, method, n_groups, n, lam, window, gen=gen, **kw)
    torch.cuda.synchronize()
    secs = time.time() - t0
    res = metrics(evaluator, x, n_groups, n)
    res.update({"method": method, "N": n, "n_batches": n_groups, "lam": lam,
                "window": window if method != "A" else None, "seed": seed,
                "seconds": round(secs, 1), **info})
    return res, x
