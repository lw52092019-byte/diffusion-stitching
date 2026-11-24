"""Acceptance check for a trained DDPM: sample independently, score each cell with the
evaluator (and judge for reference), save a sample grid.

Usage: python eval_model.py --kappa 1.0 [--n 4000]
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from common import DEVICE, NUM_CLASSES, ROOT, load_classifier, load_mnist04
from diffusion import load_model, sample_plain


@torch.no_grad()
def cell_probs(clf, x):
    """x: (n,1,28,56) in [-1,1] -> left/right softmax probs (n,5) each."""
    out = []
    for half in (x[..., :28], x[..., 28:]):
        out.append(torch.cat([F.softmax(clf(half[i:i + 4096]).float(), -1) for i in range(0, len(half), 4096)]))
    return out


def ink_ranges():
    """Per-class [0.5%, 99.5%] range of mean ink ((x+1)/2 averaged) over real MNIST-test digits."""
    x, y = load_mnist04(train=False)
    ink = x.add(1).div(2).mean(dim=(1, 2, 3))
    return torch.stack([torch.quantile(ink[y == c], torch.tensor([0.005, 0.995], device=ink.device))
                        for c in range(NUM_CLASSES)])


def ink_diag(x, pl, pr, ranges):
    """Fraction of generated cells whose ink is outside the real range for their predicted class."""
    out = {}
    for name, half, p in (("left", x[..., :28], pl), ("right", x[..., 28:], pr)):
        ink = half.add(1).div(2).mean(dim=(1, 2, 3))
        c = p.argmax(-1)
        lo, hi = ranges[c, 0], ranges[c, 1]
        out[f"{name}_ink_below_real_range"] = (ink < lo).float().mean().item()
        out[f"{name}_ink_above_real_range"] = (ink > hi).float().mean().item()
        out[f"{name}_near_blank(ink<0.02)"] = (ink < 0.02).float().mean().item()
    return out


def stats(pl, pr, kappa):
    ml, mr = pl.max(-1), pr.max(-1)
    lab_l, lab_r = ml.indices, mr.indices
    ok = (ml.values >= 0.5) & (mr.values >= 0.5)
    cells = torch.cat([ml.values, mr.values])
    joint = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for a, b in zip(lab_l[ok].tolist(), lab_r[ok].tolist()):
        joint[a, b] += 1
    joint /= max(1, int(ok.sum()))
    return {
        "frac_cells_maxprob_ge_0.9": (cells >= 0.9).float().mean().item(),
        "frac_samples_both_cells_ge_0.9": ((ml.values >= 0.9) & (mr.values >= 0.9)).float().mean().item(),
        "waste_rate_any_cell_lt_0.5": (~ok).float().mean().item(),
        "same_digit_frac (valid samples)": (lab_l[ok] == lab_r[ok]).float().mean().item(),
        "same_digit_expected": (1 - kappa) + kappa / NUM_CLASSES,
        "target_1_3_frac (valid samples)": joint[1, 3].item(),
        "target_1_3_expected": (1 / NUM_CLASSES) * ((1 - kappa) * 0 + kappa / NUM_CLASSES),
        "left_marginal": torch.bincount(lab_l[ok], minlength=NUM_CLASSES).float().div(ok.sum()).tolist(),
        "right_marginal": torch.bincount(lab_r[ok], minlength=NUM_CLASSES).float().div(ok.sum()).tolist(),
        "joint_valid": [[round(v, 4) for v in row] for row in joint.tolist()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    unet = load_model(args.kappa)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    t0 = time.time()
    x = sample_plain(unet, args.n, gen)
    torch.cuda.synchronize()
    secs = time.time() - t0

    out_png = os.path.join(ROOT, f"model_samples_kappa{args.kappa:g}.png")
    save_image(x[:64].add(1).div(2), out_png, nrow=8, padding=2, pad_value=0.5)

    res = {"kappa": args.kappa, "n": args.n, "sample_seconds": round(secs, 1)}
    ranges = ink_ranges()
    res["real_ink_ranges_per_class"] = [[round(v, 4) for v in r] for r in ranges.tolist()]
    for name in ("evaluator", "judge"):
        pl, pr = cell_probs(load_classifier(name), x)
        res[name] = stats(pl, pr, args.kappa)
        res[name]["ink_diagnostic"] = ink_diag(x, pl, pr, ranges)
    path = os.path.join(ROOT, "checkpoints", f"ddpm_kappa{args.kappa:g}_eval.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("grid:", out_png)


if __name__ == "__main__":
    main()
