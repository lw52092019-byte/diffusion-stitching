"""Milestone 2: train the judge (scores x0_hat during sampling) and the independent evaluator
(used for all final metrics) on clean 28x28 MNIST digits 0-4.

Usage:  python train_classifiers.py
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import CKPT_DIR, CLASSIFIERS, DEVICE, NUM_CLASSES, load_mnist04, seed_all

CONFIGS = {
    # name: seed, epochs, lr, max shift (px), max rotation (deg), scale range
    "judge": dict(seed=0, epochs=12, lr=2e-3, shift=2.0, rot=0.0, scale=0.0),
    "evaluator": dict(seed=1, epochs=15, lr=2e-3, shift=2.0, rot=10.0, scale=0.1),
}
BATCH = 256


def augment(x, shift, rot, scale):
    """Random affine (translation / rotation / isotropic scale) on a batch in [-1,1]."""
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(rot)
    s = 1 + (torch.rand(b, device=x.device) * 2 - 1) * scale
    t = (torch.rand(b, 2, device=x.device) * 2 - 1) * (2 * shift / 28)
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # pad with background value -1
    return F.grid_sample(x + 1, grid, align_corners=False, padding_mode="zeros") - 1


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([F.softmax(model(x[i:i + 2048]), -1) for i in range(0, len(x), 2048)])


def train(name, cfg, xtr, ytr):
    seed_all(cfg["seed"])
    model = CLASSIFIERS[name]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = cfg["epochs"] * math.ceil(len(xtr) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xtr), device=DEVICE)
        tot, n = 0.0, 0
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(xtr[idx], cfg["shift"], cfg["rot"], cfg["scale"])
            loss = F.cross_entropy(model(xb), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        print(f"[{name}] epoch {ep + 1}/{cfg['epochs']} loss {tot / n:.4f}")
    return model, time.time() - t0


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    xtr, ytr = load_mnist04(train=True)
    xte, yte = load_mnist04(train=False)
    print(f"train {len(xtr)}  test {len(xte)}")

    probs, report = {}, {}
    for name, cfg in CONFIGS.items():
        model, secs = train(name, cfg, xtr, ytr)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
        p = predict(model, xte)
        probs[name] = p
        pred = p.argmax(-1)
        report[name] = {
            "arch": type(model).__name__,
            "params": sum(q.numel() for q in model.parameters()),
            **cfg,
            "train_seconds": round(secs, 1),
            "test_acc": (pred == yte).float().mean().item(),
            "per_class_acc": [(pred[yte == c] == c).float().mean().item() for c in range(NUM_CLASSES)],
            "frac_maxprob_ge_0.9": (p.max(-1).values >= 0.9).float().mean().item(),
            "frac_maxprob_lt_0.5": (p.max(-1).values < 0.5).float().mean().item(),
        }
    pj, pe = probs["judge"].argmax(-1), probs["evaluator"].argmax(-1)
    report["agreement"] = {
        "argmax_agree": (pj == pe).float().mean().item(),
        "both_wrong": ((pj != yte) & (pe != yte)).sum().item(),
        "both_wrong_same_label": ((pj != yte) & (pe != yte) & (pj == pe)).sum().item(),
        "judge_errors": (pj != yte).sum().item(),
        "evaluator_errors": (pe != yte).sum().item(),
    }
    with open(os.path.join(CKPT_DIR, "classifiers_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
