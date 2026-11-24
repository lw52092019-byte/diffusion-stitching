"""Level 3: train digit judge, digit evaluator (on all four styles) and the thickness classifier;
report per-style test accuracy and brightness-threshold accuracy.

Usage: python level3_classifiers.py        (skips classifiers whose checkpoint exists)
Output: level3/checkpoints/{digit_judge,digit_evaluator,thick_clf}.pt, classifiers_metrics.json
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import DEVICE, EvalResNet, JudgeCNN, seed_all
from level3_common import (L3_CKPT, STYLE_NAMES, ThickCNN, apply_style, judge_brightness, load_digits,
                           save_style_params, style_parts)
from train_classifiers import augment

CONFIGS = {
    "digit_judge": dict(cls=JudgeCNN, seed=0, epochs=12, lr=2e-3, shift=2.0, rot=0.0, scale=0.0, target="digit"),
    "digit_evaluator": dict(cls=EvalResNet, seed=1, epochs=15, lr=2e-3, shift=2.0, rot=10.0, scale=0.1,
                            target="digit"),
    "thick_clf": dict(cls=ThickCNN, seed=2, epochs=6, lr=2e-3, shift=2.0, rot=10.0, scale=0.1, target="thick"),
}
BATCH = 256


def train(name, cfg, xtr, ytr):
    seed_all(cfg["seed"])
    model = cfg["cls"]().to(DEVICE)
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
            s = torch.randint(0, 4, (len(idx),), device=DEVICE)
            xb = augment(apply_style(xtr[idx], s), cfg["shift"], cfg["rot"], cfg["scale"])
            yb = ytr[idx] if cfg["target"] == "digit" else style_parts(s)[0]
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        print(f"[{name}] epoch {ep + 1}/{cfg['epochs']} loss {tot / n:.4f}", flush=True)
    return model, time.time() - t0


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([F.softmax(model(x[i:i + 2048]), -1) for i in range(0, len(x), 2048)])


def main():
    xtr, ytr = load_digits(True)
    xte, yte = load_digits(False)
    models, report = {}, {}
    for name, cfg in CONFIGS.items():
        p = os.path.join(L3_CKPT, f"{name}.pt")
        m = cfg["cls"]().to(DEVICE)
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            print(f"[{name}] loaded existing checkpoint", flush=True)
            secs = None
        else:
            m, secs = train(name, cfg, xtr, ytr)
            torch.save(m.state_dict(), p)
        m.eval()
        models[name] = m
        report[name] = {"arch": type(m).__name__, "params": sum(q.numel() for q in m.parameters()),
                        **{k: v for k, v in cfg.items() if k != "cls"},
                        "train_seconds": None if secs is None else round(secs, 1)}

    per_style = {}
    for s in range(4):
        st = torch.full((len(xte),), s, device=DEVICE)
        xs = apply_style(xte, st)
        thick, dark = style_parts(st)
        r = {}
        for name in ("digit_judge", "digit_evaluator"):
            pr = predict(models[name], xs)
            r[name + "_acc"] = (pr.argmax(-1) == yte).float().mean().item()
            r[name + "_frac_maxprob_ge_0.9"] = (pr.max(-1).values >= 0.9).float().mean().item()
        pt = predict(models["thick_clf"], xs)
        r["thickness_acc"] = (pt.argmax(-1) == thick).float().mean().item()
        r["thickness_clear_frac"] = (pt.max(-1).values >= 0.9).float().mean().item()
        bd, bc = judge_brightness(xs)
        r["brightness_acc"] = (bd == dark).float().mean().item()
        r["brightness_clear_frac"] = bc.float().mean().item()
        per_style[STYLE_NAMES[s]] = r
        print(STYLE_NAMES[s], json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
    report["test_per_style"] = per_style
    keys = ["digit_judge_acc", "digit_evaluator_acc", "thickness_acc", "brightness_acc"]
    report["min_over_styles"] = {k: min(v[k] for v in per_style.values()) for k in keys}
    ok = (report["min_over_styles"]["digit_judge_acc"] >= 0.99 and report["min_over_styles"]["digit_evaluator_acc"] >= 0.99
          and report["min_over_styles"]["thickness_acc"] >= 0.98 and report["min_over_styles"]["brightness_acc"] >= 0.98)
    report["acceptance_pass"] = ok
    json.dump(report, open(os.path.join(L3_CKPT, "classifiers_metrics.json"), "w"), indent=2)
    save_style_params()
    print("min over styles:", report["min_over_styles"], "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
