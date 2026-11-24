"""Main experiment grid for one kappa (milestones 5 / 6 / 7).

  seed 0 : A, and B and C on every (lam, window) in {1,3,10} x {full, early, late}
  seeds 1,2 : A, best B and best C (best = highest seed-0 success rate, chosen separately per method)
  --extra-N : B at the best B setting with larger N (milestone 7)

Results are appended to results/raw_kappa{k}.json after every config (resumable: configs
already present are skipped). Sample grids for the best configs go to results/.

Usage:
  python run_main.py --kappa 1.0
  python run_main.py --kappa 0.1 --extra-N 8 16
"""
import argparse
import json
import os
import shutil

import torch
from torchvision.utils import save_image

from common import ROOT, load_classifier
from diffusion import load_model
from smc import metrics, run_config, select_D

LAMS = (1.0, 3.0, 10.0)
WINDOWS = ("full", "early", "late")
OUT_DIR = os.path.join(ROOT, "results")


def cid(m, n, lam, win, seed):
    return f"{m}_N{n}_lam{lam:g}_{win}_s{seed}" if m != "A" else f"A_N{n}_s{seed}"


def seed_for(key):
    # deterministic, distinct per config id
    return sum((i + 1) * ord(c) for i, c in enumerate(key)) % (2 ** 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--extra-N", type=int, nargs="*", default=[])
    ap.add_argument("--step", type=int, default=None, help="intermediate checkpoint (debug only)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--note", default="GPU used exclusively by this run (serial)")
    args = ap.parse_args()
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"raw_kappa{args.kappa:g}.json")
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    unet = load_model(args.kappa, step=args.step)
    judge, evaluator = load_classifier("judge"), load_classifier("evaluator")

    grid_dir = os.path.join(out_dir, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    def save():
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=1)

    def run(m, n, lam, win, s):
        key = cid(m, n, lam, win, s)
        dkeys = (f"Dwhole_N{n}_s{s}", f"Dstitch_N{n}_s{s}")
        if key in raw and (m != "A" or all(k in raw for k in dkeys)):
            return raw[key]
        save_png = os.path.join(grid_dir, f"{key}_kappa{args.kappa:g}.png") if s == 0 else None
        res, x = run_config(unet, judge, evaluator, m, args.batches, n, lam, win, seed=seed_for(key))
        res.update({"id": key, "kappa": args.kappa, "seed_index": s, "rng_seed": seed_for(key),
                    "note": args.note})
        raw[key] = res
        if m == "A":
            # Method D: post-hoc judge selection on these same A samples (no intervention).
            for dk, xd in zip(dkeys, select_D(judge, x, args.batches, n)):
                rd = metrics(evaluator, xd, args.batches, 1)
                rd.update({"id": dk, "method": "D-whole" if dk.startswith("Dwhole") else "D-stitch",
                           "N": n, "n_batches": args.batches, "lam": 0.0, "window": None, "seed": res["seed"],
                           "seed_index": s, "kappa": args.kappa, "rng_seed": seed_for(key),
                           "seconds": res["seconds"], "n_resample": 0, "mean_ess_frac": None,
                           "note": "reuses the A samples of " + key + "; one output image per batch"})
                raw[dk] = rd
                print(f"{dk:<28} succ {rd['success']['mean']:.4f}±{rd['success']['se']:.4f} "
                      f"waste {rd['waste']['mean']:.4f}", flush=True)
                if save_png:
                    save_image(xd[:8].add(1).div(2), os.path.join(grid_dir, f"{dk}_kappa{args.kappa:g}.png"),
                               nrow=1, padding=2, pad_value=0.5)
        save()
        if save_png:
            rows = 8
            save_image(x[: rows * n].add(1).div(2), save_png, nrow=n, padding=2, pad_value=0.5)
        print(f"{key:<28} succ {res['success']['mean']:.4f}±{res['success']['se']:.4f} "
              f"any {res['any_success']['mean']:.3f} waste {res['waste']['mean']:.4f} "
              f"{res['seconds']}s", flush=True)
        return res

    n = args.N
    run("A", n, 0.0, "full", 0)
    # B and C alternate so a partial run always has matched pairs; lam=10/full first.
    order = [(10.0, "full")] + [(lam, win) for lam in LAMS for win in WINDOWS if (lam, win) != (10.0, "full")]
    for lam, win in order:
        for m in ("B", "C"):
            run(m, n, lam, win, 0)

    best = {}
    for m in ("B", "C"):
        cands = [(raw[cid(m, n, lam, win, 0)]["success"]["mean"], lam, win) for lam in LAMS for win in WINDOWS]
        _, lam, win = max(cands)
        best[m] = (lam, win)
    print("best settings (seed 0):", best, flush=True)
    for s in (1, 2):
        run("A", n, 0.0, "full", s)
        for m in ("B", "C"):
            run(m, n, *best[m], s)

    # sample grids (one row per batch, first 8 batches) for A and the best B / C, seed 0
    tag = f"kappa{args.kappa:g}"
    for m, key in (("A", cid("A", n, 0.0, "full", 0)), ("B", cid("B", n, *best["B"], 0)),
                   ("C", cid("C", n, *best["C"], 0))):
        shutil.copyfile(os.path.join(grid_dir, f"{key}_{tag}.png"), os.path.join(out_dir, f"samples_{m}_{tag}.png"))

    for nn in args.extra_N:
        for s in (0, 1, 2):
            run("B", nn, *best["B"], s)

    raw["_best"] = {m: {"lam": v[0], "window": v[1]} for m, v in best.items()}
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=1)
    print("done ->", raw_path)


if __name__ == "__main__":
    main()
