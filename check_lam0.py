"""Milestone 4 correctness check: with lam = 0, B and C should match A within error.

Usage:
  python check_lam0.py --kappa 1.0 [--batches 1000] [--step 3000]   # --step = intermediate ckpt
"""
import argparse
import json
import os

from common import ROOT, load_classifier
from diffusion import load_model
from smc import run_config

PER_SAMPLE = ["success", "hit_left", "hit_right", "waste", "seam",
              "success_ink", "hit_left_ink", "hit_right_ink", "waste_ink"]
# With multinomial resampling, lam = 0 still copies lineages at random, so batch-level
# quantities are expected to differ from A; they are reported but not part of the pass test.
BATCH_LEVEL = ["any_success", "diversity"]
Z_PASS = 3.0  # per-sample metrics: |z| < 3 counts as "within error" (~16 comparisons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    unet = load_model(args.kappa, step=args.step)
    judge, evaluator = load_classifier("judge"), load_classifier("evaluator")
    rows = {}
    for m in ("A", "B", "C"):
        res, _ = run_config(unet, judge, evaluator, m, args.batches, args.N, lam=0.0, window="full",
                            seed=args.seed + {"A": 0, "B": 1, "C": 2}[m])
        rows[m] = res
        print(f"{m}: {res['seconds']}s", flush=True)

    fails = []
    print(f"\nkappa={args.kappa} step={args.step or 'final'} batches={args.batches} N={args.N}, lam=0")
    print(f"{'metric':<14}" + "".join(f"{m:>18}" for m in "ABC") + "   z(B-A)  z(C-A)")
    for k in PER_SAMPLE + BATCH_LEVEL:
        cells, zs = [], []
        for m in "ABC":
            v = rows[m][k]
            if v["mean"] is None:
                cells.append("n/a")
            elif v["se"] is None:  # e.g. diversity defined in only one batch
                cells.append(f"{v['mean']:.4f}±n/a")
            else:
                cells.append(f"{v['mean']:.4f}±{v['se']:.4f}")
        for m in "BC":
            a, b = rows["A"][k], rows[m][k]
            if (a["mean"] is None or b["mean"] is None or a["se"] is None or b["se"] is None
                    or not (a["se"] or b["se"])):
                zs.append("   n/a")
            else:
                z = (b['mean'] - a['mean']) / (a['se'] ** 2 + b['se'] ** 2) ** 0.5
                zs.append(f"{z:7.2f}")
                if k in PER_SAMPLE and abs(z) >= Z_PASS:
                    fails.append(f"{m}:{k} z={z:.2f}")
        tag = "" if k in PER_SAMPLE else "  (batch-level)"
        print(f"{k:<14}" + "".join(f"{c:>18}" for c in cells) + " " + " ".join(zs) + tag)
    print("PASS (per-sample metrics |z| < 3)" if not fails else "FAIL: " + ", ".join(fails))
    rows["_check"] = {"z_pass": Z_PASS, "fails": fails}
    out = args.out or os.path.join(ROOT, "logs", f"lam0_kappa{args.kappa:g}_step{args.step or 'final'}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
