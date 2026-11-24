"""Pool several independent lam=0 check runs (equal batch counts) and recompute z-scores.

Usage: python pool_lam0.py --out logs/lam0_kappa1_pooled.json logs/lam0_kappa1_stepfinal.json logs/lam0_kappa1_seed100.json
"""
import argparse
import json
import math

from check_lam0 import BATCH_LEVEL, PER_SAMPLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    runs = [json.load(open(p)) for p in args.inputs]
    k_runs = len(runs)
    pooled = {"_sources": args.inputs}
    print(f"pooled over {k_runs} independent runs")
    print(f"{'metric':<14}{'A':>18}{'B':>18}{'C':>18}   z(B-A)  z(C-A)")
    fails = []
    for k in PER_SAMPLE + BATCH_LEVEL:
        agg = {}
        for m in "ABC":
            vals = [r[m][k] for r in runs if r[m][k]["mean"] is not None]
            if not vals:  # metric undefined in every run (e.g. A diversity at success ~0.5%)
                agg[m] = (float("nan"), float("nan"))
                continue
            ms = [v["mean"] for v in vals]
            if len(vals) < k_runs or any(v["se"] is None for v in vals):
                # undefined in some run / single-batch estimate: no usable error bar
                agg[m] = (sum(v["mean"] for v in vals) / len(vals), float("nan"))
                continue
            ses = [v["se"] for v in vals]
            agg[m] = (sum(ms) / len(vals), math.sqrt(sum(s * s for s in ses)) / len(vals))
        zs = []
        for m in "BC":
            den = math.hypot(agg["A"][1], agg[m][1])
            z = (agg[m][0] - agg["A"][0]) / den if den else float("nan")
            zs.append(z)
            if k in PER_SAMPLE and abs(z) >= 3:
                fails.append(f"{m}:{k} z={z:.2f}")
        pooled[k] = {m: {"mean": agg[m][0], "se": agg[m][1]} for m in "ABC"}
        pooled[k]["z"] = {"B": zs[0], "C": zs[1]}
        tag = "" if k in PER_SAMPLE else "  (batch-level)"
        print(f"{k:<14}" + "".join(f"{agg[m][0]:>10.4f}±{agg[m][1]:.4f}" for m in "ABC")
              + f" {zs[0]:7.2f} {zs[1]:7.2f}{tag}")
    print("PASS (pooled per-sample metrics |z| < 3)" if not fails else "FAIL: " + ", ".join(fails))
    pooled["_fails"] = fails
    with open(args.out, "w") as f:
        json.dump(pooled, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
