"""Quick smoke run of A/B/C on a trained model (small batch count, debugging only).

Usage: python smoke_smc.py --kappa 1.0 [--batches 100] [--step N]
"""
import argparse

from common import load_classifier
from diffusion import load_model
from smc import run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batches", type=int, default=100)
    args = ap.parse_args()
    u = load_model(args.kappa, step=args.step)
    j, e = load_classifier("judge"), load_classifier("evaluator")
    for m, lam, w in [("A", 0, "full"), ("B", 0, "full"), ("C", 0, "full"), ("B", 3, "full"),
                      ("C", 3, "full"), ("B", 3, "early"), ("C", 3, "early")]:
        r, _ = run_config(u, j, e, m, args.batches, 4, lam, w, seed=7)
        f = lambda k: f"{r[k]['mean']:.3f}" if r[k]["mean"] is not None else "n/a"
        ess = f"{r['mean_ess_frac']:.2f}" if r["mean_ess_frac"] is not None else "-"
        print(f"{m} lam={lam} {w:<5} succ {f('success')} any {f('any_success')} hitL {f('hit_left')} "
              f"hitR {f('hit_right')} waste {f('waste')} seam {f('seam')} div {f('diversity')} "
              f"dup {r['dup_frac_success_pairs']:.2f} ess {ess} {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
