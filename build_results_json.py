"""Collect every raw number into results.json and copy the deliverable sample grids.

Usage: python build_results_json.py
"""
import json
import os
import shutil

from common import ROOT


def load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    out = {
        "description": "All raw numbers for the stitch-vs-copy experiment. Metrics: evaluator classifier; "
                       "per-config values are mean and standard error over 1000 batches.",
        "classifiers": load("checkpoints/classifiers_metrics.json"),
        "diffusion_training": {k: load(f"checkpoints/ddpm_kappa{k}_train.json") for k in ("1", "0.1")},
        "model_acceptance": {k: load(f"checkpoints/ddpm_kappa{k}_eval.json") for k in ("1", "0.1")},
        "lam0_checks": {
            "kappa1": {"run1": load("logs/lam0_kappa1_stepfinal.json"), "run2": load("logs/lam0_kappa1_seed100.json"),
                       "pooled": load("logs/lam0_kappa1_pooled.json")},
            "kappa0.1": {"run1": load("logs/lam0_kappa0.1_stepfinal.json"),
                         "run2": load("logs/lam0_kappa0.1_seed100.json"),
                         "pooled": load("logs/lam0_kappa0.1_pooled.json")},
        },
        "configs": {k: load(f"results/raw_kappa{k}.json") for k in ("1", "0.1")},
        "excluded_partial_run_kappa1_contended": load("results/raw_kappa1_contended_partial.json"),
    }
    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote results.json")
    for m in "ABC":
        src = os.path.join(ROOT, "results", f"samples_{m}_kappa0.1.png")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(ROOT, f"samples_{m}.png"))
            print("copied", src, "-> samples_" + m + ".png")
        else:
            print("MISSING", src)


if __name__ == "__main__":
    main()
