"""Level 3 model acceptance on 4000 samples (all metrics from the evaluators).

Usage: python level3_eval_model.py [--n 4000] [--step N]
Output: level3/results/model_acceptance.json, level3/results/model_samples_l3.png
"""
import argparse
import json
import os

import torch

from common import DEVICE
from diffusion import sample_plain
from level3_common import L3_RES, STYLE_NAMES, cell_attrs, load_l3_classifiers, load_l3_model, save_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    out = os.path.join(L3_RES, "model_acceptance.json" if args.step is None else f"model_acceptance_step{args.step}.json")
    if os.path.exists(out):
        print("exists, skipping:", out)
        return
    unet = load_l3_model(step=args.step)
    _, dev, thick = load_l3_classifiers()
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    x = sample_plain(unet, args.n, gen)
    a = cell_attrs(dev, thick, x)
    L, R = a["L"], a["R"]
    f = lambda m: m.float().mean().item()
    valid = (L["dprob"] >= 0.5) & (R["dprob"] >= 0.5)
    clear_b = L["dark_clear"] & R["dark_clear"]
    clear_t = (L["tprob"] >= 0.9) & (R["tprob"] >= 0.9)
    img_style = 2 * L["thick"] + L["dark"]
    consistent = (L["dark"] == R["dark"]) & (L["thick"] == R["thick"])
    res = {
        "n": args.n, "step": args.step,
        "frac_cells_digit_maxprob_ge_0.9": f(torch.cat([L["dprob"], R["dprob"]]) >= 0.9),
        "waste_rate_any_cell_lt_0.5": f(~valid),
        "brightness_agree_all": f(L["dark"] == R["dark"]),
        "brightness_agree_both_clear": f((L["dark"] == R["dark"])[clear_b]),
        "frac_both_brightness_clear": f(clear_b),
        "thickness_agree_all": f(L["thick"] == R["thick"]),
        "thickness_agree_both_clear": f((L["thick"] == R["thick"])[clear_t]),
        "frac_both_thickness_clear": f(clear_t),
        "both_style_agree_all": f(consistent),
        "style_distribution_left_cell": {STYLE_NAMES[s]: f(img_style == s) for s in range(4)},
        "style_distribution_consistent_images": {STYLE_NAMES[s]: f((img_style == s)[consistent]) for s in range(4)},
        "same_digit_frac_valid": f((L["digit"] == R["digit"])[valid]),
        "target_1_3_frac_valid": f(((L["digit"] == 1) & (R["digit"] == 3))[valid]),
        "digit_marginal_left": [f(L["digit"][valid] == c) for c in range(5)],
        "digit_marginal_right": [f(R["digit"][valid] == c) for c in range(5)],
    }
    sd = res["style_distribution_left_cell"]
    res["acceptance"] = {
        "cells_ge_0.9_about_90pct": res["frac_cells_digit_maxprob_ge_0.9"] >= 0.88,
        "waste_low(<2%)": res["waste_rate_any_cell_lt_0.5"] < 0.02,
        "brightness_agree_ge_95": res["brightness_agree_all"] >= 0.95,
        "thickness_agree_ge_95": res["thickness_agree_all"] >= 0.95,
        "styles_about_25pct(20-30)": all(0.20 <= v <= 0.30 for v in sd.values()),
        "same_digit_about_20pct(16-24)": 0.16 <= res["same_digit_frac_valid"] <= 0.24,
    }
    res["acceptance_pass"] = all(res["acceptance"].values())
    json.dump(res, open(out, "w"), indent=2)
    save_grid(x[:64], 4, out.replace(".json", ".png").replace("model_acceptance", "model_samples_l3"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
