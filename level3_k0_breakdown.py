"""Level 3, exp1 supplement: digit survival when splicing at k (default 0 and 50), split by whether
the two donors' styles match. Reuses the exp1 reference cache. Writes level3/results/exp1_breakdown.json.
"""
import json
import os

import torch

from common import DEVICE
from diffusion import NUM_SAMPLE_STEPS
from level3_common import L3_CACHE, L3_RES, cell_attrs, load_l3_classifiers, load_l3_model
from smc import HALF
from timing import Stepper, eps_all, frac, noise_at

SEED, P = 0, 8000
OUT = os.path.join(L3_RES, "exp1_breakdown.json")


@torch.no_grad()
def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    c = torch.load(os.path.join(L3_CACHE, f"exp1_ref_seed{SEED}_n{P}.pt"), map_location=DEVICE)
    _, dev, thick = load_l3_classifiers()
    unet, st, h = load_l3_model(), Stepper(), P // 2
    R = cell_attrs(dev, thick, c["ref"])
    dL = {k: v[:h] for k, v in R["L"].items()}
    dR = {k: v[h:] for k, v in R["R"].items()}
    ok = (dL["dprob"] >= 0.5) & (dR["dprob"] >= 0.5)
    same_b, same_t = dL["dark"] == dR["dark"], dL["thick"] == dR["thick"]
    groups = {"style_both_same": same_b & same_t, "only_brightness_differs": ~same_b & same_t,
              "only_thickness_differs": same_b & ~same_t, "both_differ": ~same_b & ~same_t}
    for k in (0, 50):
        if str(k) in out:
            continue
        x = torch.cat([c["states"][k][:h][..., :HALF], c["states"][k][h:][..., HALF:]], 3)
        for i in range(k, NUM_SAMPLE_STEPS):
            n = noise_at(SEED, i, P)
            x, _ = st.step(i, x, eps_all(unet, x, st.timesteps[i]), torch.cat([n[:h][..., :HALF], n[h:][..., HALF:]], 3))
        F = cell_attrs(dev, thick, x.clamp(-1, 1))
        kL = (F["L"]["dprob"] >= 0.5) & (F["L"]["digit"] == dL["digit"]) & ok
        kR = (F["R"]["dprob"] >= 0.5) & (F["R"]["digit"] == dR["digit"]) & ok
        row = {}
        for g, m in groups.items():
            # style that the pair ended up with: which donor's cell changed style
            chL = (F["L"]["dark"] != dL["dark"]) | (F["L"]["thick"] != dL["thick"])
            chR = (F["R"]["dark"] != dR["dark"]) | (F["R"]["thick"] != dR["thick"])
            row[g] = {"n": int(m.sum()), "digits_both_kept": frac(kL & kR, m),
                      "left_kept": frac(kL, m), "right_kept": frac(kR, m),
                      "left_digit_kept_given_left_style_changed": frac(kL, m & chL),
                      "left_digit_kept_given_left_style_unchanged": frac(kL, m & ~chL),
                      "right_digit_kept_given_right_style_changed": frac(kR, m & chR),
                      "right_digit_kept_given_right_style_unchanged": frac(kR, m & ~chR)}
        out[str(k)] = row
        json.dump(out, open(OUT + ".tmp", "w"), indent=1)
        os.replace(OUT + ".tmp", OUT)
    for k, row in out.items():
        for g, r in row.items():
            print(f"k={k} {g:<24} n={r['n']:5d} both {r['digits_both_kept']['p']:.3f} "
                  f"L|changed {r['left_digit_kept_given_left_style_changed']['p']:.3f} "
                  f"L|unchanged {r['left_digit_kept_given_left_style_unchanged']['p']:.3f} "
                  f"R|changed {r['right_digit_kept_given_right_style_changed']['p']:.3f} "
                  f"R|unchanged {r['right_digit_kept_given_right_style_unchanged']['p']:.3f}")


if __name__ == "__main__":
    main()
