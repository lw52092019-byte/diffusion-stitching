"""Level 3, experiment 1: when does style (brightness, thickness) freeze? No judge involved.

Same protocol as timing.py: 8000 reference trajectories with per-step reseeded noise; donors
0..3999 = left, 4000..7999 = right; splice the k-state (and every later step's noise) at the
midline and denoise to the end; a no-splice rerun must reproduce the reference bit-for-bit.

Readability curves: every 5 steps on the reference x0_hat, per cell, digit judge / brightness
rule / thickness classifier vs the evaluators' verdict on the final image.

Usage: python level3_timing.py [--n-ref 8000] [--seed 0] [--out ...]
Resumable: every finished piece is written to level3/results/exp1.json at once; the reference
trajectories are cached in level3/cache/ (large; deleted after the whole level-3 run).
"""
import argparse
import json
import os
import time

import torch

from common import DEVICE
from diffusion import NUM_SAMPLE_STEPS
from level3_common import (BRIGHT_CLEAR, L3_CACHE, L3_RES, THICK_CLEAR, cell_attrs, judge_brightness,
                           load_l3_classifiers, load_l3_model)
from smc import HALF
from timing import JUDGE_EVERY, SPLICE_KS, Stepper, eps_all, frac, noise_at

ATTRS = {"brightness": ("dark", "dark_clear"), "thickness": ("thick", None)}


def clear_of(a, attr):
    key, ck = ATTRS[attr]
    return a[ck] if ck else a["tprob"] >= THICK_CLEAR


@torch.no_grad()
def run(unet, judge, dev, thick, n_ref, seed, out, save, cache):
    st = Stepper()
    P, h = n_ref, n_ref // 2
    if cache and os.path.exists(cache):
        c = torch.load(cache, map_location=DEVICE)
        states, x0rec, ref = c["states"], c["x0rec"], c["ref"]
        print("loaded cached reference:", cache, flush=True)
    else:
        x = noise_at(seed, -1, P)
        states, x0rec = {}, {}
        t0 = time.time()
        for i, t in enumerate(st.timesteps):
            if i in SPLICE_KS:
                states[i] = x.clone()
            x_new, x0 = st.step(i, x, eps_all(unet, x, t), noise_at(seed, i, P))
            if i % JUDGE_EVERY == 0:
                x0rec[i] = x0.half()      # predicted clean image (fp16 to keep the cache small)
            x = x_new
        ref = x.clamp(-1, 1)
        out["ref_seconds"] = round(time.time() - t0, 1)
        if cache:
            torch.save({"states": states, "x0rec": x0rec, "ref": ref}, cache)
            print("cached reference:", cache, flush=True)
    R = cell_attrs(dev, thick, ref)
    ok = (R["L"]["dprob"] >= 0.5) & (R["R"]["dprob"] >= 0.5)
    out["ref"] = {
        "waste": frac(~ok),
        "brightness_agree": frac(R["L"]["dark"] == R["R"]["dark"]),
        "thickness_agree": frac(R["L"]["thick"] == R["R"]["thick"]),
        "same_digit_valid": frac((R["L"]["digit"] == R["R"]["digit"])[ok]),
    }

    # ---- readability curves (judge / brightness rule / thickness classifier on x0_hat)
    if "readability" not in out:
        rc = {}
        for i, x0 in sorted(x0rec.items()):
            A = cell_attrs(judge, thick, x0.float())       # digit field = judge here
            row = {}
            for name, key, clear_fn in (("digit", "digit", lambda a: a["dprob"] >= 0.5),
                                        ("brightness", "dark", lambda a: a["dark_clear"]),
                                        ("thickness", "thick", lambda a: a["tprob"] >= THICK_CLEAR)):
                agree = torch.cat([(A[s][key] == R[s][key])[clear_fn(R[s])] for s in ("L", "R")])
                row[name] = frac(agree)
            row["judge_mean_maxprob"] = torch.cat([A["L"]["dprob"], A["R"]["dprob"]]).mean().item()
            row["x0hat_brightness_clear_frac"] = torch.cat([A["L"]["dark_clear"], A["R"]["dark_clear"]]).float().mean().item()
            row["x0hat_thickness_clear_frac"] = (torch.cat([A["L"]["tprob"], A["R"]["tprob"]]) >= THICK_CLEAR).float().mean().item()
            rc[str(i)] = row
        out["readability"] = rc
        save()
        for i, row in rc.items():
            print(f"read step {i:>3}: digit {row['digit']['p']:.3f} bright {row['brightness']['p']:.3f} "
                  f"thick {row['thickness']['p']:.3f}", flush=True)

    def continue_from(k, xk, splice_noise):
        x = xk
        for i in range(k, NUM_SAMPLE_STEPS):
            n = noise_at(seed, i, P)
            n = torch.cat([n[:h][..., :HALF], n[h:][..., HALF:]], 3) if splice_noise else n[:h]
            x, _ = st.step(i, x, eps_all(unet, x, st.timesteps[i]), n)
        return x.clamp(-1, 1)

    if "control_rerun_no_splice" not in out:
        ctrl = continue_from(0, states[0][:h], splice_noise=False)
        out["control_rerun_no_splice"] = {
            "max_abs_pixel_diff": (ctrl - ref[:h]).abs().max().item(),
            "frac_images_identical(<1e-3)": frac((ctrl - ref[:h]).abs().amax(dim=(1, 2, 3)) < 1e-3)}
        save()
    print("control:", out["control_rerun_no_splice"], flush=True)

    # donor reference values: left donor's left cell, right donor's right cell
    dL = {k: v[:h] for k, v in R["L"].items()}
    dR = {k: v[h:] for k, v in R["R"].items()}
    digit_ok = (dL["dprob"] >= 0.5) & (dR["dprob"] >= 0.5)
    out.setdefault("splice", {})
    out["n_pairs"] = h
    out["n_pairs_differ"] = {a: int((clear_of(dL, a) & clear_of(dR, a) & (dL[ATTRS[a][0]] != dR[ATTRS[a][0]])).sum())
                             for a in ATTRS}
    save()
    for k in SPLICE_KS:
        if str(k) in out["splice"]:
            print(f"k={k} done, skipping", flush=True)
            continue
        t0 = time.time()
        xk = torch.cat([states[k][:h][..., :HALF], states[k][h:][..., HALF:]], 3)
        F_ = cell_attrs(dev, thick, continue_from(k, xk, splice_noise=True))
        fL, fR = F_["L"], F_["R"]
        okf = (fL["dprob"] >= 0.5) & (fR["dprob"] >= 0.5)
        keepL = (fL["dprob"] >= 0.5) & (fL["digit"] == dL["digit"]) & digit_ok
        keepR = (fR["dprob"] >= 0.5) & (fR["digit"] == dR["digit"]) & digit_ok
        res = {"all_pairs": {"digits_both_kept": frac(keepL & keepR), "waste": frac(~okf),
                             "brightness_agree": frac(fL["dark"] == fR["dark"]),
                             "thickness_agree": frac(fL["thick"] == fR["thick"])}}
        for a, (key, _) in ATTRS.items():
            sub = clear_of(dL, a) & clear_of(dR, a) & (dL[key] != dR[key])
            vL, vR = dL[key], dR[key]
            uni = fL[key] == fR[key]
            res[a] = {
                "n": int(sub.sum()),
                "unified": frac(uni, sub),
                "unified_to_left_donor": frac(uni & (fL[key] == vL), sub),
                "unified_to_right_donor": frac(uni & (fL[key] == vR), sub),
                "both_kept_mismatch": frac((fL[key] == vL) & (fR[key] == vR), sub),
                "swapped": frac((fL[key] == vR) & (fR[key] == vL), sub),
                "final_both_clear": frac(clear_of(fL, a) & clear_of(fR, a), sub),
                "digit_left_kept": frac(keepL, sub), "digit_right_kept": frac(keepR, sub),
                "digits_both_kept": frac(keepL & keepR, sub),
                "waste": frac(~okf, sub),
            }
        res["seconds"] = round(time.time() - t0, 1)
        out["splice"][str(k)] = res
        save()
        b, t = res["brightness"], res["thickness"]
        print(f"k={k:3d} bright: unified {b['unified']['p']:.3f} (L {b['unified_to_left_donor']['p']:.3f} "
              f"R {b['unified_to_right_donor']['p']:.3f}) mismatch {b['both_kept_mismatch']['p']:.3f} | "
              f"thick: unified {t['unified']['p']:.3f} (L {t['unified_to_left_donor']['p']:.3f} "
              f"R {t['unified_to_right_donor']['p']:.3f}) mismatch {t['both_kept_mismatch']['p']:.3f} | "
              f"digits kept {res['all_pairs']['digits_both_kept']['p']:.3f} waste {res['all_pairs']['waste']['p']:.4f} "
              f"({res['seconds']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(L3_RES, "exp1.json"))
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    data = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for key, val in (("n_ref", args.n_ref), ("seed", args.seed)):
        if key in data and data[key] != val:
            raise SystemExit(f"{args.out} has {key}={data[key]} != {val}")
        data[key] = val

    def save():
        tmp = args.out + ".tmp"
        json.dump(data, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    cache = args.cache or os.path.join(L3_CACHE, f"exp1_ref_seed{args.seed}_n{args.n_ref}.pt")
    judge, dev, thick = load_l3_classifiers()
    run(load_l3_model(), judge, dev, thick, args.n_ref, args.seed, data, save, cache)
    save()
    print("saved", args.out)


if __name__ == "__main__":
    main()
