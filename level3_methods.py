"""Level 3, experiment 2: A / B / C / D with "coordinated success" (digits 1|3 AND both cells
share brightness AND thickness). The judge only scores digits (it knows nothing about style).

Order (each config written to level3/results/exp2.json as soon as it finishes; reruns skip):
  1. A seeds 0,1,2 (+ D-whole / D-stitch on A's samples)
  2. B and C grid, seed 0: lam in {3,10} x resampling steps in {10-90, 30-90, 30-60}
  3. best B and best C (by seed-0 coordinated success) on seeds 1,2
  4. B with N=8 at B's best setting, seeds 0,1,2
  5. lam = 0 check: two independent runs (A, B, C), resampling 10-90
Usage: python level3_methods.py [--batches 1000] [--out ...]
"""
import argparse
import json
import math
import os
import time

import torch

from common import DEVICE
from level3_common import L3_RES, STYLE_NAMES, cell_attrs, load_l3_classifiers, load_l3_model
from smc import sample, select_D

ITERS = {"10-90": list(range(10, 100, 10)), "30-90": list(range(30, 100, 10)), "30-60": [30, 40, 50, 60]}
LAMS = [3.0, 10.0]
TARGET = (1, 3)
SAMPLE_DIR = os.path.join(L3_RES, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)


def _mean_se(v):
    v = v.float()
    return {"mean": v.mean().item(), "se": (v.std() / math.sqrt(len(v))).item() if len(v) > 1 else None}


def _ratio(num, den):
    n = int(den.sum())
    p = (num & den).sum().item() / n if n else float("nan")
    return {"p": p, "se": math.sqrt(p * (1 - p) / n) if n else float("nan"), "n": n}


@torch.no_grad()
def l3_metrics(dev, thick, x, n_groups, n):
    a = cell_attrs(dev, thick, x)
    L, R = a["L"], a["R"]
    okL, okR = L["dprob"] >= 0.5, R["dprob"] >= 0.5
    hitL, hitR = okL & (L["digit"] == TARGET[0]), okR & (R["digit"] == TARGET[1])
    succ = hitL & hitR
    bag, tag = L["dark"] == R["dark"], L["thick"] == R["thick"]
    coord = succ & bag & tag
    g = lambda v: v.float().view(n_groups, n)
    res = {k: _mean_se(v) for k, v in {
        "success": g(succ).mean(1), "coord_success": g(coord).mean(1),
        "any_success": g(succ).amax(1), "any_coord_success": g(coord).amax(1),
        "hit_left": g(hitL).mean(1), "hit_right": g(hitR).mean(1),
        "waste": g(~(okL & okR)).mean(1),
        "brightness_agree": g(bag).mean(1), "thickness_agree": g(tag).mean(1),
        "style_agree": g(bag & tag).mean(1),
    }.items()}
    # among digit successes (pooled over the run)
    res["brightness_agree_given_success"] = _ratio(bag, succ)
    res["thickness_agree_given_success"] = _ratio(tag, succ)
    res["style_agree_given_success"] = _ratio(bag & tag, succ)
    # style of coordinated successes (left cell), share of each of the 4 styles
    st = 2 * L["thick"] + L["dark"]
    nc = int(coord.sum())
    res["coord_success_style_share"] = {STYLE_NAMES[s]: ((st == s) & coord).sum().item() / nc if nc else None
                                        for s in range(4)}
    # diversity among coordinated successes within a batch (mean pairwise L2), batches with >= 2
    img = x.add(1).div(2)
    flat = img.view(n_groups, n, -1)
    cm = coord.view(n_groups, n)
    if n > 1:
        d = torch.cdist(flat, flat)
        pair = cm[:, :, None] & cm[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)
        npair = pair.sum((1, 2))
        has = npair > 0
        res["diversity_coord"] = _mean_se((d * pair).sum((1, 2))[has] / npair[has]) if has.any() else \
            {"mean": None, "se": None}
        res["diversity_coord"]["n_batches"] = int(has.sum())
    res["seam"] = _mean_se((img[..., 27] - img[..., 28]).abs().mean(dim=(1, 2)))
    return res


def run_one(unet, judge, dev, thick, method, n_groups, n, lam, iters, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    x, info = sample(unet, judge, method, n_groups, n, lam=lam, gen=gen, iters=iters)
    torch.cuda.synchronize()
    res = l3_metrics(dev, thick, x, n_groups, n)
    res.update({"method": method, "N": n, "n_batches": n_groups, "lam": lam,
                "iters": None if method == "A" else iters, "rng_seed": seed,
                "seconds": round(time.time() - t0, 1), **info})
    return res, x


PER_SAMPLE = ["success", "coord_success", "hit_left", "hit_right", "waste",
              "brightness_agree", "thickness_agree", "style_agree"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(L3_RES, "exp2.json"))
    args = ap.parse_args()
    data = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if data.get("batches", args.batches) != args.batches:
        raise SystemExit("batch count differs from existing file")
    data["batches"] = args.batches
    cfg = data.setdefault("configs", {})

    def save():
        tmp = args.out + ".tmp"
        json.dump(data, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    unet = load_l3_model()
    judge, dev, thick = load_l3_classifiers()
    G = args.batches

    def do(key, method, n, lam, iters_name, seed, keep_samples=False, d_for_A=False):
        if key in cfg and (not d_for_A or f"Dwhole_s{seed}" in cfg):
            return
        r, x = run_one(unet, judge, dev, thick, method, G, n, lam,
                       None if iters_name is None else ITERS[iters_name], seed)
        r["iters_name"] = iters_name
        cfg[key] = r
        if keep_samples:
            torch.save(x[: 8 * n].cpu(), os.path.join(SAMPLE_DIR, f"{key}.pt"))
        if d_for_A:
            whole, stitched = select_D(judge, x, G, n, TARGET)
            for name, xd in (("Dwhole", whole), ("Dstitch", stitched)):
                rd = l3_metrics(dev, thick, xd, G, 1)
                rd.update({"method": name, "N": n, "n_batches": G, "rng_seed": seed, "from": key})
                cfg[f"{name}_s{seed}"] = rd
                if keep_samples:
                    torch.save(xd[:8].cpu(), os.path.join(SAMPLE_DIR, f"{name}_s{seed}.pt"))
        save()
        print(f"{key:<26} succ {r['success']['mean']:.4f} coord {r['coord_success']['mean']:.4f}"
              f"±{r['coord_success']['se']:.4f} anycoord {r['any_coord_success']['mean']:.3f} "
              f"bagree|s {r['brightness_agree_given_success']['p']:.3f} tagree|s {r['thickness_agree_given_success']['p']:.3f} "
              f"waste {r['waste']['mean']:.4f} {r['seconds']}s", flush=True)
        if d_for_A:
            for name in ("Dwhole", "Dstitch"):
                rd = cfg[f"{name}_s{seed}"]
                print(f"  {name}_s{seed}: succ {rd['success']['mean']:.4f} coord {rd['coord_success']['mean']:.4f}",
                      flush=True)

    # 1. A (+D)
    for s in range(3):
        do(f"A_N4_s{s}", "A", 4, 0.0, None, s, keep_samples=(s == 0), d_for_A=True)
    # 2. grid, seed 0, B and C alternating
    for lam in LAMS:
        for it in ITERS:
            for m in "BC":
                do(f"{m}_N4_lam{lam:g}_{it}_s0", m, 4, lam, it, 0, keep_samples=True)
    # 3. best per method by seed-0 coordinated success
    best = {}
    for m in "BC":
        cands = [(cfg[f"{m}_N4_lam{lam:g}_{it}_s0"]["coord_success"]["mean"], lam, it) for lam in LAMS for it in ITERS]
        _, lam, it = max(cands)
        best[m] = {"lam": lam, "iters": it}
        for s in (1, 2):
            do(f"{m}_N4_lam{lam:g}_{it}_s{s}", m, 4, lam, it, s)
    data["best"] = best
    save()
    print("best (seed 0, coordinated success):", best, flush=True)
    # 4. B with N = 8
    for s in range(3):
        do(f"B_N8_lam{best['B']['lam']:g}_{best['B']['iters']}_s{s}", "B", 8, best["B"]["lam"], best["B"]["iters"], s)
    # 5. lam = 0 check (two independent runs, resampling 10-90)
    for run, base in (("run1", 1000), ("run2", 2000)):
        for m, off in (("A", 0), ("B", 1), ("C", 2)):
            do(f"lam0_{run}_{m}", m, 4, 0.0, None if m == "A" else "10-90", base + off)
    lam0 = {}
    for k in PER_SAMPLE + ["any_coord_success"]:
        row = {}
        for m in "ABC":
            v = [cfg[f"lam0_{r}_{m}"][k] for r in ("run1", "run2")]
            row[m] = {"mean": sum(x["mean"] for x in v) / 2, "se": math.sqrt(sum(x["se"] ** 2 for x in v)) / 2}
        for m in "BC":
            den = math.hypot(row["A"]["se"], row[m]["se"])
            row[f"z_{m}"] = (row[m]["mean"] - row["A"]["mean"]) / den if den else float("nan")
        lam0[k] = row
    lam0["_fails"] = [f"{m}:{k} z={lam0[k]['z_' + m]:.2f}" for k in PER_SAMPLE for m in "BC"
                      if abs(lam0[k]["z_" + m]) >= 3]
    data["lam0_pooled"] = lam0
    save()
    print("lam0 pooled |z|>=3:", lam0["_fails"] or "none", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
