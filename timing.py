"""Stitch-timing experiment (three parts, both models).

  1. survival : no judge. 8000 independent reference trajectories (noise of step i drawn from a
                generator seeded by (seed, i)). Donors: 0..3999 = left, 4000..7999 = right.
                For each splice step k, hard-join left half of left donor k-state with right half of
                right donor k-state (4000 images), continue denoising to the end, using the same
                per-step noise spliced the same way. Evaluator judges whether each half still draws
                the donor's reference digit.
  2. judgecurve: on the reference trajectories, every 5 steps the judge classifies x0_hat's two
                cells; agreement with the evaluator's class of the final image, and mean max prob.
  3. sweep    : method C, lam=3, N=4, 1000 batches, seed 0, resampling every 10 steps restricted to
                "from k to the end" (k=20..90) or "from the start up to k" (k=10..80).

Usage: python timing.py --kappa 1.0 [--part all|survival|sweep] [--n-ref 8000] [--batches 1000]
Output: results/timing/timing_kappa{k}.json  (new files only; existing results untouched)
"""
import argparse
import json
import math
import os
import time

import torch

from common import DEVICE, ROOT, load_classifier
from diffusion import H, NUM_SAMPLE_STEPS, W, build_scheduler, eps_model, load_model
from smc import HALF, half_logprobs, metrics, run_config

SPLICE_KS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
JUDGE_EVERY = 5
CHUNK = 1000
OUT_DIR = os.path.join(ROOT, "results", "timing")


# ----------------------------------------------------------------------------- DDPM step with explicit noise
class Stepper:
    """Same update as diffusers DDPMScheduler.step (fixed_small, clip_sample, epsilon), but the
    Gaussian noise is passed in, so it can be spliced. Checked against scheduler.step in --selftest."""

    def __init__(self, steps=NUM_SAMPLE_STEPS):
        s = build_scheduler()
        s.set_timesteps(steps, device=DEVICE)
        self.sched = s
        self.timesteps = s.timesteps
        ac = s.alphas_cumprod.to(DEVICE)
        self.coef = []
        for t in s.timesteps:
            t = int(t)
            pt = int(s.previous_timestep(t))
            a_t = ac[t]
            a_p = ac[pt] if pt >= 0 else torch.tensor(1.0, device=DEVICE)
            b_t, b_p = 1 - a_t, 1 - a_p
            cur_a = a_t / a_p
            cur_b = 1 - cur_a
            var = torch.clamp((1 - a_p) / (1 - a_t) * cur_b, min=1e-20)
            self.coef.append(dict(sa=a_t.sqrt(), sb=b_t.sqrt(), c0=a_p.sqrt() * cur_b / b_t,
                                  ct=cur_a.sqrt() * b_p / b_t, std=var.sqrt() if t > 0 else 0.0))

    def step(self, i, x, eps, noise):
        c = self.coef[i]
        x0 = ((x - c["sb"] * eps) / c["sa"]).clamp(-1, 1)
        return c["c0"] * x0 + c["ct"] * x + c["std"] * noise, x0


def noise_at(seed, i, P):
    """Noise for step i (i = -1: initial x_T), reproducible from (seed, i)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed * 1_000_003 + i + 1)
    return torch.randn(P, 1, H, W, device=DEVICE, generator=g)


def eps_all(unet, x, t):
    return torch.cat([eps_model(unet, x[j:j + CHUNK], t) for j in range(0, len(x), CHUNK)])


@torch.no_grad()
def classify(clf, x):
    """-> (clsL, pmaxL, clsR, pmaxR)"""
    lpL, lpR = half_logprobs(clf, x)
    mL, mR = lpL.exp().max(-1), lpR.exp().max(-1)
    return mL.indices, mL.values, mR.indices, mR.values


def frac(mask, sub=None):
    m = mask if sub is None else mask[sub]
    n = int(m.numel())
    p = m.float().mean().item() if n else float("nan")
    return {"p": p, "se": math.sqrt(p * (1 - p) / n) if n else float("nan"), "n": n}


# ----------------------------------------------------------------------------- parts 1 + 2
@torch.no_grad()
def survival(unet, judge, evaluator, n_ref, seed, ks=SPLICE_KS, out=None, save=lambda: None, cache=None):
    """Resumable: `out` holds results so far; `save()` is called after every finished piece; the
    reference trajectories (states at each k, final images, judge records) are cached in `cache`
    so a resumed run continues from exactly the same reference."""
    out = {} if out is None else out
    done = set(out.get("splice", {}))
    if "control_rerun_no_splice" in out and all(str(k) in done for k in ks):
        print("survival: all splice steps already done, skipping", flush=True)
        return out
    st = Stepper()
    P, half_n = n_ref, n_ref // 2
    if cache and os.path.exists(cache):
        c = torch.load(cache, map_location=DEVICE)
        states, jrec, ref, ref_secs = c["states"], c["jrec"], c["ref"], c["ref_seconds"]
        print("loaded cached reference trajectories:", cache, flush=True)
    else:
        x = noise_at(seed, -1, P)
        states, jrec = {}, {}
        t0 = time.time()
        for i, t in enumerate(st.timesteps):
            if i in ks:
                states[i] = x.clone()
            x_new, x0 = st.step(i, x, eps_all(unet, x, t), noise_at(seed, i, P))
            if i % JUDGE_EVERY == 0:
                jrec[i] = classify(judge, x0)
            x = x_new
        ref = x.clamp(-1, 1)
        ref_secs = time.time() - t0
        jrec["final"] = classify(judge, ref)
        if cache:
            torch.save({"states": states, "jrec": jrec, "ref": ref, "ref_seconds": ref_secs,
                        "seed": seed, "n_ref": n_ref}, cache)
            print("cached reference trajectories:", cache, flush=True)
    eL, epL, eR, epR = classify(evaluator, ref)

    # ---- part 2: judge on x0_hat vs evaluator's final class
    jc = {}
    for key, (cL, pL, cR, pR) in jrec.items():
        confL, confR = epL >= 0.5, epR >= 0.5
        agree = torch.cat([(cL == eL)[confL], (cR == eR)[confR]])
        jc[str(key)] = {"agree": frac(agree),
                        "agree_left": frac((cL == eL)[confL]), "agree_right": frac((cR == eR)[confR]),
                        "mean_maxprob": torch.cat([pL, pR]).float().mean().item(),
                        "frac_maxprob_ge_0.9": torch.cat([pL, pR]).ge(0.9).float().mean().item()}

    # ---- part 1: splice
    refL, refokL = eL[:half_n], epL[:half_n] >= 0.5           # left donors' reference left digit
    refR, refokR = eR[half_n:], epR[half_n:] >= 0.5           # right donors' reference right digit
    diff = refokL & refokR & (refL != refR)
    out.update({"n_pairs": half_n, "n_diff_pairs": int(diff.sum()),
                "ref_seconds": round(ref_secs, 1),
                "ref_same_digit_frac": frac(((eL == eR) & (epL >= 0.5) & (epR >= 0.5))),
                "ref_waste": frac(~((epL >= 0.5) & (epR >= 0.5))),
                "judge_curve": jc})
    out.setdefault("splice", {})
    save()

    def continue_from(k, xk, splice_noise):
        x = xk
        for i in range(k, NUM_SAMPLE_STEPS):
            n = noise_at(seed, i, P)
            n = torch.cat([n[:half_n][..., :HALF], n[half_n:][..., HALF:]], 3) if splice_noise else n[:half_n]
            x, _ = st.step(i, x, eps_all(unet, x, st.timesteps[i]), n)
        return x.clamp(-1, 1)

    # control: re-run left donors from k=0 without splicing -> should reproduce the reference
    if "control_rerun_no_splice" not in out:
        ctrl = continue_from(0, states[0][:half_n], splice_noise=False)
        cL, _, cR, _ = classify(evaluator, ctrl)
        out["control_rerun_no_splice"] = {
            "max_abs_pixel_diff": (ctrl - ref[:half_n]).abs().max().item(),
            "mean_abs_pixel_diff": (ctrl - ref[:half_n]).abs().mean().item(),
            "frac_images_identical(<1e-3)": frac((ctrl - ref[:half_n]).abs().amax(dim=(1, 2, 3)) < 1e-3),
            "left_class_same": frac(cL == eL[:half_n]), "right_class_same": frac(cR == eR[:half_n])}
        save()
    print("control rerun:", out["control_rerun_no_splice"], flush=True)

    for k in ks:
        if str(k) in out["splice"]:
            print(f"k={k:3d} already done, skipping", flush=True)
            continue
        t0 = time.time()
        xk = torch.cat([states[k][:half_n][..., :HALF], states[k][half_n:][..., HALF:]], 3)
        xf = continue_from(k, xk, splice_noise=True)
        sL, spL, sR, spR = classify(evaluator, xf)
        okL, okR = spL >= 0.5, spR >= 0.5
        kL = okL & refokL & (sL == refL)
        kR = okR & refokR & (sR == refR)
        res = {}
        for name, sub in (("all", None), ("diff", diff)):
            res[name] = {"keep_left": frac(kL, sub), "keep_right": frac(kR, sub), "keep_both": frac(kL & kR, sub),
                         "same_digit": frac(okL & okR & (sL == sR), sub),
                         "waste": frac(~(okL & okR), sub)}
        res["seconds"] = round(time.time() - t0, 1)
        out["splice"][str(k)] = res
        save()
        d = res["diff"]
        print(f"k={k:3d} diff-pairs keepL {d['keep_left']['p']:.3f} keepR {d['keep_right']['p']:.3f} "
              f"both {d['keep_both']['p']:.3f} same {d['same_digit']['p']:.3f} waste {d['waste']['p']:.4f} "
              f"({res['seconds']}s)", flush=True)
    return out


# ----------------------------------------------------------------------------- part 3
def sweep(unet, judge, evaluator, batches, seed=0, lam=3.0, n=4, res=None, save=lambda: None):
    res = {} if res is None else res
    plans = [(f"from{k}", {i for i in range(10, 100, 10) if i >= k}) for k in range(20, 100, 10)] + \
            [(f"until{k}", {i for i in range(10, 100, 10) if i <= k}) for k in range(10, 90, 10)]
    for name, its in plans:
        if name in res:
            print(f"{name:<8} already done, skipping", flush=True)
            continue
        r, _ = run_config(unet, judge, evaluator, "C", batches, n, lam=lam, window=name, seed=seed,
                          iters=its)
        r["resample_iters"] = sorted(its)
        res[name] = r
        save()
        print(f"{name:<8} iters {sorted(its)} succ {r['success']['mean']:.4f}±{r['success']['se']:.4f} "
              f"any {r['any_success']['mean']:.3f} waste {r['waste']['mean']:.4f} {r['seconds']}s", flush=True)
    return res


def selftest():
    st = Stepper()
    torch.manual_seed(0)
    x = torch.randn(16, 1, H, W, device=DEVICE)
    eps = torch.randn_like(x)
    worst = 0.0
    for i, t in enumerate(st.timesteps):
        g1 = torch.Generator(device=DEVICE).manual_seed(i)
        g2 = torch.Generator(device=DEVICE).manual_seed(i)
        ref = st.sched.step(eps, t, x, generator=g1)
        mine, x0 = st.step(i, x, eps, torch.randn(x.shape, device=DEVICE, generator=g2))
        worst = max(worst, (mine - ref.prev_sample).abs().max().item(),
                    (x0 - ref.pred_original_sample).abs().max().item())
    print("selftest max |diff| vs scheduler.step:", worst)
    assert worst < 1e-4, worst  # float32 rounding-order differences only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--part", default="all", choices=["all", "survival", "sweep"])
    ap.add_argument("--n-ref", type=int, default=8000)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, f"timing_kappa{args.kappa:g}.json")
    data = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for key, val in (("kappa", args.kappa), ("seed", args.seed), ("n_ref", args.n_ref), ("batches", args.batches)):
        if key in data and data[key] != val:
            raise SystemExit(f"{out_path} has {key}={data[key]}, not {val}; refusing to mix runs")
        data[key] = val

    def save():  # atomic: a power cut mid-write cannot corrupt the finished pieces
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, out_path)

    unet = load_model(args.kappa)
    judge, evaluator = load_classifier("judge"), load_classifier("evaluator")
    if args.part in ("all", "survival"):
        cache = os.path.join(OUT_DIR, f"ref_kappa{args.kappa:g}_seed{args.seed}_n{args.n_ref}.pt")
        data["survival"] = survival(unet, judge, evaluator, args.n_ref, args.seed,
                                    out=data.setdefault("survival", {}), save=save, cache=cache)
        save()
    if args.part in ("all", "sweep"):
        data["sweep"] = sweep(unet, judge, evaluator, args.batches, res=data.setdefault("sweep", {}), save=save)
        save()
    print("saved", out_path)


if __name__ == "__main__":
    main()
