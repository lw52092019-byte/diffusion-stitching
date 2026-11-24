"""Level 3: train the unconditional DDPM on shared-style two-cell images (kappa = 1 digits).

Same UNet / schedule / optimiser settings as train_diffusion.py. Resumable: every --ckpt-every
steps a full training state (model, EMA, optimiser, RNG, step) is written; on start the latest
one is loaded and training continues from there.

Usage: python level3_train_diffusion.py [--steps 35000] [--ckpt-every 5000] [--bench 200]
Output: level3/checkpoints/ddpm_l3.pt (final EMA), ddpm_l3_step{N}.pt, ddpm_l3_train.json
"""
import argparse
import glob
import json
import os
import re
import time

import torch
import torch.nn.functional as F

from common import DEVICE, seed_all
from diffusion import EMA, NUM_TRAIN_T, build_scheduler, build_unet
from level3_common import L3_CKPT, StylePairSampler

FINAL = os.path.join(L3_CKPT, "ddpm_l3.pt")


def step_path(n):
    return os.path.join(L3_CKPT, f"ddpm_l3_step{n}.pt")


def latest_state():
    best = None
    for p in glob.glob(os.path.join(L3_CKPT, "ddpm_l3_step*.pt")):
        n = int(re.search(r"step(\d+)\.pt$", p).group(1))
        if best is None or n > best[0]:
            best = (n, p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(FINAL) and not args.bench:
        print("final checkpoint exists, nothing to do:", FINAL)
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    data = StylePairSampler()
    unet = build_unet().to(DEVICE)
    ema = EMA(unet, args.ema)
    abar = build_scheduler().alphas_cumprod.to(DEVICE)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / args.warmup)

    start, prev_secs = 0, 0.0
    lat = None if args.bench else latest_state()
    if lat:
        st = torch.load(lat[1], map_location=DEVICE)
        unet.load_state_dict(st["model"]); ema.model.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); gen.set_state(st["gen_state"].cpu())
        start, prev_secs = st["step"], st.get("train_seconds", 0.0)
        print(f"resumed from {lat[1]} (step {start})", flush=True)

    total = args.bench or args.steps
    t0 = time.time()
    for step in range(start, total):
        x0, _, _, _ = data.sample(args.batch, gen)
        t = torch.randint(0, NUM_TRAIN_T, (args.batch,), device=DEVICE, generator=gen)
        noise = torch.randn(x0.shape, device=DEVICE, generator=gen)
        a = abar[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(xt, t)
        loss = F.mse_loss(pred.float(), noise)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        ema.update(unet, decay=min(args.ema, (1 + step) / (10 + step)))
        if (step + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{total} loss {loss.item():.4f} {(step + 1 - start) / el:.1f} it/s "
                  f"wall {time.strftime('%H:%M:%S')}", flush=True)
        if not args.bench and (step + 1) % args.ckpt_every == 0:
            tmp = step_path(step + 1) + ".tmp"
            torch.save({"model": unet.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                        "gen_state": gen.get_state(), "step": step + 1, "args": vars(args),
                        "train_seconds": prev_secs + time.time() - t0}, tmp)
            os.replace(tmp, step_path(step + 1))
            print(f"checkpoint step {step + 1}", flush=True)

    torch.cuda.synchronize()
    el = time.time() - t0
    if args.bench:
        print(f"bench {total} steps {el:.1f}s -> {total / el:.1f} it/s")
        return
    torch.save({"ema": ema.model.state_dict(), "model": unet.state_dict(), "args": vars(args)}, FINAL)
    json.dump({**vars(args), "train_seconds_total": round(prev_secs + el, 1), "resumed_from_step": start},
              open(FINAL.replace(".pt", "_train.json"), "w"), indent=2)
    print(f"saved {FINAL}; this session {el / 60:.1f} min, total {(prev_secs + el) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
