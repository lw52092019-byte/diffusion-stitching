"""Train the unconditional DDPM for a given kappa.

Usage:
  python train_diffusion.py --kappa 1.0              # full run
  python train_diffusion.py --kappa 1.0 --bench 200  # just time 200 steps
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from common import CKPT_DIR, DEVICE, seed_all
from diffusion import EMA, NUM_TRAIN_T, PairSampler, build_scheduler, build_unet, ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="only time this many steps, save nothing")
    ap.add_argument("--ckpt-every", type=int, default=0, help="also save an intermediate checkpoint every N steps")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)

    data = PairSampler(args.kappa)
    unet = build_unet().to(DEVICE)
    nparams = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {nparams / 1e6:.2f}M")
    ema = EMA(unet, args.ema)
    sched = build_scheduler()
    abar = sched.alphas_cumprod.to(DEVICE)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.bench or args.steps
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / args.warmup)

    t0 = time.time()
    for step in range(total):
        x0, _, _ = data.sample(args.batch, gen)
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
            print(f"step {step + 1}/{total} loss(sampled) {loss.item():.4f} "
                  f"{(step + 1) / el:.1f} it/s, elapsed {el / 60:.1f} min, "
                  f"wall {time.strftime('%H:%M:%S')}", flush=True)
        if args.ckpt_every and not args.bench and (step + 1) % args.ckpt_every == 0 and step + 1 < total:
            os.makedirs(CKPT_DIR, exist_ok=True)
            p = ckpt_path(args.kappa, step + 1)
            torch.save({"ema": ema.model.state_dict(), "model": unet.state_dict(), "args": vars(args),
                        "step": step + 1}, p)
            print(f"checkpoint {p}", flush=True)

    torch.cuda.synchronize()
    el = time.time() - t0
    if args.bench:
        print(f"bench: {total} steps in {el:.1f}s -> {total / el:.1f} it/s; "
              f"{args.steps} steps would take {args.steps / (total / el) / 60:.1f} min")
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"ema": ema.model.state_dict(), "model": unet.state_dict(), "args": vars(args)},
               ckpt_path(args.kappa))
    with open(ckpt_path(args.kappa).replace(".pt", "_train.json"), "w") as f:
        json.dump({**vars(args), "params": nparams, "train_seconds": round(el, 1)}, f, indent=2)
    print(f"saved {ckpt_path(args.kappa)}; trained {el / 60:.1f} min")


if __name__ == "__main__":
    main()
