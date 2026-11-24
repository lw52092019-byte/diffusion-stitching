"""Benchmark the hand-written UNet's training step and break down where time goes.

Usage: python bench_unet.py
"""
import time

import torch
import torch.nn.functional as F

from diffusion import EMA, PairSampler, build_unet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def timed(fn, n=40, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    unet = build_unet().cuda()
    print(f"params {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    ema = EMA(unet, 0.999)
    data = PairSampler(1.0)
    abar = torch.linspace(0.999, 0.01, 1000, device="cuda")

    for bs in (64, 128, 256):
        x = torch.randn(bs, 1, 28, 56, device="cuda")
        tt = torch.randint(0, 1000, (bs,), device="cuda")

        def model_only(amp=True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.mse_loss(unet(x, tt).float(), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        def fwd_only():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                unet(x, tt)

        def full_loop():
            x0, _, _ = data.sample(bs)
            t = torch.randint(0, 1000, (bs,), device="cuda")
            noise = torch.randn_like(x0)
            a = abar[t].view(-1, 1, 1, 1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(unet(xt, t).float(), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            ema.update(unet)

        r = {
            "fwd no-grad bf16": timed(fwd_only),
            "train step fp32": timed(lambda: model_only(False)),
            "train step bf16": timed(model_only),
            "data sample": timed(lambda: data.sample(bs)),
            "EMA update": timed(lambda: ema.update(unet)),
            "full loop bf16": timed(full_loop),
        }
        print(f"batch {bs}: " + ", ".join(f"{k} {v:.1f} ms" for k, v in r.items())
              + f" -> {1000 / r['full loop bf16']:.1f} it/s, {bs * 1000 / r['full loop bf16']:.0f} img/s",
              flush=True)


if __name__ == "__main__":
    main()
