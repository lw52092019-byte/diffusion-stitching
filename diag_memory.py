"""Peak GPU memory of the pieces of one sampling step (why B/C were slower than A).

Usage: python diag_memory.py
"""
import torch

from common import load_classifier
from diffusion import build_scheduler, build_unet, eps_model
from smc import half_logprobs

P = 4000  # 1000 batches x N=4


def peak(label, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    print(f"{label:<45} peak +{(torch.cuda.max_memory_allocated() - base) / 2**30:5.2f} GB "
          f"(reserved total {torch.cuda.memory_reserved() / 2**30:5.2f} GB)")


def main():
    unet = build_unet().cuda().eval()
    judge = load_classifier("judge")
    x = torch.randn(P, 1, 28, 56, device="cuda")
    sched = build_scheduler()
    sched.set_timesteps(100, device="cuda")
    t = sched.timesteps[50]
    peak("UNet eps, chunk 2048 (A, B, C every step)",
         lambda: torch.cat([eps_model(unet, x[j:j + 2048], t) for j in range(0, P, 2048)]))
    peak("scheduler.step on 4000", lambda: sched.step(torch.randn_like(x), t, x))
    peak("judge on 8000 halves via half_logprobs (B, C)", lambda: half_logprobs(judge, x))
    for c in (2048, 1024):
        def j(c=c):
            both = torch.cat([x[..., :28], x[..., 28:]], 0)
            [judge(both[i:i + c]) for i in range(0, len(both), c)]
        peak(f"judge on 8000 halves, chunk {c}", j)


if __name__ == "__main__":
    with torch.no_grad():
        main()
