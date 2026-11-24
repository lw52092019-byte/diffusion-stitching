"""Logic tests for smc.py that need no trained diffusion model (tiny GPU use).

Usage: python test_smc.py
"""
import torch

from common import DEVICE, load_classifier, load_mnist04
from diffusion import build_unet
from smc import HALF, _ancestors, metrics, resample_iters, sample, select_D, stitch


def test_windows():
    assert sorted(resample_iters("full")) == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert sorted(resample_iters("early")) == [10, 20, 30, 40]
    assert sorted(resample_iters("late")) == [50, 60, 70, 80, 90]


def test_ancestors():
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    G, n = 3, 4
    logw = torch.full((G * n,), -1e9, device=DEVICE)
    logw[[1, 4 + 3, 8 + 0]] = 0.0          # one dominant particle per group
    a = _ancestors(logw, G, n, gen)
    assert a.tolist() == [1] * 4 + [7] * 4 + [8] * 4, a.tolist()
    # uniform weights stay inside the group
    a = _ancestors(torch.zeros(G * n * 100, device=DEVICE), G * 100, n, gen).view(G * 100, n)
    grp = a // n
    assert (grp == torch.arange(G * 100, device=DEVICE)[:, None]).all()


def test_stitch():
    x = torch.arange(3, device=DEVICE).float().view(3, 1, 1, 1).expand(3, 1, 28, 56).clone()
    x[..., HALF:] += 10                                   # left = k, right = 10 + k
    out = stitch(x, torch.tensor([2, 0, 0], device=DEVICE), torch.tensor([1, 1, 2], device=DEVICE))
    assert out[:, 0, 0, 0].tolist() == [2, 0, 0] and out[:, 0, 0, -1].tolist() == [11, 11, 12]


def test_metrics_and_stitch():
    ev = load_classifier("evaluator")
    x, y = load_mnist04(train=False)
    ones, threes, zeros = x[y == 1][:8], x[y == 3][:8], x[y == 0][:8]
    good = torch.cat([ones, threes], dim=3)              # (8,1,28,56) left 1, right 3
    bad = torch.cat([zeros, threes], dim=3)
    blank = torch.full_like(good, -1.0)                  # all-background: evaluator unsure?
    batch = torch.cat([good[:4], torch.cat([good[4:6], bad[:2]])])  # G=2, n=4
    r = metrics(ev, batch, 2, 4)
    assert abs(r["success"]["mean"] - 0.75) < 1e-6, r["success"]
    assert r["any_success"]["mean"] == 1.0
    assert abs(r["hit_left"]["mean"] - 0.75) < 1e-6 and r["hit_right"]["mean"] == 1.0
    assert r["diversity"]["n_batches"] == 2
    dup = torch.cat([good[:1].repeat(4, 1, 1, 1)])
    r2 = metrics(ev, dup, 1, 4)
    assert r2["dup_frac_success_pairs"] == 1.0 and r2["diversity"]["mean"] < 1e-4
    rb = metrics(ev, blank[:4], 1, 4)
    print("blank image waste rate (info):", rb["waste"]["mean"], "ink-checked:", rb["waste_ink"]["mean"])
    assert rb["waste_ink"]["mean"] == 1.0
    blank_left = torch.cat([torch.full_like(ones[:4], -1.0), threes[:4]], dim=3)
    rl = metrics(ev, blank_left, 1, 4)
    print("blank-left + 3: success", rl["success"]["mean"], "success_ink", rl["success_ink"]["mean"])
    assert rl["success_ink"]["mean"] == 0.0 and rl["left_near_blank"]["mean"] == 1.0
    assert r["success_ink"]["mean"] == r["success"]["mean"]   # real digits pass the ink check
    # seam: identical columns across the midline -> 0
    flat = good[:4].clone()
    flat[..., HALF] = flat[..., HALF - 1]
    assert metrics(ev, flat, 1, 4)["seam"]["mean"] < 1e-6


def test_select_D():
    judge = load_classifier("judge")
    x, y = load_mnist04(train=False)
    d = {c: x[y == c][:4] for c in range(5)}
    # batch 0: [0|3, 1|0, 2|2, 1|3]  -> whole picks index 3; stitch: left from a "1", right from a "3"
    # batch 1: [1|0, 2|3, 0|0, 4|4]  -> whole has no 1|3; stitch builds 1 (idx 4) | 3 (idx 5)
    b0 = [torch.cat([d[a][i:i + 1], d[b][i:i + 1]], 3) for i, (a, b) in enumerate([(0, 3), (1, 0), (2, 2), (1, 3)])]
    b1 = [torch.cat([d[a][i:i + 1], d[b][i:i + 1]], 3) for i, (a, b) in enumerate([(1, 0), (2, 3), (0, 0), (4, 4)])]
    xs = torch.cat(b0 + b1)
    whole, st = select_D(judge, xs, 2, 4)
    assert torch.equal(whole[0], xs[3])
    assert torch.equal(st[1][..., :HALF], xs[4][..., :HALF]) and torch.equal(st[1][..., HALF:], xs[5][..., HALF:])
    ev = load_classifier("evaluator")
    r = metrics(ev, st, 2, 1)
    assert r["success"]["mean"] == 1.0, r["success"]


def test_sampler_runs():
    torch.manual_seed(0)
    unet = build_unet().to(DEVICE).eval()                # untrained: only checks plumbing
    judge = load_classifier("judge")
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    for m, lam, win, exp in [("A", 0, "full", 0), ("B", 0, "full", 9), ("C", 3, "early", 4),
                             ("C", 10, "late", 5), ("B", 10, "late", 5)]:
        x, info = sample(unet, judge, m, 5, 4, lam, win, gen=gen, steps=100)
        assert x.shape == (20, 1, 28, 56) and torch.isfinite(x).all()
        assert info["n_resample"] == exp, (m, win, info)
        print(m, lam, win, info)


if __name__ == "__main__":
    test_windows(); print("windows ok")
    test_ancestors(); print("ancestors ok")
    test_stitch(); print("stitch ok")
    test_metrics_and_stitch(); print("metrics ok")
    test_select_D(); print("select_D ok")
    test_sampler_runs(); print("sampler plumbing ok")
