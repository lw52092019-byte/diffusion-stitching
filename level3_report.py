"""Level 3: build level3.json, level3.png, level3_samples_{A,B,C,D}.png and print summary tables.

Usage: python level3_report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from common import ROOT  # noqa: E402
from level3_common import L3_CKPT, L3_RES, save_grid  # noqa: E402

C_BRIGHT, C_THICK, C_DIGIT = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(p):
    p = os.path.join(ROOT, p) if not os.path.isabs(p) else p
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def curves(e1):
    ks = sorted(int(k) for k in e1["splice"])
    sp = lambda f: [(k, f(e1["splice"][str(k)])) for k in ks]
    rk = sorted(int(k) for k in e1["readability"])
    rd = lambda a: [(k, e1["readability"][str(k)][a]["p"]) for k in rk]
    return {
        "brightness_unified": sp(lambda r: r["brightness"]["unified"]["p"]),
        "thickness_unified": sp(lambda r: r["thickness"]["unified"]["p"]),
        "brightness_mismatch_kept": sp(lambda r: r["brightness"]["both_kept_mismatch"]["p"]),
        "thickness_mismatch_kept": sp(lambda r: r["thickness"]["both_kept_mismatch"]["p"]),
        "digits_both_kept_all_pairs": sp(lambda r: r["all_pairs"]["digits_both_kept"]["p"]),
        "read_digit": rd("digit"), "read_brightness": rd("brightness"), "read_thickness": rd("thickness"),
    }


def plot(c, out):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, facecolor=SURF)
    panels = [
        ("拼接于第 k 步后，最终图上（不用裁判）", "拼接时刻 k", [
            (c["brightness_unified"], C_BRIGHT, "o", "-", "亮度被统一（两供体亮度不同的对子）"),
            (c["thickness_unified"], C_THICK, "s", "-", "粗细被统一（两供体粗细不同的对子）"),
            (c["digits_both_kept_all_pairs"], C_DIGIT, "^", "-", "两格数字都保持供体（全部对子）"),
        ]),
        ("参考轨迹第 k 步的 x0_hat 是否已可读", "采样步 k", [
            (c["read_brightness"], C_BRIGHT, "o", "--", "亮度判定 与最终一致"),
            (c["read_thickness"], C_THICK, "s", "--", "粗细判定 与最终一致"),
            (c["read_digit"], C_DIGIT, "^", "--", "数字裁判 与最终一致"),
        ]),
    ]
    for ax, (title, xl, series) in zip(axes, panels):
        ax.set_facecolor(SURF)
        ax.axvspan(30, 60, color="#ecebe6", zorder=0)
        ax.text(45, 1.035, "C 的有效窗口 30–60", ha="center", va="bottom", fontsize=9, color=INK2)
        for pts, col, mk, ls, lab in series:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2, ls=ls, marker=mk, ms=6, mec=SURF, mew=1.5, label=lab, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=18)
        ax.set_xlabel(f"{xl}（0 = 纯噪声，100 = 结束）", color=INK2)
        ax.set_xlim(-2, 102); ax.set_ylim(0, 1.08)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
        if ax is axes[0]:
            ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.74), frameon=False, fontsize=9)
        else:
            ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel("比例", color=INK2)
    fig.suptitle("第三级（共享风格）：风格何时凝固、何时可读", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURF)
    print("wrote", out)


def sample_grids(e2):
    sd = os.path.join(L3_RES, "samples")
    best = e2.get("best", {})
    srcs = {"A": "A_N4_s0"}
    for m in "BC":
        if m in best:
            srcs[m] = f"{m}_N4_lam{best[m]['lam']:g}_{best[m]['iters']}_s0"
    for m, key in srcs.items():
        p = os.path.join(sd, key + ".pt")
        if os.path.exists(p):
            save_grid(torch.load(p), 4, os.path.join(ROOT, f"level3_samples_{m}.png"))
            print(f"level3_samples_{m}.png <- {key}")
    pa, pw, ps = (os.path.join(sd, f) for f in ("A_N4_s0.pt", "Dwhole_s0.pt", "Dstitch_s0.pt"))
    if all(os.path.exists(p) for p in (pa, pw, ps)):
        a, w, s = torch.load(pa).view(8, 4, 1, 28, 56), torch.load(pw), torch.load(ps)
        blank = torch.full_like(w, -1)
        rows = [torch.cat([a[i], blank[i:i + 1], w[i:i + 1], s[i:i + 1]]) for i in range(8)]
        save_grid(torch.cat(rows), 7, os.path.join(ROOT, "level3_samples_D.png"))
        print("level3_samples_D.png: each row = A's 4 samples | blank | D-whole | D-stitch")


def main():
    e1, e2 = load("level3/results/exp1.json"), load("level3/results/exp2.json")
    out = {
        "description": "Level 3 (shared style). exp1: style survival/unification after splicing at step k (no judge) "
                       "and readability of x0_hat; exp2: A/B/C/D with coordinated success "
                       "(digits 1|3 and both cells same brightness and thickness), all from the evaluators.",
        "style_params": load(os.path.join(L3_CKPT, "style_params.json")),
        "classifiers": load(os.path.join(L3_CKPT, "classifiers_metrics.json")),
        "classifiers_first_attempt_dilate3_failed": load("level3/logs/classifiers_metrics_dilate3_failed.json"),
        "diffusion_training": load(os.path.join(L3_CKPT, "ddpm_l3_train.json")),
        "model_acceptance": load("level3/results/model_acceptance.json"),
        "exp1": e1, "exp2": e2,
        "curves": curves(e1) if e1 and "readability" in e1 and e1.get("splice") else None,
    }
    json.dump(out, open(os.path.join(ROOT, "level3.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote level3.json")
    if out["curves"]:
        plot(out["curves"], os.path.join(ROOT, "level3.png"))
    if e2:
        sample_grids(e2)


if __name__ == "__main__":
    main()
