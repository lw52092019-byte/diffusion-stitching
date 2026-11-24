"""Combine results/timing/timing_kappa{1,0.1}.json -> timing.json, timing.png, and print summary tables.

Usage: python timing_report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT  # noqa: E402

KAPPAS = ["1", "0.1"]
# validated categorical slots 1-4 (dataviz reference palette, light mode)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(k):
    p = os.path.join(ROOT, "results", "timing", f"timing_kappa{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def curves(d):
    sv = d["survival"]
    ks = sorted(int(k) for k in sv["splice"])
    surv = [(k, sv["splice"][str(k)]["diff"]["keep_both"]["p"]) for k in ks]
    pulled = [(k, sv["splice"][str(k)]["diff"]["same_digit"]["p"]) for k in ks]
    jk = sorted(int(k) for k in sv["judge_curve"] if k != "final")
    judge = [(k, sv["judge_curve"][str(k)]["agree"]["p"]) for k in jk]
    judge_p = [(k, sv["judge_curve"][str(k)]["mean_maxprob"]) for k in jk]
    sw = d["sweep"]
    frm = sorted((int(n[4:]), v["success"]["mean"]) for n, v in sw.items() if n.startswith("from"))
    unt = sorted((int(n[5:]), v["success"]["mean"]) for n, v in sw.items() if n.startswith("until"))
    return dict(surv=surv, pulled=pulled, judge=judge, judge_p=judge_p, frm=frm, unt=unt)


def plot(all_curves):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor="#fcfcfb")
    for ax, k in zip(axes, KAPPAS):
        c = all_curves.get(k)
        ax.set_facecolor("#fcfcfb")
        if c is None:
            ax.set_title(f"kappa = {k}（无数据）")
            continue
        series = [
            (c["surv"], C1, "o", "-", "① 存活：拼接于第 k 步后两半都保持供体数字\n   （仅两供体数字不同的对子）"),
            (c["judge"], C2, "s", "-", "② 裁判在第 k 步对 x0_hat 的判断\n   与最终评估器类别一致"),
            (c["frm"], C3, "^", "-", "③ C 成功率：从第 k 步拼到结束"),
            (c["unt"], C4, "v", "--", "③ C 成功率：从头拼到第 k 步"),
        ]
        for pts, col, mk, ls, lab in series:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2, ls=ls, marker=mk, ms=6, mec="#fcfcfb", mew=1.5,
                    label=lab, zorder=3)
            if col in (C3, C4):  # curves 1-2 both end at ~1.00; labelling them would collide
                ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
        ax.set_title(f"kappa = {k}", color=INK, loc="left", fontsize=12)
        ax.set_xlabel("采样步 k（0 = 纯噪声，100 = 结束；共 100 步）", color=INK2)
        ax.set_xlim(-2, 104)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("比例", color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("拼接时机实验：存活曲线、裁判可读性、方法 C 成功率（lam=3，N=4）", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    out = os.path.join(ROOT, "timing.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print("wrote", out)


def table(d, k):
    sv = d["survival"]
    print(f"\n===== kappa={k}  n_pairs={sv['n_pairs']} diff_pairs={sv['n_diff_pairs']}  "
          f"ref same-digit {sv['ref_same_digit_frac']['p']:.3f} ref waste {sv['ref_waste']['p']:.4f}")
    print("control:", json.dumps(sv["control_rerun_no_splice"]))
    print("| k | 全部:左保持 | 全部:右保持 | 全部:两边 | 不同对:左保持 | 不同对:右保持 | 不同对:两边 | 不同对:被拉回同数字 | 不同对:废品 | 全部:废品 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for kk in sorted(sv["splice"], key=int):
        a, b = sv["splice"][kk]["all"], sv["splice"][kk]["diff"]
        f = lambda v: f"{v['p']:.3f}"
        print(f"| {kk} | {f(a['keep_left'])} | {f(a['keep_right'])} | {f(a['keep_both'])} | {f(b['keep_left'])} | "
              f"{f(b['keep_right'])} | {f(b['keep_both'])} | {f(b['same_digit'])} | {b['waste']['p']:.4f} | {a['waste']['p']:.4f} |")
    print("\n| 步 | 裁判与最终类别一致 | 左 | 右 | 裁判平均最大概率 | 最大概率≥0.9 |")
    print("|---|---|---|---|---|---|")
    for kk in sorted(sv["judge_curve"], key=lambda s: 999 if s == "final" else int(s)):
        j = sv["judge_curve"][kk]
        print(f"| {kk} | {j['agree']['p']:.3f} | {j['agree_left']['p']:.3f} | {j['agree_right']['p']:.3f} | "
              f"{j['mean_maxprob']:.3f} | {j['frac_maxprob_ge_0.9']:.3f} |")
    print("\n| 方案 | 重采样点 | 成功率 | 批内≥1成功 | 左命中 | 右命中 | 废图率 | 耗时 |")
    print("|---|---|---|---|---|---|---|---|")
    for n, v in d["sweep"].items():
        g = lambda m, p=3: f"{v[m]['mean']:.{p}f} ± {v[m]['se']:.{p}f}"
        print(f"| {n} | {','.join(map(str, v['resample_iters']))} | {g('success')} | {g('any_success')} | "
              f"{v['hit_left']['mean']:.3f} | {v['hit_right']['mean']:.3f} | {g('waste', 4)} | {v['seconds']:.0f} |")


def main():
    data = {k: load(k) for k in KAPPAS}
    cs = {k: curves(d) for k, d in data.items() if d and "survival" in d and "sweep" in d}
    out = {"description": "Stitch-timing experiment. survival: no judge, splice left/right donors at step k "
                          "(noise spliced the same way) and check whether each half keeps the donor's reference "
                          "digit (evaluator). judge_curve: judge on reference x0_hat every 5 steps vs evaluator "
                          "class of the final image. sweep: method C lam=3 N=4 1000 batches seed 0 with resampling "
                          "restricted to steps >= k (fromK) or <= k (untilK).",
           "curves": cs, "raw": data}
    json.dump(out, open(os.path.join(ROOT, "timing.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote timing.json")
    plot(cs)
    for k, d in data.items():
        if d and "survival" in d and "sweep" in d:
            table(d, k)


if __name__ == "__main__":
    main()
