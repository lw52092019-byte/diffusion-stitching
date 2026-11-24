"""Turn results/raw_kappa{k}.json into markdown tables (results/table_kappa{k}.md, Chinese labels).

Usage: python make_table.py --kappa 1.0 [--lam0 logs/lam0_kappa1_stepfinal.json ...] [--lam0-pooled logs/..._pooled.json]
"""
import argparse
import json
import math
import os
import statistics

from common import ROOT

OUT_DIR = os.path.join(ROOT, "results")
WIN_ZH = {"full": "全程", "early": "前45%", "late": "后55%", None: "-"}


def fm(d, k, p=3, se=True):
    v = d.get(k) or {}
    if v.get("mean") is None:
        return "n/a"
    s = f"{v['mean']:.{p}f}"
    return s + (f" ± {v['se']:.{p}f}" if se and v.get("se") is not None else "")


def seed_vals(rows, k):
    return [r[k]["mean"] for r in rows if r.get(k) and r[k]["mean"] is not None]


def seeds_summary(rows, k, p=3):
    """mean over seeds ± std over seeds, using per-seed means."""
    vals = seed_vals(rows, k)
    if not vals:
        return "n/a"
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    return f"{statistics.mean(vals):.{p}f} ± {sd:.{p}f}"


def equiv_N(points, target):
    """points: [(N, value)] increasing N; log2-linear interpolation of N where value == target."""
    pts = sorted(points)
    if target <= pts[0][1]:
        return f"≤{pts[0][0]}"
    for (n0, v0), (n1, v1) in zip(pts, pts[1:]):
        if v0 <= target <= v1 and v1 > v0:
            f = (target - v0) / (v1 - v0)
            return f"≈{2 ** (math.log2(n0) + f * (math.log2(n1) - math.log2(n0))):.1f}"
    return f">{pts[-1][0]}"


def build(kappa, in_dir, lam0_files, lam0_pooled):
    raw = json.load(open(os.path.join(in_dir, f"raw_kappa{kappa:g}.json")))
    best = raw.get("_best", {})
    runs = {k: v for k, v in raw.items() if not k.startswith("_")}

    def rows_for(m, n, lam=None, win=None):
        return [v for v in runs.values() if v["method"] == m and v["N"] == n
                and (m in ("A", "D-whole", "D-stitch") or (v["lam"] == lam and v["window"] == win))]

    def mean_sec(rs):
        return f"{statistics.mean(r['seconds'] for r in rs):.0f}" if rs else "n/a"

    L = [f"# kappa = {kappa:g}\n",
         "全部指标由评估分类器计算。单配置的数值是「均值 ± 批间标准误」（1000 批）；三种子汇总是「各种子均值的平均 ± 种子间标准差」。"
         "`_ink` 列额外要求每格墨水量落在该类真实数字的 0.5%–99.5% 区间（TASK.md 之外的补充指标，"
         "用于排除分类器把空白格判成「1」的情况）。耗时单位为秒/配置（1000 批），除另注明外均在独占 GPU 时测得。\n"]

    # ---- main comparison, 3 seeds
    L.append("## 主对比（N = 4，3 个种子）\n")
    hdr = ["做法", "种子数", "成功率", "批内至少一个成功", "左格命中", "右格命中", "废图率",
           "接缝", "多样性", "成功率_ink", "废图率_ink", "耗时"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    mains = [("A", None, None)] + [(m, best[m]["lam"], best[m]["window"]) for m in ("B", "C") if m in best] \
        + [("D-whole", None, None), ("D-stitch", None, None)]
    names = {"A": "A 独立", "D-whole": "D-整张（事后挑一张）", "D-stitch": "D-拼接（事后挑半格拼）"}
    for m, lam, win in mains:
        rs = rows_for(m, 4, lam, win)
        if not rs:
            continue
        name = names.get(m) or f"{'B 整张复制' if m == 'B' else 'C 拼接'}（lam={lam:g}，{WIN_ZH[win]}）"
        L.append("| " + " | ".join([name, str(len(rs))] + [seeds_summary(rs, k, 4 if k in ('seam', 'waste', 'waste_ink') else 3) for k in (
            "success", "any_success", "hit_left", "hit_right", "waste", "seam", "diversity",
            "success_ink", "waste_ink")] + [mean_sec(rs)]) + " |")
    L.append("\nD 每批只输出一张图，所以它的「成功率」就是「这一批的输出是否成功」，与其他做法的「批内至少一个成功」可直接比较；"
             "D 的耗时就是 A 的耗时（复用同一批样本，裁判打分可忽略）。\n")

    # ---- batch-level comparison
    L.append("## 「一批至少成一张」横向比较（N = 4 除非注明，3 个种子）\n")
    L.append("| 做法 | 一批的产出至少一张成功 | 同上（_ink） | 说明 |")
    L.append("|---|---|---|---|")
    for m, lam, win in mains:
        rs = rows_for(m, 4, lam, win)
        if not rs:
            continue
        key, key_i = ("success", "success_ink") if m.startswith("D") else ("any_success", "any_success_ink")
        name = names.get(m) or f"{m}（lam={lam:g}，{WIN_ZH[win]}）"
        note = "每批输出 1 张" if m.startswith("D") else "每批 4 张中至少 1 张成功"
        L.append(f"| {name} | {seeds_summary(rs, key)} | {seeds_summary(rs, key_i)} | {note} |")
    if "B" in best:
        for n in sorted({v["N"] for v in runs.values() if v["N"] != 4}):
            rs = rows_for("B", n, best["B"]["lam"], best["B"]["window"])
            if rs:
                L.append(f"| B N={n}（lam={best['B']['lam']:g}，{WIN_ZH[best['B']['window']]}） | "
                         f"{seeds_summary(rs, 'any_success')} | {seeds_summary(rs, 'any_success_ink')} | 每批 {n} 张中至少 1 张成功 |")
    L.append("")

    # ---- extra N for B
    extra = sorted({v["N"] for v in runs.values() if v["N"] != 4})
    if extra and "B" in best and "C" in best:
        bl, bw = best["B"]["lam"], best["B"]["window"]
        L.append(f"## 「拼接 4 张相当于整张复制多少张」（B 固定为其 N=4 最佳参数 lam={bl:g}，{WIN_ZH[bw]}；3 个种子）\n")
        L.append("| 做法 | 成功率 | 批内至少一个成功 | 废图率 | 多样性 | 耗时 |")
        L.append("|---|---|---|---|---|---|")
        pts_s, pts_a = [], []
        for n in [4] + extra:
            rs = rows_for("B", n, bl, bw)
            if rs:
                pts_s.append((n, statistics.mean(seed_vals(rs, "success"))))
                pts_a.append((n, statistics.mean(seed_vals(rs, "any_success"))))
                L.append(f"| B N={n} | {seeds_summary(rs, 'success')} | {seeds_summary(rs, 'any_success')} | "
                         f"{seeds_summary(rs, 'waste', 4)} | {seeds_summary(rs, 'diversity')} | {mean_sec(rs)} |")
        cs = rows_for("C", 4, best["C"]["lam"], best["C"]["window"])
        if cs:
            L.append(f"| C N=4（lam={best['C']['lam']:g}，{WIN_ZH[best['C']['window']]}） | {seeds_summary(cs, 'success')} | "
                     f"{seeds_summary(cs, 'any_success')} | {seeds_summary(cs, 'waste', 4)} | "
                     f"{seeds_summary(cs, 'diversity')} | {mean_sec(cs)} |")
            cS, cA = statistics.mean(seed_vals(cs, "success")), statistics.mean(seed_vals(cs, "any_success"))
            L.append(f"\n按 log2(N) 线性插值：按成功率，C(N=4) 相当于 B 的 N {equiv_N(pts_s, cS)}；"
                     f"按批内至少一个成功，相当于 B 的 N {equiv_N(pts_a, cA)}。"
                     "注意 B 在 N=8/16 时沿用 N=4 的最佳参数，没有重新调参；B N=8/16 的每批计算量是 C 的 2/4 倍。\n")

    # ---- windows: early vs full vs late
    L.append("## 重采样窗口对比（种子 0：成功率 / 废图率）\n")
    L.append("| 做法 | lam | 全程 成功率 | 全程 废图率 | 前45% 成功率 | 前45% 废图率 | 后55% 成功率 | 后55% 废图率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in ("B", "C"):
        for lam in (1.0, 3.0, 10.0):
            cells = []
            for win in ("full", "early", "late"):
                r = [v for v in rows_for(m, 4, lam, win) if v["seed_index"] == 0]
                cells += [fm(r[0], "success"), fm(r[0], "waste", 4)] if r else ["n/a", "n/a"]
            L.append(f"| {m} | {lam:g} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- full grid, seed 0
    L.append("## 完整网格（N = 4，种子 0）\n")
    hdr = ["做法", "lam", "窗口", "成功率", "批内≥1成功", "左格命中", "右格命中", "废图率",
           "接缝", "多样性（批数）", "成功对中完全重复", "成功率_ink", "废图率_ink", "左格近空白",
           "平均ESS/N", "耗时"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    order = {"A": 0, "B": 1, "C": 2, "D-whole": 3, "D-stitch": 4}
    grid = sorted((v for v in runs.values() if v["N"] == 4 and v["seed_index"] == 0),
                  key=lambda v: (order[v["method"]], v["lam"], ["full", "early", "late", None].index(v["window"])))
    for v in grid:
        div = v["diversity"]
        L.append("| " + " | ".join([
            v["method"], f"{v['lam']:g}" if v["method"] in ("B", "C") else "-", WIN_ZH[v["window"]],
            fm(v, "success"), fm(v, "any_success"), fm(v, "hit_left"), fm(v, "hit_right"),
            fm(v, "waste", 4), fm(v, "seam", 4),
            (f"{div['mean']:.2f} ({div['n_batches']})" if div.get("mean") is not None else "n/a"),
            f"{v['dup_frac_success_pairs']:.2f}", fm(v, "success_ink"), fm(v, "waste_ink", 4),
            fm(v, "left_near_blank", 4, se=False),
            f"{v['mean_ess_frac']:.2f}" if v.get("mean_ess_frac") is not None else "-",
            f"{v['seconds']:.0f}"]) + " |")
    L.append("")

    # ---- lam = 0 checks
    for p in lam0_files:
        if not os.path.exists(p):
            L.append(f"## lam = 0 检查：{os.path.basename(p)} 不存在（该步未完成）\n")
            continue
        d = json.load(open(p))
        L.append(f"## lam = 0 正确性检查（{os.path.basename(p)}，各 1000 批）\n")
        L.append("| 指标 | A | B (lam=0) | C (lam=0) | z(B−A) | z(C−A) |")
        L.append("|---|---|---|---|---|---|")
        for k in ("success", "hit_left", "hit_right", "waste", "seam", "success_ink", "hit_left_ink",
                  "hit_right_ink", "waste_ink", "any_success", "diversity"):
            cells, zs = [fm(d[m], k, 4) for m in "ABC"], []
            for m in "BC":
                a, b = d["A"][k], d[m][k]
                if None in (a["mean"], b["mean"], a["se"], b["se"]):
                    zs.append("n/a")
                    continue
                den = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
                zs.append(f"{(b['mean'] - a['mean']) / den:.2f}" if den else "n/a")
            L.append(f"| {k} | " + " | ".join(cells + zs) + " |")
        if "_check" in d:
            L.append(f"\n逐样本指标中 |z| ≥ 3 的：{d['_check']['fails'] or '无'}\n")
    if lam0_pooled and os.path.exists(lam0_pooled):
        d = json.load(open(lam0_pooled))
        L.append(f"## lam = 0 检查：两次独立运行合并（{os.path.basename(lam0_pooled)}）\n")
        L.append("| 指标 | A | B | C | z(B−A) | z(C−A) |")
        L.append("|---|---|---|---|---|---|")
        for k, v in d.items():
            if k.startswith("_"):
                continue
            def _f(x, fmt):
                return "n/a" if x is None or x != x else format(x, fmt)
            L.append(f"| {k} | " + " | ".join(f"{_f(v[m]['mean'], '.4f')} ± {_f(v[m]['se'], '.4f')}"
                                              for m in "ABC")
                     + f" | {_f(v['z']['B'], '.2f')} | {_f(v['z']['C'], '.2f')} |")
        L.append(f"\n合并后逐样本指标中 |z| ≥ 3 的：{d['_fails'] or '无'}"
                 "（批次级指标 any_success、diversity 不参与判定，见 results.md 说明）\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--lam0", nargs="*", default=[])
    ap.add_argument("--lam0-pooled", default=None)
    ap.add_argument("--in-dir", default=OUT_DIR)
    args = ap.parse_args()
    text = build(args.kappa, args.in_dir, args.lam0, args.lam0_pooled)
    out = os.path.join(args.in_dir, f"table_kappa{args.kappa:g}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("->", out)


if __name__ == "__main__":
    main()
