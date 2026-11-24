"""Draw the two figures used in paper.md (English labels) from the JSON result files.

    python make_figures_en.py

Reads  timing.json, level3.json   (repository root)
Writes figures/timing_en.png, figures/style_en.png
Needs only matplotlib; no GPU and no model checkpoints.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "text.color": INK,
})


def xy(pairs):
    return [p[0] for p in pairs], [p[1] for p in pairs]


def line(ax, pairs, color, marker, label, ls="-"):
    x, y = xy(pairs)
    ax.plot(x, y, ls, color=color, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, label=label,
            clip_on=False, zorder=3)


def style_axes(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlim(-2, 100); ax.set_ylim(0, 1.03)
    ax.set_xticks(range(0, 101, 20)); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)


def end_label(ax, pairs, dy=0.0):
    x, y = pairs[-1]
    ax.annotate(f"{100 * y:.0f}%", (x, y), xytext=(6, dy), textcoords="offset points",
                fontsize=9, color=INK2, va="center")


# ---------------------------------------------------------------- figure 1: timing
t = json.load(open(os.path.join(ROOT, "timing.json")))["curves"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
for ax, key, title in zip(axes, ["1", "0.1"],
                          ["Setting 1: independent cells (kappa = 1)",
                           "Setting 2: content-coupled cells (kappa = 0.1)"]):
    c = t[key]
    line(ax, c["surv"], BLUE, "o", "Retention of spliced content (splice at step k, no judge)")
    line(ax, c["judge"], ORANGE, "s", "Intermediate reward: judge on the step-k estimate agrees with final digit")
    line(ax, c["frm"], AQUA, "^", "Success of C when resampling only from step k onward")
    line(ax, c["unt"], YELLOW, "v", "Success of C when resampling only up to step k", ls="--")
    end_label(ax, c["frm"]); end_label(ax, c["unt"])
    style_axes(ax, title, "Sampling step k  (0 = pure noise, 100 = finished image)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("When does resampling matter?  (C at selection strength 3, N = 4; curves 1-2 from 8000 reference trajectories)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0.11, 1, 0.95))
fig.savefig(os.path.join(OUT, "timing_en.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: style
c = json.load(open(os.path.join(ROOT, "level3.json")))["curves"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
ax = axes[0]
line(ax, c["brightness_unified"], BLUE, "o", "Brightness made uniform\n(pairs whose donors differ in brightness)")
line(ax, c["thickness_unified"], ORANGE, "s", "Stroke thickness made uniform\n(pairs whose donors differ in thickness)")
line(ax, c["digits_both_kept_all_pairs"], AQUA, "^", "Both digits kept (all pairs)")
style_axes(ax, "After splicing two samples at step k (no judge): final image", "Splice step k")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False, fontsize=9)
ax = axes[1]
line(ax, c["read_brightness"], BLUE, "o", "Brightness", ls="--")
line(ax, c["read_thickness"], ORANGE, "s", "Stroke thickness", ls="--")
line(ax, c["read_digit"], AQUA, "^", "Digit (judge classifier)", ls="--")
style_axes(ax, "Step-k estimate already agrees with the final image", "Sampling step k")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), frameon=False, fontsize=9)
fig.suptitle("Setting 3 (shared style): when content and style become fixed, and when they become readable",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0.0, 1, 0.95))
fig.savefig(os.path.join(OUT, "style_en.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", os.listdir(OUT))
