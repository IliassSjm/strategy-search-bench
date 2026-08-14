"""Figure 4 — Evolution optimizes the illusion (F12).

Per-seed paired differences (GP minus random) of the champion's BELIEVED
score (CV fitness), one panel per world, sorted lollipops around zero.
The believed score separates (Wilcoxon p = .004 gbm / .044 ar1); the paid
score never does (p = .981 / .156) — the figure shows the first, the
footnote carries the second.

Usage:  python make_fig4.py runs/ [--no-title]
Writes: figs/fig4_believed_inflation.png (300 dpi) and .pdf (vector).

Colors are the bench's validated dataviz palette (categorical slots 1-2 on
the light surface; adjacent-pair CVD dE 9.1, documented pass).
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

BLUE = "#2a78d6"      # series: the paired difference
ORANGE = "#eb6834"    # accent: the median line
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

WORLDS = [
    ("gbm", "Null world (GBM): fitness is pure noise"),
    ("ar1", "Planted-edge world (AR(1), φ = 0.10)"),
]


def load(runs_dir):
    by = {}
    for f in glob.glob(os.path.join(runs_dir, "*_summary.json")):
        s = json.load(open(f))
        by[(s["world"], s["operator"], s["seed"])] = s
    return by


def diffs(by, world):
    seeds = sorted(s for (w, op, s) in by if w == world and op == "gp")
    return np.array([
        by[(world, "gp", s)]["champion"]["cv_fitness"]
        - by[(world, "random", s)]["champion"]["cv_fitness"]
        for s in seeds
    ])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    with_title = "--no-title" not in sys.argv
    os.makedirs("figs", exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.edgecolor": BASELINE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    by = load(runs_dir)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.9), dpi=300, sharey=True,
        facecolor=SURFACE, constrained_layout=True)

    for ax, (world, label) in zip(axes, WORLDS):
        d = np.sort(diffs(by, world))
        n = len(d)
        med = float(np.median(d))
        p_bel = stats.wilcoxon(d)[1]
        # paid differences on the same seeds, for the footnote
        seeds = sorted(s for (w, op, s) in by if w == world and op == "gp")
        paid = np.array([
            by[(world, "gp", s)]["champion"]["holdout_sharpe"]
            - by[(world, "random", s)]["champion"]["holdout_sharpe"]
            for s in seeds
        ])
        p_paid = stats.wilcoxon(paid)[1]
        wins = int((d > 0).sum())

        x = np.arange(n)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.vlines(x, 0, d, color=BLUE, linewidth=1.1, alpha=0.55, zorder=2)
        ax.scatter(x, d, s=16, color=BLUE, zorder=3)
        ax.axhline(med, color=ORANGE, linewidth=1.2, zorder=2)
        ax.annotate(f"median {med:+.3f}",
                    xy=(0.02, 0.93), xycoords="axes fraction",
                    color=ORANGE, fontsize=8, fontweight="bold")
        ax.annotate(f"Wilcoxon p = {p_bel:.3f} · {wins}/{n} seeds > 0",
                    xy=(0.02, 0.84), xycoords="axes fraction",
                    color=INK2, fontsize=7)
        ax.annotate(f"same seeds, PAID difference: p = {p_paid:.3f} (n.s.)",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                    color=MUTED, fontsize=7, fontstyle="italic")
        ax.margins(y=0.10)
        ax.set_title(label, fontsize=8.5, color=INK, pad=6)
        ax.set_xticks([])
        ax.set_xlabel(f"{n} paired seeds, sorted by difference",
                      fontsize=7, color=MUTED)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="y", length=0)

    axes[0].set_ylabel("believed Sharpe, GP − random\n(paired, same world)",
                       fontsize=7.5)
    if with_title:
        fig.suptitle("GP raises what the search believes, "
                     "never what the data pays", fontsize=9.5, color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(f"figs/fig4_believed_inflation.{ext}",
                    facecolor=SURFACE, bbox_inches="tight")
    print("wrote figs/fig4_believed_inflation.{png,pdf}")


if __name__ == "__main__":
    main()
