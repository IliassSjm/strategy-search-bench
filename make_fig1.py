"""Figure 1 — the benchmark pipeline, one glance.

Interchangeable proposal operators (the only moving part) feed a fixed
loop: propose STRUCTURE -> DE fits VALUES -> purged-CV score -> observe;
the loop ends after 60 SCORED candidates; then champion refit and ONE
read of the walled-off holdout. Worlds have known ground truth.

Usage:  python make_fig1.py
Writes: figs/fig1_pipeline.png (300 dpi) and .pdf (vector).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#2a78d6"
BLUE_100 = "#cde2fb"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
BORDER = "#c3c2b7"

W, H = 126.0, 30.0


def box(ax, x0, x1, y0, y1, title, body, edge=BORDER, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=lw, edgecolor=edge, facecolor=SURFACE, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 2.2, title, ha="center", va="center", zorder=3,
            fontsize=8, fontweight="bold", color=INK)
    ax.text(cx, (y0 + y1) / 2 - 1.8, body, ha="center", va="center", zorder=3,
            fontsize=5.9, color=INK2, linespacing=1.55)


def arrow(ax, xy_from, xy_to, color=MUTED, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, zorder=4, shrinkA=0, shrinkB=0))


def main():
    os.makedirs("figs", exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif"})

    fig, ax = plt.subplots(figsize=(7.2, 2.35), dpi=300, facecolor=SURFACE)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    # --- operator chips (the only moving part) ---------------------------
    for name, y in (("random", 23.4), ("tree GP", 19.4), ("LLM (7B)", 15.4)):
        ax.add_patch(FancyBboxPatch(
            (2.5, y), 11.0, 3.2, boxstyle="round,pad=0.4,rounding_size=1.0",
            linewidth=1.0, edgecolor=BLUE, facecolor=BLUE_100, zorder=2))
        ax.text(8.0, y + 1.6, name, ha="center", va="center",
                fontsize=6.6, color=INK, zorder=3)
    ax.text(8.0, 12.8, "proposal operator:\nthe ONLY moving part",
            ha="center", va="top", fontsize=6.2, color=BLUE,
            fontweight="bold", linespacing=1.4)
    arrow(ax, (13.9, 19.0), (17.4, 19.0), color=BLUE)

    # --- the search loop --------------------------------------------------
    box(ax, 18.0, 38.0, 13.0, 25.0, "PROPOSE",
        "one JSON tree\nstructure, never values\ngrammar of 5.6M trees")
    box(ax, 41.5, 61.5, 13.0, 25.0, "FIT",
        "differential evolution\nsets every parameter\nsame optimizer for all")
    box(ax, 65.0, 85.0, 13.0, 25.0, "SCORE",
        "median OOS Sharpe\n24 quarter splits\n1-bar purge at entries")
    arrow(ax, (38.4, 19.0), (41.1, 19.0))
    arrow(ax, (61.9, 19.0), (64.6, 19.0))

    # return path: score -> observe -> propose
    ax.plot([75.0, 75.0, 28.0], [12.4, 8.0, 8.0],
            color=MUTED, linewidth=1.2, zorder=1)
    arrow(ax, (28.0, 8.0), (28.0, 12.4))
    ax.text(51.5, 7.3, "observe: scored history feeds the next proposal",
            ha="center", va="top", fontsize=6.0, color=INK2)
    ax.text(51.5, 4.6,
            "loop ends after 60 SCORED candidates · "
            "duplicates & invalid replies are logged overhead, not budget",
            ha="center", va="top", fontsize=6.0, color=INK2,
            fontstyle="italic")

    # --- after the loop ---------------------------------------------------
    arrow(ax, (85.4, 19.0), (88.1, 19.0))
    ax.text(86.75, 20.2, "budget\nspent", ha="center", va="bottom",
            fontsize=5.2, color=MUTED, linespacing=1.2)
    box(ax, 88.5, 103.5, 13.0, 25.0, "REFIT",
        "champion only\nDE budget ×3")

    # the wall, then the holdout
    ax.plot([105.8, 105.8], [11.0, 27.0], color=ORANGE, linewidth=1.4,
            linestyle=(0, (4, 3)), zorder=2)
    ax.text(105.8, 27.8, "wall", ha="center", va="bottom", fontsize=6.2,
            color=ORANGE, fontweight="bold")
    arrow(ax, (103.9, 19.0), (107.6, 19.0), color=ORANGE)
    box(ax, 108.0, 124.0, 13.0, 25.0, "HOLDOUT",
        "final 350 bars\nunseen by search\nread ONCE per run",
        edge=ORANGE, lw=1.3)

    # --- ground-truth strip ----------------------------------------------
    ax.plot([2.5, 124.0], [2.9, 2.9], color=BORDER, linewidth=0.7, zorder=1)
    ax.text(63.0, 1.2,
            "Worlds with known truth: GBM (no timing edge exists) · "
            "AR(1) φ = 0.10 (planted edge, net ceiling +1.10)  |  "
            "paired seeds: every operator faces identical worlds, splits, "
            "DE randomness, costs",
            ha="center", va="center", fontsize=5.9, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(f"figs/fig1_pipeline.{ext}", facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.06)
    print("wrote figs/fig1_pipeline.{png,pdf}")


if __name__ == "__main__":
    main()
