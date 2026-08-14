#!/usr/bin/env python3
"""Aggregate run summaries into the fool's-gold figure + a text table.

    python make_report.py runs/

For each world, plots every run's champion twice on one Sharpe axis:
what the search believed (CV fitness) vs what the walled-off holdout paid.
On no-edge worlds the gap between those clouds IS the manufactured edge.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (dataviz reference, light mode, slots 1-2).
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
OPERATOR_ORDER = ["random", "gp", "llm", "mockllm"]


def load(run_dir):
    by_world = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(run_dir, "*_summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        if s.get("champion"):
            by_world[s["world"]][s["operator"]].append(s)
    return by_world


def table(by_world):
    lines = []
    for world, ops in by_world.items():
        lines.append(f"\nworld={world}")
        lines.append(f"{'operator':<10}{'runs':>5}{'med CV':>9}{'med holdout':>13}"
                     f"{'rejects':>9}{'dups':>7}{'med sec':>9}")
        for op in sorted(ops, key=lambda o: OPERATOR_ORDER.index(o)
                         if o in OPERATOR_ORDER else 99):
            runs = ops[op]
            cv = np.median([r["champion"]["cv_fitness"] for r in runs])
            ho = np.median([r["champion"]["holdout_sharpe"] for r in runs])
            rej = sum(r["n_rejected"] for r in runs)
            dup = sum(r["n_duplicate"] for r in runs)
            sec = np.median([r["elapsed_sec"] for r in runs])
            lines.append(f"{op:<10}{len(runs):>5}{cv:>+9.3f}{ho:>+13.3f}"
                         f"{rej:>9}{dup:>7}{sec:>9.0f}")
    return "\n".join(lines)


def figure(world, ops, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)

    names = [o for o in OPERATOR_ORDER if o in ops] + \
            [o for o in ops if o not in OPERATOR_ORDER]
    for x, op in enumerate(names):
        cv = [r["champion"]["cv_fitness"] for r in ops[op]]
        ho = [r["champion"]["holdout_sharpe"] for r in ops[op]]
        for vals, color, dx in ((cv, BLUE, -0.16), (ho, ORANGE, 0.16)):
            jitter = rng.uniform(-0.05, 0.05, len(vals))
            ax.scatter(x + dx + jitter, vals, s=34, color=color, zorder=3,
                       edgecolors=SURFACE, linewidths=1.0, alpha=0.9)
            med = float(np.median(vals))
            ax.hlines(med, x + dx - 0.12, x + dx + 0.12, color=color,
                      linewidth=2, zorder=4)
            ax.annotate(f"{med:+.2f}", (x + dx, med), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=INK)

    ax.axhline(0.0, color=INK2, linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("champion Sharpe (annualized)", color=INK)
    ax.set_title(f"What the search believed vs. what the holdout paid — "
                 f"world: {world}", fontsize=11, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=INK2, alpha=0.15, linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE,
                          label="CV fitness (what the search saw)"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE,
                          label="holdout Sharpe (read once)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best",
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    by_world = load(run_dir)
    if not by_world:
        sys.exit(f"no *_summary.json found in {run_dir}/")
    print(table(by_world))
    for world, ops in by_world.items():
        out = os.path.join(run_dir, f"fools_gold_{world}.png")
        print(f"\nwrote {figure(world, ops, out)}")


if __name__ == "__main__":
    main()
