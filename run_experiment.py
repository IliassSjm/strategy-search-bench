#!/usr/bin/env python3
"""Run one experiment cell: operator x world x seeds.

Weekend MVP (fool's gold on provably edgeless markets):
    python run_experiment.py --operator random --world gbm --budget 60 --seeds 0 1 2 3 4
    python run_experiment.py --operator llm    --world gbm --budget 60 --seeds 0 1 2 3 4
    python make_report.py runs/

LLM config: export LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5-coder:7b
            (or unset those and set ANTHROPIC_API_KEY)
"""

import argparse
import os

from ssbench import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", required=True,
                    choices=["random", "gp", "llm", "mockllm"])
    ap.add_argument("--world", required=True,
                    choices=["gbm", "bootstrap", "ar1"])
    ap.add_argument("--budget", type=int, default=60,
                    help="scored candidates per run (default 60)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one run per seed (world + operator both seeded)")
    ap.add_argument("--splits", type=int, default=24,
                    help="quarter CV splits per candidate (default 24)")
    ap.add_argument("--de-iter", type=int, default=6)
    ap.add_argument("--de-pop", type=int, default=8)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--n-days", type=int, default=2500)
    ap.add_argument("--holdout-days", type=int, default=350)
    ap.add_argument("--source-csv", default=None,
                    help="daily close CSV for the bootstrap world")
    ap.add_argument("--phi", type=float, default=0.10,
                    help="AR(1) coefficient for the ar1 world")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes for split-level parallelism "
                         "(default: cores - 2)")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    for seed in args.seeds:
        runner.run(args.operator, args.world, args.budget, seed,
                   out_dir=args.out, n_splits=args.splits,
                   de_iter=args.de_iter, de_pop=args.de_pop,
                   cost_bps=args.cost_bps, n_days=args.n_days,
                   holdout_days=args.holdout_days,
                   source_csv=args.source_csv, phi=args.phi,
                   workers=args.workers)


if __name__ == "__main__":
    main()
