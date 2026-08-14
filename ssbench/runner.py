"""Experiment loop: world x operator x budget -> JSONL records + summary.

Budget = number of SCORED candidates (duplicates and rejects are logged as
overhead but don't consume budget — matched budgets across operators is the
fairness contract). The final `holdout_days` bars are walled off from the
search; the champion is measured there exactly once, after the loop ends.
"""

import json
import os
import time

import numpy as np

from . import backtest, evaluate, grammar, operators, worlds


def run(operator_name: str, world_name: str, budget: int, seed: int,
        out_dir: str, n_splits: int = 24, train_frac: float = 0.75,
        de_iter: int = 6, de_pop: int = 8, cost_bps: float = 5.0,
        n_days: int = worlds.DEFAULT_DAYS, holdout_days: int = 350,
        source_csv: str = None, phi: float = 0.10,
        max_overhead: int = 200, workers: int = 1, quiet: bool = False):
    t0 = time.time()
    close_full = worlds.make_world(world_name, seed=seed, n_days=n_days,
                                   source_csv=source_csv, phi=phi)
    close_search = close_full.iloc[:-holdout_days]
    splits = evaluate.make_splits(close_search.index, n_splits, train_frac,
                                  seed=seed)
    op = operators.make_operator(operator_name, seed=seed)

    run_id = f"{operator_name}_{world_name}_s{seed}"
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, f"{run_id}.jsonl")

    history, seen = [], set()
    n_rejected = n_duplicate = 0
    best = None

    with open(records_path, "w") as fh:
        while len(history) < budget:
            if n_rejected + n_duplicate > max_overhead:
                print(f"[{run_id}] stopping early: overhead cap hit "
                      f"({n_rejected} rejects, {n_duplicate} duplicates)")
                break
            try:
                tree = op.propose(history)
                grammar.validate(tree)
            except (operators.ProposalError, grammar.InvalidTree) as e:
                n_rejected += 1
                fh.write(json.dumps({"event": "reject", "error": str(e)}) + "\n")
                continue
            h = grammar.tree_hash(tree)
            if h in seen:
                n_duplicate += 1
                fh.write(json.dumps({"event": "duplicate", "hash": h}) + "\n")
                continue
            seen.add(h)

            i = len(history)
            fitness, oos = evaluate.cv_fitness(
                tree, close_search, splits, cost_bps=cost_bps,
                de_iter=de_iter, de_pop=de_pop, seed=seed * 100003 + i,
                workers=workers)
            record = {"event": "scored", "i": i, "hash": h, "tree": tree,
                      "desc": grammar.describe(tree), "fitness": fitness,
                      "n_params": grammar.n_params(tree),
                      "oos_quartiles": [float(np.percentile(oos, q))
                                        for q in (25, 50, 75)]}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            history.append(record)
            op.observe(record)
            if best is None or fitness > best["fitness"]:
                best = record
            if not quiet:
                print(f"[{run_id}] {i + 1}/{budget} cv={fitness:+.3f} "
                      f"best={best['fitness']:+.3f}  {record['desc']}",
                      flush=True)

    # --- champion: refit on the whole search region, read holdout ONCE ----
    summary = {"run_id": run_id, "operator": operator_name,
               "world": world_name, "seed": seed, "budget": budget,
               "n_scored": len(history), "n_rejected": n_rejected,
               "n_duplicate": n_duplicate, "n_splits": n_splits,
               "de_iter": de_iter, "de_pop": de_pop, "cost_bps": cost_bps,
               "elapsed_sec": None, "champion": None}
    if best is not None:
        all_mask = np.ones(len(close_search), dtype=bool)
        values, train_sharpe = evaluate.fit_params(
            best["tree"], close_search, all_mask, cost_bps,
            de_iter=de_iter * 3, de_pop=de_pop * 2, seed=seed)
        sig_full = grammar.build_signal(best["tree"], close_full, values)
        rets_full = backtest.strategy_returns(close_full, sig_full, cost_bps)
        # skip the first holdout bar: its position was formed on a search bar
        # (same 1-bar purge as the CV evaluator)
        holdout_sharpe = backtest.sharpe(rets_full.iloc[-(holdout_days - 1):])
        bh_holdout = backtest.sharpe(
            close_full.pct_change().fillna(0.0).iloc[-(holdout_days - 1):])
        summary["champion"] = {
            "tree": best["tree"], "desc": best["desc"],
            "cv_fitness": best["fitness"], "refit_train_sharpe": train_sharpe,
            "holdout_sharpe": holdout_sharpe,
            "buyhold_holdout_sharpe": bh_holdout}
    summary["elapsed_sec"] = round(time.time() - t0, 1)

    with open(os.path.join(out_dir, f"{run_id}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if not quiet and summary["champion"]:
        c = summary["champion"]
        print(f"[{run_id}] DONE in {summary['elapsed_sec']}s  "
              f"champion cv={c['cv_fitness']:+.3f} "
              f"holdout={c['holdout_sharpe']:+.3f} "
              f"(buy&hold holdout={c['buyhold_holdout_sharpe']:+.3f})  "
              f"overhead: {n_rejected} rejects, {n_duplicate} dups")
    return summary
