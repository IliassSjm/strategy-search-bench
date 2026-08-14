"""Fitness = median out-of-sample Sharpe over random quarter splits.

For each split: parameters are fit (differential evolution) on the train
quarters and the fitted strategy is scored on the withheld test quarters.
The candidate's fitness is the median test-quarter Sharpe across splits.

Signals are always computed on the FULL price history (indicators need
contiguous data); train/test masks select which bars' returns enter each
objective. Splits and DE seeds are deterministic per (run_seed, candidate).
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from . import backtest, grammar


def quarter_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer quarter label per bar."""
    q = index.year * 4 + (index.month - 1) // 3
    return q.to_numpy() - q.min()


def make_splits(index: pd.DatetimeIndex, n_splits: int, train_frac: float,
                seed: int):
    """Random assignments of whole quarters to train/test.

    Returns list of (train_mask, test_mask) boolean arrays over bars.
    """
    qid = quarter_ids(index)
    quarters = np.unique(qid)
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(quarters)
        n_train = max(1, int(round(train_frac * len(quarters))))
        train_q = set(perm[:n_train].tolist())
        train_mask = np.isin(qid, list(train_q))
        splits.append((train_mask, ~train_mask))
    return splits


def _objective_factory(tree, close, mask, cost_bps):
    def neg_sharpe(x):
        sig = grammar.build_signal(tree, close, x)
        rets = backtest.strategy_returns(close, sig, cost_bps)
        return -backtest.sharpe(rets[mask])
    return neg_sharpe


def fit_params(tree, close, mask, cost_bps, de_iter, de_pop, seed):
    """Fit the tree's parameters on the masked bars. Returns (values, train_sharpe)."""
    names, _, bounds = grammar.param_bounds(tree)
    if not names:                      # parameterless tree (e.g. const_long)
        sig = grammar.build_signal(tree, close, [])
        rets = backtest.strategy_returns(close, sig, cost_bps)
        return [], backtest.sharpe(rets[mask])
    res = differential_evolution(
        _objective_factory(tree, close, mask, cost_bps),
        bounds=bounds, seed=seed, maxiter=de_iter, popsize=de_pop,
        tol=0.01, polish=False, init="sobol", updating="deferred")
    return list(res.x), -float(res.fun)


def _purge_entry_bars(test_mask: np.ndarray) -> np.ndarray:
    """Drop the first bar of each contiguous test block: that bar's position
    was formed on a train bar (a 1-bar purge, after Lopez de Prado). Rolling
    indicator windows still span the boundary — inherent to this CV design
    and disclosed, not fixable by purging."""
    prev = np.concatenate([[False], test_mask[:-1]])
    return test_mask & prev


def _one_split(args):
    """Fit on one split's train quarters, score its test quarters (picklable)."""
    tree, close, train_mask, test_mask, cost_bps, de_iter, de_pop, seed = args
    values, _ = fit_params(tree, close, train_mask, cost_bps,
                           de_iter, de_pop, seed=seed)
    sig = grammar.build_signal(tree, close, values)
    rets = backtest.strategy_returns(close, sig, cost_bps)
    return backtest.sharpe(rets[_purge_entry_bars(test_mask)])


def cv_fitness(tree, close, splits, cost_bps=5.0, de_iter=6, de_pop=8,
               seed=0, workers=1):
    """Median out-of-sample Sharpe of `tree` across quarter splits.

    workers > 1 parallelizes across splits with a process pool (each split's
    fit is independent); results are identical to the sequential path.
    """
    jobs = [(tree, close, tr, te, cost_bps, de_iter, de_pop, seed * 1000 + k)
            for k, (tr, te) in enumerate(splits)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            oos = list(ex.map(_one_split, jobs))
    else:
        oos = [_one_split(j) for j in jobs]
    return float(np.median(oos)), oos
