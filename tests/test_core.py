"""Core correctness tests. The look-ahead test is the credibility anchor:
if it fails, every result the bench produces is worthless."""

import json

import numpy as np
import pandas as pd
import pytest

from ssbench import backtest, evaluate, grammar, operators, worlds


RNG = np.random.default_rng(7)


def _random_trees(n, seed=11):
    rng = np.random.default_rng(seed)
    return [grammar.random_tree(rng) for _ in range(n)]


def _mid_values(tree):
    _, _, bounds = grammar.param_bounds(tree)
    return [(lo + hi) / 2.0 for lo, hi in bounds]


# ---------------------------------------------------------------- grammar --

def test_random_trees_valid():
    for tree in _random_trees(200):
        grammar.validate(tree)                       # must not raise
        assert grammar.n_params(tree) <= 20


def test_roundtrip_serialization():
    close = worlds.gbm(500, seed=1)
    for tree in _random_trees(30):
        clone = json.loads(grammar.canonical(tree))
        a = grammar.build_signal(tree, close, _mid_values(tree))
        b = grammar.build_signal(clone, close, _mid_values(clone))
        pd.testing.assert_series_equal(a, b)


def test_signal_bounded():
    close = worlds.gbm(600, seed=2)
    for tree in _random_trees(50):
        sig = grammar.build_signal(tree, close, _mid_values(tree))
        assert sig.notna().all()
        assert (sig.abs() <= 1.0 + 1e-9).all()


# ------------------------------------------------------------- look-ahead --

def test_no_lookahead():
    """Signal up to bar k must not change when data after k changes."""
    full = worlds.gbm(800, seed=3)
    k = 500
    tampered = full.copy()
    tampered.iloc[k:] = tampered.iloc[k:] * np.linspace(1.0, 3.0, len(full) - k)
    for tree in _random_trees(60):
        v = _mid_values(tree)
        a = grammar.build_signal(tree, full, v).iloc[:k]
        b = grammar.build_signal(tree, tampered, v).iloc[:k]
        pd.testing.assert_series_equal(a, b, check_exact=True)


def test_backtest_uses_next_bar_return():
    """A position taken at bar t must earn bar t+1's return, not bar t's."""
    close = pd.Series([100.0, 100.0, 110.0, 110.0],
                      index=pd.bdate_range("2020-01-01", periods=4))
    sig = pd.Series([0.0, 1.0, 0.0, 0.0], index=close.index)
    rets = backtest.strategy_returns(close, sig, cost_bps=0.0)
    assert rets.iloc[2] == pytest.approx(0.10)       # long during the +10% bar
    assert rets.iloc[1] == 0.0                        # nothing earned same-bar


# --------------------------------------------------------------- backtest --

def test_costs_charged_on_turnover():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=10))
    sig = pd.Series([0, 1, 1, -1, -1, 0, 0, 0, 0, 0], index=close.index, dtype=float)
    rets = backtest.strategy_returns(close, sig, cost_bps=10.0)
    # flat prices: pnl is pure cost = 10bp * total turnover (1 + 2 + 1)
    assert rets.sum() == pytest.approx(-(10 / 1e4) * 4)


def test_sharpe_degenerate():
    assert backtest.sharpe(pd.Series(np.zeros(100))) == 0.0
    assert backtest.sharpe(pd.Series([0.01] * 5)) == 0.0     # too short


# --------------------------------------------------------------- evaluate --

def test_cv_fitness_smoke():
    close = worlds.gbm(700, seed=4)
    splits = evaluate.make_splits(close.index, n_splits=4, train_frac=0.75, seed=0)
    tree = {"type": "trend", "children": []}
    fit, oos = evaluate.cv_fitness(tree, close, splits, de_iter=2, de_pop=4, seed=0)
    assert np.isfinite(fit) and len(oos) == 4


def test_purge_drops_entry_bars():
    mask = np.array([False, True, True, False, True])
    purged = evaluate._purge_entry_bars(mask)
    assert (purged == np.array([False, False, True, False, False])).all()


def test_splits_are_whole_quarters():
    close = worlds.gbm(1000, seed=5)
    qid = evaluate.quarter_ids(close.index)
    for train, test in evaluate.make_splits(close.index, 8, 0.75, seed=1):
        assert not (train & test).any()
        assert (train | test).all()
        for q in np.unique(qid):                     # no quarter straddles both
            in_q = qid == q
            assert train[in_q].all() or test[in_q].all()


# ---------------------------------------------------------------- worlds ---

def test_worlds_shapes():
    for w in (worlds.gbm(300, seed=0), worlds.ar1(300, seed=0),
              worlds.bootstrap(RNG.normal(0, 0.02, 2000), 300, seed=0)):
        assert len(w) == 300 and w.notna().all() and (w > 0).all()


def test_bootstrap_demeans_source():
    biased = RNG.normal(0.005, 0.02, 3000)           # strong positive drift
    close = worlds.bootstrap(biased, 5000, seed=2)
    ann_ret = close.pct_change().mean() * 252
    assert abs(ann_ret) < 0.25                        # drift edge destroyed

def test_ar1_oracle_recovers_planted_edge():
    sharpes = [worlds.oracle_ar1_sharpe(worlds.ar1(2500, phi=0.10, seed=s))
               for s in range(5)]
    assert np.median(sharpes) > 1.0                  # phi=0.10 is a real edge
    null = [worlds.oracle_ar1_sharpe(worlds.gbm(2500, seed=s)) for s in range(5)]
    assert abs(np.median(null)) < 0.6                # no edge for the oracle on GBM


# -------------------------------------------------------------- operators --

def test_random_and_gp_propose_valid():
    for name in ("random", "gp"):
        op = operators.make_operator(name, seed=0)
        history = []
        for i in range(40):
            tree = op.propose(history)
            grammar.validate(tree)
            rec = {"tree": tree, "fitness": float(np.sin(i))}
            history.append(rec)
            op.observe(rec)


def test_gp_population_evicts_worst():
    op = operators.GPOp(seed=0, pop_size=5, init_size=2)
    for i in range(10):
        tree = op.propose([])
        op.observe({"tree": tree, "fitness": float(i)})
    assert len(op.pop) == 5
    assert min(r["fitness"] for r in op.pop) == 5.0


def test_mock_llm_parse_and_repair():
    op = operators.MockLLMOp(seed=0)
    ok = 0
    for _ in range(30):
        try:
            grammar.validate(op.propose([]))
            ok += 1
        except operators.ProposalError:
            pass
    assert ok >= 25                                   # repair path mostly saves it


def test_llm_extract_tree_variants():
    raw = '{"type": "trend", "children": []}'
    fenced = f"```json\n{raw}\n```"
    chatty = f"Sure! Here you go: {raw} Hope it beats the best."
    for text in (raw, fenced, chatty):
        tree = operators.LLMOp._extract_tree(text)
        assert tree["type"] == "trend"
    with pytest.raises((grammar.InvalidTree, json.JSONDecodeError)):
        operators.LLMOp._extract_tree("no json here")
