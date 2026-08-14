"""The strategy grammar shared by ALL operators (random, GP, LLM).

A strategy is an expression tree of dict nodes: {"type": <name>, "children": [...]}.
Node types declare their tunable parameters and fixed search ranges; the tree
only chooses STRUCTURE. Parameter VALUES are always fit by the optimizer
(ssbench.evaluate), never by the proposer. This is the fairness core of the
bench: every operator searches exactly the same space.

Signals returned by leaves/combinators are roughly in [-1, 1]; the final
signal is clipped to [-1, 1] and NaNs (indicator warmup) become 0 = flat.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from . import indicators as ind

MAX_DEPTH = 3          # root at depth 0; deepest node allowed at depth 3
MAX_NODES = 9

# ---------------------------------------------------------------------------
# Node specs: name -> (n_children, [(param, kind, lo, hi), ...])
# Ranges are FIXED per type so the search space is identical for all operators.
# ---------------------------------------------------------------------------
NODE_SPECS = {
    # leaves
    "const_long": (0, []),                                         # always long (buy & hold)
    "trend":      (0, [("fast", "int", 5, 60), ("slow", "int", 60, 250)]),
    "momentum":   (0, [("n", "int", 1, 120)]),   # n=1 reaches lag-1 (AR(1)) edges
    "meanrev_z":  (0, [("n", "int", 5, 60)]),
    "rsi_gate":   (0, [("n", "int", 5, 30), ("lvl", "float", 10.0, 40.0)]),
    "breakout":   (0, [("n", "int", 10, 120)]),
    # combinators
    "wsum":       (2, [("w1", "float", -1.0, 1.0), ("w2", "float", -1.0, 1.0)]),
    "vol_gate":   (2, [("n", "int", 10, 120), ("th", "float", -1.5, 1.5)]),
    "switch_z":   (2, [("n", "int", 10, 120)]),
}

LEAF_TYPES = [t for t, (c, _) in NODE_SPECS.items() if c == 0]
COMBINATOR_TYPES = [t for t, (c, _) in NODE_SPECS.items() if c > 0]


class InvalidTree(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation / traversal
# ---------------------------------------------------------------------------

def validate(tree) -> None:
    """Raise InvalidTree unless `tree` is a well-formed tree within size caps."""
    n_nodes = _validate_node(tree, depth=0)
    if n_nodes > MAX_NODES:
        raise InvalidTree(f"{n_nodes} nodes > MAX_NODES={MAX_NODES}")


def _validate_node(node, depth: int) -> int:
    if not isinstance(node, dict):
        raise InvalidTree(f"node is {type(node).__name__}, expected dict")
    ntype = node.get("type")
    if ntype not in NODE_SPECS:
        raise InvalidTree(f"unknown node type {ntype!r}")
    if depth > MAX_DEPTH:
        raise InvalidTree(f"depth > MAX_DEPTH={MAX_DEPTH}")
    n_children, _ = NODE_SPECS[ntype]
    children = node.get("children", [])
    if not isinstance(children, list) or len(children) != n_children:
        raise InvalidTree(f"{ntype} needs {n_children} children, got {children!r}")
    extra = set(node) - {"type", "children"}
    if extra:
        raise InvalidTree(f"unexpected keys {extra} (parameters are fit, not chosen)")
    return 1 + sum(_validate_node(c, depth + 1) for c in children)


def param_bounds(tree):
    """Flatten the tree's parameters into (names, kinds, bounds) for the optimizer.

    Names are position-prefixed ("0.1.fast") so identical node types at
    different positions get independent parameters.
    """
    names, kinds, bounds = [], [], []

    def walk(node, path):
        _, params = NODE_SPECS[node["type"]]
        for pname, kind, lo, hi in params:
            names.append(f"{path}.{pname}" if path else pname)
            kinds.append(kind)
            bounds.append((float(lo), float(hi)))
        for i, child in enumerate(node.get("children", [])):
            walk(child, f"{path}.{i}" if path else str(i))

    walk(tree, "")
    return names, kinds, bounds


def canonical(tree) -> str:
    return json.dumps(tree, sort_keys=True, separators=(",", ":"))


def tree_hash(tree) -> str:
    return hashlib.md5(canonical(tree).encode()).hexdigest()[:12]


def n_params(tree) -> int:
    return len(param_bounds(tree)[0])


# ---------------------------------------------------------------------------
# Evaluation: tree + fitted parameter values -> position signal in [-1, 1]
# ---------------------------------------------------------------------------

def build_signal(tree, close: pd.Series, values) -> pd.Series:
    """Evaluate the tree into a position series. `values` is the flat vector
    in param_bounds() order (floats; int params are rounded here)."""
    names, kinds, _ = param_bounds(tree)
    p = {}
    for name, kind, v in zip(names, kinds, values):
        p[name] = int(round(v)) if kind == "int" else float(v)
    sig = _eval_node(tree, close, p, path="")
    return sig.clip(-1.0, 1.0).fillna(0.0)


def _eval_node(node, close, p, path):
    t = node["type"]
    q = lambda pname: p[f"{path}.{pname}" if path else pname]  # noqa: E731
    child = lambda i: _eval_node(  # noqa: E731
        node["children"][i], close, p, f"{path}.{i}" if path else str(i))

    if t == "const_long":
        return pd.Series(1.0, index=close.index)
    if t == "trend":
        fast, slow = q("fast"), max(q("slow"), q("fast") + 1)
        return np.sign(ind.sma(close, fast) - ind.sma(close, slow))
    if t == "momentum":
        return np.sign(ind.roc(close, q("n")))
    if t == "meanrev_z":
        return (-ind.zscore(close, q("n")) / 2.0).clip(-1.0, 1.0)
    if t == "rsi_gate":
        r, lvl = ind.rsi(close, q("n")), q("lvl")
        return (r < lvl).astype(float) - (r > 100.0 - lvl).astype(float)
    if t == "breakout":
        n = q("n")
        up = (close >= ind.rolling_high(close, n)).astype(float)
        dn = (close <= ind.rolling_low(close, n)).astype(float)
        return up - dn
    if t == "wsum":
        return q("w1") * child(0) + q("w2") * child(1)
    if t == "vol_gate":
        # calm-vs-wild regime switch on z-scored trailing vol
        vz = ind.zscore(ind.realized_vol(close, q("n")), q("n"))
        calm = (vz < q("th")).astype(float)
        return calm * child(0) + (1.0 - calm) * child(1)
    if t == "switch_z":
        z = ind.zscore(close, q("n"))
        a, b = child(0), child(1)
        return pd.Series(np.where(z < 0, a, b), index=close.index).where(z.notna())
    raise InvalidTree(f"unknown node type {t!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Random sampling (the random-search operator and GP's raw material)
# ---------------------------------------------------------------------------

def random_tree(rng: np.random.Generator, depth: int = 0):
    """Sample a random valid tree. Combinator probability decays with depth."""
    p_comb = {0: 0.75, 1: 0.45, 2: 0.2}.get(depth, 0.0)
    if rng.random() < p_comb:
        t = COMBINATOR_TYPES[rng.integers(len(COMBINATOR_TYPES))]
        n_children, _ = NODE_SPECS[t]
        node = {"type": t,
                "children": [random_tree(rng, depth + 1) for _ in range(n_children)]}
    else:
        t = LEAF_TYPES[rng.integers(len(LEAF_TYPES))]
        node = {"type": t, "children": []}
    if depth == 0:
        try:
            validate(node)
        except InvalidTree:      # over MAX_NODES: fall back to a single leaf
            t = LEAF_TYPES[rng.integers(len(LEAF_TYPES))]
            node = {"type": t, "children": []}
    return node


def describe(tree) -> str:
    """Compact human-readable form, e.g. vol_gate(trend, meanrev_z)."""
    ch = tree.get("children", [])
    if not ch:
        return tree["type"]
    return f"{tree['type']}({', '.join(describe(c) for c in ch)})"
