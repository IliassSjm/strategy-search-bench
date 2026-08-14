"""Price worlds with known ground truth.

- gbm:        zero-drift geometric Brownian motion. NO timing edge exists.
- bootstrap:  stationary block bootstrap (Politis-Romano) of a real,
              DEMEANED daily-return series. Preserves vol clustering and fat
              tails; destroys drift and cross-block predictability. NOT a
              strict null: autocorrelation shorter than the average block
              survives resampling, so treat it as "realistic, approximately
              edgeless" and use gbm for hard no-edge claims.
- ar1:        planted edge — AR(1) autocorrelation in returns with known
              phi. The oracle (sign of yesterday's return) gives the
              recoverable Sharpe ceiling, measured empirically.

All worlds return a close-price Series on a business-day index so quarter
splits work identically to real data.
"""

import numpy as np
import pandas as pd

DEFAULT_DAYS = 2500          # ~10 years
ANN_VOL = 0.30               # single-stock-ish annualized vol
START = "2015-01-02"


def _index(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=n_days)


def _to_close(daily_rets: np.ndarray, n_days: int) -> pd.Series:
    close = 100.0 * np.cumprod(1.0 + daily_rets)
    return pd.Series(close, index=_index(n_days), name="close")


def gbm(n_days: int = DEFAULT_DAYS, ann_vol: float = ANN_VOL,
        seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_vol = ann_vol / np.sqrt(252)
    rets = rng.normal(0.0, daily_vol, n_days)
    return _to_close(rets, n_days)


def bootstrap(source_rets: np.ndarray, n_days: int = DEFAULT_DAYS,
              avg_block: int = 20, seed: int = 0) -> pd.Series:
    """Stationary block bootstrap of a demeaned real return series."""
    src = np.asarray(source_rets, dtype=float)
    src = src[np.isfinite(src)]
    src = src - src.mean()
    m = len(src)
    if m < 100:
        raise ValueError(f"source series too short ({m} returns)")
    rng = np.random.default_rng(seed)
    out = np.empty(n_days)
    i = int(rng.integers(m))
    p = 1.0 / avg_block           # geometric block lengths, mean avg_block
    for t in range(n_days):
        out[t] = src[i]
        if rng.random() < p:
            i = int(rng.integers(m))     # start a new block
        else:
            i = (i + 1) % m              # continue block (wrap around)
    return _to_close(out, n_days)


def ar1(n_days: int = DEFAULT_DAYS, phi: float = 0.10,
        ann_vol: float = ANN_VOL, seed: int = 0) -> pd.Series:
    """Returns with AR(1) structure: r_t = phi * r_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    daily_vol = ann_vol / np.sqrt(252)
    eps = rng.normal(0.0, daily_vol * np.sqrt(1.0 - phi ** 2), n_days)
    rets = np.empty(n_days)
    prev = 0.0
    for t in range(n_days):
        rets[t] = phi * prev + eps[t]
        prev = rets[t]
    return _to_close(rets, n_days)


def oracle_ar1_sharpe(close: pd.Series, cost_bps: float = 0.0) -> float:
    """Empirical Sharpe of the AR(1) oracle: hold sign(yesterday's return).

    The oracle flips sign roughly every other day, so costs bite hard: pass
    the bench's cost_bps to get the NET ceiling — that, not the gross
    number, is what operator recoveries should be compared against.
    """
    from . import backtest
    rets = close.pct_change().fillna(0.0)
    pos = np.sign(rets)              # formed at t; strategy_returns shifts it
    return backtest.sharpe(backtest.strategy_returns(close, pos, cost_bps))


def load_returns_csv(path: str) -> np.ndarray:
    """Load a daily close CSV (columns: date, close) -> return array."""
    df = pd.read_csv(path, parse_dates=[0])
    close = df.iloc[:, 1].astype(float)
    return close.pct_change().dropna().to_numpy()


def make_world(name: str, seed: int, n_days: int = DEFAULT_DAYS,
               source_csv: str = None, phi: float = 0.10) -> pd.Series:
    if name == "gbm":
        return gbm(n_days=n_days, seed=seed)
    if name == "bootstrap":
        if not source_csv:
            raise ValueError("bootstrap world needs --source-csv "
                             "(run scripts/fetch_data.py first)")
        return bootstrap(load_returns_csv(source_csv), n_days=n_days, seed=seed)
    if name == "ar1":
        return ar1(n_days=n_days, phi=phi, seed=seed)
    raise ValueError(f"unknown world {name!r}")
