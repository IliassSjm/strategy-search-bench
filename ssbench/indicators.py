"""Causal indicator library.

Every function maps a close-price Series to a Series aligned on the same
index, using ONLY data at or before each bar (rolling windows are
trailing). Causality is enforced by tests/test_core.py::test_no_lookahead.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(int(n), min_periods=int(n)).mean()


def roc(close: pd.Series, n: int) -> pd.Series:
    """n-bar rate of change (fractional)."""
    return close.pct_change(int(n))


def zscore(close: pd.Series, n: int) -> pd.Series:
    n = int(n)
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std()
    return (close - m) / s.replace(0.0, np.nan)


def rsi(close: pd.Series, n: int) -> pd.Series:
    n = int(n)
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(n, min_periods=n).mean()
    down = (-delta.clip(upper=0.0)).rolling(n, min_periods=n).mean()
    rs = up / down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When down is 0 over the window but up is not, RSI is 100 by convention.
    out = out.where(~(down.eq(0.0) & up.gt(0.0)), 100.0)
    return out


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """Annualized trailing volatility of daily returns."""
    n = int(n)
    return close.pct_change().rolling(n, min_periods=n).std() * np.sqrt(TRADING_DAYS)


def rolling_high(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(int(n), min_periods=int(n)).max()


def rolling_low(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(int(n), min_periods=int(n)).min()
