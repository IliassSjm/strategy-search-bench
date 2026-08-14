"""Vectorized daily backtest: position signal -> net returns -> Sharpe.

The position formed at bar t earns the return from t to t+1 (signal is
shifted by one bar before being applied), so there is no same-bar look-ahead.
Costs are charged per unit of turnover.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def strategy_returns(close: pd.Series, signal: pd.Series,
                     cost_bps: float = 5.0) -> pd.Series:
    """Net daily strategy returns for a [-1, 1] position signal."""
    rets = close.pct_change().fillna(0.0)
    pos = signal.clip(-1.0, 1.0).shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    return pos * rets - (cost_bps / 1e4) * turnover


def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe. 0.0 for degenerate (flat / too short) series."""
    r = np.asarray(returns, dtype=float)
    if r.size < 20:
        return 0.0
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))
