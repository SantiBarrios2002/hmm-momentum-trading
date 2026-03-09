"""Backtesting utilities for HMM-derived trading strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.metrics import annualized_return, max_drawdown, sharpe_ratio


def backtest(
    returns: NDArray[np.floating],
    signals: NDArray[np.floating],
    transaction_cost_bps: float = 5,
) -> dict[str, Any]:
    """
    Backtest a signal-driven strategy with one-period execution lag (Paper §7).

    Strategy return:
        r^strat_t = s_{t-1} * r_t
    with convention s_{-1} = 0.

    Transaction costs are charged on executed position changes:
        tc_t = |s^lag_t - s^lag_{t-1}| * c / 10000
    where s^lag_t = s_{t-1} and c is transaction_cost_bps.

    Net return:
        r^net_t = r^strat_t - tc_t

    Parameters:
        returns: np.ndarray, shape (T,)
            Asset returns.
        signals: np.ndarray, shape (T,)
            Trading signals in [-1, 1], where signals[t] is computed after
            observing return[t].
        transaction_cost_bps: float, default 5
            Transaction cost in basis points per unit position change.

    Returns:
        dict with keys:
            "net_returns": np.ndarray, shape (T,)
            "cumulative": np.ndarray, shape (T,)
            "metrics": dict with keys "sharpe", "annualized_return",
                "max_drawdown", "turnover"

    Raises:
        ValueError: If inputs are empty/non-1D/length-mismatched, or cost is
            negative or non-finite.
    """
    returns = np.asarray(returns, dtype=float)
    signals = np.asarray(signals, dtype=float)
    transaction_cost_bps = float(transaction_cost_bps)

    if returns.ndim != 1 or signals.ndim != 1:
        raise ValueError("returns and signals must be 1D arrays")
    if returns.size == 0 or signals.size == 0:
        raise ValueError("returns and signals must be non-empty")
    if returns.size != signals.size:
        raise ValueError("returns and signals must have the same length")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be a non-negative finite value")

    T = returns.size

    lagged_signals = np.empty(T, dtype=float)
    lagged_signals[0] = 0.0
    lagged_signals[1:] = signals[:-1]

    strategy_returns = lagged_signals * returns

    prev_lagged = np.empty(T, dtype=float)
    prev_lagged[0] = 0.0
    prev_lagged[1:] = lagged_signals[:-1]
    tc = np.abs(lagged_signals - prev_lagged) * (transaction_cost_bps / 10_000.0)

    net_returns = strategy_returns - tc
    cumulative = np.cumprod(1.0 + net_returns)

    if T > 1:
        turnover = float(np.mean(np.abs(np.diff(signals))))
    else:
        turnover = 0.0

    metrics = {
        "sharpe": float(sharpe_ratio(net_returns)),
        "annualized_return": float(annualized_return(net_returns)),
        "max_drawdown": float(max_drawdown(cumulative)),
        "turnover": turnover,
    }

    return {
        "net_returns": net_returns,
        "cumulative": cumulative,
        "metrics": metrics,
    }
