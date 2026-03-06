"""Performance metric utilities for trading strategies."""

import numpy as np


def sharpe_ratio(daily_returns):
    """
    Compute annualized Sharpe ratio from daily returns.

    Implements:
        Sharpe = sqrt(252) * mean(r_t) / std(r_t)

    Parameters:
        daily_returns: 1D array-like of daily strategy returns.

    Returns:
        Annualized Sharpe ratio as float. Returns 0.0 if the standard
        deviation is zero.

    Raises:
        ValueError: If input is not 1D or is empty.
    """
    values = np.asarray(daily_returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("daily_returns must be a 1D array-like input")
    if values.size == 0:
        raise ValueError("daily_returns must contain at least 1 observation")

    std = values.std(ddof=0)
    if std == 0.0:
        return 0.0
    return float(np.sqrt(252.0) * values.mean() / std)


def max_drawdown(cumulative_returns):
    """
    Compute maximum drawdown from a cumulative return or equity curve.

    Implements:
        drawdown_t = cumulative_t / max_{s<=t}(cumulative_s) - 1
        max_drawdown = -min_t(drawdown_t)

    Parameters:
        cumulative_returns: 1D array-like cumulative curve values.

    Returns:
        Maximum drawdown as a non-negative float.

    Raises:
        ValueError: If input is not 1D or is empty.
    """
    values = np.asarray(cumulative_returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("cumulative_returns must be a 1D array-like input")
    if values.size == 0:
        raise ValueError("cumulative_returns must contain at least 1 observation")

    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    return float(-np.min(drawdowns))


def annualized_return(daily_returns):
    """
    Compute geometric annualized return from daily returns.

    Implements:
        annualized = (prod_t (1 + r_t))^(252 / T) - 1

    Parameters:
        daily_returns: 1D array-like of daily returns.

    Returns:
        Annualized return as float.

    Raises:
        ValueError: If input is not 1D or is empty.
    """
    values = np.asarray(daily_returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("daily_returns must be a 1D array-like input")
    if values.size == 0:
        raise ValueError("daily_returns must contain at least 1 observation")

    growth = np.prod(1.0 + values)
    return float(growth ** (252.0 / values.size) - 1.0)
