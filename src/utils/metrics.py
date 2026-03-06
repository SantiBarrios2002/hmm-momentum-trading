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
