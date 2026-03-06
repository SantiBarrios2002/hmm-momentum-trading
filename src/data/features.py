"""Feature engineering utilities for price series."""

import numpy as np
import pandas as pd


def log_returns(prices):
    """
    Compute log-returns from a price series.

    Implements:
        r_t = log(p_t / p_{t-1}), for t = 2, ..., T

    Parameters:
        prices: 1D array-like or pandas Series of strictly positive prices.

    Returns:
        If input is a pandas Series, returns a pandas Series indexed from the
        second observation onward. Otherwise returns a 1D numpy array of length
        T-1.

    Raises:
        ValueError: If input has fewer than 2 observations or contains
            non-positive prices.
    """
    price_values = np.asarray(prices, dtype=float)

    if price_values.ndim != 1:
        raise ValueError("prices must be a 1D array-like input")
    if price_values.size < 2:
        raise ValueError("prices must contain at least 2 observations")
    if np.any(price_values <= 0):
        raise ValueError("prices must be strictly positive")

    result = np.log(price_values[1:] / price_values[:-1])

    if isinstance(prices, pd.Series):
        return pd.Series(result, index=prices.index[1:], name=prices.name)
    return result
