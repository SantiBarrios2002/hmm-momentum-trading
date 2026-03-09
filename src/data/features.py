"""Feature engineering utilities for price series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def log_returns(prices: NDArray[np.floating] | pd.Series) -> NDArray[np.floating] | pd.Series:
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


def ewma_volatility(
    returns: NDArray[np.floating] | pd.Series,
    lambda_param: float = 0.94,
) -> NDArray[np.floating] | pd.Series:
    """
    Compute one-step-ahead EWMA conditional variance from returns.

    Implements the recursion:
        sigma2[0] = returns[0]^2
        sigma2[t] = lambda * sigma2[t-1] + (1 - lambda) * returns[t-1]^2
    for t = 1, ..., T-1.

    Parameters:
        returns: 1D array-like or pandas Series of returns.
        lambda_param: Forgetting factor in [0, 1).

    Returns:
        If input is a pandas Series, returns a pandas Series aligned with the
        same index. Otherwise returns a 1D numpy array of length T.

    Raises:
        ValueError: If input is not 1D, is empty, or lambda_param is outside
            [0, 1).
    """
    values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("returns must be a 1D array-like input")
    if values.size == 0:
        raise ValueError("returns must contain at least 1 observation")
    if not (0.0 <= lambda_param < 1.0):
        raise ValueError("lambda_param must satisfy 0 <= lambda_param < 1")

    sigma2 = np.empty_like(values, dtype=float)
    sigma2[0] = values[0] ** 2
    for t in range(1, values.size):
        sigma2[t] = (
            lambda_param * sigma2[t - 1]
            + (1.0 - lambda_param) * (values[t - 1] ** 2)
        )

    if isinstance(returns, pd.Series):
        return pd.Series(sigma2, index=returns.index, name=returns.name)
    return sigma2


def normalize_returns(
    returns: NDArray[np.floating] | pd.Series,
    window: int = 20,
) -> NDArray[np.floating] | pd.Series:
    """
    Standardize returns with rolling mean and rolling standard deviation.

    Implements:
        z_t = (r_t - mean(r_{t-window+1:t})) / std(r_{t-window+1:t})
    using a trailing window of fixed length.

    Parameters:
        returns: 1D array-like or pandas Series of returns.
        window: Positive integer rolling window size.

    Returns:
        If input is a pandas Series, returns a pandas Series with the same
        index. Otherwise returns a 1D numpy array. Values before the first full
        window are NaN; zero rolling standard deviation also yields NaN.

    Raises:
        ValueError: If input is not 1D, is empty, or if window < 1.
    """
    values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("returns must be a 1D array-like input")
    if values.size == 0:
        raise ValueError("returns must contain at least 1 observation")
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer")

    series = pd.Series(values)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    normalized = (series - rolling_mean) / rolling_std
    normalized = normalized.mask(rolling_std == 0.0)

    if isinstance(returns, pd.Series):
        return pd.Series(normalized.to_numpy(), index=returns.index, name=returns.name)
    return normalized.to_numpy()
