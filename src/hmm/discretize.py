"""Return discretization to tick-size grid (Paper §2.1, Eq. 3).

The paper models returns on a discrete grid defined by the security's tick size α.
Raw continuous returns are rounded to the nearest multiple of α, and the Gaussian
emission is normalized over the grid to form a proper PMF:

    φ_k(Δy) = N(Δy; μ_k, σ²_k) / Σ_{Δy' ∈ Y} N(Δy'; μ_k, σ²_k)

where Y = {..., -2α, -α, 0, α, 2α, ...} is the set of all possible returns on the grid.

The variance floor is tied to tick size (Paper §3.2):
    min_variance = α² / 2

For ES (E-mini S&P 500), α = 0.25 index points.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def discretize_returns(returns: np.ndarray, tick_size: float) -> np.ndarray:
    """Round returns to the nearest tick-size multiple (Paper §2.1).

    Δy_disc = round(Δy / α) · α

    Parameters
    ----------
    returns : (T,) array of raw continuous returns.
    tick_size : α, the minimum price increment.

    Returns
    -------
    disc_returns : (T,) array of discretized returns on the grid {..., -α, 0, α, ...}.
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    returns = np.asarray(returns, dtype=np.float64)
    return np.round(returns / tick_size) * tick_size


def tick_variance_floor(tick_size: float) -> float:
    """Minimum emission variance tied to tick size (Paper §3.2).

    min_variance = α² / 2

    The Baum-Welch algorithm must never allow σ²_k to fall below this value,
    as the model cannot predict with accuracy smaller than the grid size.

    Parameters
    ----------
    tick_size : α, the minimum price increment.

    Returns
    -------
    min_var : float, the variance floor α²/2.
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    return tick_size ** 2 / 2.0


def discretized_log_gaussian(
    x: np.ndarray,
    mu: float,
    sigma2: float,
    tick_size: float,
    n_sigma: float = 10.0,
) -> np.ndarray:
    """Discretized Gaussian log-PMF (Paper §2.1, Eq. 3).

    φ_k(Δy) = N(Δy; μ_k, σ²_k) / Z_k

    where Z_k = Σ_{Δy' ∈ Y} N(Δy'; μ_k, σ²_k) is the normalization constant
    computed over the grid Y truncated to ±n_sigma·σ_k around μ_k.

    Returns log φ_k(Δy) for each observation.

    Parameters
    ----------
    x : (T,) array of discretized observations (must be on the grid).
    mu : emission mean μ_k for state k.
    sigma2 : emission variance σ²_k for state k.
    tick_size : α, the grid spacing.
    n_sigma : number of standard deviations for grid truncation (default 10).

    Returns
    -------
    log_pmf : (T,) array of log φ_k(Δy_t) values.
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")

    x = np.asarray(x, dtype=np.float64)
    sigma = np.sqrt(sigma2)

    # Build the grid around mu, truncated to ±n_sigma*sigma
    half_range = n_sigma * sigma
    grid_lo = np.floor((mu - half_range) / tick_size) * tick_size
    grid_hi = np.ceil((mu + half_range) / tick_size) * tick_size
    n_points = int(round((grid_hi - grid_lo) / tick_size)) + 1
    grid = grid_lo + np.arange(n_points) * tick_size

    # Continuous log-PDF at grid points: log N(g; mu, sigma2)
    log_pdf_grid = -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (grid - mu) ** 2 / sigma2

    # Log normalization constant: log Z_k = logsumexp(log_pdf_grid)
    log_Z = logsumexp(log_pdf_grid)

    # Log-PMF for each observation
    log_pdf_x = -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2
    log_pmf = log_pdf_x - log_Z

    return log_pmf
