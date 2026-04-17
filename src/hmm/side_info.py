"""Side information features for IOHMM — volatility ratio + seasonality (Paper §4).

The paper introduces two extrinsic predictors that condition the IOHMM's
transition matrix, boosting Sharpe from ~0.5 (HMM alone) to ~2.0:

1. Volatility ratio (§4.2): x_t = σ_{short}(t) / σ_{long}(t)
   where σ is estimated via IGARCH(1,1) / EWMA (Eq. 4).
   Economic meaning: ratio < 0.6 → risk falling → buy;
                      ratio > 0.8 → risk rising → sell.

2. Intraday seasonality (§4.3): x_t = normalized time-of-day ∈ [0, 1].
   The paper finds: buy early morning, sell afternoon.

Both features are fed into B-splines (§4.1) to capture their non-linear
relationship with future returns. Spline roots discretize the feature space
into R buckets for the IOHMM (§5, Algorithm 3).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_lsq_spline


def ewma_volatility(
    returns: np.ndarray,
    lam: float = 0.79,
    window: int | None = None,
) -> np.ndarray:
    """EWMA volatility estimate (Paper §4.2, Eq. 4).

    σ_{t+1|t} = sqrt((1 - λ) · Σ_{τ=0}^{Ω-1} λ^τ · Δy²_{t-τ})

    This is IGARCH(1,1) / J.P. Morgan RiskMetrics. The paper uses λ=0.79
    for 1-min data (more reactive than the daily default of λ=0.94).

    The recursion is:
        σ²_{t+1|t} = λ · σ²_{t|t-1} + (1 - λ) · Δy²_t

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log-returns Δy_1, ..., Δy_T.
    lam : float
        Decay factor λ ∈ (0, 1). Paper §4.2: λ=0.79 for 1-min data.
    window : int or None
        If given, use only the last `window` observations for each estimate.
        If None, use the full history (infinite window).

    Returns
    -------
    sigma : np.ndarray, shape (T,)
        Conditional volatility estimates σ_{t+1|t} for t = 1, ..., T.
        sigma[t] is the forecast for time t+1, based on data up to time t.
    """
    if len(returns) < 1:
        raise ValueError(f"returns must have at least 1 observation, got {len(returns)}")
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}")

    returns = np.asarray(returns, dtype=np.float64)
    T = len(returns)
    sigma2 = np.empty(T)

    # Initialize with first squared return
    sigma2[0] = returns[0] ** 2
    if sigma2[0] == 0.0:
        sigma2[0] = np.mean(returns ** 2) if T > 1 else 1e-10

    # EWMA recursion: σ²_{t+1} = λ·σ²_t + (1-λ)·Δy²_t
    for t in range(1, T):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t] ** 2

    # Apply finite window by re-initializing if needed
    if window is not None and window < T:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        # Re-compute with window-limited sums
        for t in range(T):
            start = max(0, t - window + 1)
            weights = lam ** np.arange(t - start, -1, -1)
            sigma2[t] = (1 - lam) * np.sum(weights * returns[start:t + 1] ** 2)
            if sigma2[t] == 0.0:
                sigma2[t] = 1e-10

    return np.sqrt(np.maximum(sigma2, 0.0))


def volatility_ratio(
    returns: np.ndarray,
    window_short: int = 50,
    window_long: int = 100,
    lam: float = 0.79,
) -> np.ndarray:
    """Ratio of short-window to long-window EWMA volatility (Paper §4.2).

    x_t = σ_{t+1|t}(Ω_fast) / σ_{t+1|t}(Ω_slow)

    The paper sweeps parameters and selects Ω_fast=50, Ω_slow=100.

    Economic interpretation:
        x_t < 0.6 → recent vol falling relative to history → risk falling → buy
        x_t > 0.8 → recent vol rising relative to history → risk rising → sell

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log-returns.
    window_short : int
        Short EWMA window Ω_fast (Paper: 50).
    window_long : int
        Long EWMA window Ω_slow (Paper: 100).
    lam : float
        EWMA decay factor λ (Paper: 0.79).

    Returns
    -------
    ratio : np.ndarray, shape (T,)
        Volatility ratio x_t = σ_short / σ_long. Values in (0, ∞).
    """
    if window_short >= window_long:
        raise ValueError(
            f"window_short ({window_short}) must be < window_long ({window_long})"
        )
    sigma_short = ewma_volatility(returns, lam=lam, window=window_short)
    sigma_long = ewma_volatility(returns, lam=lam, window=window_long)
    # Avoid division by zero
    sigma_long = np.maximum(sigma_long, 1e-15)
    return sigma_short / sigma_long


def seasonality_feature(
    timestamps: np.ndarray,
    market_open_minutes: int = 0,
    market_close_minutes: int = 390,
) -> np.ndarray:
    """Normalized time-of-day feature ∈ [0, 1] (Paper §4.3).

    x_t = (minutes_since_midnight(t) - market_open) / (market_close - market_open)

    For ES RTH: open=09:30 ET (570 min), close=16:00 ET (960 min).

    Parameters
    ----------
    timestamps : array-like
        DatetimeIndex or array of datetime-like objects.
    market_open_minutes : int
        Market open as minutes since midnight (default 0 = use raw hour/minute).
    market_close_minutes : int
        Market close as minutes since midnight (default 390 = 6.5h session).

    Returns
    -------
    feature : np.ndarray, shape (T,)
        Normalized time-of-day ∈ [0, 1].
    """
    import pandas as pd

    ts = pd.DatetimeIndex(timestamps)
    minutes = ts.hour * 60 + ts.minute

    if market_open_minutes == 0 and market_close_minutes == 390:
        # Auto-detect: use min/max of observed minutes
        market_open_minutes = int(minutes.min())
        market_close_minutes = int(minutes.max())

    span = market_close_minutes - market_open_minutes
    if span <= 0:
        raise ValueError(
            f"market_close ({market_close_minutes}) must be > market_open ({market_open_minutes})"
        )

    feature = (minutes - market_open_minutes) / span
    return np.clip(feature.values.astype(np.float64), 0.0, 1.0)


def fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    n_knots: int = 6,
    degree: int = 3,
) -> object:
    """Fit a zero-mean B-spline to (x, y) pairs (Paper §4.1, Algorithm 2).

    The paper (§4.1): "Each spline is forced to be zero mean by setting
    the integral of the spline to be zero."

    Uses scipy's make_lsq_spline for least-squares B-spline fitting.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Predictor values (vol ratio or time-of-day).
    y : np.ndarray, shape (T,)
        Normalized returns ȳ (zero mean, unit variance).
    n_knots : int
        Number of interior knots. Paper: 6 for vol ratio, 10 for seasonality.
    degree : int
        B-spline degree (default 3 = cubic).

    Returns
    -------
    spline : BSpline
        Fitted B-spline callable: spline(x_new) → predicted return.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    if len(x) < n_knots + degree + 1:
        raise ValueError(
            f"Need at least {n_knots + degree + 1} observations for {n_knots} knots, "
            f"got {len(x)}"
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort by x for spline fitting
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Interior knots: evenly spaced quantiles
    knot_positions = np.linspace(0, 1, n_knots + 2)[1:-1]
    interior_knots = np.quantile(x_sorted, knot_positions)

    # Full knot vector: degree+1 copies of endpoints + interior knots
    t = np.concatenate([
        np.full(degree + 1, x_sorted[0]),
        interior_knots,
        np.full(degree + 1, x_sorted[-1]),
    ])

    spline = make_lsq_spline(x_sorted, y_sorted, t, k=degree)

    # Zero-mean correction: subtract mean of spline over data range
    # "the integral of the spline to be zero" (Paper §4.1)
    x_eval = np.linspace(x_sorted[0], x_sorted[-1], 1000)
    spline_mean = np.mean(spline(x_eval))
    spline.c -= spline_mean

    return spline


def evaluate_spline(spline, x_new: np.ndarray) -> np.ndarray:
    """Evaluate fitted spline on new data (Paper §4.1, Algorithm 2 line 9).

    ŷ_t = G(x_t)

    Parameters
    ----------
    spline : BSpline
        Fitted spline from fit_spline().
    x_new : np.ndarray, shape (T,)
        New predictor values.

    Returns
    -------
    signal : np.ndarray, shape (T,)
        Predicted return signal.
    """
    x_new = np.asarray(x_new, dtype=np.float64)
    # Clip to spline domain to avoid extrapolation
    x_lo = spline.t[0]
    x_hi = spline.t[-1]
    x_clipped = np.clip(x_new, x_lo, x_hi)
    return spline(x_clipped)


def spline_buckets(
    spline,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find spline roots → R buckets, assign x values to buckets (Paper §5, Algorithm 3).

    From the paper (§5): "discretizing the spline according to its roots,
    with R-1 roots giving R 'buckets' of spline."

    Algorithm 3, line 1: R = NewtonRaphson(G)  {Find the roots of spline G}
    Algorithm 3, line 2: Z_{1:R} = map(Y, X, R)  {Map Y to buckets}

    Parameters
    ----------
    spline : BSpline
        Fitted spline from fit_spline().
    x : np.ndarray, shape (T,)
        Feature values to assign to buckets.

    Returns
    -------
    boundaries : np.ndarray, shape (n_roots,)
        Spline root positions (bucket boundaries), sorted.
    bucket_indices : np.ndarray, shape (T,), dtype int
        Bucket index for each observation (0, 1, ..., R-1).
    """
    x = np.asarray(x, dtype=np.float64)

    # Find roots by evaluating spline on a fine grid and detecting sign changes
    x_lo = spline.t[0]
    x_hi = spline.t[-1]
    x_grid = np.linspace(x_lo, x_hi, 10000)
    y_grid = spline(x_grid)

    # Detect sign changes
    sign_changes = np.where(np.diff(np.sign(y_grid)))[0]
    roots = []
    for idx in sign_changes:
        # Linear interpolation to find root
        x0, x1 = x_grid[idx], x_grid[idx + 1]
        y0, y1 = y_grid[idx], y_grid[idx + 1]
        if y1 != y0:
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            roots.append(root)

    boundaries = np.array(sorted(set(np.round(roots, 10))))

    # Assign to buckets using np.searchsorted
    # Bucket 0: x < boundaries[0]
    # Bucket r: boundaries[r-1] <= x < boundaries[r]
    # Bucket R-1: x >= boundaries[-1]
    bucket_indices = np.searchsorted(boundaries, x, side="right").astype(np.intp)

    return boundaries, bucket_indices
