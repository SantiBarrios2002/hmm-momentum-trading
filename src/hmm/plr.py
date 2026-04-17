"""Piecewise Linear Regression — Default HMM baseline (Paper §3.1).

The paper's "Default HMM" uses PLR as a naive parameter estimation method:
    1. Detect change points in prices via sequential t-tests on regression slopes.
    2. Fit OLS on each segment → μ_k (slope), σ²_k (residual variance).
    3. Build a "sticky" transition matrix: A[k,k] = β, A[k,j] = (1-β)/(K-1).

From the paper (§3.1):
    "PLR is simply ordinary least squares carried out over segmented data,
     with change points tested for by t-stats."

    "For each segment of data that contains a trend, μ_k is the gradient
     of the regression and σ²_k the variance, found from the maximum
     likelihood estimate for a Gaussian noise model:
         σ = sqrt(Σ ε²_t / T)
     where ε are the regression residuals."

Cross-validation selects K=2 for PLR (§3.1, §3.4).
"""

from __future__ import annotations

import numpy as np


def piecewise_linear_regression(
    prices: np.ndarray,
    min_segment_length: int = 50,
    significance: float = 0.05,
) -> dict:
    """Detect change points in a price series using sequential OLS with t-tests (Paper §3.1).

    Algorithm:
        For each candidate split point τ in [min_seg, T - min_seg]:
            1. Fit OLS on segment [start, τ]:  y_t = a + b·t + ε_t
            2. Fit OLS on segment [τ+1, end]:  y_t = a' + b'·t + ε_t
            3. Compare to single-segment OLS on [start, end]
            4. Test whether the split is significant via F-test (Chow test):
                   F = ((RSS_pooled - RSS_split) / 2) / (RSS_split / (n - 4))
               Under H₀ (no change point), F ~ F(2, n-4).
               Reject H₀ if p-value < significance.
        Pick τ* = argmin RSS_split among significant splits.
        Recurse on each sub-segment.

    The paper references Oh (2011) for PLR and notes that change points P
    "represent breaks between latent momentum states (i.e. trends)."

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price series y_1, ..., y_T (NOT log-returns; PLR fits trends on prices).
    min_segment_length : int
        Minimum number of observations per segment. Prevents overfitting.
    significance : float
        p-value threshold for the Chow F-test. Lower = fewer change points.

    Returns
    -------
    result : dict with keys:
        "change_points" : list of int
            Indices where regime changes occur (0-indexed into prices).
        "n_segments" : int
            Number of segments (= len(change_points) + 1).
        "segments" : list of dict, each with:
            "start" : int — start index (inclusive)
            "end" : int — end index (exclusive)
            "slope" : float — OLS slope (μ_k in paper)
            "intercept" : float — OLS intercept
            "residual_variance" : float — MLE variance σ²_k = Σε²/T
            "residuals" : np.ndarray — regression residuals ε_t
    """
    if len(prices) < 2:
        raise ValueError(f"prices must have at least 2 observations, got {len(prices)}")
    if min_segment_length < 2:
        raise ValueError(f"min_segment_length must be >= 2, got {min_segment_length}")
    if not (0.0 < significance < 1.0):
        raise ValueError(f"significance must be in (0, 1), got {significance}")

    prices = np.asarray(prices, dtype=np.float64)

    # Find change points recursively
    change_points = _find_change_points(
        prices, 0, len(prices), min_segment_length, significance
    )
    change_points.sort()

    # Build segments
    boundaries = [0] + change_points + [len(prices)]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_prices = prices[start:end]
        slope, intercept, residuals = _fit_ols(seg_prices)
        # MLE variance: σ² = Σε²/T (Paper §3.1)
        residual_var = np.sum(residuals ** 2) / len(residuals)
        segments.append({
            "start": start,
            "end": end,
            "slope": slope,
            "intercept": intercept,
            "residual_variance": residual_var,
            "residuals": residuals,
        })

    return {
        "change_points": change_points,
        "n_segments": len(segments),
        "segments": segments,
    }


def _fit_ols(y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit OLS: y_t = a + b·t + ε_t.

    Uses the closed-form solution for simple linear regression:
        b = (T·Σ(t·y) - Σt·Σy) / (T·Σ(t²) - (Σt)²)
        a = ȳ - b·t̄

    Parameters
    ----------
    y : np.ndarray, shape (n,)

    Returns
    -------
    slope : float
    intercept : float
    residuals : np.ndarray, shape (n,)
    """
    n = len(y)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y.mean()
    # Avoid division by zero for n=1 (degenerate segment)
    ss_tt = np.sum((t - t_mean) ** 2)
    if ss_tt == 0.0:
        return 0.0, y_mean, y - y_mean
    slope = np.sum((t - t_mean) * (y - y_mean)) / ss_tt
    intercept = y_mean - slope * t_mean
    residuals = y - (intercept + slope * t)
    return slope, intercept, residuals


def _chow_test_p_value(
    y: np.ndarray, split: int
) -> float:
    """Chow test for structural break at index `split` (Paper §3.1).

    Tests H₀: no change point vs H₁: different linear trends before/after split.

    F = ((RSS_pooled - RSS_split) / 2) / (RSS_split / (n - 4))

    Under H₀, F ~ F(2, n-4). Returns the p-value.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Price segment to test.
    split : int
        Candidate split index (0-indexed, exclusive for first segment).

    Returns
    -------
    p_value : float
        p-value of the F-test. Small = evidence for change point.
    """
    from scipy.stats import f as f_dist

    n = len(y)

    # Pooled (single regression on full segment)
    _, _, res_pooled = _fit_ols(y)
    rss_pooled = np.sum(res_pooled ** 2)

    # Split regressions
    _, _, res_left = _fit_ols(y[:split])
    _, _, res_right = _fit_ols(y[split:])
    rss_split = np.sum(res_left ** 2) + np.sum(res_right ** 2)

    # F-statistic: 2 extra parameters (slope + intercept for second segment)
    df_num = 2
    df_den = n - 4  # 4 params total in split model (2 per segment)
    if df_den <= 0:
        return 1.0  # Not enough data to test

    if rss_split <= 0.0:
        if rss_pooled <= 0.0:
            return 1.0  # Both fits perfect — no discriminating power
        return 0.0  # Perfect split fit, imperfect pooled fit → definite change point

    f_stat = ((rss_pooled - rss_split) / df_num) / (rss_split / df_den)
    if f_stat <= 0.0:
        return 1.0

    p_value = 1.0 - f_dist.cdf(f_stat, df_num, df_den)
    return float(p_value)


def _find_change_points(
    prices: np.ndarray,
    global_start: int,
    global_end: int,
    min_seg: int,
    significance: float,
) -> list[int]:
    """Recursively find change points via binary segmentation.

    At each level, scan all candidate splits in [min_seg, n - min_seg],
    pick the one with lowest RSS, test with Chow F-test.
    If significant, recurse on each sub-segment.
    """
    y = prices[global_start:global_end]
    n = len(y)

    if n < 2 * min_seg:
        return []  # Too short to split

    # Scan candidate splits
    best_split = None
    best_rss = np.inf

    for s in range(min_seg, n - min_seg + 1):
        _, _, res_left = _fit_ols(y[:s])
        _, _, res_right = _fit_ols(y[s:])
        rss = np.sum(res_left ** 2) + np.sum(res_right ** 2)
        if rss < best_rss:
            best_rss = rss
            best_split = s

    if best_split is None:
        return []

    # Test significance
    p_value = _chow_test_p_value(y, best_split)
    if p_value >= significance:
        return []  # Not significant — no change point here

    # Convert to global index
    cp_global = global_start + best_split

    # Recurse on left and right sub-segments
    left_cps = _find_change_points(
        prices, global_start, cp_global, min_seg, significance
    )
    right_cps = _find_change_points(
        prices, cp_global, global_end, min_seg, significance
    )

    return left_cps + [cp_global] + right_cps


def plr_to_hmm_params(
    plr_result: dict,
    K: int = 2,
    beta: float = 0.5,
) -> dict:
    """Construct Default HMM parameters from PLR output (Paper §3.1).

    The Default HMM uses a "sticky" transition matrix:
        A[k, k] = β
        A[k, j] = (1 - β) / (K - 1)   for j ≠ k

    Emission parameters come from PLR segments:
        μ_k = slope of segment k (trend gradient)
        σ²_k = MLE residual variance of segment k

    If PLR produces more segments than K, the K segments with the most
    distinct slopes are selected (via K-means-style clustering on slopes).
    If fewer segments, parameters are duplicated symmetrically.

    The initial distribution π is the ergodic (stationary) distribution of A.
    For the sticky matrix with uniform off-diagonal, π = [1/K, ..., 1/K].

    Parameters
    ----------
    plr_result : dict
        Output of piecewise_linear_regression().
    K : int
        Number of hidden states (Paper §3.1: K=2 via cross-validation).
    beta : float
        Diagonal stickiness parameter (Paper §3.1: β=0.5).
        "β is the probability of the state staying in its current state."

    Returns
    -------
    params : dict with keys:
        "A" : np.ndarray, shape (K, K)
            Sticky transition matrix.
        "pi" : np.ndarray, shape (K,)
            Initial state distribution (ergodic distribution of A).
        "mu" : np.ndarray, shape (K,)
            Emission means (segment slopes, sorted ascending).
        "sigma2" : np.ndarray, shape (K,)
            Emission variances (segment residual variances).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    segments = plr_result["segments"]
    if len(segments) == 0:
        raise ValueError("PLR result has no segments")

    # Extract slopes and variances
    slopes = np.array([s["slope"] for s in segments])
    variances = np.array([s["residual_variance"] for s in segments])
    lengths = np.array([s["end"] - s["start"] for s in segments])

    if len(segments) >= K:
        # Select K representative segments by clustering slopes
        mu, sigma2 = _select_k_segments(slopes, variances, lengths, K)
    else:
        # Fewer segments than K: spread evenly
        mu, sigma2 = _expand_segments(slopes, variances, K)

    # Sort by ascending emission mean (consistent state ordering)
    order = np.argsort(mu)
    mu = mu[order]
    sigma2 = sigma2[order]

    # Floor zero variances to prevent forward() crash (Rule 11, 13)
    sigma2 = np.maximum(sigma2, 1e-10)

    # K=1: trivial single-state model
    if K == 1:
        return {
            "A": np.array([[1.0]]),
            "pi": np.array([1.0]),
            "mu": mu,
            "sigma2": sigma2,
        }

    # Sticky transition matrix (Paper §3.1)
    A = np.full((K, K), (1.0 - beta) / (K - 1))
    np.fill_diagonal(A, beta)

    # Ergodic distribution of symmetric sticky matrix is uniform
    pi = np.full(K, 1.0 / K)

    return {"A": A, "pi": pi, "mu": mu, "sigma2": sigma2}


def _select_k_segments(
    slopes: np.ndarray,
    variances: np.ndarray,
    lengths: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select K representative (μ, σ²) from PLR segments.

    Uses length-weighted K-means on slopes to cluster segments,
    then computes weighted mean slope and variance per cluster.
    """
    if len(slopes) == K:
        return slopes.copy(), variances.copy()

    # Simple approach: sort slopes, split into K quantile groups
    order = np.argsort(slopes)
    sorted_slopes = slopes[order]
    sorted_vars = variances[order]
    sorted_lengths = lengths[order]

    # Split into K roughly equal groups
    indices = np.array_split(np.arange(len(slopes)), K)
    mu = np.empty(K)
    sigma2 = np.empty(K)
    for k, idx in enumerate(indices):
        w = sorted_lengths[idx]
        w = w / w.sum()
        mu[k] = np.sum(w * sorted_slopes[idx])
        sigma2[k] = np.sum(w * sorted_vars[idx])
    return mu, sigma2


def _expand_segments(
    slopes: np.ndarray,
    variances: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand fewer-than-K segments to K states.

    Spreads segment parameters symmetrically around the observed values.
    """
    if len(slopes) == 1:
        # Single segment: create K states centered on the slope
        mu_center = slopes[0]
        var = variances[0]
        # Spread states: use ±σ/2, ±σ, etc.
        spread = np.sqrt(var) if var > 0 else 1e-6
        offsets = np.linspace(-spread, spread, K)
        mu = mu_center + offsets
        sigma2 = np.full(K, var)
        return mu, sigma2

    # Multiple segments but fewer than K: interpolate
    mu = np.interp(
        np.linspace(0, len(slopes) - 1, K),
        np.arange(len(slopes)),
        np.sort(slopes),
    )
    sigma2 = np.interp(
        np.linspace(0, len(variances) - 1, K),
        np.arange(len(variances)),
        variances[np.argsort(slopes)],
    )
    return mu, sigma2


def durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson test statistic for autocorrelation (Paper §3.1).

    DW = Σ_{t=2}^{T} (ε_t - ε_{t-1})² / Σ_{t=1}^{T} ε_t²

    DW ≈ 2 indicates no autocorrelation.
    DW < 2 indicates positive autocorrelation.
    DW > 2 indicates negative autocorrelation.

    The paper states: "The presence of autocorrelation would suggest
    that PLR was not working correctly and so is checked for using
    the Durbin-Watson test."

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)
        Regression residuals ε_1, ..., ε_T.

    Returns
    -------
    dw : float
        Durbin-Watson statistic in [0, 4].
    """
    if len(residuals) < 2:
        raise ValueError(f"Need at least 2 residuals, got {len(residuals)}")
    ss_res = np.sum(residuals ** 2)
    if ss_res == 0.0:
        return 2.0  # Perfect fit — no autocorrelation to detect
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / ss_res)
