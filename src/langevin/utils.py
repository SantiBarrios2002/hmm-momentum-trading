"""Utility functions bridging RBPF output to trading strategy (Paper §IV-C, §IV-D).

Paper: Christensen, Turner & Godsill (2020), arXiv:2006.08307.
"""

import numpy as np
from numpy.typing import NDArray


def estimate_langevin_params(
    log_returns: NDArray,
    dt: float = 1.0,
) -> dict:
    """Estimate Langevin jump-diffusion parameters from log-returns via method of moments (Paper §IV-C, Table I).

    The paper manually tunes parameters from SPY data; this function provides
    a principled initialization using method-of-moments heuristics:

        1. theta:     estimated from AR(1) coefficient of differenced returns
                      theta = log(phi) / dt, where phi = autocorrelation lag-1
        2. sigma:     residual volatility of the OU trend process
        3. sigma_obs: observation noise std, estimated as fraction of return vol
        4. lambda_J:  jump rate from excess kurtosis
        5. mu_J:      mean jump size (set to 0 for symmetric jumps)
        6. sigma_J:   jump size std from tail behavior

    These are heuristic estimates — the RBPF can be run with these as starting
    values, then refined via marginal likelihood optimization if desired.

    Note: lambda_J and sigma_J are expressed per unit time.  If dt changes
    (e.g. daily vs hourly), the estimated values scale accordingly because
    the underlying Poisson rate and jump variance decomposition depend on dt.

    Parameters
    ----------
    log_returns : np.ndarray, shape (T,)
        Log-returns (Δy_t = log(P_t / P_{t-1})). Must have T >= 3.
    dt : float
        Timestep in appropriate units (e.g. 1.0 for daily). Must be > 0.

    Returns
    -------
    params : dict with keys
        'theta'     : float — mean-reversion parameter (< 0)
        'sigma'     : float — diffusion coefficient (> 0)
        'sigma_obs' : float — observation noise std (> 0)
        'lambda_J'  : float — jump intensity per unit time (>= 0)
        'mu_J'      : float — mean jump size
        'sigma_J'   : float — jump size std (>= 0)
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    log_returns = np.asarray(log_returns, dtype=float)
    if len(log_returns) < 3:
        raise ValueError(f"Need at least 3 returns, got {len(log_returns)}")

    T = len(log_returns)
    ret_std = np.std(log_returns)

    # Floor ret_std to avoid zero sigma/sigma_obs which would crash downstream
    # (zero Q matrix in discretize_langevin, division by zero in kalman_update)
    ret_std = max(ret_std, 1e-10)

    # --- theta from AR(1) on differenced returns ---
    # Approximate the trend as an OU process: delta_t ~ AR(1) with phi = exp(theta*dt)
    # Use lag-1 autocorrelation of returns as proxy for phi
    returns_centered = log_returns - np.mean(log_returns)
    autocov_0 = np.sum(returns_centered**2) / T
    autocov_1 = np.sum(returns_centered[:-1] * returns_centered[1:]) / T

    if autocov_0 > 0 and autocov_1 / autocov_0 > 0:
        phi = autocov_1 / autocov_0
        # phi > 1 is a finite-sample artifact; clamp to (0, 1) to ensure theta < 0
        phi = min(phi, 1.0 - 1e-9)
        theta = np.log(phi) / dt
    else:
        # Default to moderate mean-reversion if autocorrelation is non-positive
        theta = -0.5 / dt

    # Clamp theta to be strictly negative (stable dynamics)
    theta = min(theta, -0.01 / dt)

    # --- sigma from residual volatility ---
    # For an OU process, stationary variance = sigma^2 / (2 |theta|)
    # Approximate: sigma = ret_std * sqrt(2 * |theta|)
    sigma = ret_std * np.sqrt(2.0 * abs(theta))

    # --- sigma_obs: observation noise as fraction of total volatility ---
    # Heuristic: observation noise is ~10% of return std
    sigma_obs = 0.1 * ret_std

    # --- Jump parameters from tail behavior ---
    # Excess kurtosis indicates jump activity
    if ret_std > 0:
        kurtosis = np.mean(returns_centered**4) / ret_std**4 - 3.0
    else:
        kurtosis = 0.0

    if kurtosis > 0:
        # Higher kurtosis → more jump activity
        # lambda_J: jump rate per unit time, capped at reasonable values
        lambda_J = min(kurtosis / (3.0 * dt), 10.0 / dt)
    else:
        lambda_J = 0.0

    # mu_J: symmetric jumps by default
    mu_J = 0.0

    # sigma_J: jump size std from tail contribution
    if lambda_J > 0:
        # Rough decomposition: total variance ≈ diffusion var + jump var
        # jump_var_contribution ≈ lambda_J * sigma_J^2 * dt
        # Use a fraction of total variance for jump contribution
        jump_var_fraction = min(kurtosis / (kurtosis + 3.0), 0.5)
        sigma_J = np.sqrt(jump_var_fraction * ret_std**2 / (lambda_J * dt))
    else:
        sigma_J = 0.0

    return {
        'theta': theta,
        'sigma': sigma,
        'sigma_obs': sigma_obs,
        'lambda_J': lambda_J,
        'mu_J': mu_J,
        'sigma_J': sigma_J,
    }


def trend_to_trading_signal(
    trend_estimates: NDArray,
    sigma_delta: float,
) -> NDArray:
    """Nonlinear transfer function from trend changes to trading signals (Paper §IV-D, Eq 46).

    Converts raw trend changes into bounded trading signals using the
    soft-sign transfer function:

        delta_t = x2_t - x2_{t-1}           (trend change)

        Z_t = sign(delta_t) / sqrt(delta_t^2 + sigma_delta^2)

    This is NOT a simple sign function — it has two key properties:
    1. Bounded in [-1/sigma_delta, 1/sigma_delta] — clips extreme signals
    2. Smooth transition through zero — avoids excessive position flipping

    The signal is further normalized so Z_t ∈ [-1, 1] by multiplying by
    sigma_delta:

        Z_t = delta_t / sqrt(delta_t^2 + sigma_delta^2)

    Note: the paper writes sign(delta) / sqrt(delta^2 + sigma_delta^2), but
    this is equivalent to delta / sqrt(delta^2 + sigma_delta^2) when delta != 0,
    and gives Z = 0 when delta = 0. We use the latter form as it is continuous
    and differentiable everywhere.

    Parameters
    ----------
    trend_estimates : np.ndarray, shape (T,)
        RBPF-filtered trend component x2_t at each timestep.
    sigma_delta : float
        Smoothing parameter controlling signal sensitivity. Must be > 0.
        Smaller sigma_delta → sharper transitions (closer to sign function).
        Larger sigma_delta → more gradual signals.

    Returns
    -------
    signals : np.ndarray, shape (T-1,)
        Trading signals Z_t ∈ [-1, 1] for t = 1, ..., T-1.
        Length is T-1 because the first trend change requires two points.
    """
    if sigma_delta <= 0:
        raise ValueError(f"sigma_delta must be > 0, got {sigma_delta}")
    trend_estimates = np.asarray(trend_estimates, dtype=float)
    if len(trend_estimates) < 2:
        raise ValueError(f"Need at least 2 trend estimates, got {len(trend_estimates)}")

    # Trend changes
    delta = np.diff(trend_estimates)  # shape (T-1,)

    # Soft-sign transfer function: delta / sqrt(delta^2 + sigma_delta^2)
    signals = delta / np.sqrt(delta**2 + sigma_delta**2)

    return signals
