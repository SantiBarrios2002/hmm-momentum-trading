"""Utility functions bridging RBPF output to trading strategy (Paper §IV-C, §IV-D).

2020 Paper: Christensen, Turner & Godsill (2020),
    "Hidden Markov Models Applied To Intraday Momentum Trading",
    arXiv:2006.08307.
2012 Paper: Christensen, Murphy & Godsill (2012),
    "Forecasting High-Frequency Futures Returns Using Online
    Langevin Dynamics", IEEE JSTSP, vol. 6, no. 7, pp. 727-737.
"""

import itertools

import numpy as np
from numpy.typing import NDArray

from src.langevin.rbpf import run_rbpf
from src.strategy.backtest import backtest


def estimate_langevin_params(
    log_returns: NDArray,
    dt: float = 1.0,
) -> dict:
    """Estimate Langevin jump-diffusion parameters from log-returns via method-of-moments heuristics.

    This is NOT a reproduction of a specific paper formula.  The paper (§IV-C,
    Table I) manually tunes parameters from SPY data; this function provides
    a principled data-driven initialization for those same parameters:

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
    # For an OU process at stationarity: Var[x2] = sigma^2 / (2*|theta|)
    # So sigma = sqrt(2*|theta|) * ret_std.
    # If theta came from the AR(1) path, ret_std ≈ sqrt(Var[x2]), giving a
    # consistent estimate.  If theta is the default (-0.5/dt), this sets
    # sigma so the stationary std of x2 equals ret_std.
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


def estimate_langevin_params_raw(
    price_changes: NDArray,
    price_level: float,
    dt: float = 1.0,
) -> dict:
    """Estimate Langevin parameters from raw price differences (2012 Paper §IV-C, Table I).

    The 2012 paper (Christensen, Murphy & Godsill, IEEE JSTSP 2012) defines the
    state as (raw price, trend) — not (log-price, trend).  Parameters in Table I
    are "scale factors" (SF) expressed as fractions of the initial price level P_0.

    Estimation steps (method-of-moments heuristics on raw price changes ΔP_t):

        1. theta (mean-reversion rate, dimensionless):
             phi = Cov(ΔP_t, ΔP_{t-1}) / Var(ΔP_t)   (lag-1 autocorrelation)
             theta = log(phi) / dt                      (OU decay rate)
           Clamped to theta <= -0.01/dt for stability.

        2. sigma (diffusion coefficient, price units):
             sigma = std(ΔP) * sqrt(2 * |theta|)
           From the OU stationary variance: Var[x2] = sigma^2 / (2|theta|),
           so sigma = sqrt(2|theta|) * sqrt(Var[x2]) ≈ sqrt(2|theta|) * std(ΔP).

        3. sigma_obs (observation noise, price units):
             sigma_obs = 0.1 * std(ΔP)

        4. lambda_J (jump intensity, per unit time):
             kurtosis_excess = E[(ΔP - mean)^4] / std(ΔP)^4 - 3
             lambda_J = min(kurtosis_excess / (3 * dt), 10 / dt)
           Excess kurtosis > 0 indicates jump activity.

        5. mu_J = 0 (symmetric jumps by default).

        6. sigma_J (jump size std, price units):
             jump_var_fraction = min(kurtosis / (kurtosis + 3), 0.5)
             sigma_J = sqrt(jump_var_fraction * std(ΔP)^2 / (lambda_J * dt))
           Decomposes total variance into diffusion + jump contributions.

    Scale factors for Table I comparison:
        SF(x) = x / P_0

    Parameters
    ----------
    price_changes : np.ndarray, shape (T,)
        Raw price differences ΔP_t = P_t - P_{t-1}. Must have T >= 3.
    price_level : float
        Initial price level P_0 for computing scale factors. Must be > 0.
    dt : float
        Timestep in appropriate units (e.g. 1.0 for daily). Must be > 0.

    Returns
    -------
    params : dict with keys
        'theta'     : float — mean-reversion parameter (< 0)
        'sigma'     : float — diffusion coefficient in price units (> 0)
        'sigma_obs' : float — observation noise std in price units (> 0)
        'lambda_J'  : float — jump intensity per unit time (>= 0)
        'mu_J'      : float — mean jump size in price units
        'sigma_J'   : float — jump size std in price units (>= 0)
        'scale_factors' : dict — SF values relative to price_level for
                          comparison with Table I ('sigma', 'sigma_obs', 'sigma_J')
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if price_level <= 0:
        raise ValueError(f"price_level must be > 0, got {price_level}")
    price_changes = np.asarray(price_changes, dtype=float)
    if len(price_changes) < 3:
        raise ValueError(f"Need at least 3 price changes, got {len(price_changes)}")

    T = len(price_changes)
    change_std = np.std(price_changes)

    # Floor to avoid zero sigma/sigma_obs
    change_std = max(change_std, 1e-10)

    # --- theta from AR(1) on price changes ---
    changes_centered = price_changes - np.mean(price_changes)
    autocov_0 = np.sum(changes_centered**2) / T
    autocov_1 = np.sum(changes_centered[:-1] * changes_centered[1:]) / T

    if autocov_0 > 0 and autocov_1 / autocov_0 > 0:
        phi = autocov_1 / autocov_0
        phi = min(phi, 1.0 - 1e-9)
        theta = np.log(phi) / dt
    else:
        theta = -0.5 / dt

    theta = min(theta, -0.01 / dt)

    # --- sigma from residual volatility (in price units) ---
    sigma = change_std * np.sqrt(2.0 * abs(theta))

    # --- sigma_obs: observation noise as fraction of price change volatility ---
    sigma_obs = 0.1 * change_std

    # --- Jump parameters from tail behavior ---
    if change_std > 0:
        kurtosis = np.mean(changes_centered**4) / change_std**4 - 3.0
    else:
        kurtosis = 0.0

    if kurtosis > 0:
        lambda_J = min(kurtosis / (3.0 * dt), 10.0 / dt)
    else:
        lambda_J = 0.0

    mu_J = 0.0

    if lambda_J > 0:
        jump_var_fraction = min(kurtosis / (kurtosis + 3.0), 0.5)
        sigma_J = np.sqrt(jump_var_fraction * change_std**2 / (lambda_J * dt))
    else:
        sigma_J = 0.0

    # Scale factors relative to initial price (for comparison with Table I)
    scale_factors = {
        'sigma': sigma / price_level,
        'sigma_obs': sigma_obs / price_level,
        'sigma_J': sigma_J / price_level if sigma_J > 0 else 0.0,
    }

    return {
        'theta': theta,
        'sigma': sigma,
        'sigma_obs': sigma_obs,
        'lambda_J': lambda_J,
        'mu_J': mu_J,
        'sigma_J': sigma_J,
        'scale_factors': scale_factors,
    }


def trend_to_trading_signal(
    trend_estimates: NDArray,
    sigma_delta: float,
) -> NDArray:
    """Nonlinear transfer function from trend changes to trading signals (Paper §IV-D, Eq 46).

    The paper defines the signal as (Eq 46):

        Z_t = sign(delta_t) / sqrt(delta_t^2 + sigma_delta^2)

    where delta_t = x2_t - x2_{t-1} (trend change).  This maps to
    [-1/sigma_delta, 1/sigma_delta] and is discontinuous at delta = 0.

    We implement the normalized, continuous variant:

        Z_t = delta_t / sqrt(delta_t^2 + sigma_delta^2)

    which maps to [-1, 1] and is differentiable everywhere.  The two forms
    agree in sign and monotonicity; they differ only in scale (ours is
    bounded by 1, the paper's by 1/sigma_delta) and smoothness at zero.

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


def fir_momentum_signal(
    trend_estimates: NDArray,
    n_taps: int = 4,
) -> NDArray:
    """FIR-smoothed momentum direction signal (2012 Paper §IV-D, Steps 1-2).

    The 2012 paper's signal pipeline starts by converting the RBPF-filtered
    trend into a persistence-of-direction indicator via two steps:

        Step 1 — Binary direction:
            b_t = sign(x2_t)                  (is the trend up or down?)

        Step 2 — Uniform FIR moving average:
            M_t = (1/n) * sum_{j=0}^{n-1} b_{t-j}

    where n = n_taps (paper uses n=4).  The output M_t measures the fraction
    of the last n periods where momentum was positive:

        M_t = +1.0 → all n periods uptrend   (strong long)
        M_t = +0.5 → 3 of 4 periods up       (moderate long)
        M_t =  0.0 → equal up/down           (no signal)
        M_t = -0.5 → 3 of 4 periods down     (moderate short)
        M_t = -1.0 → all n periods downtrend  (strong short)

    This is the key turnover reducer: a single noisy trend reversal lasting
    one step only moves M_t by 1/n (= 0.25 for n=4), rather than causing
    a full position flip.

    Note: uses sign(x2_t) (trend level, 1st derivative of price), NOT
    sign(Δx2_t) (trend change, 2nd derivative).  The trend must cross zero
    to flip the binary signal — not just slow down.

    Alignment note: output has length T - n_taps + 1 (the FIR warmup
    consumes the first n_taps - 1 elements).  Prepend n_taps - 1 zeros
    when aligning with the original price series for backtesting.

    Parameters
    ----------
    trend_estimates : np.ndarray, shape (T,)
        RBPF-filtered trend component x2_t at each timestep.
    n_taps : int
        Number of FIR taps (moving average window). Must be >= 1.
        Paper uses n_taps=4. n_taps=1 reduces to sign(x2_t).

    Returns
    -------
    momentum : np.ndarray, shape (T - n_taps + 1,)
        FIR-smoothed momentum indicator M_t ∈ [-1, 1].
    """
    if n_taps < 1:
        raise ValueError(f"n_taps must be >= 1, got {n_taps}")
    trend_estimates = np.asarray(trend_estimates, dtype=float)
    if len(trend_estimates) < n_taps:
        raise ValueError(
            f"Need at least n_taps={n_taps} trend estimates, got {len(trend_estimates)}"
        )

    # Step 1: binary direction signal
    b = np.sign(trend_estimates)  # shape (T,), values in {-1, 0, +1}

    # Step 2: uniform FIR moving average (n-tap box filter)
    # Use cumsum for O(T) computation instead of convolution
    cumsum = np.cumsum(b)
    # M_t = (cumsum[t] - cumsum[t - n_taps]) / n_taps for t >= n_taps - 1
    momentum = np.empty(len(trend_estimates) - n_taps + 1)
    momentum[0] = cumsum[n_taps - 1] / n_taps
    momentum[1:] = (cumsum[n_taps:] - cumsum[:-n_taps]) / n_taps

    return momentum


def igarch_volatility_scale(
    signals: NDArray,
    returns: NDArray,
    alpha: float = 0.06,
    sigma2_init: float | None = None,
) -> NDArray:
    """IGARCH(1,1) volatility scaling of trading signals (2012 Paper §IV-D, Step 4).

    Normalizes position sizes by a time-varying volatility estimate so that
    risk contribution is approximately constant over time.  The integrated
    GARCH(1,1) model (IGARCH) is:

        sigma2_t = alpha * r_{t-1}^2 + (1 - alpha) * sigma2_{t-1}

    where r_{t-1} is the previous observed return and the constraint
    alpha + beta = 1 makes the variance a unit-root process (no mean
    reversion).  This is appropriate for financial volatility which is
    highly persistent (RiskMetrics model uses alpha ≈ 0.06).

    The volatility-scaled position is:

        position_t = signal_t / sqrt(sigma2_t)

    For the multi-contract case (2012 paper, 75 futures), this normalizes
    signals across contracts with different volatility levels.  For single-
    contract (SPY), it reduces to a time-varying position sizer that shrinks
    during volatile periods.

    Parameters
    ----------
    signals : np.ndarray, shape (T,)
        Raw trading signals (e.g. from the transfer function applied to FIR output).
    returns : np.ndarray, shape (T,)
        Observed returns r_t for volatility estimation.  Must have same length
        as signals.
    alpha : float
        IGARCH weight on the squared return innovation.  Must be in (0, 1).
        Default 0.06 (RiskMetrics convention).
    sigma2_init : float or None
        Initial variance estimate sigma2_0.  If None, uses np.var(returns)
        as a full-sample fallback.  Must be > 0 if provided.

    Returns
    -------
    positions : np.ndarray, shape (T,)
        Volatility-scaled positions.  The scale depends on the input signal
        scale and the volatility level.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    signals = np.asarray(signals, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if len(signals) != len(returns):
        raise ValueError(
            f"signals and returns must have same length, got {len(signals)} and {len(returns)}"
        )
    if len(signals) == 0:
        raise ValueError("signals and returns must be non-empty")

    if sigma2_init is None:
        # Use first return squared as a causal initial variance estimate.
        # np.var(returns) would use future data (lookahead bias).
        sigma2_init = returns[0] ** 2
        if sigma2_init == 0.0:
            raise ValueError(
                "First return is zero — cannot auto-initialize sigma2. "
                "Provide sigma2_init explicitly."
            )
    if not np.isfinite(sigma2_init) or sigma2_init <= 0:
        raise ValueError(f"sigma2_init must be a finite positive float, got {sigma2_init}")

    T = len(signals)
    positions = np.empty(T)
    sigma2 = sigma2_init

    for t in range(T):
        positions[t] = signals[t] / np.sqrt(sigma2)
        # Update variance for next step using current return
        sigma2 = alpha * returns[t] ** 2 + (1.0 - alpha) * sigma2

    return positions


def grid_search_params(
    prices: np.ndarray,
    param_grid: dict,
    train_frac: float = 0.7,
    N_particles: int = 200,
    dt: float = 1.0,
    n_taps: int = 4,
    use_fir: bool = True,
    transaction_cost_bps: float = 5.0,
    seed: int = 42,
) -> dict:
    """Grid search over RBPF parameters to maximise training-set Sharpe ratio (Paper §IV-C).

    The 2012 paper (§IV-C) states that parameters are "hand-tuned based on
    measurements of the Sharpe ratio on earlier time periods of data".  This
    function implements the systematic equivalent: exhaustive grid search over
    (theta, lambda_J, sigma_J, sigma_delta), evaluating each combination on
    a training split and reporting overfitting via the train/test Sharpe gap.

    Pipeline for each parameter combination:
        1. Run RBPF on training prices → filtered trend x2_t            (Paper §III-B)
        2. FIR smoothing (n_taps=4) on sign(x2_t) → M_t                 (Paper §IV-D)
           OR direct transfer function trend_to_trading_signal if use_fir=False
        3. Backtest M_t with transaction_cost_bps → train Sharpe

    The best parameter set is selected by maximum train Sharpe.  Test Sharpe
    is computed once on the held-out split for the best parameters only, to
    avoid look-ahead bias through the test set.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Raw price series (or log-price series).  The RBPF is applied directly
        to this sequence.  For raw prices use estimate_langevin_params_raw()
        to initialise sigma/sigma_obs; for log-prices use estimate_langevin_params().
    param_grid : dict
        Keys must be a subset of: 'theta', 'lambda_J', 'sigma_J', 'sigma_delta',
        'sigma', 'sigma_obs_sq'.
        Values are lists of candidate values to evaluate.
        Parameters not in the grid are derived from the training data via
        estimate_langevin_params() using the raw returns.
        Example::

            param_grid = {
                'theta':       [-0.01, -0.1, -0.5, -1.0],
                'lambda_J':    [0.0, 0.5, 1.0, 2.0],
                'sigma_J':     [0.0, 0.01, 0.02],
                'sigma_delta': [0.001, 0.005, 0.01],
            }

    train_frac : float
        Fraction of prices used for training.  Default 0.7 (70/30 split).
        Must be in (0, 1).
    N_particles : int
        Number of RBPF particles per evaluation.  Lower = faster search;
        higher = lower variance estimates.  Default 200.
    dt : float
        Timestep in consistent units (e.g. 1/390 for 1-min bars). Must be > 0.
    n_taps : int
        FIR filter length for fir_momentum_signal (used when use_fir=True).
    use_fir : bool
        If True, use fir_momentum_signal (Paper §IV-D Steps 1-2).
        If False, use trend_to_trading_signal with sigma_delta from the grid.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points. Must be >= 0.
    seed : int
        Base random seed.  Each grid point uses seed + grid_index to ensure
        reproducibility while avoiding identical RNG states.

    Returns
    -------
    result : dict with keys
        'best_params' : dict
            Parameter values that achieved the highest training Sharpe.
            Includes both grid params and data-derived params (sigma, sigma_obs_sq).
        'best_train_sharpe' : float
            Sharpe ratio on the training split for the best parameters.
        'test_sharpe' : float
            Sharpe ratio on the test split for the best parameters only.
            Computed once — not used for selection (avoids look-ahead bias).
        'train_test_gap' : float
            best_train_sharpe - test_sharpe.  Large positive gap signals overfit.
        'all_results' : list of dict
            One entry per grid point, each containing 'params' and 'train_sharpe'.
            Sorted by descending train_sharpe.
        'n_evaluated' : int
            Total number of grid points evaluated.

    Raises
    ------
    ValueError
        If param_grid is empty, train_frac is out of (0, 1), or dt <= 0.
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if transaction_cost_bps < 0:
        raise ValueError(f"transaction_cost_bps must be >= 0, got {transaction_cost_bps}")
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 10:
        raise ValueError(f"Need at least 10 prices, got {len(prices)}")

    # ── Train / test split ────────────────────────────────────────────
    T = len(prices)
    T_train = int(T * train_frac)
    train_prices = prices[:T_train]
    test_prices = prices[T_train:]

    # Returns aligned with prices[1:] (length T-1)
    train_returns = np.diff(train_prices) / train_prices[:-1]
    test_returns = np.diff(test_prices) / test_prices[:-1]

    # ── Data-derived defaults (method-of-moments on training data) ────
    log_returns_train = np.diff(np.log(np.maximum(train_prices, 1e-10)))
    base_params = estimate_langevin_params(log_returns_train, dt=dt)

    # ── Build Cartesian product of the grid ───────────────────────────
    grid_keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    grid_points = list(itertools.product(*grid_values))

    def _evaluate(point: tuple, prices_eval: np.ndarray, returns_eval: np.ndarray,
                  point_seed: int) -> float:
        """Run RBPF + signal pipeline + backtest; return Sharpe."""
        # Build full parameter dict: grid overrides base_params defaults
        p = dict(base_params)
        for key, val in zip(grid_keys, point):
            p[key] = val

        # sigma_obs_sq may come from the grid directly, or from sigma_obs
        sigma_obs_sq = p.get('sigma_obs_sq', p['sigma_obs'] ** 2)

        # Stable prior: diffuse price, stationary variance for trend
        trend_var = p['sigma'] ** 2 / (2.0 * abs(p['theta']))
        mu0 = np.array([prices_eval[0], 0.0])
        C0 = np.diag([prices_eval[0] ** 2 * 0.01, trend_var])

        try:
            filtered_means, _, _, _, _ = run_rbpf(
                prices_eval,
                N_particles,
                theta=p['theta'],
                sigma=p['sigma'],
                sigma_obs_sq=sigma_obs_sq,
                lambda_J=p.get('lambda_J', 0.0),
                mu_J=p.get('mu_J', 0.0),
                sigma_J=p.get('sigma_J', 0.0),
                mu0=mu0,
                C0=C0,
                dt=dt,
                rng=np.random.default_rng(point_seed),
            )
        except Exception:
            return np.nan

        trend = filtered_means[:, 1]  # x2_t

        # Signal pipeline
        T_eval = len(prices_eval)
        if use_fir:
            if len(trend) < n_taps:
                return np.nan
            m = fir_momentum_signal(trend, n_taps=n_taps)
            # Align to full length: prepend n_taps-1 zeros (warmup)
            signals = np.zeros(T_eval)
            signals[n_taps - 1:] = m
        else:
            sigma_delta = p.get('sigma_delta', 0.005)
            if len(trend) < 2:
                return np.nan
            raw = trend_to_trading_signal(trend, sigma_delta=sigma_delta)
            signals = np.zeros(T_eval)
            signals[1:] = raw

        if len(returns_eval) == 0 or np.all(signals == 0):
            return np.nan

        # Backtest on returns aligned to prices[1:]
        # signals[t] is based on prices[t], applied to returns[t+1]
        # backtest() handles the 1-period lag internally
        signals_bt = signals[:-1]  # drop last (no return after it)
        result = backtest(returns_eval, signals_bt,
                          transaction_cost_bps=transaction_cost_bps)
        sharpe = result['metrics']['sharpe']
        return sharpe if np.isfinite(sharpe) else np.nan

    # ── Grid search on training data ──────────────────────────────────
    all_results = []
    best_train_sharpe = -np.inf
    best_point = grid_points[0]

    for i, point in enumerate(grid_points):
        train_sharpe = _evaluate(point, train_prices, train_returns, seed + i)
        entry = {
            'params': dict(zip(grid_keys, point)),
            'train_sharpe': train_sharpe,
        }
        all_results.append(entry)
        if np.isfinite(train_sharpe) and train_sharpe > best_train_sharpe:
            best_train_sharpe = train_sharpe
            best_point = point

    # Sort by descending train Sharpe
    all_results.sort(key=lambda x: x['train_sharpe'] if np.isfinite(x['train_sharpe']) else -np.inf,
                     reverse=True)

    # ── Evaluate best params on test set (once, no peeking) ──────────
    best_params_full = dict(base_params)
    for key, val in zip(grid_keys, best_point):
        best_params_full[key] = val
    best_params_full['sigma_obs_sq'] = best_params_full.get(
        'sigma_obs_sq', best_params_full['sigma_obs'] ** 2
    )

    test_sharpe = _evaluate(best_point, test_prices, test_returns, seed + len(grid_points))

    return {
        'best_params': best_params_full,
        'best_train_sharpe': best_train_sharpe,
        'test_sharpe': test_sharpe,
        'train_test_gap': best_train_sharpe - (test_sharpe if np.isfinite(test_sharpe) else 0.0),
        'all_results': all_results,
        'n_evaluated': len(grid_points),
    }

    return positions
