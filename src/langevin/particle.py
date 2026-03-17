"""Standard (bootstrap) particle filter for the Langevin jump-diffusion model (2012 Paper §III-B)."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from src.langevin.model import discretize_langevin, observation_matrix


def initialize_particles(
    N: int,
    mu0: NDArray,
    C0: NDArray,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Draw N particles from the prior and assign uniform weights (2012 Paper §III-B, initialization).

    Each particle is a sample from the prior state distribution:

        x_i ~ N(mu0, C0),   i = 1, ..., N
        w_i = 1/N            (uniform weights)

    In the bootstrap PF, particles represent possible state trajectories.
    At initialization, all particles are equally likely since we have
    no observations yet.

    Parameters
    ----------
    N : int
        Number of particles. Must be >= 1.
    mu0 : np.ndarray, shape (2,)
        Prior state mean [price, trend].
    C0 : np.ndarray, shape (2, 2)
        Prior state covariance (must be symmetric, PSD).
    rng : np.random.Generator or None
        Random number generator. If None, a default is created.

    Returns
    -------
    states : np.ndarray, shape (N, 2)
        Initial particle states sampled from N(mu0, C0).
    log_weights : np.ndarray, shape (N,)
        Log particle weights, all equal to -log(N) (uniform).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    if rng is None:
        rng = np.random.default_rng()

    states = rng.multivariate_normal(mu0, C0, size=N)  # shape (N, 2)
    log_weights = np.full(N, -np.log(N))  # uniform: log(1/N) = -log(N)

    return states, log_weights


def propose_jump_times(
    N: int,
    dt: float,
    lambda_J: float,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Sample jump occurrence and times for N particles (2012 Paper §II-A, Eq 12, 17).

    Jumps follow a homogeneous Poisson process with rate lambda_J.
    In an interval of length dt, the probability of at least one jump is:

        p(jump in [0, dt]) = 1 - exp(-lambda_J * dt)          (Eq 17)

    Conditioned on a jump occurring, the jump time tau within [0, dt] is
    uniformly distributed (property of Poisson process with a single event):

        tau | jump ~ Uniform(0, dt)                            (Eq 12)

    For simplicity (and following the paper's assumption of at most one jump
    per interval), we model at most one jump per particle per timestep.

    Parameters
    ----------
    N : int
        Number of particles.
    dt : float
        Timestep length. Must be > 0.
    lambda_J : float
        Jump intensity (rate of the Poisson process). Must be >= 0.
    rng : np.random.Generator or None
        Random number generator. If None, a default is created.

    Returns
    -------
    jump_occurred : np.ndarray, shape (N,), dtype bool
        Whether a jump occurred for each particle.
    jump_times : np.ndarray, shape (N,)
        Jump time within [0, dt] for each particle. Only meaningful where
        jump_occurred is True; set to 0.0 where no jump occurred.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if lambda_J < 0:
        raise ValueError(f"lambda_J must be >= 0, got {lambda_J}")

    if rng is None:
        rng = np.random.default_rng()

    # Probability of at least one jump in [0, dt]
    p_jump = 1.0 - np.exp(-lambda_J * dt)

    # Sample whether each particle has a jump
    jump_occurred = rng.random(N) < p_jump  # shape (N,), bool

    # Sample jump times uniformly in [0, dt] for particles that jump
    jump_times = np.zeros(N)
    n_jumps = np.sum(jump_occurred)
    if n_jumps > 0:
        jump_times[jump_occurred] = rng.uniform(0.0, dt, size=n_jumps)

    return jump_occurred, jump_times


def propagate_particles(
    states: NDArray,
    jump_occurred: NDArray,
    jump_times: NDArray,
    theta: float,
    sigma: float,
    dt: float,
    mu_J: float,
    sigma_J: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Propagate particles forward one timestep via the transition density (2012 Paper §II-A, Eq 9, 14-16).

    Each particle's state is advanced by sampling from the conditional
    transition density. Two cases:

    Case 1 — No jump (Eq 4-6):

        x_new = F @ x_old + w,   w ~ N(0, Q)

        where F = exp(A * dt), Q from Van Loan's method.

    Case 2 — Jump at time tau (Eq 14-16):

        1. Diffuse from 0 to tau:    x_mid = F1 @ x_old + w1,    w1 ~ N(0, Q1)
        2. Add jump:                 x_mid[1] += J,               J ~ N(mu_J, sigma_J^2)
        3. Diffuse from tau to dt:   x_new = F2 @ x_mid + w2,    w2 ~ N(0, Q2)

    This samples the FULL state (price + trend) with noise — this is what
    makes the standard PF suboptimal compared to the RBPF which analytically
    integrates out the continuous state via a Kalman filter.

    Parameters
    ----------
    states : np.ndarray, shape (N, 2)
        Current particle states [price, trend].
    jump_occurred : np.ndarray, shape (N,), dtype bool
        Whether each particle experiences a jump this timestep.
    jump_times : np.ndarray, shape (N,)
        Jump time tau within [0, dt] for each particle.
    theta : float
        Mean-reversion parameter (theta <= 0).
    sigma : float
        Diffusion coefficient.
    dt : float
        Timestep length.
    mu_J : float
        Mean of jump size in the trend component.
    sigma_J : float
        Std of jump size in the trend component.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    new_states : np.ndarray, shape (N, 2)
        Propagated particle states.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if np.any(jump_times[jump_occurred] < 0) or np.any(jump_times[jump_occurred] > dt):
        raise ValueError("jump_times must be in [0, dt] for all jumping particles")

    if rng is None:
        rng = np.random.default_rng()

    N = states.shape[0]
    new_states = np.zeros_like(states)

    # --- Non-jumping particles: single-step diffusion ---
    no_jump = ~jump_occurred
    n_no_jump = np.sum(no_jump)
    if n_no_jump > 0:
        F, Q = discretize_langevin(theta, sigma, dt)
        noise = rng.multivariate_normal(np.zeros(2), Q, size=n_no_jump)
        # x_new = F @ x_old + w  for each particle
        new_states[no_jump] = (F @ states[no_jump].T).T + noise

    # --- Jumping particles: pre-jump diffusion + jump + post-jump diffusion ---
    jump_indices = np.where(jump_occurred)[0]
    for i in jump_indices:
        tau = jump_times[i]

        # Pre-jump diffusion: 0 to tau (skip if tau=0, identity diffusion)
        if tau > 0:
            F1, Q1 = discretize_langevin(theta, sigma, tau)
            w1 = rng.multivariate_normal(np.zeros(2), Q1)
            x_mid = F1 @ states[i] + w1
        else:
            x_mid = states[i].copy()

        # Jump perturbation in trend component
        J = rng.normal(mu_J, sigma_J) if sigma_J > 0 else mu_J
        x_mid[1] += J

        # Post-jump diffusion: tau to dt (skip if tau=dt, identity diffusion)
        dt2 = dt - tau
        if dt2 > 0:
            F2, Q2 = discretize_langevin(theta, sigma, dt2)
            w2 = rng.multivariate_normal(np.zeros(2), Q2)
            new_states[i] = F2 @ x_mid + w2
        else:
            new_states[i] = x_mid

    return new_states


def weight_particles(
    states: NDArray,
    log_weights: NDArray,
    observation: float,
    G: NDArray,
    sigma_obs_sq: float,
) -> NDArray:
    """Update particle log-weights using the observation likelihood (2012 Paper §III-B, Eq 41-43).

    In the bootstrap particle filter, the proposal distribution equals the
    prior (transition density). The importance weight update then simplifies to
    the observation likelihood (Eq 43):

        log w_i += log p(y_t | x_i)
                 = log N(y_t; G @ x_i, sigma_obs^2)

    where:

        log N(y; mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - 0.5 * (y - mu)^2 / sigma^2

    The weights are NOT normalized here — normalization is done separately
    (e.g., via logsumexp) to allow the caller to compute the effective sample
    size or marginal likelihood estimate.

    Parameters
    ----------
    states : np.ndarray, shape (N, 2)
        Current particle states [price, trend].
    log_weights : np.ndarray, shape (N,)
        Current log weights (from previous step or initialization).
    observation : float
        Observed value y_t.
    G : np.ndarray, shape (1, 2)
        Observation matrix. G @ x extracts the observed component.
    sigma_obs_sq : float
        Observation noise variance. Must be > 0.

    Returns
    -------
    log_weights_new : np.ndarray, shape (N,)
        Updated (unnormalized) log weights.
    """
    if sigma_obs_sq <= 0:
        raise ValueError(f"sigma_obs_sq must be > 0, got {sigma_obs_sq}")

    # Predicted observation for each particle: y_pred_i = G @ x_i
    y_pred = (G @ states.T).ravel()  # shape (N,)

    # Log-likelihood: log N(y; y_pred_i, sigma_obs_sq)
    log_lik = (
        -0.5 * np.log(2.0 * np.pi * sigma_obs_sq)
        - 0.5 * (observation - y_pred) ** 2 / sigma_obs_sq
    )

    return log_weights + log_lik


def resample_particles(
    states: NDArray,
    log_weights: NDArray,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Systematic resampling to combat weight degeneracy (2012 Paper §III-B, Eq 38-39).

    Resampling replaces the weighted particle set {x_i, w_i} with an
    equally-weighted set {x_j*, 1/N} by drawing N indices from the
    categorical distribution defined by the normalized weights.

    Systematic resampling uses a single uniform random number u ~ U(0, 1/N)
    and then selects particles at equally-spaced points:

        u_i = (u + i) / N,   i = 0, ..., N-1

    The i-th resampled particle is the one whose cumulative weight bracket
    contains u_i. This has lower variance than multinomial resampling.

    Parameters
    ----------
    states : np.ndarray, shape (N, 2)
        Current particle states.
    log_weights : np.ndarray, shape (N,)
        Current log weights (unnormalized).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    new_states : np.ndarray, shape (N, 2)
        Resampled particle states (may contain duplicates).
    new_log_weights : np.ndarray, shape (N,)
        Uniform log weights: all equal to -log(N).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(log_weights)

    # Normalize weights in log-space
    log_w_max = np.max(log_weights)
    weights = np.exp(log_weights - log_w_max)
    weights /= np.sum(weights)

    # Cumulative sum of normalized weights
    cumsum = np.cumsum(weights)

    # Systematic resampling: single uniform offset
    u = rng.random() / N
    positions = u + np.arange(N) / N  # shape (N,)

    # Select particles
    indices = np.searchsorted(cumsum, positions)
    # Clip to valid range (numerical edge case: position > cumsum[-1])
    indices = np.clip(indices, 0, N - 1)

    new_states = states[indices].copy()
    new_log_weights = np.full(N, -np.log(N))

    return new_states, new_log_weights


def run_particle_filter(
    observations: NDArray,
    N_particles: int,
    theta: float,
    sigma: float,
    sigma_obs_sq: float,
    lambda_J: float,
    mu_J: float,
    sigma_J: float,
    mu0: NDArray,
    C0: NDArray,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Run the bootstrap particle filter over an observation sequence (2012 Paper §III, Algorithm 1).

    The full algorithm for each timestep t:

        1. Propose jumps:  sample whether each particle jumps (Poisson, Eq 17)
        2. Propagate:      advance state through transition density (Eq 9, 14-16)
                           x_i^new ~ p(x_t | x_{t-1}^i, jump_info_i)
        3. Weight:         log w_i += log N(y_t; G @ x_i^new, sigma_obs^2)  (Eq 43)
        4. Estimate:       compute weighted mean state E[x_t | y_{1:t}]
        5. Resample:       systematic resampling to reset weights to 1/N (Eq 38-39)

    The per-step marginal likelihood estimate is (Eq 42):

        p_hat(y_t | y_{1:t-1}) = (1/N) * sum_i p(y_t | x_i)

    computed from the unnormalized weight increment.

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Observed values y_1, ..., y_T.
    N_particles : int
        Number of particles.
    theta : float
        Mean-reversion parameter (theta <= 0).
    sigma : float
        Diffusion coefficient.
    sigma_obs_sq : float
        Observation noise variance.
    lambda_J : float
        Jump intensity (Poisson rate).
    mu_J : float
        Mean of jump size in trend.
    sigma_J : float
        Std of jump size in trend.
    mu0 : np.ndarray, shape (2,)
        Prior state mean.
    C0 : np.ndarray, shape (2, 2)
        Prior state covariance.
    dt : float
        Timestep length.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    filtered_means : np.ndarray, shape (T, 2)
        Weighted mean state estimates [price, trend] at each timestep.
    filtered_stds : np.ndarray, shape (T, 2)
        Weighted std of state estimates at each timestep.
    log_likelihoods : np.ndarray, shape (T,)
        Per-step log marginal likelihood estimates.
    total_log_likelihood : float
        Sum of per-step log-likelihoods.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(observations)
    G = observation_matrix()

    filtered_means = np.zeros((T, 2))
    filtered_stds = np.zeros((T, 2))
    log_likelihoods = np.zeros(T)

    # Initialize particles from prior
    states, log_weights = initialize_particles(N_particles, mu0, C0, rng=rng)

    for t in range(T):
        if t > 0:
            # 1. Propose jumps
            jump_occurred, jump_times = propose_jump_times(
                N_particles, dt, lambda_J, rng=rng,
            )

            # 2. Propagate particles
            states = propagate_particles(
                states, jump_occurred, jump_times,
                theta=theta, sigma=sigma, dt=dt,
                mu_J=mu_J, sigma_J=sigma_J, rng=rng,
            )

        # 3. Weight particles by observation likelihood
        log_weights = weight_particles(
            states, log_weights, observations[t], G, sigma_obs_sq,
        )

        # Per-step marginal likelihood estimate (bootstrap PF, Eq 42):
        #   log p_hat(y_t) = log( (1/N) * sum_i p(y_t | x_i) )
        #                  = logsumexp(log_lik_i) - log(N)
        # After resampling, log_weights = -log(N) (uniform), so after
        # weight_particles: log_weights = -log(N) + log_lik_i, and
        # logsumexp(log_weights) = logsumexp(log_lik_i) - log(N) = correct.
        log_likelihoods[t] = logsumexp(log_weights)

        # 4. Compute weighted estimates
        # Normalize weights for estimation
        w_normalized = np.exp(log_weights - logsumexp(log_weights))
        filtered_means[t] = np.average(states, weights=w_normalized, axis=0)
        # Weighted std
        deviations = states - filtered_means[t]
        filtered_stds[t] = np.sqrt(
            np.average(deviations**2, weights=w_normalized, axis=0)
        )

        # 5. Resample
        states, log_weights = resample_particles(states, log_weights, rng=rng)

    total_log_likelihood = np.sum(log_likelihoods)
    return filtered_means, filtered_stds, log_likelihoods, total_log_likelihood
