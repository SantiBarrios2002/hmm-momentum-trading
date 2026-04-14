"""Rao-Blackwellized Particle Filter for the Langevin jump-diffusion model (2012 Paper §III-B).

NOTE: Supplementary module — see experiments/supplementary/ for usage.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from src.langevin.model import discretize_langevin, observation_matrix
from src.langevin.kalman import kalman_update
from src.langevin.particle import propose_jump_times, resample_particles


def initialize_rbpf_particles(
    N: int,
    mu0: NDArray,
    C0: NDArray,
) -> dict:
    """Initialize N RBPF particles, each carrying a Kalman filter state (2012 Paper §III-B, Eq 37).

    In the RBPF, each particle represents a *jump history* hypothesis, NOT a
    sampled state vector.  The continuous state (price, trend) is tracked
    analytically by a Kalman filter conditioned on that particle's jump history.

    At initialization no observations have been seen, so every particle starts
    from the same prior:

        mu_i  = mu0          (prior mean, shape (2,))
        C_i   = C0           (prior covariance, shape (2, 2))
        w_i   = 1 / N        (uniform weight)

    This is the key difference from the standard PF: instead of sampling
    x_i ~ N(mu0, C0), we keep the full Gaussian belief (mu, C) per particle.
    The Rao-Blackwell theorem guarantees that this analytical marginalization
    yields lower variance estimates than sampling.

    Parameters
    ----------
    N : int
        Number of particles.  Must be >= 1.
    mu0 : np.ndarray, shape (2,)
        Prior state mean [price, trend].
    C0 : np.ndarray, shape (2, 2)
        Prior state covariance (must be symmetric, PSD).

    Returns
    -------
    particles : dict with keys
        'mu'          : np.ndarray, shape (N, 2)
            Kalman filter means for each particle.
        'C'           : np.ndarray, shape (N, 2, 2)
            Kalman filter covariances for each particle.
        'log_weights' : np.ndarray, shape (N,)
            Log particle weights, all equal to -log(N) (uniform).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    mu0 = np.asarray(mu0, dtype=float)
    C0 = np.asarray(C0, dtype=float)
    if mu0.shape != (2,):
        raise ValueError(f"mu0 must have shape (2,), got {mu0.shape}")
    if C0.shape != (2, 2):
        raise ValueError(f"C0 must have shape (2, 2), got {C0.shape}")

    # All particles start with the same prior
    mus = np.tile(mu0, (N, 1))          # shape (N, 2)
    Cs = np.tile(C0, (N, 1, 1))         # shape (N, 2, 2)
    log_weights = np.full(N, -np.log(N))  # uniform: log(1/N)

    return {
        'mu': mus,
        'C': Cs,
        'log_weights': log_weights,
    }


def rbpf_predict_update(
    particles: dict,
    observation: float,
    theta: float,
    sigma: float,
    dt: float,
    lambda_J: float,
    mu_J: float,
    sigma_J: float,
    G: NDArray,
    sigma_obs_sq: float,
    rng: np.random.Generator | None = None,
) -> dict:
    """One-step RBPF predict-update: sample jumps, Kalman filter, reweight (2012 Paper §III-B, Eq 35-44).

    This is the core Rao-Blackwellization step. For each particle i:

        1. Sample jump from prior:  (jump_i, tau_i) ~ Poisson(lambda_J * dt)   (Eq 40)
        2. Compute transition matrices conditioned on jump:
           - No jump:   F = exp(A*dt),  Q from Van Loan                        (Eq 5-8)
           - Jump at tau: F = F2 @ F1,  Q_total (see below)                    (Eq 14-16)
        3. Kalman predict (conditioned on jump):
           - mu_pred = F @ mu_i  [+ F2 @ [0, mu_J]' if jump]                  (Eq 29-30)
           - C_pred  = F @ C_i @ F' + Q_total                                 (Eq 30)
        4. Kalman update with PED log-likelihood:
           - S = G C_pred G' + sigma_obs^2                                     (Eq 31)
           - K = C_pred G' / S                                                 (Eq 32)
           - mu_new = mu_pred + K (y - G mu_pred)                              (Eq 33)
           - C_new  = (I-KG) C_pred (I-KG)' + K sigma_obs^2 K'   (Joseph)     (Eq 33)
           - ll = log N(y; G mu_pred, S)                                       (Eq 26)
        5. Update weight:
           - log w_i += ll   (PED as importance weight)                        (Eq 43)

    KEY INSIGHT: Step 3-4 replace the state sampling in the standard PF with
    an exact Kalman filter.  The PED log-likelihood (step 4) is the marginal
    likelihood of the observation under particle i's jump hypothesis,
    analytically integrating over the continuous state.  This is the
    Rao-Blackwellization: the Rao-Blackwell theorem guarantees that this
    yields lower-variance estimates than the standard PF.

    For the jump case, the combined process noise covariance is:

        Q_total = F2 @ Q1 @ F2' + F2 @ [[0,0],[0,sigma_J^2]] @ F2' + Q2      (Eq 15)

    where F1, Q1 = discretize(tau) and F2, Q2 = discretize(dt - tau).

    Parameters
    ----------
    particles : dict
        Particle state with keys 'mu' (N,2), 'C' (N,2,2), 'log_weights' (N,).
    observation : float
        Observed value y_t.
    theta : float
        Mean-reversion parameter (theta <= 0).
    sigma : float
        Diffusion coefficient.
    dt : float
        Timestep length. Must be > 0.
    lambda_J : float
        Jump intensity (Poisson rate). Must be >= 0.
    mu_J : float
        Mean of jump size in the trend component.
    sigma_J : float
        Std of jump size in the trend component.
    G : np.ndarray, shape (1, 2)
        Observation matrix.
    sigma_obs_sq : float
        Observation noise variance. Must be > 0.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    particles_new : dict
        Updated particle state with same keys. Weights are unnormalized
        (caller handles normalization / resampling).
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if sigma_obs_sq <= 0:
        raise ValueError(f"sigma_obs_sq must be > 0, got {sigma_obs_sq}")
    if theta > 0:
        raise ValueError(f"theta must be <= 0 for stable dynamics, got {theta}")
    if lambda_J < 0:
        raise ValueError(f"lambda_J must be >= 0, got {lambda_J}")
    if sigma_J < 0:
        raise ValueError(f"sigma_J must be >= 0, got {sigma_J}")

    if rng is None:
        rng = np.random.default_rng()

    N = particles['mu'].shape[0]
    mu_all = particles['mu']        # (N, 2)
    C_all = particles['C']          # (N, 2, 2)
    log_w = particles['log_weights'].copy()  # (N,)

    # Allocate output arrays
    mu_new = np.zeros_like(mu_all)
    C_new = np.zeros_like(C_all)

    # 1. Sample jumps for all particles at once
    jump_occurred, jump_times = propose_jump_times(N, dt, lambda_J, rng=rng)

    # Pre-compute F, Q for no-jump case (shared by all non-jumping particles)
    F_nj, Q_nj = discretize_langevin(theta, sigma, dt)

    for i in range(N):
        mu_i = mu_all[i]   # (2,)
        C_i = C_all[i]     # (2, 2)

        if not jump_occurred[i]:
            # --- No jump: standard Kalman predict ---
            F = F_nj
            Q_total = Q_nj
            mu_pred = F @ mu_i
        else:
            # --- Jump at tau: split diffusion + jump ---
            tau = jump_times[i]

            # Pre-jump diffusion: 0 to tau
            if tau > 0:
                F1, Q1 = discretize_langevin(theta, sigma, tau)
            else:
                F1, Q1 = np.eye(2), np.zeros((2, 2))

            # Post-jump diffusion: tau to dt
            dt2 = dt - tau
            if dt2 > 0:
                F2, Q2 = discretize_langevin(theta, sigma, dt2)
            else:
                F2, Q2 = np.eye(2), np.zeros((2, 2))

            # Combined transition
            F = F2 @ F1

            # Jump covariance contribution
            jump_cov = np.array([[0.0, 0.0], [0.0, sigma_J**2]])

            # Q_total = F2 Q1 F2' + F2 jump_cov F2' + Q2  (Eq 15)
            Q_total = F2 @ Q1 @ F2.T + F2 @ jump_cov @ F2.T + Q2
            Q_total = (Q_total + Q_total.T) / 2.0  # enforce symmetry

            # Predicted mean includes jump mean contribution
            mu_pred = F @ mu_i + F2 @ np.array([0.0, mu_J])

        # Kalman predict covariance (same formula for both cases)
        C_pred = F @ C_i @ F.T + Q_total
        C_pred = (C_pred + C_pred.T) / 2.0

        # 4. Kalman update — reuse existing function (Joseph form, PED)
        mu_new[i], C_new[i], ll = kalman_update(
            mu_pred, C_pred, G, sigma_obs_sq, observation,
        )

        # 5. Weight update: log w_i += log p(y | jump_history_i, y_{1:t-1})
        log_w[i] += ll

    return {
        'mu': mu_new,
        'C': C_new,
        'log_weights': log_w,
    }


def extract_rbpf_signal(particles: dict) -> tuple[NDArray, NDArray]:
    """Extract the RBPF filtered estimate as a weighted mixture of Kalman means (2012 Paper §III-B, Eq 44).

    The RBPF posterior is a Gaussian mixture:

        p(x_t | y_{1:t}) = sum_i  W_i * N(x_t; mu_i, C_i)

    The filtered point estimate is the mixture mean:

        E[x_t | y_{1:t}] = sum_i  W_i * mu_i                          (Eq 44)

    The filtered standard deviation accounts for both within-particle
    and between-particle variance (law of total variance):

        Var[x_t | y_{1:t}] = sum_i W_i * C_i                          (within)
                            + sum_i W_i * (mu_i - E[x])^2             (between)

    Parameters
    ----------
    particles : dict
        Particle state with keys 'mu' (N,2), 'C' (N,2,2), 'log_weights' (N,).

    Returns
    -------
    mean : np.ndarray, shape (2,)
        Weighted mean state estimate [price, trend].
    std : np.ndarray, shape (2,)
        Weighted standard deviation of state estimate [price, trend].
    """
    # Normalize weights in probability space
    log_w = particles['log_weights']
    w = np.exp(log_w - logsumexp(log_w))  # (N,)

    mus = particles['mu']   # (N, 2)
    Cs = particles['C']     # (N, 2, 2)

    # Mixture mean: sum_i W_i * mu_i
    mean = np.average(mus, weights=w, axis=0)  # (2,)

    # Law of total variance: Var = E[Var] + Var[E]
    # Within-particle variance: sum_i W_i * diag(C_i)
    within_var = np.average(
        np.array([np.diag(Cs[i]) for i in range(len(w))]),
        weights=w, axis=0,
    )
    # Between-particle variance: sum_i W_i * (mu_i - mean)^2
    deviations = mus - mean  # (N, 2)
    between_var = np.average(deviations**2, weights=w, axis=0)

    std = np.sqrt(within_var + between_var)

    return mean, std


def run_rbpf(
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
) -> tuple[NDArray, NDArray, NDArray, float, NDArray]:
    """Run the full RBPF over an observation sequence (2012 Paper §III-B, Algorithm 1).

    The RBPF loop for each timestep t:

        t = 0 (initialization):
            1. Initialize particles with prior (mu0, C0)
            2. Kalman update only (no predict — prior IS the prediction)
            3. Reweight by PED log-likelihood
            4. Resample

        t > 0:
            1. Predict-update: sample jumps, Kalman predict+update, reweight  (Eq 35-44)
            2. Extract signal: weighted mean of particle Kalman states          (Eq 44)
            3. Resample: systematic resampling to combat weight degeneracy     (Eq 38-39)

    Resampling operates on particle INDICES — after resampling, particles with
    high weight are duplicated and low-weight particles are discarded, but the
    Kalman filter states (mu, C) are carried along with each particle.

    The per-step marginal likelihood estimate is:

        log p_hat(y_t | y_{1:t-1}) = logsumexp(log_weights)              (Eq 42)

    computed from the unnormalized weights before resampling.

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
    n_eff_history : np.ndarray, shape (T,)
        Effective sample size at each timestep: N_eff = 1 / sum(W_i^2).
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if sigma_obs_sq <= 0:
        raise ValueError(f"sigma_obs_sq must be > 0, got {sigma_obs_sq}")
    if theta > 0:
        raise ValueError(f"theta must be <= 0 for stable dynamics, got {theta}")
    if lambda_J < 0:
        raise ValueError(f"lambda_J must be >= 0, got {lambda_J}")
    if sigma_J < 0:
        raise ValueError(f"sigma_J must be >= 0, got {sigma_J}")

    if rng is None:
        rng = np.random.default_rng()

    T = len(observations)
    G = observation_matrix()

    filtered_means = np.zeros((T, 2))
    filtered_stds = np.zeros((T, 2))
    log_likelihoods = np.zeros(T)
    n_eff_history = np.zeros(T)

    # Initialize particles from prior
    particles = initialize_rbpf_particles(N_particles, mu0, C0)

    for t in range(T):
        if t == 0:
            # t=0: Kalman update only (prior as prediction, no predict step)
            N = N_particles
            mu_new = np.zeros((N, 2))
            C_new = np.zeros((N, 2, 2))
            log_w = particles['log_weights'].copy()

            for i in range(N):
                mu_new[i], C_new[i], ll = kalman_update(
                    particles['mu'][i], particles['C'][i],
                    G, sigma_obs_sq, observations[0],
                )
                log_w[i] += ll

            particles = {'mu': mu_new, 'C': C_new, 'log_weights': log_w}
        else:
            # t>0: full predict-update (sample jumps + Kalman predict + update)
            particles = rbpf_predict_update(
                particles, observation=observations[t],
                theta=theta, sigma=sigma, dt=dt,
                lambda_J=lambda_J, mu_J=mu_J, sigma_J=sigma_J,
                G=G, sigma_obs_sq=sigma_obs_sq, rng=rng,
            )

        # Per-step marginal likelihood (before resampling)
        log_likelihoods[t] = logsumexp(particles['log_weights'])

        # Effective sample size: N_eff = 1 / sum(W_i^2)
        log_w_norm = particles['log_weights'] - logsumexp(particles['log_weights'])
        n_eff_history[t] = 1.0 / np.sum(np.exp(2.0 * log_w_norm))

        # Extract filtered signal
        filtered_means[t], filtered_stds[t] = extract_rbpf_signal(particles)

        # Resample (carries mu, C along with indices)
        log_w = particles['log_weights']
        w = np.exp(log_w - logsumexp(log_w))

        # Systematic resampling (reuse logic from particle.py but for dicts)
        cumsum = np.cumsum(w)
        u = rng.random() / N_particles
        positions = u + np.arange(N_particles) / N_particles
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N_particles - 1)

        particles = {
            'mu': particles['mu'][indices].copy(),
            'C': particles['C'][indices].copy(),
            'log_weights': np.full(N_particles, -np.log(N_particles)),
        }

    total_log_likelihood = np.sum(log_likelihoods)
    return filtered_means, filtered_stds, log_likelihoods, total_log_likelihood, n_eff_history
