"""Numba-accelerated Rao-Blackwellized Particle Filter (2012 Paper §III-B).

This module reimplements run_rbpf from src/langevin/rbpf.py using Numba @njit
JIT compilation.  The Python loop over T * N iterations is the bottleneck: for
T=400 000 (1-min RTH bars over 5 years) and N=100 particles, this is 40M Kalman
updates — roughly 99 minutes in pure Python, but ~3 minutes after Numba compiles
the inner loop to native machine code.

Key implementation differences from rbpf.py
--------------------------------------------
1. **Closed-form F, Q** — scipy.linalg.expm cannot be called inside @njit.
   For the 2×2 Langevin A matrix the matrix exponential and Van Loan integral
   have closed-form solutions (derived below), so no scipy is needed.

2. **Explicit 2×2 ops** — np.outer is not supported in all @njit contexts;
   all 2×2 products are written out element-by-element or via the @ operator
   on small explicit arrays.

3. **Pre-generated random numbers** — Numba's numpy.random support in @njit is
   limited.  All random draws (jump decisions, jump times, resampling offsets)
   are generated in the Python wrapper using numpy.random.Generator, then
   passed as plain float/int arrays into the compiled core.

4. **Manual logsumexp** — scipy.special.logsumexp is unavailable inside @njit;
   the three-line numerically stable version is implemented directly.

Closed-form F and Q (2012 Paper §II-A, Eq 4-8)
-----------------------------------------------
Continuous model:  dx = A x dt + b dW,
    A = [[0, 1], [0, θ]],  b = [[0], [σ]]

Discrete transition:  x_{t+dt} = F x_t + w_t,  w_t ~ N(0, Q)

**F** = exp(A dt):
    F = [[1,  (e^{θdt} - 1)/θ],   θ ≠ 0
         [0,  e^{θdt}         ]]
    F = [[1, dt], [0, 1]]           θ = 0  (limit)

**Q** (via Van Loan / direct integration of exp(As) bb' exp(A's) ds):
    Let e1 = e^{θdt},  e2 = e1²

    Q[1,1] = σ² (e2 - 1) / (2θ)
    Q[0,1] = σ² (e1 - 1)² / (2θ²)          (symmetric)
    Q[0,0] = σ²/θ² [ (e2-1)/(2θ) - 2(e1-1)/θ + dt ]

    Limits as θ→0 (Taylor):
    Q[1,1] → σ² dt
    Q[0,1] → σ² dt²/2
    Q[0,0] → σ² dt³/3

References
----------
Christensen, Murphy & Godsill (2012), IEEE JSTSP, §III-B.
Van Loan (1978), "Computing Integrals Involving the Matrix Exponential".
"""

from __future__ import annotations

import numpy as np
import numba as nb


# ── Numba @njit helpers ────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp (replaces scipy.special.logsumexp).

    Computes log Σ_i exp(a_i) = max(a) + log Σ_i exp(a_i - max(a)).

    Parameters
    ----------
    a : np.ndarray, shape (N,)
        Log-values to sum over.

    Returns
    -------
    float
        log Σ_i exp(a_i)
    """
    a_max = np.max(a)
    s = 0.0
    for i in range(len(a)):
        s += np.exp(a[i] - a_max)
    return np.log(s) + a_max


@nb.njit(cache=True)
def _discretize_cf(theta: float, sigma: float, dt: float):
    """Closed-form discrete transition (F, Q) for 2×2 Langevin SDE.

    Implements the closed-form solution of exp(A*dt) and the Van Loan
    integral Q = ∫₀^dt exp(As) bb' exp(A's) ds for:

        A = [[0, 1], [0, θ]],   b = [[0], [σ]]

    See module docstring for the full derivation.

    Parameters
    ----------
    theta : float   Mean-reversion parameter (≤ 0 for stable dynamics).
    sigma : float   Diffusion coefficient of the trend.
    dt    : float   Discretisation step (trading-day units).

    Returns
    -------
    F : np.ndarray, shape (2, 2)  State transition matrix.
    Q : np.ndarray, shape (2, 2)  Process noise covariance (symmetric, PSD).
    """
    e1 = np.exp(theta * dt)
    e2 = e1 * e1
    s2 = sigma * sigma

    # --- F ---
    F = np.zeros((2, 2))
    F[0, 0] = 1.0
    F[1, 1] = e1
    if abs(theta) < 1e-12:
        F[0, 1] = dt
    else:
        F[0, 1] = (e1 - 1.0) / theta

    # --- Q ---
    Q = np.zeros((2, 2))
    if abs(theta) < 1e-12:
        Q[1, 1] = s2 * dt
        Q[0, 1] = s2 * dt * dt * 0.5
        Q[0, 0] = s2 * dt * dt * dt / 3.0
    else:
        Q[1, 1] = s2 * (e2 - 1.0) / (2.0 * theta)
        Q[0, 1] = s2 * (e1 - 1.0) * (e1 - 1.0) / (2.0 * theta * theta)
        Q[0, 0] = s2 / (theta * theta) * (
            (e2 - 1.0) / (2.0 * theta) - 2.0 * (e1 - 1.0) / theta + dt
        )
    Q[1, 0] = Q[0, 1]

    return F, Q


@nb.njit(cache=True)
def _kalman_update(
    mu_pred: np.ndarray,   # (2,)
    C_pred:  np.ndarray,   # (2, 2)
    y:       float,
    sigma_obs_sq: float,
):
    """Kalman measurement update with Joseph-form covariance (2012 Paper §III-B).

    Observation model: y_t = G x_t + v_t, v_t ~ N(0, σ_obs²),  G = [1, 0].

    Innovation covariance (scalar because G is 1×2 and y is scalar):
        S = G C_pred G' + σ_obs² = C_pred[0,0] + σ_obs²

    Kalman gain (2×1 column):
        K = C_pred G' / S  →  K[0] = C_pred[0,0]/S, K[1] = C_pred[1,0]/S

    Mean update:
        μ_new = μ_pred + K (y - G μ_pred)

    Joseph-form covariance (numerically stable, preserves PSD):
        C_new = (I - KG) C_pred (I - KG)' + σ_obs² K K'

    PED log-likelihood contribution:
        log p(y_t | y_{1:t-1}) = -½ log(2π S) - ½ (y - G μ_pred)² / S

    Parameters
    ----------
    mu_pred     : np.ndarray (2,)   Predicted mean [price, trend].
    C_pred      : np.ndarray (2,2)  Predicted covariance.
    y           : float             Observed price at time t.
    sigma_obs_sq: float             Observation noise variance σ_obs².

    Returns
    -------
    mu_new : np.ndarray (2,)   Updated mean.
    C_new  : np.ndarray (2,2)  Updated covariance (Joseph form).
    ll     : float             PED log-likelihood log p(y_t | y_{1:t-1}).
    """
    innov = y - mu_pred[0]
    S     = C_pred[0, 0] + sigma_obs_sq

    K0 = C_pred[0, 0] / S
    K1 = C_pred[1, 0] / S

    mu_new    = np.empty(2)
    mu_new[0] = mu_pred[0] + K0 * innov
    mu_new[1] = mu_pred[1] + K1 * innov

    # I - K G  (G = [[1, 0]])  →  [[1-K0, 0], [-K1, 1]]
    IKG = np.zeros((2, 2))
    IKG[0, 0] =  1.0 - K0
    IKG[1, 0] = -K1
    IKG[1, 1] =  1.0

    # Joseph form: (I-KG) C_pred (I-KG)' + σ_obs² K K'
    IKG_C = IKG @ C_pred                 # (2,2)
    C_new  = IKG_C @ IKG.T               # (2,2)
    C_new[0, 0] += sigma_obs_sq * K0 * K0
    C_new[0, 1] += sigma_obs_sq * K0 * K1
    C_new[1, 0] += sigma_obs_sq * K1 * K0
    C_new[1, 1] += sigma_obs_sq * K1 * K1
    # Enforce symmetry
    sym = (C_new[0, 1] + C_new[1, 0]) * 0.5
    C_new[0, 1] = sym
    C_new[1, 0] = sym

    ll = -0.5 * np.log(2.0 * np.pi * S) - 0.5 * innov * innov / S

    return mu_new, C_new, ll


# ── Core @njit RBPF loop ───────────────────────────────────────────────────────

@nb.njit(cache=True)
def _rbpf_core(
    obs:          np.ndarray,   # (T,)   float64 — raw price observations
    mu_arr:       np.ndarray,   # (N, 2) float64 — particle means (mutated)
    C_arr:        np.ndarray,   # (N, 2, 2) float64 — particle covs (mutated)
    log_w:        np.ndarray,   # (N,)  float64 — log weights (mutated)
    theta:        float,
    sigma:        float,
    sigma_obs_sq: float,
    lambda_J:     float,
    mu_J:         float,
    sigma_J:      float,
    dt:           float,
    jump_counts:  np.ndarray,   # (T, N) int64 — Poisson(lambda_J*dt) draws
    tau_frac:     np.ndarray,   # (T, N) float64 — Uniform(0,1) jump time fracs
    resample_u:   np.ndarray,   # (T,)   float64 — Uniform(0,1) for resampling
):
    """Compiled RBPF inner loop (2012 Paper §III-B, Algorithm 1).

    Runs T steps of the Rao-Blackwellized Particle Filter.  At each step:

      For each particle i:
        1. Sample jump: jump_counts[t,i] ~ Poisson(λ_J dt) (pre-generated)
        2. If jump: compute F_jump, Q_jump for the split-interval dynamics
           (Eq 10-16); otherwise use the pre-computed no-jump F_nj, Q_nj.
        3. Kalman predict: μ_pred = F μ_i, C_pred = F C_i F' + Q
        4. Kalman update (Joseph form): obtain μ_new, C_new, log-weight ll_i
        5. Accumulate: log_w[i] += ll_i

      Aggregate:
        6. Compute ESS = 1 / Σ w_i²  (normalised weights)
        7. Compute weighted mean (law of total expectation) and std
           (law of total variance)
        8. Systematic resampling when t > 0
        9. Reset log-weights to log(1/N)

    Parameters are documented in run_rbpf_numba.

    Returns
    -------
    filtered_means : (T, 2) float64
    filtered_stds  : (T, 2) float64
    log_ll         : (T,)   float64  — per-step log marginal likelihood
    n_eff          : (T,)   float64  — effective sample size
    """
    T = obs.shape[0]
    N = mu_arr.shape[0]

    filtered_means = np.zeros((T, 2))
    filtered_stds  = np.zeros((T, 2))
    log_ll         = np.zeros(T)
    n_eff          = np.zeros(T)

    # Pre-compute no-jump F, Q (reused every step for non-jumping particles)
    F_nj, Q_nj = _discretize_cf(theta, sigma, dt)

    sigma_J_sq = sigma_J * sigma_J
    log_1_over_N = -np.log(float(N))

    for t in range(T):
        # ── Predict-update for each particle ──────────────────────────────────
        for i in range(N):
            if t == 0:
                # At t=0: no predict step — just update from the prior
                mu_new, C_new, ll = _kalman_update(
                    mu_arr[i], C_arr[i], obs[0], sigma_obs_sq
                )
            else:
                if jump_counts[t, i] > 0:
                    # ── Jump path (2012 Paper §II-A, Eq 10-16) ────────────────
                    tau  = tau_frac[t, i] * dt       # jump time within [0, dt]
                    dt2  = dt - tau

                    # Pre-jump and post-jump sub-transitions
                    F1, Q1 = _discretize_cf(theta, sigma, tau)
                    F2, Q2 = _discretize_cf(theta, sigma, dt2)
                    F_j    = F2 @ F1

                    # Additional variance from jump: J ~ N(μ_J, σ_J²) on x2
                    jump_cov       = np.zeros((2, 2))
                    jump_cov[1, 1] = sigma_J_sq

                    # Q_total = F2 Q1 F2' + F2 Σ_J F2' + Q2
                    Q_j = F2 @ Q1 @ F2.T + F2 @ jump_cov @ F2.T + Q2
                    sym = (Q_j[0, 1] + Q_j[1, 0]) * 0.5
                    Q_j[0, 1] = sym
                    Q_j[1, 0] = sym

                    # Mean shift from jump on x2 component
                    jump_mean    = np.zeros(2)
                    jump_mean[1] = mu_J
                    mu_pred      = F_j @ mu_arr[i] + F2 @ jump_mean
                    C_pred       = F_j @ C_arr[i] @ F_j.T + Q_j
                else:
                    # ── No-jump path (standard Kalman predict) ────────────────
                    mu_pred = F_nj @ mu_arr[i]
                    C_pred  = F_nj @ C_arr[i] @ F_nj.T + Q_nj

                # Enforce C_pred symmetry
                sym = (C_pred[0, 1] + C_pred[1, 0]) * 0.5
                C_pred[0, 1] = sym
                C_pred[1, 0] = sym

                mu_new, C_new, ll = _kalman_update(
                    mu_pred, C_pred, obs[t], sigma_obs_sq
                )

            mu_arr[i] = mu_new
            C_arr[i]  = C_new
            log_w[i] += ll

        # ── Aggregate: log-likelihood, weights, ESS ───────────────────────────
        lse       = _logsumexp(log_w)
        log_ll[t] = lse

        log_w_norm = log_w - lse           # normalised log weights
        w = np.exp(log_w_norm)             # (N,) normalised weights

        sum_w2 = 0.0
        for i in range(N):
            sum_w2 += w[i] * w[i]
        n_eff[t] = 1.0 / (sum_w2 + 1e-300)

        # Weighted mean (law of total expectation)
        mean0 = 0.0
        mean1 = 0.0
        for i in range(N):
            mean0 += w[i] * mu_arr[i, 0]
            mean1 += w[i] * mu_arr[i, 1]
        filtered_means[t, 0] = mean0
        filtered_means[t, 1] = mean1

        # Weighted variance (law of total variance):
        #   Var = E[Var_i] + Var(E_i)  =  Σ w_i C_i[j,j] + Σ w_i (μ_i[j] - μ̄[j])²
        var0 = 0.0
        var1 = 0.0
        for i in range(N):
            var0 += w[i] * C_arr[i, 0, 0]
            var1 += w[i] * C_arr[i, 1, 1]
            d0    = mu_arr[i, 0] - mean0
            d1    = mu_arr[i, 1] - mean1
            var0 += w[i] * d0 * d0
            var1 += w[i] * d1 * d1
        filtered_stds[t, 0] = np.sqrt(var0)
        filtered_stds[t, 1] = np.sqrt(var1)

        # ── Systematic resampling ─────────────────────────────────────────────
        # Build CDF
        cumsum    = np.zeros(N)
        cumsum[0] = w[0]
        for i in range(1, N):
            cumsum[i] = cumsum[i - 1] + w[i]

        # N equally-spaced positions starting from U[0, 1/N]
        u0        = resample_u[t] / float(N)
        positions = np.empty(N)
        for i in range(N):
            positions[i] = u0 + float(i) / float(N)

        indices = np.searchsorted(cumsum, positions)
        # Clip to valid range (handles floating-point overshoot at right edge)
        for i in range(N):
            if indices[i] >= N:
                indices[i] = N - 1

        # Copy selected particles
        mu_tmp = np.zeros((N, 2))
        C_tmp  = np.zeros((N, 2, 2))
        for i in range(N):
            mu_tmp[i] = mu_arr[indices[i]]
            C_tmp[i]  = C_arr[indices[i]]
        mu_arr = mu_tmp
        C_arr  = C_tmp

        # Reset to uniform log-weights
        for i in range(N):
            log_w[i] = log_1_over_N

    return filtered_means, filtered_stds, log_ll, n_eff


# ── Public Python wrapper (same signature as run_rbpf) ────────────────────────

def run_rbpf_numba(
    observations:  np.ndarray,
    N_particles:   int,
    theta:         float,
    sigma:         float,
    sigma_obs_sq:  float,
    lambda_J:      float,
    mu_J:          float,
    sigma_J:       float,
    mu0:           np.ndarray,
    C0:            np.ndarray,
    dt:            float,
    rng:           np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Numba-compiled Rao-Blackwellized Particle Filter (2012 Paper §III-B).

    Drop-in replacement for src.langevin.rbpf.run_rbpf with an identical
    call signature and return type.  Uses Numba @njit compilation for the
    inner T × N loop, enabling 1-minute frequency runs (~400 k bars).

    The first call triggers JIT compilation (~20-30 s for @njit(cache=True)).
    Subsequent calls (or after the cache warms up) run at native speed.

    Algorithm (per timestep t, per particle i):
        1. Sample jump indicator:  n_j ~ Poisson(λ_J dt)
        2. If n_j > 0: compute split-interval F_jump, Q_jump
           (2012 Paper §II-A, Eq 10-16)
        3. Kalman predict: μ_pred, C_pred
        4. Kalman update (Joseph form): get weight contribution ll_i
        5. Systematic resampling; reset weights to uniform

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Raw price observations y_1 … y_T (NOT log-prices).
    N_particles  : int
        Number of particles N.
    theta        : float
        Mean-reversion parameter (< 0 for stable dynamics).
    sigma        : float
        Diffusion coefficient of the trend process x_2.
    sigma_obs_sq : float
        Observation noise variance σ_obs².
    lambda_J     : float
        Jump rate (jumps per trading day).
    mu_J         : float
        Mean of jump size on x_2.
    sigma_J      : float
        Std of jump size on x_2.
    mu0          : np.ndarray, shape (2,)
        Prior mean of x_0 = [price_0, trend_0].
    C0           : np.ndarray, shape (2, 2)
        Prior covariance of x_0.
    dt           : float
        Timestep in trading-day units (e.g. 1/390 for 1-min bars).
    rng          : np.random.Generator
        Seeded numpy random generator (created via np.random.default_rng).

    Returns
    -------
    filtered_means : np.ndarray, shape (T, 2)
        Weighted mean of particle Kalman means: E[x_t | y_{1:t}].
        Column 0 = filtered price, column 1 = filtered trend x_2.
    filtered_stds  : np.ndarray, shape (T, 2)
        Per-component std via law of total variance.
    log_likelihoods : np.ndarray, shape (T,)
        Per-step log marginal likelihood log p(y_t | y_{1:t-1}).
    total_log_likelihood : float
        Total log p(y_{1:T}) = Σ_t log p(y_t | y_{1:t-1}).
    n_eff_history : np.ndarray, shape (T,)
        Effective sample size 1 / Σ_i w_i² at each timestep.
    """
    observations = np.asarray(observations, dtype=np.float64)
    T = len(observations)
    N = N_particles

    # ── Initialise particle arrays ────────────────────────────────────────────
    mu_arr = np.tile(mu0.astype(np.float64), (N, 1))          # (N, 2)
    C_arr  = np.tile(C0.astype(np.float64),  (N, 1, 1))       # (N, 2, 2)
    log_w  = np.full(N, -np.log(float(N)), dtype=np.float64)  # uniform

    # ── Pre-generate all random numbers ───────────────────────────────────────
    # Jump decisions: Poisson(λ_J * dt) — how many jumps occur in bar (t, i).
    # For λ_J * dt << 1 this is nearly Bernoulli, but exact Poisson is used
    # to match the original run_rbpf implementation.
    p_jump   = lambda_J * dt
    if p_jump > 0.0:
        jump_counts = rng.poisson(p_jump, size=(T, N)).astype(np.int64)
    else:
        jump_counts = np.zeros((T, N), dtype=np.int64)

    # Jump timing: Uniform(0, 1) scaled to [0, dt] inside the @njit core.
    tau_frac   = rng.random((T, N))

    # Systematic resampling offsets: one Uniform(0, 1) per timestep.
    resample_u = rng.random(T)

    # ── Call compiled core ────────────────────────────────────────────────────
    filtered_means, filtered_stds, log_ll, n_eff = _rbpf_core(
        observations,
        mu_arr, C_arr, log_w,
        theta, sigma, sigma_obs_sq,
        lambda_J, mu_J, sigma_J, dt,
        jump_counts, tau_frac, resample_u,
    )

    total_ll = float(np.sum(log_ll))

    return filtered_means, filtered_stds, log_ll, total_ll, n_eff
