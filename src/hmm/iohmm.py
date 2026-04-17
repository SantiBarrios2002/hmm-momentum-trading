"""Input-Output HMM — bucket-based training + inference (Paper §5-6).

The IOHMM conditions the HMM parameters on extrinsic side information x_t
(volatility ratio or intraday seasonality).  The side info is discretized
into R buckets via B-spline roots (Algorithm 3), and a separate HMM is
trained per bucket using standard Baum-Welch.

At inference time (Algorithm 5), the predict-update filter selects
parameters Θ_{r_t} based on which bucket the current side info x_t falls in.

    ω̂_t(k) = Σ_i ω_{t-1}(i) · A_{r_t}[i, k]      (predict with bucket A)
    ŷ_t   = Σ_k ω̂_t(k) · μ_{r_t}(k)               (one-step-ahead prediction)
    ω_t(k) ∝ ω̂_t(k) · N(Δy_t; μ_{r_t}(k), σ²_{r_t}(k))  (update)

Note: this implementation trains *fully separate* HMMs per bucket (each with
its own A, pi, mu, sigma2).  A more refined approach would share emission
parameters across buckets and only vary the transition matrix — this is left
as a possible extension.

Public API
----------
train_iohmm(obs, side_info, K, ...) -> dict
    Bucket-based BW training.  Returns spline + per-bucket parameters.

run_inference_iohmm(obs, side_info, iohmm_params) -> predictions, state_probs
    Online inference with bucket-switching.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src.hmm.baum_welch_numba import train_hmm_numba
from src.hmm.side_info import fit_spline, spline_buckets


def train_iohmm(
    observations: np.ndarray,
    side_info: np.ndarray,
    K: int,
    *,
    n_knots: int = 6,
    min_obs_per_bucket: int | None = None,
    n_restarts: int = 10,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_variance: float = 1e-8,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """Train IOHMM via bucket-based Baum-Welch (Paper §5, Algorithm 3).

    Algorithm:
        1. Fit B-spline G(x) to (side_info, observations)   (§4.1, Algorithm 2)
        2. Find roots of G → R bucket boundaries              (§5, Alg 3 line 1)
        3. Map observations to buckets: Z_t = bucket(x_t)     (§5, Alg 3 line 2)
        4. For each bucket r ∈ {0, ..., R-1}:
             Θ_r = BaumWelch(obs[Z == r], K)                  (§5, Alg 3 line 3)

    Buckets with fewer than *min_obs_per_bucket* observations fall back
    to parameters trained on the full dataset.

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Log-returns Δy_1, ..., Δy_T.
    side_info : np.ndarray, shape (T,)
        Extrinsic predictor values x_1, ..., x_T (vol ratio or time-of-day).
    K : int
        Number of hidden states.
    n_knots : int
        Number of interior knots for B-spline (Paper: 6 for vol ratio).
    min_obs_per_bucket : int or None
        Minimum observations required per bucket.  Buckets with fewer
        observations fall back to global (full-data) parameters.
        Default: max(2·K, 10).
    n_restarts : int
        BW restarts per bucket.
    max_iter : int
        Max BW iterations per restart.
    tol : float
        BW convergence tolerance.
    min_variance : float
        Variance floor for emissions.
    random_state : int
        RNG seed.
    verbose : bool
        Print training progress.

    Returns
    -------
    iohmm_params : dict
        "spline"        : fitted BSpline object
        "boundaries"    : np.ndarray, shape (n_roots,) — bucket boundaries
        "bucket_params" : list of R dicts, each with "A", "pi", "mu", "sigma2"
        "bucket_ll"     : list of R floats — final log-likelihood per bucket
        "K"             : int — number of states
        "R"             : int — number of buckets
    """
    observations = np.asarray(observations, dtype=np.float64)
    side_info = np.asarray(side_info, dtype=np.float64)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if side_info.ndim != 1:
        raise ValueError("side_info must be a 1D array")
    if len(observations) != len(side_info):
        raise ValueError(
            f"observations and side_info must have same length, "
            f"got {len(observations)} and {len(side_info)}"
        )
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    if min_obs_per_bucket is None:
        min_obs_per_bucket = max(2 * K, 10)

    # Step 1: Fit spline G(x) to (side_info, observations)
    spline = fit_spline(side_info, observations, n_knots=n_knots)

    # Step 2-3: Find roots → buckets, assign observations
    boundaries, bucket_idx = spline_buckets(spline, side_info)
    R = len(boundaries) + 1

    if verbose:
        print(f"IOHMM: {R} buckets from {len(boundaries)} spline roots")

    # Step 4: Train BW per bucket
    bucket_params = []
    bucket_ll = []
    # Lazy global fallback: only trained if a bucket has too few observations.
    global_params = None
    global_best_ll = float("nan")

    for r in range(R):
        mask = bucket_idx == r
        obs_r = observations[mask]

        if len(obs_r) < min_obs_per_bucket:
            # Too few observations — fall back to global parameters
            if verbose:
                print(
                    f"  bucket {r}: {len(obs_r)} obs "
                    f"(< {min_obs_per_bucket}), using global fallback"
                )
            if global_params is None:
                global_params, global_ll_hist = train_hmm_numba(
                    observations, K,
                    n_restarts=n_restarts, max_iter=max_iter, tol=tol,
                    min_variance=min_variance, random_state=random_state,
                    verbose=False,
                )
                global_best_ll = global_ll_hist[0]
            # Copy so each bucket slot owns its own arrays
            bucket_params.append({k: v.copy() for k, v in global_params.items()})
            bucket_ll.append(global_best_ll)
            continue

        if verbose:
            print(f"  bucket {r}: {len(obs_r)} obs — training BW")

        params_r, ll_r = train_hmm_numba(
            obs_r, K,
            n_restarts=n_restarts, max_iter=max_iter, tol=tol,
            min_variance=min_variance,
            random_state=random_state + r,
            verbose=False,
        )
        bucket_params.append(params_r)
        bucket_ll.append(ll_r[0])

    return {
        "spline": spline,
        "boundaries": boundaries,
        "bucket_params": bucket_params,
        "bucket_ll": bucket_ll,
        "K": K,
        "R": R,
    }


def run_inference_iohmm(
    observations: np.ndarray,
    side_info: np.ndarray,
    iohmm_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Online predict-update filter with bucket-switching (Paper §6, Algorithm 5).

    At each time step t ≥ 1:
        1. Look up bucket r_t = bucket(x_t) from the side-info spline roots
        2. Retrieve Θ_{r_t} = (A_{r_t}, π_{r_t}, μ_{r_t}, σ²_{r_t})
        3. Predict:  ω̂_t(k) = Σ_i ω_{t-1}(i) · A_{r_t}[i, k]
        4. Forecast: ŷ_t = Σ_k ω̂_t(k) · μ_{r_t}(k)
        5. Update:   ω_t(k) ∝ ω̂_t(k) · N(Δy_t; μ_{r_t}(k), σ²_{r_t}(k))

    At t = 0 there is no prior posterior, so:
        predictions[0] = Σ_k π_{r_0}(k) · μ_{r_0}(k)   (prior prediction)
        ω_0(k) ∝ π_{r_0}(k) · N(Δy_0; μ_{r_0}(k), σ²_{r_0}(k))

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Log-returns Δy_1, ..., Δy_T.
    side_info : np.ndarray, shape (T,)
        Side information x_1, ..., x_T.
    iohmm_params : dict
        Output from train_iohmm().  Required keys:
        "boundaries", "bucket_params", "K", "R".

    Returns
    -------
    predictions : np.ndarray, shape (T,)
        predictions[0] = E[Δy_0 | x_0]  (unconditional prior prediction).
        predictions[t] = E[Δy_t | Δy_{0:t-1}, x_t]  for t ≥ 1.
    state_probs : np.ndarray, shape (T, K)
        Filtered state posteriors p(m_t | Δy_{0:t}, x_{0:t}).
    """
    observations = np.asarray(observations, dtype=np.float64)
    side_info = np.asarray(side_info, dtype=np.float64)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if side_info.ndim != 1:
        raise ValueError("side_info must be a 1D array")
    if len(observations) != len(side_info):
        raise ValueError(
            f"observations and side_info must have same length, "
            f"got {len(observations)} and {len(side_info)}"
        )

    # Validate iohmm_params structure
    required_keys = {"boundaries", "bucket_params", "K", "R"}
    missing = required_keys - set(iohmm_params)
    if missing:
        raise ValueError(f"iohmm_params is missing keys: {missing}")

    boundaries = iohmm_params["boundaries"]
    bucket_params = iohmm_params["bucket_params"]
    K = iohmm_params["K"]
    R = iohmm_params["R"]

    if len(bucket_params) != R:
        raise ValueError(
            f"bucket_params length {len(bucket_params)} != R={R}"
        )

    # Assign observations to buckets via the trained spline boundaries
    bucket_idx = np.searchsorted(boundaries, side_info, side="right").astype(np.intp)
    # Clamp to valid range [0, R-1]
    np.clip(bucket_idx, 0, R - 1, out=bucket_idx)

    T = len(observations)
    predictions = np.empty(T)
    state_probs = np.empty((T, K))

    # ── t = 0: use pi from bucket r_0 ───────────────────────────────────────
    r0 = bucket_idx[0]
    p0 = bucket_params[r0]
    pi = p0["pi"]
    mu = p0["mu"]
    sigma2 = p0["sigma2"]

    predictions[0] = float(np.dot(pi, mu))

    log_omega = np.log(pi) + _log_gauss_vec(observations[0], mu, sigma2)
    log_omega -= logsumexp(log_omega)
    omega = np.exp(log_omega)
    state_probs[0] = omega

    # ── t ≥ 1: predict-update with bucket-dependent parameters ──────────────
    for t in range(1, T):
        r_t = bucket_idx[t]
        p_t = bucket_params[r_t]
        A_t = p_t["A"]
        mu_t = p_t["mu"]
        sigma2_t = p_t["sigma2"]

        # Predict: ω̂_t = ω_{t-1} · A_{r_t}
        omega_pred = omega @ A_t

        # One-step-ahead prediction: ŷ_t = Σ_k ω̂_t(k) · μ_{r_t}(k)
        predictions[t] = float(np.dot(omega_pred, mu_t))

        # Update: ω_t(k) ∝ ω̂_t(k) · N(Δy_t; μ_{r_t}(k), σ²_{r_t}(k))
        log_omega = (
            np.log(np.maximum(omega_pred, 1e-300))
            + _log_gauss_vec(observations[t], mu_t, sigma2_t)
        )
        log_omega -= logsumexp(log_omega)
        omega = np.exp(log_omega)
        state_probs[t] = omega

    return predictions, state_probs


def _log_gauss_vec(
    x: float,
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """log N(x; μ_k, σ²_k) for each state k — vectorized.

    Parameters
    ----------
    x : float
        Observation.
    mu : np.ndarray, shape (K,)
        Emission means.
    sigma2 : np.ndarray, shape (K,)
        Emission variances (must be > 0).

    Returns
    -------
    log_pdf : np.ndarray, shape (K,)
        log N(x; μ_k, σ²_k) for k = 0, ..., K-1.
    """
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2
