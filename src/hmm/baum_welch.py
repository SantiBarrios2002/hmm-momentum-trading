"""Baum-Welch (EM) training for Gaussian-emission Hidden Markov Models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.hmm.forward_backward import compute_posteriors


def m_step(
    observations: NDArray[np.floating],
    gamma: NDArray[np.floating],
    xi: NDArray[np.floating],
    min_variance: float = 1e-8,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    M-step updates for Gaussian HMM parameters (Paper §3.2, Algorithm 1).

    Update equations:
        pi_k = gamma_1(k)
        A_ij = sum_t xi_t(i,j) / sum_t gamma_t(i),   t = 1..T-1
        mu_k = sum_t gamma_t(k) * y_t / sum_t gamma_t(k), t = 1..T
        sigma2_k = sum_t gamma_t(k) * (y_t - mu_k)^2 / sum_t gamma_t(k)

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence y_1, ..., y_T.
        gamma: np.ndarray, shape (T, K)
            State posterior probabilities.
        xi: np.ndarray, shape (T-1, K, K)
            Transition posterior probabilities.
        min_variance: float
            Lower bound applied to each variance for stability.

    Returns:
        A_new: np.ndarray, shape (K, K)
        pi_new: np.ndarray, shape (K,)
        mu_new: np.ndarray, shape (K,)
        sigma2_new: np.ndarray, shape (K,)
    """
    observations = np.asarray(observations, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    xi = np.asarray(xi, dtype=float)

    T = observations.size
    K = gamma.shape[1]

    pi_new = gamma[0].copy()
    pi_new /= pi_new.sum()

    numerator_A = xi.sum(axis=0)
    denominator_A = gamma[:-1].sum(axis=0)[:, None]
    A_new = numerator_A / np.maximum(denominator_A, 1e-15)
    A_new = A_new / A_new.sum(axis=1, keepdims=True)

    gamma_sum = gamma.sum(axis=0)
    mu_new = (gamma * observations[:, None]).sum(axis=0) / np.maximum(gamma_sum, 1e-15)

    residual2 = (observations[:, None] - mu_new[None, :]) ** 2
    sigma2_new = (gamma * residual2).sum(axis=0) / np.maximum(gamma_sum, 1e-15)
    sigma2_new = np.maximum(sigma2_new, min_variance)

    return A_new, pi_new, mu_new, sigma2_new


def baum_welch(
    observations: NDArray[np.floating],
    K: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_restarts: int = 10,
    random_state: int | None = None,
    min_variance: float = 1e-8,
) -> tuple[dict[str, NDArray[np.floating]], list[float], NDArray[np.floating]]:
    """
    Baum-Welch EM optimization for Gaussian HMM parameters.

    For each restart:
        1. Initialize (A, pi, mu, sigma2)
        2. Repeat:
           - E-step: compute_posteriors(...)
           - M-step: m_step(...)
        3. Stop when |L_t - L_{t-1}| < tol

    Keeps the restart with the largest final log-likelihood.

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence y_1, ..., y_T.
        K: int
            Number of hidden states.
        max_iter: int
            Maximum EM iterations per restart.
        tol: float
            Convergence tolerance on log-likelihood increments.
        n_restarts: int
            Number of random restarts.
        random_state: int or None
            Optional seed for deterministic initialization.
        min_variance: float
            Lower bound for emission variances.

    Returns:
        best_params: dict
            Keys: "A", "pi", "mu", "sigma2".
        best_history: list[float]
            Log-likelihood history for best restart.
        best_gamma: np.ndarray, shape (T, K)
            Final state posteriors for best restart.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if K < 1:
        raise ValueError("K must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    rng = np.random.default_rng(random_state)
    obs_mean = float(np.mean(observations))
    obs_std = float(np.std(observations) + 1e-12)

    best_ll = -np.inf
    best_params = None
    best_history = None
    best_gamma = None

    for _ in range(n_restarts):
        A = rng.dirichlet(alpha=np.ones(K), size=K)
        pi = rng.dirichlet(alpha=np.ones(K))
        mu = obs_mean + rng.normal(loc=0.0, scale=obs_std, size=K)
        sigma2 = np.full(K, max(obs_std**2, min_variance), dtype=float)

        history = []
        for _ in range(max_iter):
            gamma, xi, log_likelihood = compute_posteriors(observations, A, pi, mu, sigma2)
            history.append(float(log_likelihood))
            A, pi, mu, sigma2 = m_step(
                observations, gamma, xi, min_variance=min_variance
            )
            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                break

        if history[-1] > best_ll:
            best_ll = history[-1]
            best_history = history
            best_gamma = gamma
            best_params = {"A": A, "pi": pi, "mu": mu, "sigma2": sigma2}

    return best_params, best_history, best_gamma
