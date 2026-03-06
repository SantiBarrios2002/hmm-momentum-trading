"""Forward algorithm for Gaussian-emission Hidden Markov Models."""

import numpy as np
from scipy.special import logsumexp


def forward(observations, A, pi, mu, sigma2):
    """
    Forward algorithm in log-space (Paper §3.2, Algorithm 1 lines 6-9).

    Computes log alpha_t(k) for all t and k, where:
        alpha_1(k) = pi_k * N(y_1; mu_k, sigma2_k)
        alpha_t(k) = [sum_i alpha_{t-1}(i) * A_{ik}] * N(y_t; mu_k, sigma2_k)

    The recursion is evaluated in log-space for numerical stability:
        log alpha_t(k) = logsumexp_i(log alpha_{t-1}(i) + log A_{ik})
                         + log N(y_t; mu_k, sigma2_k)

    Parameters:
        observations: np.ndarray, shape (T,)
            Sequence of observed log-returns y_1, ..., y_T.
        A: np.ndarray, shape (K, K)
            Transition matrix. A[i, j] = p(m_t = j | m_{t-1} = i).
        pi: np.ndarray, shape (K,)
            Initial state probabilities. pi[k] = p(m_1 = k).
        mu: np.ndarray, shape (K,)
            State-dependent Gaussian means.
        sigma2: np.ndarray, shape (K,)
            State-dependent Gaussian variances.

    Returns:
        log_alpha: np.ndarray, shape (T, K)
            Forward log-probabilities.
        log_likelihood: float
            log p(y_1:T | Theta) = logsumexp_k(log alpha_T(k)).

    Raises:
        ValueError: If shapes are inconsistent, inputs are empty, probabilities
            are invalid, or variances are non-positive.
    """
    observations = np.asarray(observations, dtype=float)
    A = np.asarray(A, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    K = A.shape[0]
    if pi.shape != (K,) or mu.shape != (K,) or sigma2.shape != (K,):
        raise ValueError("pi, mu, sigma2 must all have shape (K,)")
    if np.any(A <= 0.0) or np.any(pi <= 0.0):
        raise ValueError("A and pi must have strictly positive entries")
    if not np.allclose(A.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("rows of A must sum to 1")
    if not np.isclose(pi.sum(), 1.0, atol=1e-10):
        raise ValueError("pi must sum to 1")
    if np.any(sigma2 <= 0.0):
        raise ValueError("sigma2 entries must be strictly positive")

    log_A = np.log(A)
    log_pi = np.log(pi)

    log_emissions = (
        -0.5 * np.log(2.0 * np.pi * sigma2)[None, :]
        -0.5 * ((observations[:, None] - mu[None, :]) ** 2) / sigma2[None, :]
    )

    T = observations.size
    log_alpha = np.empty((T, K), dtype=float)
    log_alpha[0] = log_pi + log_emissions[0]

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = (
                logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_emissions[t, k]
            )

    log_likelihood = float(logsumexp(log_alpha[-1]))
    return log_alpha, log_likelihood
