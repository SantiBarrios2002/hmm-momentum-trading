"""Backward algorithm for Gaussian-emission Hidden Markov Models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


def backward(
    observations: NDArray[np.floating],
    A: NDArray[np.floating],
    mu: NDArray[np.floating],
    sigma2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Backward algorithm in log-space (Paper §3.2, Algorithm 1 lines 11-14).

    Computes log beta_t(k) for all t and k, where:
        beta_T(k) = 1
        beta_t(k) = sum_j A_{kj} * N(y_{t+1}; mu_j, sigma2_j) * beta_{t+1}(j)

    In log-space:
        log beta_T(k) = 0
        log beta_t(k) = logsumexp_j(
            log A_{kj} + log N(y_{t+1}; mu_j, sigma2_j) + log beta_{t+1}(j)
        )

    Parameters:
        observations: np.ndarray, shape (T,)
            Sequence of observed log-returns y_1, ..., y_T.
        A: np.ndarray, shape (K, K)
            Transition matrix. A[i, j] = p(m_t = j | m_{t-1} = i).
        mu: np.ndarray, shape (K,)
            State-dependent Gaussian means.
        sigma2: np.ndarray, shape (K,)
            State-dependent Gaussian variances.

    Returns:
        log_beta: np.ndarray, shape (T, K)
            Backward log-probabilities.

    Raises:
        ValueError: If shapes are inconsistent, inputs are empty, probabilities
            are invalid, or variances are non-positive.
    """
    observations = np.asarray(observations, dtype=float)
    A = np.asarray(A, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    K = A.shape[0]
    if mu.shape != (K,) or sigma2.shape != (K,):
        raise ValueError("mu and sigma2 must both have shape (K,)")
    if np.any(A <= 0.0):
        raise ValueError("A must have strictly positive entries")
    if not np.allclose(A.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("rows of A must sum to 1")
    if np.any(sigma2 <= 0.0):
        raise ValueError("sigma2 entries must be strictly positive")

    log_A = np.log(A)
    log_emissions = (
        -0.5 * np.log(2.0 * np.pi * sigma2)[None, :]
        -0.5 * ((observations[:, None] - mu[None, :]) ** 2) / sigma2[None, :]
    )

    T = observations.size
    log_beta = np.empty((T, K), dtype=float)
    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        for k in range(K):
            log_beta[t, k] = logsumexp(
                log_A[k, :] + log_emissions[t + 1, :] + log_beta[t + 1, :]
            )

    return log_beta
