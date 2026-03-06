"""Viterbi decoding for Gaussian-emission Hidden Markov Models."""

import numpy as np


def viterbi(observations, A, pi, mu, sigma2):
    """
    Compute the MAP hidden-state path via the Viterbi algorithm.

    Recursion (log-space):
        delta_1(k) = log pi_k + log N(y_1; mu_k, sigma2_k)
        delta_t(k) = max_i [delta_{t-1}(i) + log A_{ik}] + log N(y_t; mu_k, sigma2_k)
        psi_t(k) = argmax_i [delta_{t-1}(i) + log A_{ik}]

    Backtracking:
        m_T = argmax_k delta_T(k)
        m_t = psi_{t+1}(m_{t+1})

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence y_1, ..., y_T.
        A: np.ndarray, shape (K, K)
            Transition matrix.
        pi: np.ndarray, shape (K,)
            Initial state distribution.
        mu: np.ndarray, shape (K,)
            Emission means.
        sigma2: np.ndarray, shape (K,)
            Emission variances.

    Returns:
        states: np.ndarray, shape (T,)
            MAP state sequence.
        log_prob: float
            Log-probability of MAP path.
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
    delta = np.empty((T, K), dtype=float)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        for k in range(K):
            candidates = delta[t - 1] + log_A[:, k]
            psi[t, k] = int(np.argmax(candidates))
            delta[t, k] = candidates[psi[t, k]] + log_emissions[t, k]

    states = np.empty(T, dtype=int)
    states[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    log_prob = float(np.max(delta[T - 1]))
    return states, log_prob
