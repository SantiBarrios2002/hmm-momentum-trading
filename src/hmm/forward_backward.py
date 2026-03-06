"""Forward-backward posterior computations for Gaussian HMMs."""

import numpy as np
from scipy.special import logsumexp

from src.hmm.backward import backward
from src.hmm.forward import forward


def compute_posteriors(observations, A, pi, mu, sigma2):
    """
    Compute state and transition posteriors via forward-backward.

    E-step quantities (Paper §3.2, Algorithm 1):
        gamma_t(k) = p(m_t = k | y_1:T, Theta)
        xi_t(i, j) = p(m_t = i, m_{t+1} = j | y_1:T, Theta)

    Uses log-space forward/backward variables and log-sum-exp normalization.

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed log-returns y_1, ..., y_T.
        A: np.ndarray, shape (K, K)
            Transition matrix.
        pi: np.ndarray, shape (K,)
            Initial state distribution.
        mu: np.ndarray, shape (K,)
            Emission means.
        sigma2: np.ndarray, shape (K,)
            Emission variances.

    Returns:
        gamma: np.ndarray, shape (T, K)
            State posterior probabilities.
        xi: np.ndarray, shape (T-1, K, K)
            Transition posterior probabilities.
        log_likelihood: float
            Sequence log-likelihood.

    Raises:
        ValueError: If observations length is zero.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")

    log_alpha, log_likelihood = forward(observations, A, pi, mu, sigma2)
    log_beta = backward(observations, A, mu, sigma2)

    # gamma_t(k) proportional to alpha_t(k) * beta_t(k)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    T, K = log_alpha.shape
    xi = np.empty((T - 1, K, K), dtype=float)

    log_A = np.log(np.asarray(A, dtype=float))
    log_emissions = (
        -0.5 * np.log(2.0 * np.pi * np.asarray(sigma2, dtype=float))[None, :]
        -0.5
        * (
            (observations[:, None] - np.asarray(mu, dtype=float)[None, :]) ** 2
            / np.asarray(sigma2, dtype=float)[None, :]
        )
    )

    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t, :, None]
            + log_A
            + log_emissions[t + 1, None, :]
            + log_beta[t + 1, None, :]
        )
        log_xi_t -= logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t)

    return gamma, xi, float(log_likelihood)
