"""Online inference routines for Gaussian-emission HMMs."""

import numpy as np
from scipy.special import logsumexp


def predict_update_step(omega_prev, A, mu, sigma2, observation):
    """
    Run one online predict-update step (Paper §6, Algorithm 4).

    Predict:
        omega_pred(k) = sum_{k'} A[k', k] * omega_prev(k')

    Update:
        omega_new(k) proportional to omega_pred(k) * N(y_t; mu_k, sigma2_k)

    Prediction:
        y_hat_t = sum_k omega_pred(k) * mu_k

    Parameters:
        omega_prev: np.ndarray, shape (K,)
            Previous filtered state distribution p(m_{t-1} | y_{1:t-1}).
        A: np.ndarray, shape (K, K)
            Transition matrix.
        mu: np.ndarray, shape (K,)
            Emission means.
        sigma2: np.ndarray, shape (K,)
            Emission variances.
        observation: float
            Current observation y_t.

    Returns:
        omega_new: np.ndarray, shape (K,)
            Updated filtered distribution p(m_t | y_{1:t}).
        prediction: float
            One-step-ahead return prediction from predictive distribution.
    """
    omega_prev = np.asarray(omega_prev, dtype=float)
    A = np.asarray(A, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    observation = float(observation)

    if omega_prev.ndim != 1:
        raise ValueError("omega_prev must be a 1D array")
    K = omega_prev.size
    if A.shape != (K, K) or mu.shape != (K,) or sigma2.shape != (K,):
        raise ValueError("inconsistent shapes between omega_prev, A, mu, sigma2")
    if np.any(omega_prev < 0.0) or not np.isclose(omega_prev.sum(), 1.0, atol=1e-10):
        raise ValueError("omega_prev must be a valid probability vector")
    if np.any(A <= 0.0) or not np.allclose(A.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("A must have positive entries and row sums equal to 1")
    if np.any(sigma2 <= 0.0):
        raise ValueError("sigma2 entries must be strictly positive")

    omega_pred = omega_prev @ A
    prediction = float(np.dot(omega_pred, mu))

    log_omega_pred = np.log(omega_pred)
    log_likelihood = -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (
        (observation - mu) ** 2
    ) / sigma2
    log_omega_new = log_omega_pred + log_likelihood
    log_omega_new -= logsumexp(log_omega_new)
    omega_new = np.exp(log_omega_new)

    return omega_new, prediction


def run_inference(observations, A, pi, mu, sigma2):
    """
    Run online inference across all observations.

    Applies predict_update_step iteratively and returns:
        predictions[t] = E[y_t | y_{1:t-1}]
        state_probs[t] = p(m_t | y_{1:t})

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        A: np.ndarray, shape (K, K)
            Transition matrix.
        pi: np.ndarray, shape (K,)
            Initial state distribution for t=1.
        mu: np.ndarray, shape (K,)
            Emission means.
        sigma2: np.ndarray, shape (K,)
            Emission variances.

    Returns:
        predictions: np.ndarray, shape (T,)
        state_probs: np.ndarray, shape (T, K)
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")

    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1:
        raise ValueError("pi must be a 1D array")
    if np.any(pi < 0.0) or not np.isclose(pi.sum(), 1.0, atol=1e-10):
        raise ValueError("pi must be a valid probability vector")

    T = observations.size
    K = pi.size
    predictions = np.empty(T, dtype=float)
    state_probs = np.empty((T, K), dtype=float)

    # t = 0: no transition yet; update directly from the prior pi
    predictions[0] = float(np.dot(pi, np.asarray(mu, dtype=float)))
    log_pi = np.log(pi)
    log_likelihood0 = -0.5 * np.log(2.0 * np.pi * np.asarray(sigma2, dtype=float)) - 0.5 * (
        (observations[0] - np.asarray(mu, dtype=float)) ** 2
    ) / np.asarray(sigma2, dtype=float)
    log_omega0 = log_pi + log_likelihood0
    log_omega0 -= logsumexp(log_omega0)
    omega = np.exp(log_omega0)
    state_probs[0] = omega

    for t in range(1, T):
        omega, pred = predict_update_step(omega, A, mu, sigma2, observations[t])
        predictions[t] = pred
        state_probs[t] = omega

    return predictions, state_probs
