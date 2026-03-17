"""Kalman filter for linear Gaussian state-space models (2012 Paper §III-A)."""

import numpy as np
from numpy.typing import NDArray


def kalman_predict(
    mu_prev: NDArray, C_prev: NDArray, F: NDArray, Q: NDArray
) -> tuple[NDArray, NDArray]:
    """One-step Kalman prediction (2012 Paper §III-A, Eq 29-30).

    Propagates the state estimate through the transition model:

        mu_pred = F @ mu_prev                       (Eq 29)
        C_pred  = F @ C_prev @ F' + Q               (Eq 30)

    This is the "time update" step: given the filtered estimate at t,
    compute the predicted estimate at t+1 before seeing the observation.

    Parameters
    ----------
    mu_prev : np.ndarray, shape (2,)
        Filtered state mean at time t. [price, trend].
    C_prev : np.ndarray, shape (2, 2)
        Filtered state covariance at time t.
    F : np.ndarray, shape (2, 2)
        State transition matrix from discretize_langevin.
    Q : np.ndarray, shape (2, 2)
        Process noise covariance from discretize_langevin.

    Returns
    -------
    mu_pred : np.ndarray, shape (2,)
        Predicted state mean at time t+1.
    C_pred : np.ndarray, shape (2, 2)
        Predicted state covariance at time t+1 (symmetric, PSD).
    """
    mu_pred = F @ mu_prev
    C_pred = F @ C_prev @ F.T + Q
    C_pred = (C_pred + C_pred.T) / 2.0  # enforce symmetry
    return mu_pred, C_pred


def kalman_update(
    mu_pred: NDArray,
    C_pred: NDArray,
    G: NDArray,
    sigma_obs_sq: float,
    observation: float,
) -> tuple[NDArray, NDArray, float]:
    """One-step Kalman update with PED log-likelihood (2012 Paper §III-A, Eq 26-33).

    Incorporates a new observation to refine the state estimate:

        S   = G @ C_pred @ G' + sigma_obs^2       innovation covariance  (Eq 31)
        K   = C_pred @ G' / S                      Kalman gain            (Eq 32)
        mu  = mu_pred + K * (y - G @ mu_pred)      filtered mean          (Eq 33)
        C   = (I - K @ G) @ C_pred                 filtered covariance    (Eq 33)

    The prediction error decomposition (PED) log-likelihood is:

        log p(y_t | y_{1:t-1}) = log N(y_t; G @ mu_pred, S)             (Eq 26)

    This is the quantity used as particle weights in the RBPF (each particle
    runs its own Kalman filter, and the PED gives the marginal likelihood
    of the observation under that particle's jump history).

    Parameters
    ----------
    mu_pred : np.ndarray, shape (2,)
        Predicted state mean (from kalman_predict).
    C_pred : np.ndarray, shape (2, 2)
        Predicted state covariance (from kalman_predict).
    G : np.ndarray, shape (1, 2)
        Observation matrix. G @ x extracts the observed component.
    sigma_obs_sq : float
        Observation noise variance.
    observation : float
        Observed value y_t.

    Returns
    -------
    mu_new : np.ndarray, shape (2,)
        Filtered state mean at time t.
    C_new : np.ndarray, shape (2, 2)
        Filtered state covariance at time t (symmetric, PSD).
    log_likelihood : float
        PED log-likelihood log p(y_t | y_{1:t-1}).
    """
    # Innovation (prediction error)
    y_pred = (G @ mu_pred).item()  # scalar
    innovation = observation - y_pred

    # Innovation covariance (scalar for 1-D observation)
    S = (G @ C_pred @ G.T).item() + sigma_obs_sq

    # Kalman gain: shape (2, 1) then squeeze to (2,)
    K = (C_pred @ G.T) / S  # shape (2, 1)
    K = K.ravel()  # shape (2,)

    # Filtered state
    mu_new = mu_pred + K * innovation
    C_new = (np.eye(2) - np.outer(K, G.ravel())) @ C_pred
    C_new = (C_new + C_new.T) / 2.0  # enforce symmetry

    # PED log-likelihood: log N(y; y_pred, S)
    log_likelihood = -0.5 * np.log(2.0 * np.pi * S) - 0.5 * innovation**2 / S

    return mu_new, C_new, log_likelihood


def kalman_filter(
    observations: NDArray,
    F: NDArray,
    Q: NDArray,
    G: NDArray,
    sigma_obs_sq: float,
    mu0: NDArray,
    C0: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, float]:
    """Run a full Kalman filter over a sequence of observations (2012 Paper §III-A).

    For each timestep t = 0, ..., T-1:
        1. Predict:  mu_pred, C_pred = kalman_predict(mu_{t-1}, C_{t-1}, F, Q)
        2. Update:   mu_t, C_t, ll_t = kalman_update(mu_pred, C_pred, G, sigma_obs^2, y_t)

    The total log-likelihood is the sum of per-step PED terms:

        log p(y_{1:T}) = sum_{t=1}^{T} log p(y_t | y_{1:t-1})

    This is the exact marginal likelihood for the linear Gaussian model,
    and is used in the RBPF to weight particles by their observation fit.

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Observed values y_1, ..., y_T.
    F : np.ndarray, shape (2, 2)
        State transition matrix.
    Q : np.ndarray, shape (2, 2)
        Process noise covariance.
    G : np.ndarray, shape (1, 2)
        Observation matrix.
    sigma_obs_sq : float
        Observation noise variance.
    mu0 : np.ndarray, shape (2,)
        Prior state mean.
    C0 : np.ndarray, shape (2, 2)
        Prior state covariance.

    Returns
    -------
    predicted_means : np.ndarray, shape (T, 2)
        Predicted state means (before incorporating observation at each step).
    predicted_covs : np.ndarray, shape (T, 2, 2)
        Predicted state covariances.
    filtered_means : np.ndarray, shape (T, 2)
        Filtered state means (after incorporating observation at each step).
    filtered_covs : np.ndarray, shape (T, 2, 2)
        Filtered state covariances.
    log_likelihoods : np.ndarray, shape (T,)
        Per-step PED log-likelihoods.
    total_log_likelihood : float
        Sum of per-step log-likelihoods = log p(y_{1:T}).
    """
    T = len(observations)
    predicted_means = np.zeros((T, 2))
    predicted_covs = np.zeros((T, 2, 2))
    filtered_means = np.zeros((T, 2))
    filtered_covs = np.zeros((T, 2, 2))
    log_likelihoods = np.zeros(T)

    mu = mu0.copy()
    C = C0.copy()

    for t in range(T):
        # Predict
        mu_pred, C_pred = kalman_predict(mu, C, F, Q)
        predicted_means[t] = mu_pred
        predicted_covs[t] = C_pred

        # Update
        mu, C, ll = kalman_update(mu_pred, C_pred, G, sigma_obs_sq, observations[t])
        filtered_means[t] = mu
        filtered_covs[t] = C
        log_likelihoods[t] = ll

    total_log_likelihood = np.sum(log_likelihoods)
    return (
        predicted_means,
        predicted_covs,
        filtered_means,
        filtered_covs,
        log_likelihoods,
        total_log_likelihood,
    )
