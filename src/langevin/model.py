"""Langevin state-space model discretization (2012 Paper §II-A)."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


def discretize_langevin(
    theta: float, sigma: float, dt: float
) -> tuple[NDArray, NDArray]:
    """Discretize the continuous-time Langevin SDE (2012 Paper §II-A, Eq 1-8).

    The continuous-time model is (Eq 1-3):

        dx1 = x2 dt                        (price = integral of trend)
        dx2 = theta * x2 dt + sigma * dW   (trend follows Ornstein-Uhlenbeck)

    In matrix form:

        dx = A x dt + b dW
        A = [[0, 1],      b = [[0    ],
             [0, theta]]       [sigma]]

    Integrating from t to t+dt gives the discrete transition (Eq 4-8):

        x_{t+dt} = F x_t + w_t,   w_t ~ N(0, Q)

    where:

        F = exp(A * dt)                                      (Eq 5)
        Q = integral_0^dt exp(A*s) b b' exp(A'*s) ds         (Eq 6)

    Q is computed via Van Loan's matrix fraction decomposition (Eq 7-8):
    construct the block matrix H = [[-A, b*b'], [0, A']] * dt, compute
    exp(H), then Q = F @ (upper-right block of exp(H)).

    Parameters
    ----------
    theta : float
        Mean-reversion parameter (theta < 0 for stable dynamics).
        Controls how quickly the trend reverts to zero.
    sigma : float
        Diffusion coefficient of the trend process.
        Controls the magnitude of random fluctuations in the trend.
    dt : float
        Discretization timestep (e.g. 1.0 for daily data).

    Returns
    -------
    F : np.ndarray, shape (2, 2)
        Discrete-time state transition matrix.
    Q : np.ndarray, shape (2, 2)
        Discrete-time process noise covariance (symmetric, PSD).
    """
    if theta > 0:
        raise ValueError(
            f"theta must be <= 0 for stable dynamics, got theta={theta}"
        )

    A = np.array([[0.0, 1.0], [0.0, theta]])
    b = np.array([[0.0], [sigma]])
    bbT = b @ b.T  # shape (2, 2), = [[0, 0], [0, sigma^2]]

    # --- F via matrix exponential ---
    F = expm(A * dt)

    # --- Q via Van Loan's method ---
    # Build 4x4 block matrix H = [[-A, bbT], [0, A']] * dt
    H = np.zeros((4, 4))
    H[:2, :2] = -A * dt
    H[:2, 2:] = bbT * dt
    H[2:, 2:] = A.T * dt

    eH = expm(H)
    # Upper-right block of exp(H) = F^{-1} Q, so Q = F @ eH[:2, 2:]
    Q = F @ eH[:2, 2:]

    # Enforce exact symmetry (numerical round-off can break it)
    Q = (Q + Q.T) / 2.0

    return F, Q


def observation_matrix() -> NDArray:
    """Return the observation matrix G for the Langevin model (2012 Paper §II-B, Eq 18-19, 23).

    The observation equation is:

        y_t = G @ x_t + v_t,   v_t ~ N(0, sigma_obs^2)

    where x_t = [x1_t, x2_t]' (price, trend) and we observe only the price:

        G = [1, 0]

    So y_t = x1_t + v_t (observed price = true price + noise).

    Returns
    -------
    G : np.ndarray, shape (1, 2)
        Observation matrix. G @ x extracts the price component.
    """
    return np.array([[1.0, 0.0]])


def transition_density_with_jump(
    x_prev: NDArray,
    theta: float,
    sigma: float,
    dt: float,
    jump_occurred: bool,
    tau: float = 0.0,
    mu_J: float = 0.0,
    sigma_J: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """State transition density conditioned on jump occurrence (2012 Paper §II-A, Eq 10-16).

    Computes the conditional mean and covariance of x_{t+dt} given x_t.

    Case 1 — No jump (Eq 4-6):

        x_{t+dt} ~ N(F x_t, Q)

        where F, Q come from discretize_langevin(theta, sigma, dt).

    Case 2 — Jump at time tau after t (Eq 10-16):

        The state evolves as:
          1. Diffuse from t to t+tau:         F1 = exp(A*tau),   Q1
          2. Jump perturbation at t+tau:       add [0, J]' to trend, J ~ N(mu_J, sigma_J^2)
          3. Diffuse from t+tau to t+dt:       F2 = exp(A*(dt-tau)), Q2

        Combined mean:
            E[x_{t+dt}] = F x_t + F2 @ [0, mu_J]'                  (Eq 14)

        Combined covariance:
            Cov = F2 Q1 F2' + F2 [[0,0],[0,sigma_J^2]] F2' + Q2    (Eq 15)

        where F = F2 @ F1 (the full-interval transition).

    Parameters
    ----------
    x_prev : np.ndarray, shape (2,)
        Previous state [price, trend].
    theta : float
        Mean-reversion parameter.
    sigma : float
        Diffusion coefficient.
    dt : float
        Full timestep.
    jump_occurred : bool
        Whether a jump occurs during [t, t+dt].
    tau : float
        Time offset of jump within the interval [0, dt]. Only used if jump_occurred.
    mu_J : float
        Mean of jump size in the trend component. Only used if jump_occurred.
    sigma_J : float
        Std of jump size in the trend component. Only used if jump_occurred.

    Returns
    -------
    mean : np.ndarray, shape (2,)
        Conditional mean E[x_{t+dt} | x_t, jump info].
    cov : np.ndarray, shape (2, 2)
        Conditional covariance Cov[x_{t+dt} | x_t, jump info].
    """
    if not jump_occurred:
        F, Q = discretize_langevin(theta, sigma, dt)
        mean = F @ x_prev
        return mean, Q

    # Jump at time tau within [0, dt]
    if not (0.0 <= tau <= dt):
        raise ValueError(f"tau must be in [0, dt={dt}], got tau={tau}")

    # Pre-jump diffusion: t to t+tau
    F1, Q1 = discretize_langevin(theta, sigma, tau)
    # Post-jump diffusion: t+tau to t+dt
    dt2 = dt - tau
    F2, Q2 = discretize_langevin(theta, sigma, dt2)

    # Full transition F = F2 @ F1
    F = F2 @ F1

    # Mean: F x_prev + F2 @ [0, mu_J]'
    jump_mean_vec = np.array([0.0, mu_J])
    mean = F @ x_prev + F2 @ jump_mean_vec

    # Covariance: F2 Q1 F2' + F2 diag(0, sigma_J^2) F2' + Q2
    jump_cov = np.array([[0.0, 0.0], [0.0, sigma_J**2]])
    cov = F2 @ Q1 @ F2.T + F2 @ jump_cov @ F2.T + Q2
    cov = (cov + cov.T) / 2.0  # enforce symmetry

    return mean, cov
