"""Model selection utilities for Gaussian HMMs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.hmm.baum_welch import baum_welch


def _num_parameters(K: int) -> int:
    """Number of free parameters in K-state Gaussian HMM with 1D emissions."""
    # Transition: K rows each with K-1 free params => K*(K-1)
    # Means: K
    # Variances: K
    # Initial distribution: K-1
    return K * (K - 1) + 2 * K + (K - 1)


def compute_aic(log_likelihood: float, K: int) -> float:
    """
    Compute Akaike Information Criterion for a K-state HMM.

    AIC = -2 log(L) + 2p, where p is number of free parameters.
    """
    p = _num_parameters(int(K))
    return float(-2.0 * log_likelihood + 2.0 * p)


def compute_bic(log_likelihood: float, K: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion for a K-state HMM.

    BIC = -2 log(L) + p log(n), where p is number of free parameters.
    """
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    p = _num_parameters(int(K))
    return float(-2.0 * log_likelihood + p * np.log(n_obs))


def select_K(
    observations: NDArray[np.floating],
    K_range: Any = range(1, 11),
    criterion: str = "bic",
    max_iter: int = 100,
    tol: float = 1e-6,
    n_restarts: int = 5,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Fit HMMs across K values and select best K by AIC or BIC.

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        K_range: iterable[int]
            Candidate hidden-state counts.
        criterion: str
            Either "aic" or "bic".
        max_iter, tol, n_restarts, random_state:
            Passed to baum_welch().

    Returns:
        result: dict with keys:
            - "best_K": int
            - "criterion": str
            - "scores": dict[K -> float]
            - "log_likelihoods": dict[K -> float]
            - "models": dict[K -> params-dict]
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be either 'aic' or 'bic'")

    K_values = [int(k) for k in K_range]
    if len(K_values) == 0 or any(k < 1 for k in K_values):
        raise ValueError("K_range must contain integers >= 1")

    scores = {}
    log_likelihoods = {}
    models = {}

    for idx, K in enumerate(K_values):
        seed = None if random_state is None else int(random_state) + idx
        params, history, _ = baum_welch(
            observations,
            K=K,
            max_iter=max_iter,
            tol=tol,
            n_restarts=n_restarts,
            random_state=seed,
        )
        ll = history[-1]
        log_likelihoods[K] = float(ll)
        models[K] = params
        if criterion == "aic":
            scores[K] = compute_aic(ll, K)
        else:
            scores[K] = compute_bic(ll, K, n_obs=observations.size)

    best_K = min(scores, key=scores.get)
    return {
        "best_K": int(best_K),
        "criterion": criterion,
        "scores": scores,
        "log_likelihoods": log_likelihoods,
        "models": models,
    }
