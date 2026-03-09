"""Shared HMM training utilities for experiment scripts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.hmm.baum_welch import baum_welch


EPS: float = 1e-12


def sort_states(params: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
    """
    Sort HMM states in ascending order of emission mean.

    After reordering, transition probabilities and initial distribution are
    clipped to ``EPS`` and renormalized so that downstream inference routines
    never encounter zero-probability entries.

    Parameters:
        params: dict with keys "A", "pi", "mu", "sigma2".

    Returns:
        New dict with the same keys, states reordered by ``mu``.
    """
    order = np.argsort(params["mu"])
    sorted_params = {
        "A": params["A"][np.ix_(order, order)],
        "pi": params["pi"][order],
        "mu": params["mu"][order],
        "sigma2": params["sigma2"][order],
    }

    A = np.clip(sorted_params["A"], EPS, None)
    sorted_params["A"] = A / A.sum(axis=1, keepdims=True)

    pi = np.clip(sorted_params["pi"], EPS, None)
    sorted_params["pi"] = pi / pi.sum()

    sorted_params["sigma2"] = np.clip(sorted_params["sigma2"], EPS, None)
    return sorted_params


def train_best_model(
    observations: NDArray[np.floating],
    K: int,
    *,
    successful_restarts: int = 10,
    max_attempts: int = 150,
    max_iter: int = 200,
    tol: float = 1e-6,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, NDArray[np.floating]], list[float], NDArray[np.floating]]:
    """
    Collect *successful_restarts* single-restart EM fits and keep the best.

    Each attempt runs ``baum_welch`` with ``n_restarts=1`` and a unique seed.
    If a restart raises a ``ValueError`` mentioning "strictly positive entries"
    (a known numerical edge-case), it is silently skipped.

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        K: int
            Number of hidden states.
        successful_restarts: int
            Target number of restarts that converge without error.
        max_attempts: int
            Upper bound on total attempts (including failures).
        max_iter: int
            Maximum EM iterations per restart.
        tol: float
            EM convergence tolerance.
        random_state: int
            Base random seed; each attempt uses ``random_state + attempt``.
        verbose: bool
            If True, print progress for each successful restart.

    Returns:
        (params, history, gamma) for the restart with highest final
        log-likelihood.

    Raises:
        RuntimeError: If no restart succeeds within *max_attempts*.
    """
    best = None
    best_ll = -np.inf
    successes = 0

    for attempt in range(max_attempts):
        if successes >= successful_restarts:
            break

        seed = random_state + attempt
        try:
            params, history, gamma = baum_welch(
                observations,
                K=K,
                max_iter=max_iter,
                tol=tol,
                n_restarts=1,
                random_state=seed,
            )
        except ValueError as exc:
            if "strictly positive entries" not in str(exc):
                raise
            continue

        successes += 1
        ll = float(history[-1])
        if verbose:
            print(f"  successful restart {successes}/{successful_restarts} (LL={ll:.1f})")

        if ll > best_ll:
            best_ll = ll
            best = (params, history, gamma)

    if best is None:
        raise RuntimeError(
            "Training failed: no successful restart produced valid parameters"
        )

    if successes < successful_restarts:
        print(
            f"Warning: used {successes}/{successful_restarts} successful restarts "
            f"after {max_attempts} attempts"
        )

    return best
