"""Rolling-window daily retraining for HMM at 1-min frequency (Paper §2.3, §7).

The paper states (§2.3): "parameter estimation can be done using the previous H
days of market data, when the market is shut."  This implies daily retraining on a
rolling window of H days — not training once and freezing parameters.

Training protocol:
    For each trading day d >= H:
        1. Collect 1-min returns from days [d-H, d-1]  (training window)
        2. Run Baum-Welch on concatenated training returns → Θ_d
        3. Use Θ_d for online inference on day d
        4. Advance to d+1, repeat

This adapts the HMM to current market microstructure, preventing parameter
staleness that caused Sharpe -1.22 with frozen parameters.
"""

from __future__ import annotations

import numpy as np


def split_by_day(
    returns: np.ndarray,
    day_indices: np.ndarray,
) -> list[np.ndarray]:
    """Split a flat returns array into per-day arrays.

    Parameters
    ----------
    returns : (T,) 1-min log-returns.
    day_indices : (T,) integer day label for each return (e.g. 0, 0, ..., 1, 1, ...).

    Returns
    -------
    daily_arrays : list of 1-D arrays, one per unique day (sorted by day index).
    """
    if len(returns) != len(day_indices):
        raise ValueError(
            f"returns and day_indices must have same length, "
            f"got {len(returns)} and {len(day_indices)}"
        )
    unique_days = np.unique(day_indices)
    return [returns[day_indices == d] for d in unique_days]


def rolling_hmm(
    daily_returns: list[np.ndarray],
    K: int,
    H: int,
    train_fn,
    inference_fn,
    *,
    min_variance: float = 1e-8,
    n_restarts: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Rolling-window daily retraining for HMM (Paper §2.3, §7).

    For each trading day d >= H:
        training_data = concat(daily_returns[d-H : d])
        Θ_d = BaumWelch(training_data, K)
        predictions_d, states_d = Inference(daily_returns[d], Θ_d)

    Parameters
    ----------
    daily_returns : list of (N_d,) arrays, one per trading day.
        Each array contains the 1-min log-returns for that day.
    K : int
        Number of hidden states.
    H : int
        Lookback window in trading days.
    train_fn : callable
        Baum-Welch training function with signature:
            train_fn(obs, K, n_restarts=..., max_iter=..., tol=...,
                     min_variance=..., random_state=...) -> (params_dict, ll_history)
    inference_fn : callable
        Online inference function with signature:
            inference_fn(obs, A, pi, mu, sigma2) -> (predictions, state_probs)
    min_variance : float
        Variance floor for Baum-Welch.
    n_restarts : int
        Number of random restarts per daily training.
    max_iter : int
        Max EM iterations per restart.
    tol : float
        EM convergence tolerance.
    verbose : bool
        Print per-day progress.

    Returns
    -------
    result : dict with keys:
        "predictions" : (T_test,) concatenated one-step-ahead predictions.
        "state_probs" : (T_test, K) concatenated filtered state posteriors.
        "daily_params" : list of D_test dicts, each with "A", "pi", "mu", "sigma2".
        "daily_ll" : list of D_test floats, training LL for each day.
        "n_train_days" : int, number of days used for first training window.
        "n_test_days" : int, number of days with predictions.
    """
    D = len(daily_returns)
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    if H >= D:
        raise ValueError(
            f"H ({H}) must be less than total days ({D})"
        )
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    all_predictions = []
    all_state_probs = []
    daily_params = []
    daily_ll = []

    for d in range(H, D):
        # Training window: days [d-H, d-1]
        train_obs = np.concatenate(daily_returns[d - H : d])

        if len(train_obs) == 0:
            continue

        # Train
        try:
            params, ll_hist = train_fn(
                train_obs,
                K,
                n_restarts=n_restarts,
                max_iter=max_iter,
                tol=tol,
                min_variance=min_variance,
                random_state=d,  # different seed per day for diversity
            )
        except RuntimeError:
            # All restarts failed — skip this day
            if verbose:
                print(f"  Day {d}/{D}: training failed, skipping")
            continue

        daily_params.append(params)
        daily_ll.append(ll_hist[-1] if ll_hist else float("nan"))

        # Inference on day d
        test_obs = daily_returns[d]
        if len(test_obs) == 0:
            continue

        preds, sprobs = inference_fn(
            test_obs, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        all_predictions.append(preds)
        all_state_probs.append(sprobs)

        if verbose and (d - H) % 50 == 0:
            print(
                f"  Day {d}/{D}: train LL={daily_ll[-1]:.1f}, "
                f"μ={params['mu']}, test bars={len(test_obs)}"
            )

    if not all_predictions:
        raise RuntimeError("No days produced predictions")

    return {
        "predictions": np.concatenate(all_predictions),
        "state_probs": np.vstack(all_state_probs),
        "daily_params": daily_params,
        "daily_ll": daily_ll,
        "n_train_days": H,
        "n_test_days": len(daily_params),
    }
