"""Experiment 04: regime detection via Viterbi and forward-backward posteriors."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import log_returns
from src.data.loader import load_daily_prices
from src.hmm.baum_welch import baum_welch
from src.hmm.forward_backward import compute_posteriors
from src.hmm.viterbi import viterbi
from src.utils.plotting import plot_regime_colored_prices

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
MAX_ITER = 200
TOL = 1e-6
N_RESTARTS = 10
RANDOM_STATE = 42
EPS = 1e-12
FIGURES_DIR = Path("figures")


def _extract_close_series(prices):
    if "Close" in prices.columns:
        close = prices["Close"]
    elif "Adj Close" in prices.columns:
        close = prices["Adj Close"]
    else:
        raise ValueError("Expected 'Close' or 'Adj Close' in downloaded data")

    if hasattr(close, "ndim") and close.ndim != 1:
        close = close.iloc[:, 0]

    return close.astype(float)


def _sort_states(params):
    order = np.argsort(params["mu"])
    sorted_params = {
        "A": params["A"][np.ix_(order, order)],
        "pi": params["pi"][order],
        "mu": params["mu"][order],
        "sigma2": params["sigma2"][order],
    }

    # Keep strictly positive probabilities to satisfy inference validation.
    A = np.clip(sorted_params["A"], EPS, None)
    sorted_params["A"] = A / A.sum(axis=1, keepdims=True)

    pi = np.clip(sorted_params["pi"], EPS, None)
    sorted_params["pi"] = pi / pi.sum()

    sorted_params["sigma2"] = np.clip(sorted_params["sigma2"], EPS, None)
    return sorted_params


def _train_best_model(observations, *, successful_restarts, max_attempts=150):
    best = None
    best_ll = -np.inf
    successes = 0

    for attempt in range(max_attempts):
        if successes >= successful_restarts:
            break

        seed = RANDOM_STATE + attempt
        try:
            params, history, gamma = baum_welch(
                observations,
                K=K,
                max_iter=MAX_ITER,
                tol=TOL,
                n_restarts=1,
                random_state=seed,
            )
        except ValueError as exc:
            if "strictly positive entries" not in str(exc):
                raise
            continue

        successes += 1
        ll = float(history[-1])
        if ll > best_ll:
            best_ll = ll
            best = (params, history, gamma)

    if best is None:
        raise RuntimeError("Training failed: no successful restart produced valid parameters")

    if successes < successful_restarts:
        print(
            f"Warning: used {successes}/{successful_restarts} successful restarts "
            f"after {max_attempts} attempts"
        )

    return best


def _average_durations(states, k_states):
    durations = [[] for _ in range(k_states)]
    run_state = int(states[0])
    run_len = 1

    for current in states[1:]:
        current = int(current)
        if current == run_state:
            run_len += 1
        else:
            durations[run_state].append(run_len)
            run_state = current
            run_len = 1
    durations[run_state].append(run_len)

    return [float(np.mean(d)) if len(d) > 0 else 0.0 for d in durations]


def _save_regime_price_figure(aligned_prices, states):
    fig, ax = plot_regime_colored_prices(
        aligned_prices.to_numpy(),
        states,
        title="SPY Price Colored by HMM Regime (Viterbi, K=3)",
    )
    ax.set_xlabel("Time index")
    fig.savefig(FIGURES_DIR / "04_regime_prices.png", dpi=150)
    plt.close(fig)


def _save_state_posterior_figure(returns, gamma):
    fig, axes = plt.subplots(K, 1, figsize=(12, 2.6 * K), sharex=True)
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        ax.plot(returns.index, gamma[:, k], linewidth=1.1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"P(S={k})")
        ax.grid(alpha=0.3)

    axes[0].set_title("State Posterior Probabilities (Forward-Backward)")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_state_posteriors.png", dpi=150)
    plt.close(fig)


def main():
    print("=== Experiment 04: Regime Detection ===")
    print("Training HMM and running Viterbi decoding...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        print(f"Data load failed: {exc}")
        return 1

    close = _extract_close_series(prices)
    returns = log_returns(close)
    returns_values = returns.to_numpy()

    params, _, _ = _train_best_model(returns_values, successful_restarts=N_RESTARTS)
    params = _sort_states(params)

    A = params["A"]
    pi = params["pi"]
    mu = params["mu"]
    sigma2 = params["sigma2"]

    states, log_prob = viterbi(returns_values, A, pi, mu, sigma2)
    gamma, _, ll_fb = compute_posteriors(returns_values, A, pi, mu, sigma2)

    counts = np.bincount(states, minlength=K)
    total_days = len(states)
    durations = _average_durations(states, K)

    print("\n--- Regime Statistics (Viterbi) ---")
    labels = ["bearish", "neutral", "bullish"] if K == 3 else [f"state-{i}" for i in range(K)]
    for k in range(K):
        mask = states == k
        if np.any(mask):
            mean_k = float(np.mean(returns_values[mask]))
            vol_k = float(np.std(returns_values[mask], ddof=0))
        else:
            mean_k = 0.0
            vol_k = 0.0

        pct = 100.0 * counts[k] / total_days
        print(
            f"  State {k} ({labels[k]}): {counts[k]:4d} days ({pct:5.1f}%), "
            f"mean return = {mean_k * 100: .3f}%, vol = {vol_k * 100: .3f}%"
        )

    print("\nAverage regime durations:")
    for k in range(K):
        print(f"  State {k}: {durations[k]:.2f} days")

    print(f"\nViterbi path log-probability: {log_prob:.2f}")
    print(f"Forward-backward log-likelihood: {ll_fb:.2f}")

    # Align prices with returns index: return_t corresponds to close[t] - close[t-1].
    aligned_prices = close.iloc[1:]
    _save_regime_price_figure(aligned_prices, states)
    _save_state_posterior_figure(returns, gamma)

    print("\nFigures saved to figures/04_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
