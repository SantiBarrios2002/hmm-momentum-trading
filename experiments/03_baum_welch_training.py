"""Experiment 03: Baum-Welch training and parameter interpretation."""

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

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
MAX_ITER = 200
TOL = 1e-6
N_RESTARTS = 10
RANDOM_STATE = 42
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
    return {
        "A": params["A"][np.ix_(order, order)],
        "pi": params["pi"][order],
        "mu": params["mu"][order],
        "sigma2": params["sigma2"][order],
    }


def _train_best_model(observations, *, successful_restarts, max_attempts=150):
    """Collect successful single-restart fits and keep the best LL."""
    best_params = None
    best_history = None
    best_gamma = None
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
            print(f"  restart attempt {attempt + 1}: numerical retry")
            continue

        ll = float(history[-1])
        successes += 1
        print(f"  successful restart {successes}/{successful_restarts} (LL={ll:.1f})")

        if ll > best_ll:
            best_ll = ll
            best_params = params
            best_history = history
            best_gamma = gamma

    if best_params is None:
        raise RuntimeError("Training failed: no successful restart produced valid parameters")

    if successes < successful_restarts:
        print(
            f"Warning: used {successes}/{successful_restarts} successful restarts "
            f"after {max_attempts} attempts"
        )

    return best_params, best_history, best_gamma


def _format_transition_table(A):
    lines = []
    header = "         " + "  ".join([f"State {j}" for j in range(A.shape[1])])
    lines.append(header)
    for i in range(A.shape[0]):
        row = "  ".join([f"{A[i, j]:.3f}" for j in range(A.shape[1])])
        lines.append(f"State {i}  {row}")
    return "\n".join(lines)


def _save_convergence_figure(history):
    iterations = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iterations, history, marker="o", linewidth=1.6, markersize=3)
    ax.set_title("Baum-Welch EM Convergence (K=3)")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_em_convergence.png", dpi=150)
    plt.close(fig)


def main():
    print("=== Experiment 03: Baum-Welch Training ===")
    print(
        f"Training {K}-state HMM on {TICKER} returns "
        f"({N_RESTARTS} successful random restarts target)..."
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        print(f"Data load failed: {exc}")
        return 1

    close = _extract_close_series(prices)
    returns = log_returns(close).to_numpy()

    params, history, _ = _train_best_model(returns, successful_restarts=N_RESTARTS)
    params = _sort_states(params)

    A = params["A"]
    pi = params["pi"]
    mu = params["mu"]
    sigma = np.sqrt(params["sigma2"])

    ll_initial = float(history[0])
    ll_final = float(history[-1])
    improvement = ll_final - ll_initial
    durations = 1.0 / np.maximum(1.0 - np.diag(A), 1e-12)

    state_labels = ["bearish", "neutral", "bullish"] if K == 3 else [f"state-{i}" for i in range(K)]

    print(f"Converged after {len(history)} EM iterations.")

    print("\n--- Learned Parameters ---")
    print("Transition matrix A:")
    print(_format_transition_table(A))
    print("\nInitial distribution pi:")
    print(np.array2string(pi, precision=4, suppress_small=False))

    print("\nEmission parameters:")
    for i in range(K):
        print(
            f"  State {i} ({state_labels[i]}): "
            f"mu = {mu[i]: .6f}, sigma = {sigma[i]: .6f}"
        )

    print("\nExpected regime durations (days):")
    for i in range(K):
        print(f"  State {i}: {durations[i]:.2f} days")

    print(
        f"\nLog-likelihood: {ll_final:.1f} "
        f"(initial: {ll_initial:.1f}, improvement: {improvement:.1f})"
    )

    _save_convergence_figure(history)
    print("Figure saved to figures/03_em_convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
