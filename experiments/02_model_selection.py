"""Experiment 02: HMM model selection via AIC/BIC over K=1..10."""

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
from src.hmm.model_selection import compute_aic, compute_bic

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K_VALUES = list(range(1, 11))
N_RESTARTS = 5
MAX_ITER = 100
TOL = 1e-6
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


def _save_model_selection_figure(k_values, aic_scores, bic_scores, best_aic_k, best_bic_k):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_values, aic_scores, marker="o", linewidth=2.0, label="AIC")
    ax.plot(k_values, bic_scores, marker="s", linewidth=2.0, label="BIC")

    ax.axvline(best_aic_k, linestyle="--", linewidth=1.0, alpha=0.7, color="tab:blue")
    ax.axvline(best_bic_k, linestyle="--", linewidth=1.0, alpha=0.7, color="tab:orange")

    ax.set_title("Model Selection: AIC and BIC vs Number of States")
    ax.set_xlabel("Number of states (K)")
    ax.set_ylabel("Information criterion")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_model_selection.png", dpi=150)
    plt.close(fig)


def _fit_hmm_with_retries(observations, k, *, max_attempts=80):
    """
    Fit a K-state HMM, retrying with different seeds on numerical failures.

    The current core implementation may occasionally hit a strict-positivity
    check during EM for unlucky random initializations.
    """
    last_error = None
    best = None
    best_ll = -np.inf
    successful_restarts = 0

    for attempt in range(max_attempts):
        if successful_restarts >= N_RESTARTS:
            break

        seed = RANDOM_STATE + (100 * k) + attempt
        try:
            params, history, gamma = baum_welch(
                observations,
                K=k,
                max_iter=MAX_ITER,
                tol=TOL,
                n_restarts=1,
                random_state=seed,
            )
            successful_restarts += 1
            ll = float(history[-1])
            if ll > best_ll:
                best_ll = ll
                best = (params, history, gamma)
        except ValueError as exc:
            if "strictly positive entries" not in str(exc):
                raise
            last_error = exc
            print(f"retry {attempt + 1}/{max_attempts}", end=" ", flush=True)

    if best is None:
        raise RuntimeError(
            f"Failed to fit K={k} after {max_attempts} attempts due to numerical instability"
        ) from last_error

    if successful_restarts < N_RESTARTS:
        print(f"(used {successful_restarts}/{N_RESTARTS} successful restarts)", end=" ")

    return best


def main():
    print("=== Experiment 02: Model Selection ===")
    print(f"Ticker: {TICKER} | Period: {START} to {END}")
    print(
        f"Fitting HMMs for K={K_VALUES[0]}..{K_VALUES[-1]} "
        f"({N_RESTARTS} restarts each, max_iter={MAX_ITER})"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        print(f"Data load failed: {exc}")
        return 1

    close = _extract_close_series(prices)
    returns = log_returns(close).to_numpy()

    log_likelihoods = {}
    aic_scores = {}
    bic_scores = {}

    for k in K_VALUES:
        print(f"  Fitting K={k}...", end=" ", flush=True)
        _, history, _ = _fit_hmm_with_retries(returns, k)
        ll = float(history[-1])
        aic = float(compute_aic(ll, k))
        bic = float(compute_bic(ll, k, n_obs=returns.size))

        log_likelihoods[k] = ll
        aic_scores[k] = aic
        bic_scores[k] = bic

        print(f"done (LL={ll:.1f})")

    print("\nK   LogLik      AIC         BIC")
    print("-" * 36)
    for k in K_VALUES:
        print(
            f"{k:<2d}  {log_likelihoods[k]:>8.1f}  "
            f"{aic_scores[k]:>10.1f}  {bic_scores[k]:>10.1f}"
        )

    best_aic_k = min(aic_scores, key=aic_scores.get)
    best_bic_k = min(bic_scores, key=bic_scores.get)

    print(f"\nBest K by AIC: {best_aic_k}")
    print(f"Best K by BIC: {best_bic_k}")

    _save_model_selection_figure(
        K_VALUES,
        [aic_scores[k] for k in K_VALUES],
        [bic_scores[k] for k in K_VALUES],
        best_aic_k,
        best_bic_k,
    )

    print("Figure saved to figures/02_model_selection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
