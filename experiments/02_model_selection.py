"""Experiment 02: HMM model selection via AIC/BIC over K=1..10."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import log_returns
from src.data.loader import extract_close_series, load_daily_prices
from src.hmm.baum_welch import baum_welch
from src.hmm.model_selection import compute_aic, compute_bic

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K_VALUES = list(range(1, 11))
N_RESTARTS = 1
MAX_ITER = 40
TOL = 1e-6
RANDOM_STATE = 42
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


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


def _fit_hmm_with_retries(observations, k, *, max_attempts=20):
    """
    Fit a K-state HMM, retrying with different seeds on numerical failures.

    Increased budget (max_attempts=20) compared to original to handle high-K
    models that are more prone to numerical issues.
    """
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
            print(f"retry {attempt + 1}/{max_attempts}", end=" ", flush=True)

    if best is None:
        print(f"failed after {max_attempts} attempts", end=" ")
        return None

    if successful_restarts < N_RESTARTS:
        print(f"(used {successful_restarts}/{N_RESTARTS} successful restarts)", end=" ")

    return best


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 02: Model Selection ===")
    log(f"Ticker: {TICKER} | Period: {START} to {END}")
    log(
        f"Fitting HMMs for K={K_VALUES[0]}..{K_VALUES[-1]} "
        f"({N_RESTARTS} restarts each, max_iter={MAX_ITER})"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close).to_numpy()
    log(f"Observations: {returns.size} daily log-returns")

    log_likelihoods = {}
    aic_scores = {}
    bic_scores = {}

    for k in K_VALUES:
        t_k = time.time()
        print(f"  Fitting K={k}...", end=" ", flush=True)
        fit_result = _fit_hmm_with_retries(returns, k)
        elapsed_k = time.time() - t_k
        if fit_result is None:
            log_likelihoods[k] = np.nan
            aic_scores[k] = np.nan
            bic_scores[k] = np.nan
            log(f"  K={k}: no stable fit ({elapsed_k:.1f}s)")
            continue

        _, history, _ = fit_result
        ll = float(history[-1])
        aic = float(compute_aic(ll, k))
        bic = float(compute_bic(ll, k, n_obs=returns.size))

        log_likelihoods[k] = ll
        aic_scores[k] = aic
        bic_scores[k] = bic

        log(f"  K={k}: LL={ll:.1f}, AIC={aic:.1f}, BIC={bic:.1f} ({elapsed_k:.1f}s)")

    log("\n--- Model Selection Table ---")
    log(f"{'K':<4}{'LogLik':>10}{'AIC':>12}{'BIC':>12}")
    log("-" * 38)
    for k in K_VALUES:
        log(
            f"{k:<4}{log_likelihoods[k]:>10.1f}  "
            f"{aic_scores[k]:>10.1f}  {bic_scores[k]:>10.1f}"
        )

    valid_aic = {k: v for k, v in aic_scores.items() if np.isfinite(v)}
    valid_bic = {k: v for k, v in bic_scores.items() if np.isfinite(v)}
    if len(valid_aic) == 0 or len(valid_bic) == 0:
        log("\nNo stable model fit found for any K.")
        return 1

    best_aic_k = min(valid_aic, key=valid_aic.get)
    best_bic_k = min(valid_bic, key=valid_bic.get)

    log(f"\nBest K by AIC: {best_aic_k}")
    log(f"Best K by BIC: {best_bic_k}")

    log("\n--- Delta from best ---")
    for k in K_VALUES:
        if np.isfinite(aic_scores[k]):
            delta_aic = aic_scores[k] - valid_aic[best_aic_k]
            delta_bic = bic_scores[k] - valid_bic[best_bic_k]
            log(f"  K={k}: ΔAIC={delta_aic:>8.1f}  ΔBIC={delta_bic:>8.1f}")

    _save_model_selection_figure(
        K_VALUES,
        [aic_scores[k] for k in K_VALUES],
        [bic_scores[k] for k in K_VALUES],
        best_aic_k,
        best_bic_k,
    )

    elapsed = time.time() - t_start
    log(f"\nFigure saved to figures/02_model_selection.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "02_model_selection.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
