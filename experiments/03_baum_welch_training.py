"""Experiment 03: Baum-Welch training and parameter interpretation."""

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
from src.hmm.utils import sort_states, train_best_model

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
MAX_ITER = 200
TOL = 1e-6
N_RESTARTS = 10
RANDOM_STATE = 42
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _format_transition_table(A):
    lines = []
    header = "         " + "  ".join([f"State {j}" for j in range(A.shape[1])])
    lines.append(header)
    for i in range(A.shape[0]):
        row = "  ".join([f"{A[i, j]:.4f}" for j in range(A.shape[1])])
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
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 03: Baum-Welch Training ===")
    log(
        f"Training {K}-state HMM on {TICKER} returns "
        f"({N_RESTARTS} successful random restarts target)..."
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

    t_train = time.time()
    params, history, _ = train_best_model(
        returns,
        K,
        successful_restarts=N_RESTARTS,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    params = sort_states(params)
    elapsed_train = time.time() - t_train
    log(f"Training completed in {elapsed_train:.1f}s")

    A = params["A"]
    pi = params["pi"]
    mu = params["mu"]
    sigma = np.sqrt(params["sigma2"])

    ll_initial = float(history[0])
    ll_final = float(history[-1])
    improvement = ll_final - ll_initial
    durations = 1.0 / np.maximum(1.0 - np.diag(A), 1e-12)

    state_labels = ["bearish", "neutral", "bullish"] if K == 3 else [f"state-{i}" for i in range(K)]

    log(f"Converged after {len(history)} EM iterations.")

    log("\n--- Learned Parameters ---")
    log("Transition matrix A:")
    log(_format_transition_table(A))
    log("\nInitial distribution pi:")
    log(np.array2string(pi, precision=4, suppress_small=False))

    log("\nEmission parameters:")
    for i in range(K):
        drift_mu = mu[i] * 252
        ann_sigma = sigma[i] * np.sqrt(252)
        log(
            f"  State {i} ({state_labels[i]}): "
            f"mu = {mu[i]: .6f} (drift mu*252 = {drift_mu * 100: .2f}%), "
            f"sigma = {sigma[i]: .6f} (ann. {ann_sigma * 100: .2f}%)"
        )

    log("\nExpected regime durations (days):")
    for i in range(K):
        log(f"  State {i} ({state_labels[i]}): {durations[i]:.2f} days")

    log(
        f"\nLog-likelihood: {ll_final:.1f} "
        f"(initial: {ll_initial:.1f}, improvement: {improvement:.1f})"
    )

    log("\n--- Stationary Distribution ---")
    # Compute stationary distribution from transition matrix
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    for i in range(K):
        log(f"  State {i} ({state_labels[i]}): {stationary[i]:.4f}")

    _save_convergence_figure(history)

    elapsed = time.time() - t_start
    log(f"\nFigure saved to figures/03_em_convergence.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "03_baum_welch_training.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
