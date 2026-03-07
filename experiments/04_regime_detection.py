"""Experiment 04: regime detection via Viterbi and forward-backward posteriors."""

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
from src.hmm.forward_backward import compute_posteriors
from src.hmm.utils import sort_states, train_best_model
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
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _average_durations(states, k_states):
    durations = [[] for _ in range(k_states)]
    current_state = int(states[0])
    run_len = 1

    for t in range(1, len(states)):
        s = int(states[t])
        if s == current_state:
            run_len += 1
        else:
            durations[current_state].append(run_len)
            current_state = s
            run_len = 1
    durations[current_state].append(run_len)

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
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 04: Regime Detection ===")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close)
    returns_values = returns.to_numpy()
    log(f"Observations: {returns_values.size} daily log-returns")

    t_train = time.time()
    log("Training HMM...")
    params, _, _ = train_best_model(
        returns_values,
        K,
        successful_restarts=N_RESTARTS,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    params = sort_states(params)
    log(f"Training completed in {time.time() - t_train:.1f}s")

    A = params["A"]
    pi = params["pi"]
    mu = params["mu"]
    sigma2 = params["sigma2"]

    t_decode = time.time()
    log("Running Viterbi and forward-backward decoding...")
    states, log_prob = viterbi(returns_values, A, pi, mu, sigma2)
    gamma, _, ll_fb = compute_posteriors(returns_values, A, pi, mu, sigma2)
    log(f"Decoding completed in {time.time() - t_decode:.1f}s")

    counts = np.bincount(states, minlength=K)
    total_days = len(states)
    durations = _average_durations(states, K)

    labels = ["bearish", "neutral", "bullish"] if K == 3 else [f"state-{i}" for i in range(K)]

    log("\n--- Regime Statistics (Viterbi) ---")
    log(f"{'State':<22}{'Days':>6}{'Pct':>8}{'MeanRet':>10}{'Vol':>10}{'AvgDur':>10}")
    log("-" * 66)
    for k in range(K):
        mask = states == k
        if np.any(mask):
            mean_k = float(np.mean(returns_values[mask]))
            vol_k = float(np.std(returns_values[mask], ddof=0))
        else:
            mean_k = 0.0
            vol_k = 0.0

        pct = 100.0 * counts[k] / total_days
        log(
            f"  {k} ({labels[k]:<10})"
            f"{counts[k]:>6d}"
            f"{pct:>7.1f}%"
            f"{mean_k * 100:>9.3f}%"
            f"{vol_k * 100:>9.3f}%"
            f"{durations[k]:>9.2f}d"
        )

    log("\n--- Annualized per-regime statistics ---")
    for k in range(K):
        mask = states == k
        if np.any(mask):
            mean_k = float(np.mean(returns_values[mask]))
            vol_k = float(np.std(returns_values[mask], ddof=0))
            ann_ret = mean_k * 252
            ann_vol = vol_k * np.sqrt(252)
            log(
                f"  {k} ({labels[k]}): "
                f"ann. return = {ann_ret * 100: .2f}%, "
                f"ann. vol = {ann_vol * 100: .2f}%"
            )

    log(f"\nViterbi path log-probability: {log_prob:.2f}")
    log(f"Forward-backward log-likelihood: {ll_fb:.2f}")

    log("\n--- Transition matrix (sorted) ---")
    header = "         " + "  ".join([f"State {j}" for j in range(K)])
    log(header)
    for i in range(K):
        row = "  ".join([f"{A[i, j]:.4f}" for j in range(K)])
        log(f"State {i}  {row}")

    log("\n--- Regime transitions count ---")
    n_transitions = np.sum(np.diff(states) != 0)
    log(f"Total regime transitions: {n_transitions}")
    log(f"Average days between transitions: {total_days / max(n_transitions, 1):.1f}")

    # Align prices with returns index: return_t corresponds to close[t] - close[t-1].
    aligned_prices = close.iloc[1:]
    _save_regime_price_figure(aligned_prices, states)
    _save_state_posterior_figure(returns, gamma)

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/04_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "04_regime_detection.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
