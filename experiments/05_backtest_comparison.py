"""Experiment 05: out-of-sample backtest comparison against buy-and-hold."""

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
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states, train_best_model
from src.strategy.backtest import backtest
from src.strategy.signals import predictions_to_signal, states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
MAX_ITER = 200
TOL = 1e-6
N_RESTARTS = 10
RANDOM_STATE = 42
TRANSACTION_COST_BPS = 5
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _print_metric_row(name, metrics):
    return (
        f"{name:<16}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_comparison_figure(test_index, cumulative_sign, cumulative_vote, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cumulative_sign, label="Sign Signal", linewidth=1.8)
    ax.plot(test_index, cumulative_vote, label="Weighted Vote", linewidth=1.8)
    ax.plot(test_index, cumulative_bh, label="Buy-and-Hold", linewidth=1.8, linestyle="--")

    ax.set_title("Out-of-Sample Backtest: HMM Strategies vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_backtest_comparison.png", dpi=150)
    plt.close(fig)


def _save_signal_figure(test_index, signals_sign, signals_vote):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(test_index, signals_sign, linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("Sign signal")
    axes[0].set_ylim(-1.3, 1.3)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(test_index, signals_vote, linewidth=0.8, alpha=0.8, color="tab:orange")
    axes[1].set_ylabel("Weighted vote")
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    axes[0].set_title("Trading Signals Over Test Period")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_signals.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 05: Backtest Comparison ===")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close)

    split = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]

    log(
        f"Train period: {train_returns.index.min().date()} to "
        f"{train_returns.index.max().date()} ({len(train_returns)} days)"
    )
    log(
        f"Test period:  {test_returns.index.min().date()} to "
        f"{test_returns.index.max().date()} ({len(test_returns)} days)"
    )

    t_train = time.time()
    log(f"\nTraining {K}-state HMM on training data...")
    params, history, _ = train_best_model(
        train_returns.to_numpy(),
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

    log("\n--- Learned Parameters (sorted by mu) ---")
    labels = ["bearish", "neutral", "bullish"] if K == 3 else [f"state-{i}" for i in range(K)]
    for i in range(K):
        log(
            f"  State {i} ({labels[i]}): "
            f"mu = {mu[i]: .6f} (ann. {mu[i] * 252 * 100: .2f}%), "
            f"sigma = {np.sqrt(sigma2[i]): .6f} (ann. {np.sqrt(sigma2[i]) * np.sqrt(252) * 100: .2f}%)"
        )

    t_infer = time.time()
    log("\nRunning inference on test data...")
    predictions, state_probs = run_inference(test_returns.to_numpy(), A, pi, mu, sigma2)
    log(f"Inference completed in {time.time() - t_infer:.1f}s")

    signals_sign = predictions_to_signal(predictions, transfer_fn="sign")
    signals_vote = states_to_signal(state_probs, mu)

    result_sign = backtest(
        test_returns.to_numpy(),
        signals_sign,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    result_vote = backtest(
        test_returns.to_numpy(),
        signals_vote,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    bh_signals = np.ones_like(test_returns.to_numpy())
    result_bh = backtest(test_returns.to_numpy(), bh_signals, transaction_cost_bps=0)

    log(
        f"\n--- Out-of-Sample Performance ({TRANSACTION_COST_BPS} bps transaction costs) ---"
    )
    log(f"{'Strategy':<16}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 60)
    log(_print_metric_row("Sign signal", result_sign["metrics"]))
    log(_print_metric_row("Weighted vote", result_vote["metrics"]))
    log(_print_metric_row("Buy-and-hold", result_bh["metrics"]))

    # Check one-period lag convention: first net return should not use a prior signal.
    log(
        f"\nFirst-day net return check (should be ~0 for HMM strategies): "
        f"sign={result_sign['net_returns'][0]:.6f}, "
        f"vote={result_vote['net_returns'][0]:.6f}, "
        f"buy-hold={result_bh['net_returns'][0]:.6f}"
    )

    log("\n--- Signal Statistics ---")
    for name, sig in [("Sign", signals_sign), ("Vote", signals_vote)]:
        log(
            f"  {name}: mean={np.mean(sig):.4f}, "
            f"std={np.std(sig):.4f}, "
            f"frac_long={np.mean(sig > 0):.2%}, "
            f"frac_short={np.mean(sig < 0):.2%}, "
            f"frac_flat={np.mean(sig == 0):.2%}"
        )

    log("\n--- Final Cumulative Values ---")
    log(f"  Sign signal:   {result_sign['cumulative'][-1]:.4f}")
    log(f"  Weighted vote: {result_vote['cumulative'][-1]:.4f}")
    log(f"  Buy-and-hold:  {result_bh['cumulative'][-1]:.4f}")

    _save_comparison_figure(
        test_returns.index,
        result_sign["cumulative"],
        result_vote["cumulative"],
        result_bh["cumulative"],
    )
    _save_signal_figure(test_returns.index, signals_sign, signals_vote)

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/05_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "05_backtest_comparison.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
