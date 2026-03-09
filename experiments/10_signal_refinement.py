"""Experiment 10: Signal refinement — no-trade zone and EMA smoothing.

Grid-searches over (neutral_threshold x ema_alpha) to find the best
signal post-processing configuration, then compares the best variant
against the baseline weighted-vote signal and buy-and-hold.
"""

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
from src.strategy.signals import apply_no_trade_zone, smooth_signal, states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
NEUTRAL_IDX = 1  # middle state after sorting by mu
TRANSACTION_COST_BPS = 5
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")

# Grid search parameters
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # 1.0 = no-trade zone disabled
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 1.0 = smoothing disabled


def _print_metric_row(name, metrics):
    return (
        f"{name:<30}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_grid_heatmap(grid_sharpe, thresholds, alphas):
    """Save heatmap of Sharpe ratios across the (threshold, alpha) grid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid_sharpe, aspect="auto", cmap="RdYlGn", origin="lower")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.1f}" for t in thresholds])
    ax.set_xlabel("EMA alpha (1.0 = no smoothing)")
    ax.set_ylabel("Neutral threshold (1.0 = no-trade zone disabled)")
    ax.set_title("Out-of-Sample Sharpe Ratio: Signal Refinement Grid Search")

    for i in range(len(thresholds)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{grid_sharpe[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    fig.colorbar(im, ax=ax, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_signal_grid_sharpe.png", dpi=150)
    plt.close(fig)


def _save_backtest_figure(test_index, results_dict):
    """Save cumulative return comparison of selected strategies."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, result in results_dict.items():
        linestyle = "--" if label == "Buy-and-hold" else "-"
        ax.plot(test_index, result["cumulative"], label=label, linewidth=1.8,
                linestyle=linestyle)

    ax.set_title("Signal Refinement: Best Variant vs Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_signal_refinement_backtest.png", dpi=150)
    plt.close(fig)


def _save_signal_comparison(test_index, signals_dict):
    """Save signal time series for baseline vs best refined."""
    n = len(signals_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, sig) in zip(axes, signals_dict.items()):
        ax.plot(test_index, sig, linewidth=0.7, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.3)

    axes[0].set_title("Trading Signals: Baseline vs Refined")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_signal_comparison.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 10: Signal Refinement ===")
    log(f"Ticker: {TICKER} | Period: {START} to {END} | K={K}")
    log(f"Thresholds: {THRESHOLDS}")
    log(f"EMA alphas: {ALPHAS}")

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
    test_np = test_returns.to_numpy()

    log(f"Observations: {len(returns)} | Train={len(train_returns)} | Test={len(test_returns)}")
    log(f"Train period: {train_returns.index.min().date()} to {train_returns.index.max().date()}")
    log(f"Test period:  {test_returns.index.min().date()} to {test_returns.index.max().date()}")

    # --- Train model ---
    log("\nTraining model...")
    t_train = time.time()
    params, history, _ = train_best_model(
        train_returns.to_numpy(), K,
        successful_restarts=10, max_iter=200, tol=1e-6, random_state=42,
    )
    params = sort_states(params)
    log(f"Training done in {time.time() - t_train:.1f}s | final LL={history[-1]:.2f}")

    A, pi, mu, sigma2 = params["A"], params["pi"], params["mu"], params["sigma2"]

    # --- Run inference ---
    predictions, state_probs = run_inference(test_np, A, pi, mu, sigma2)

    # --- Baseline: weighted vote (no refinement) ---
    signals_baseline = states_to_signal(state_probs, mu)
    result_baseline = backtest(test_np, signals_baseline,
                               transaction_cost_bps=TRANSACTION_COST_BPS)

    # --- Buy-and-hold ---
    bh_signals = np.ones_like(test_np)
    result_bh = backtest(test_np, bh_signals, transaction_cost_bps=0)

    # --- Grid search ---
    log("\n--- Grid Search: threshold x alpha ---")
    grid_sharpe = np.zeros((len(THRESHOLDS), len(ALPHAS)))
    grid_annret = np.zeros_like(grid_sharpe)
    grid_maxdd = np.zeros_like(grid_sharpe)
    grid_turnover = np.zeros_like(grid_sharpe)

    best_sharpe = -np.inf
    best_config = None
    best_result = None
    best_signals = None

    for i, threshold in enumerate(THRESHOLDS):
        for j, alpha in enumerate(ALPHAS):
            sig = states_to_signal(state_probs, mu)
            sig = apply_no_trade_zone(sig, state_probs, NEUTRAL_IDX, threshold)
            sig = smooth_signal(sig, alpha)

            result = backtest(test_np, sig,
                              transaction_cost_bps=TRANSACTION_COST_BPS)
            m = result["metrics"]

            grid_sharpe[i, j] = m["sharpe"]
            grid_annret[i, j] = m["annualized_return"]
            grid_maxdd[i, j] = m["max_drawdown"]
            grid_turnover[i, j] = m["turnover"]

            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_config = (threshold, alpha)
                best_result = result
                best_signals = sig

    log(f"\nBest config: threshold={best_config[0]:.1f}, alpha={best_config[1]:.1f}")
    log(f"Best Sharpe: {best_sharpe:.2f}")

    # --- Print grid ---
    log("\nSharpe ratio grid (rows=threshold, cols=alpha):")
    header = f"{'thr\\alpha':>10}" + "".join(f"{a:>8.1f}" for a in ALPHAS)
    log(header)
    log("-" * len(header))
    for i, threshold in enumerate(THRESHOLDS):
        row = f"{threshold:>10.1f}" + "".join(f"{grid_sharpe[i, j]:>8.2f}" for j in range(len(ALPHAS)))
        log(row)

    # --- Top 5 configurations ---
    log("\n--- Top 5 Configurations ---")
    flat_indices = np.argsort(grid_sharpe.ravel())[::-1][:5]
    log(f"{'Rank':<6}{'Threshold':>10}{'Alpha':>8}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDD':>10}{'Turnover':>10}")
    log("-" * 64)
    for rank, idx in enumerate(flat_indices, 1):
        i, j = divmod(idx, len(ALPHAS))
        log(f"{rank:<6}{THRESHOLDS[i]:>10.1f}{ALPHAS[j]:>8.1f}"
            f"{grid_sharpe[i, j]:>8.2f}"
            f"{grid_annret[i, j] * 100:>11.2f}%"
            f"{grid_maxdd[i, j] * 100:>9.2f}%"
            f"{grid_turnover[i, j]:>10.4f}")

    # --- Comparison table ---
    log(f"\n--- Out-of-sample comparison ({TRANSACTION_COST_BPS} bps costs) ---")
    log(f"{'Strategy':<30}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 74)
    log(_print_metric_row("Baseline (vote, no refine)", result_baseline["metrics"]))
    best_label = f"Best (thr={best_config[0]:.1f}, a={best_config[1]:.1f})"
    log(_print_metric_row(best_label, best_result["metrics"]))
    log(_print_metric_row("Buy-and-hold", result_bh["metrics"]))

    delta_sharpe = best_result["metrics"]["sharpe"] - result_baseline["metrics"]["sharpe"]
    delta_annret = best_result["metrics"]["annualized_return"] - result_baseline["metrics"]["annualized_return"]
    log(f"\nBest vs baseline: delta Sharpe={delta_sharpe:+.3f}, "
        f"delta ann.return={delta_annret * 100:+.2f}%")

    # --- Effect decomposition ---
    log("\n--- Effect Decomposition ---")
    # No-trade zone only (best threshold, alpha=1.0)
    sig_ntz = apply_no_trade_zone(signals_baseline, state_probs, NEUTRAL_IDX, best_config[0])
    result_ntz = backtest(test_np, sig_ntz, transaction_cost_bps=TRANSACTION_COST_BPS)
    log(_print_metric_row(f"NTZ only (thr={best_config[0]:.1f})", result_ntz["metrics"]))

    # Smoothing only (threshold=1.0, best alpha)
    sig_ema = smooth_signal(signals_baseline, best_config[1])
    result_ema = backtest(test_np, sig_ema, transaction_cost_bps=TRANSACTION_COST_BPS)
    log(_print_metric_row(f"EMA only (a={best_config[1]:.1f})", result_ema["metrics"]))

    log(_print_metric_row("Combined (best)", best_result["metrics"]))

    # --- Save figures ---
    _save_grid_heatmap(grid_sharpe, THRESHOLDS, ALPHAS)

    _save_backtest_figure(test_returns.index, {
        "Baseline (weighted vote)": result_baseline,
        best_label: best_result,
        "Buy-and-hold": result_bh,
    })

    _save_signal_comparison(test_returns.index, {
        "Baseline": signals_baseline,
        best_label: best_signals,
    })

    elapsed = time.time() - t_start
    log(f"\nFigures saved to:")
    log(f"  figures/10_signal_grid_sharpe.png")
    log(f"  figures/10_signal_refinement_backtest.png")
    log(f"  figures/10_signal_comparison.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "10_signal_refinement.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
