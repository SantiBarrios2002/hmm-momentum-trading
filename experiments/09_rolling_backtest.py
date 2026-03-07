"""Experiment 09: expanding-window rolling backtest with periodic retraining."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import log_returns
from src.data.loader import extract_close_series, load_daily_prices
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states, train_best_model
from src.strategy.backtest import backtest
from src.strategy.signals import states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
INITIAL_TRAIN_YEARS = 5
RETRAIN_MONTHS = 6
TEST_MONTHS = 6
MAX_ITER = 80
TOL = 1e-6
SUCCESSFUL_RESTARTS = 1
MAX_ATTEMPTS = 20
RANDOM_STATE = 123
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


def _window_label(start_date, end_date):
    return f"{start_date:%Y-%m-%d} -> {end_date:%Y-%m-%d}"


def _save_cumulative_figure(index, cumulative_vote, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(index, cumulative_vote, linewidth=1.8, label="Weighted vote (rolling)")
    ax.plot(index, cumulative_bh, linewidth=1.8, linestyle="--", label="Buy-and-hold")
    ax.set_title("Rolling Backtest: Expanding Window + 6M Retraining")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_rolling_cumulative.png", dpi=150)
    plt.close(fig)


def _save_window_sharpe_figure(labels, sharpe_vote, sharpe_bh):
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, sharpe_vote, width, label="Weighted vote")
    ax.bar(x + width / 2, sharpe_bh, width, label="Buy-and-hold")
    ax.set_title("Per-Window Sharpe Ratio")
    ax.set_xlabel("Window")
    ax.set_ylabel("Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_window_sharpe.png", dpi=150)
    plt.close(fig)


def _save_excess_return_figure(labels, excess_ann_returns):
    x = np.arange(len(labels))
    colors = ["tab:green" if val >= 0 else "tab:red" for val in excess_ann_returns]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    ax.bar(x, excess_ann_returns * 100.0, color=colors)
    ax.set_title("Per-Window Excess Annualized Return (Vote - Buy-and-Hold)")
    ax.set_xlabel("Window")
    ax.set_ylabel("Excess annualized return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_window_excess_return.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 09: Rolling Backtest ===")
    log(
        f"Ticker: {TICKER} | Period: {START} to {END} | "
        f"initial_train={INITIAL_TRAIN_YEARS}y, retrain_every={RETRAIN_MONTHS}m, "
        f"test_horizon={TEST_MONTHS}m"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close)
    returns = returns.sort_index()
    log(f"Observations: {returns.size} daily log-returns")

    earliest_test_date = returns.index.min() + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    if earliest_test_date > returns.index.max():
        log("Not enough history for the configured initial training window.")
        return 1

    rolling_start = returns.index[returns.index >= earliest_test_date]
    if rolling_start.empty:
        log("Could not align first test window to available market dates.")
        return 1

    test_start = rolling_start[0]
    window_idx = 0

    per_window = []
    all_test_returns = []
    all_vote_signals = []
    all_bh_signals = []
    all_test_index = []

    log("\n--- Rolling windows ---")
    while test_start <= returns.index.max():
        test_end_exclusive = test_start + pd.DateOffset(months=TEST_MONTHS)
        test_mask = (returns.index >= test_start) & (returns.index < test_end_exclusive)
        if not np.any(test_mask):
            test_start = test_start + pd.DateOffset(months=RETRAIN_MONTHS)
            continue

        train_mask = returns.index < test_start
        train_window = returns.loc[train_mask]
        test_window = returns.loc[test_mask]
        if train_window.empty or test_window.empty:
            test_start = test_start + pd.DateOffset(months=RETRAIN_MONTHS)
            continue

        label = _window_label(test_window.index.min(), test_window.index.max())
        log(
            f"Window {window_idx + 1:02d}: train={len(train_window):4d} obs, "
            f"test={len(test_window):3d} obs ({label})"
        )

        train_values = train_window.to_numpy()
        test_values = test_window.to_numpy()

        params, _history, _ = train_best_model(
            train_values,
            K,
            successful_restarts=SUCCESSFUL_RESTARTS,
            max_attempts=MAX_ATTEMPTS,
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=RANDOM_STATE + window_idx * 100,
        )
        params = sort_states(params)

        _, state_probs = run_inference(
            test_values, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        vote_signals = states_to_signal(state_probs, params["mu"])

        result_vote = backtest(
            test_values, vote_signals, transaction_cost_bps=TRANSACTION_COST_BPS
        )
        result_bh = backtest(test_values, np.ones_like(test_values), transaction_cost_bps=0)

        per_window.append(
            {
                "window": window_idx + 1,
                "label": label,
                "start": test_window.index.min(),
                "end": test_window.index.max(),
                "n_obs": len(test_window),
                "vote_sharpe": result_vote["metrics"]["sharpe"],
                "vote_ann_return": result_vote["metrics"]["annualized_return"],
                "vote_mdd": result_vote["metrics"]["max_drawdown"],
                "vote_turnover": result_vote["metrics"]["turnover"],
                "bh_sharpe": result_bh["metrics"]["sharpe"],
                "bh_ann_return": result_bh["metrics"]["annualized_return"],
                "bh_mdd": result_bh["metrics"]["max_drawdown"],
            }
        )

        all_test_returns.append(test_values)
        all_vote_signals.append(vote_signals)
        all_bh_signals.append(np.ones_like(test_values))
        all_test_index.append(test_window.index.to_numpy())

        window_idx += 1
        test_start = test_start + pd.DateOffset(months=RETRAIN_MONTHS)

    if len(per_window) == 0:
        log("No rolling windows were produced with the current configuration.")
        return 1

    test_concat = np.concatenate(all_test_returns)
    vote_signals_concat = np.concatenate(all_vote_signals)
    bh_signals_concat = np.concatenate(all_bh_signals)
    index_concat = pd.DatetimeIndex(np.concatenate(all_test_index))

    overall_vote = backtest(
        test_concat, vote_signals_concat, transaction_cost_bps=TRANSACTION_COST_BPS
    )
    overall_bh = backtest(test_concat, bh_signals_concat, transaction_cost_bps=0)

    df = pd.DataFrame(per_window).sort_values("start").reset_index(drop=True)
    excess_ann_return = df["vote_ann_return"].to_numpy() - df["bh_ann_return"].to_numpy()
    win_rate = float(np.mean(excess_ann_return > 0.0))

    log(f"\nWindows evaluated: {len(df)}")
    log(
        f"Overall outperformance frequency (annualized return): "
        f"{win_rate * 100:.1f}%"
    )

    log("\n--- Overall rolling out-of-sample metrics ---")
    log(f"{'Strategy':<16}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 60)
    log(_print_metric_row("Weighted vote", overall_vote["metrics"]))
    log(_print_metric_row("Buy-and-hold", overall_bh["metrics"]))

    log("\n--- Per-window summary ---")
    log(
        f"{'Win':<5}{'Period':<24}{'Sharpe(V)':>10}{'Sharpe(BH)':>12}"
        f"{'AnnRet(V)':>12}{'AnnRet(BH)':>12}{'Delta':>9}"
    )
    log("-" * 84)
    for row in df.itertuples(index=False):
        delta = (row.vote_ann_return - row.bh_ann_return) * 100.0
        log(
            f"{row.window:<5d}{row.label:<24}"
            f"{row.vote_sharpe:>10.2f}{row.bh_sharpe:>12.2f}"
            f"{row.vote_ann_return * 100:>11.2f}%"
            f"{row.bh_ann_return * 100:>11.2f}%"
            f"{delta:>8.2f}%"
        )

    _save_cumulative_figure(index_concat, overall_vote["cumulative"], overall_bh["cumulative"])
    _save_window_sharpe_figure(
        df["window"].astype(str).to_list(),
        df["vote_sharpe"].to_numpy(),
        df["bh_sharpe"].to_numpy(),
    )
    _save_excess_return_figure(
        df["window"].astype(str).to_list(),
        excess_ann_return,
    )

    elapsed = time.time() - t_start
    log("\nFigures saved to:")
    log("  figures/09_rolling_cumulative.png")
    log("  figures/09_window_sharpe.png")
    log("  figures/09_window_excess_return.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "09_rolling_backtest.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
