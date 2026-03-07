"""Experiment 07: multi-asset HMM regime analysis and portfolio backtest."""

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
from src.data.loader import extract_close_series, load_multiple
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states, train_best_model
from src.hmm.viterbi import viterbi
from src.strategy.backtest import backtest
from src.strategy.signals import states_to_signal
from src.utils.metrics import annualized_return, max_drawdown, sharpe_ratio

TICKERS = ["SPY", "QQQ", "IWM", "TLT"]
START = "2015-01-01"
END = "2024-12-31"
K = 3
TRAIN_FRACTION = 0.7
MAX_ITER = 120
TOL = 1e-6
SUCCESSFUL_RESTARTS = 2
MAX_ATTEMPTS = 40
RANDOM_STATE = 42
TRANSACTION_COST_BPS = 5
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _print_metric_row(name, metrics):
    return (
        f"{name:<10}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_heatmap(matrix, labels, title, out_path, *, cmap="Blues", fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Ticker")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_asset_metric_bar_figure(tickers, sharpe_vote, sharpe_bh):
    x = np.arange(len(tickers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, sharpe_vote, width, label="HMM weighted vote")
    ax.bar(x + width / 2, sharpe_bh, width, label="Buy-and-hold")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Per-Asset Out-of-Sample Sharpe")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_asset_sharpe_comparison.png", dpi=150)
    plt.close(fig)


def _save_portfolio_backtest_figure(index, strategy_cumulative, bh_cumulative):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(index, strategy_cumulative, linewidth=1.9, label="Cross-asset HMM portfolio")
    ax.plot(index, bh_cumulative, linewidth=1.9, linestyle="--", label="Equal-weight buy-and-hold")
    ax.set_title("Cross-Asset Portfolio Backtest (Out-of-Sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_cross_asset_backtest.png", dpi=150)
    plt.close(fig)


def _portfolio_metrics(net_returns, signals_matrix):
    cumulative = np.cumprod(1.0 + net_returns)
    turnover = float(np.mean(np.abs(np.diff(signals_matrix, axis=0))))
    return {
        "sharpe": float(sharpe_ratio(net_returns)),
        "annualized_return": float(annualized_return(net_returns)),
        "max_drawdown": float(max_drawdown(cumulative)),
        "turnover": turnover,
        "cumulative": cumulative,
    }


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 07: Multi-Asset Analysis ===")
    log(f"Tickers: {', '.join(TICKERS)} | Period: {START} to {END} | K={K}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_prices = load_multiple(TICKERS, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    returns_dict = {}
    for ticker in TICKERS:
        close = extract_close_series(raw_prices[ticker])
        returns_dict[ticker] = log_returns(close).rename(ticker)

    # Keep dates common to all assets.
    returns_df = pd.concat(returns_dict.values(), axis=1, join="inner").dropna()
    if returns_df.empty:
        log("No overlapping return dates across all tickers.")
        return 1

    split = int(len(returns_df) * TRAIN_FRACTION)
    if split <= 1 or split >= len(returns_df):
        log("Invalid train/test split on aligned multi-asset data.")
        return 1

    train_df = returns_df.iloc[:split]
    test_df = returns_df.iloc[split:]
    log(
        f"Aligned observations: {len(returns_df)} | Train={len(train_df)} | Test={len(test_df)}"
    )
    log(
        f"Train period: {train_df.index.min().date()} to {train_df.index.max().date()} | "
        f"Test period: {test_df.index.min().date()} to {test_df.index.max().date()}"
    )

    models = {}
    test_states = {}
    test_bearish_probs = {}
    test_signals = {}
    test_backtests = {}

    log("\n--- Per-asset model training and backtesting ---")
    for idx, ticker in enumerate(TICKERS):
        t0 = time.time()
        train_values = train_df[ticker].to_numpy()
        test_values = test_df[ticker].to_numpy()

        params, history, _ = train_best_model(
            train_values,
            K=K,
            successful_restarts=SUCCESSFUL_RESTARTS,
            max_attempts=MAX_ATTEMPTS,
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=RANDOM_STATE + idx * 1000,
        )
        params = sort_states(params)
        models[ticker] = params

        states, _ = viterbi(
            test_values, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        _, probs = run_inference(
            test_values, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        signals = states_to_signal(probs, params["mu"])

        strategy_bt = backtest(
            test_values,
            signals,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        bh_bt = backtest(test_values, np.ones_like(test_values), transaction_cost_bps=0)

        test_states[ticker] = states
        test_bearish_probs[ticker] = probs[:, 0]  # sorted states: 0 is most bearish
        test_signals[ticker] = signals
        test_backtests[ticker] = {"strategy": strategy_bt, "buy_hold": bh_bt}

        log(
            f"{ticker}: train LL={history[-1]:.2f}, "
            f"elapsed={time.time() - t0:.1f}s, "
            f"mu={np.array2string(params['mu'], precision=5)}"
        )

    # Regime agreement matrix from test Viterbi paths.
    n = len(TICKERS)
    agreement = np.empty((n, n), dtype=float)
    bearish_corr = np.empty((n, n), dtype=float)
    for i, ti in enumerate(TICKERS):
        si = test_states[ti]
        bi = test_bearish_probs[ti]
        for j, tj in enumerate(TICKERS):
            sj = test_states[tj]
            bj = test_bearish_probs[tj]
            agreement[i, j] = float(np.mean(si == sj))
            corr = float(np.corrcoef(bi, bj)[0, 1])
            bearish_corr[i, j] = corr if np.isfinite(corr) else 0.0

    log("\n--- Regime agreement (test period, same-state frequency) ---")
    for i, ti in enumerate(TICKERS):
        row = "  ".join(f"{agreement[i, j]:.2f}" for j in range(n))
        log(f"{ti}: {row}")

    log("\n--- Bearish posterior correlation matrix (test period) ---")
    for i, ti in enumerate(TICKERS):
        row = "  ".join(f"{bearish_corr[i, j]:+.2f}" for j in range(n))
        log(f"{ti}: {row}")

    # Equal-weight cross-asset portfolio from asset-level strategy net returns.
    strategy_matrix = np.column_stack(
        [test_backtests[t]["strategy"]["net_returns"] for t in TICKERS]
    )
    signal_matrix = np.column_stack([test_signals[t] for t in TICKERS])
    bh_matrix = np.column_stack([test_df[t].to_numpy() for t in TICKERS])

    portfolio_net = np.mean(strategy_matrix, axis=1)
    portfolio_bh = np.mean(bh_matrix, axis=1)
    portfolio_metrics = _portfolio_metrics(portfolio_net, signal_matrix)
    portfolio_bh_metrics = _portfolio_metrics(portfolio_bh, np.ones_like(signal_matrix))

    log(f"\n--- Per-asset out-of-sample metrics ({TRANSACTION_COST_BPS} bps costs) ---")
    log(f"{'Ticker':<10}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 54)
    for ticker in TICKERS:
        log(_print_metric_row(ticker, test_backtests[ticker]["strategy"]["metrics"]))

    log("\n--- Cross-asset portfolio metrics ---")
    log(f"{'Strategy':<22}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 66)
    log(_print_metric_row("HMM portfolio", portfolio_metrics))
    log(_print_metric_row("EW buy-and-hold", portfolio_bh_metrics))

    # TLT vs equity bearish-correlation diagnostic (expected often negative).
    equity = ["SPY", "QQQ", "IWM"]
    tlt_idx = TICKERS.index("TLT")
    mean_tlt_equity_corr = float(
        np.mean([bearish_corr[TICKERS.index(e), tlt_idx] for e in equity])
    )
    log(
        f"\nMean bearish posterior correlation: equities vs TLT = "
        f"{mean_tlt_equity_corr:+.3f}"
    )

    _save_heatmap(
        agreement,
        TICKERS,
        "Regime Agreement (Viterbi, Test Period)",
        FIGURES_DIR / "07_regime_agreement.png",
        cmap="Blues",
        fmt="{:.2f}",
    )
    _save_heatmap(
        bearish_corr,
        TICKERS,
        "Bearish Posterior Correlation (Test Period)",
        FIGURES_DIR / "07_bearish_posterior_corr.png",
        cmap="RdBu_r",
        fmt="{:+.2f}",
    )
    _save_asset_metric_bar_figure(
        TICKERS,
        np.array([test_backtests[t]["strategy"]["metrics"]["sharpe"] for t in TICKERS]),
        np.array([test_backtests[t]["buy_hold"]["metrics"]["sharpe"] for t in TICKERS]),
    )
    _save_portfolio_backtest_figure(
        test_df.index,
        portfolio_metrics["cumulative"],
        portfolio_bh_metrics["cumulative"],
    )

    elapsed = time.time() - t_start
    log("\nFigures saved to:")
    log("  figures/07_regime_agreement.png")
    log("  figures/07_bearish_posterior_corr.png")
    log("  figures/07_asset_sharpe_comparison.png")
    log("  figures/07_cross_asset_backtest.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "07_multi_asset.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
