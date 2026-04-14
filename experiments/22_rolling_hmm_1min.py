#!/usr/bin/env python3
"""Experiment 22 — Rolling-window BW HMM at 1-min frequency.

Tests whether daily retraining on a rolling H-day window fixes the HMM's
Sharpe of -1.22 (frozen parameters) on 1-min ES futures data.

The paper (§2.3) says: "parameter estimation can be done using the previous H
days of market data, when the market is shut."

We sweep H ∈ {20, 40, 60, 120} and compare:
    1. Rolling BW HMM (retrained daily)
    2. Frozen BW HMM (trained once on 70%, from experiment 16)
    3. Buy-and-hold

Outputs:
    figures/22_rolling_sharpe_vs_H.png
    figures/22_rolling_vs_frozen.png
    figures/22_param_evolution.png
    reports/22_rolling_hmm_1min.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.futures_loader import filter_rth, load_futures_1m
from src.hmm.baum_welch_numba import run_inference_numba, train_hmm_numba
from src.hmm.rolling import rolling_hmm, split_by_day
from src.strategy.backtest import backtest
from src.strategy.signals import predictions_to_signal

FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")
FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

K = 3  # 3 hidden states (bearish / neutral / bullish)
TC_BPS = 2  # transaction cost in bps (1-min)
N_RESTARTS = 2  # minimal restarts to keep runtime feasible (~1500 daily fits per H)
MAX_ITER = 50  # parameters change slowly day-to-day; 50 iters suffices


def main():
    print("=" * 70)
    print("EXPERIMENT 22: Rolling-Window BW HMM at 1-min Frequency")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading 1-min ES data...")
    df = load_futures_1m("ES")
    df = filter_rth(df)
    close = df["close"].values.astype(np.float64)
    timestamps = df.index

    # Log-returns
    returns = np.diff(np.log(close))
    ret_dates = np.array([t.date() for t in timestamps[1:]])

    # Split into per-day arrays
    daily_arrays = split_by_day(returns, ret_dates)
    n_days = len(daily_arrays)
    print(f"  {len(returns):,} returns across {n_days} trading days")
    print(f"  Avg bars/day: {len(returns) / n_days:.0f}")

    # ── Sweep H ──────────────────────────────────────────────────────────
    H_values = [20, 60]
    results = {}
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("EXPERIMENT 22: Rolling-Window BW HMM at 1-min Frequency")
    report_lines.append("=" * 70)

    print(f"\n[2/4] Running rolling HMM for H ∈ {H_values}...")
    print(f"  K={K}, n_restarts={N_RESTARTS}, max_iter={MAX_ITER}, TC={TC_BPS} bps")

    for H in H_values:
        print(f"\n  --- H = {H} days ---")
        t0 = time.time()

        result = rolling_hmm(
            daily_arrays,
            K=K,
            H=H,
            train_fn=train_hmm_numba,
            inference_fn=run_inference_numba,
            n_restarts=N_RESTARTS,
            max_iter=MAX_ITER,
            min_variance=1e-8,
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s ({result['n_test_days']} test days)")

        # Generate signal and backtest
        preds = result["predictions"]
        signal = predictions_to_signal(preds, "sign")

        # Get corresponding returns for the test period
        # Test period starts at day H, so collect returns from day H onward
        test_returns = np.concatenate(daily_arrays[H:])
        # Ensure lengths match (rolling_hmm may skip failed days)
        min_len = min(len(test_returns), len(signal))
        test_returns = test_returns[:min_len]
        signal = signal[:min_len]

        bt = backtest(test_returns, signal, transaction_cost_bps=TC_BPS)
        metrics = bt["metrics"]

        results[H] = {
            "result": result,
            "backtest": bt,
            "metrics": metrics,
            "signal": signal,
            "test_returns": test_returns,
            "elapsed": elapsed,
        }

        print(f"  Sharpe: {metrics['sharpe']:.3f}, "
              f"Ann. Return: {metrics['annualized_return']:.2%}, "
              f"Max DD: {metrics['max_drawdown']:.2%}, "
              f"Turnover: {metrics['turnover']:.4f}")

    # ── Frozen baseline ──────────────────────────────────────────────────
    print("\n[3/4] Running frozen BW HMM baseline (train once on 70%)...")
    split_day = int(n_days * 0.7)
    train_obs_frozen = np.concatenate(daily_arrays[:split_day])
    test_obs_frozen = np.concatenate(daily_arrays[split_day:])

    t0 = time.time()
    params_frozen, _ = train_hmm_numba(
        train_obs_frozen, K,
        n_restarts=5, max_iter=200, min_variance=1e-8
    )
    preds_frozen, _ = run_inference_numba(
        test_obs_frozen,
        params_frozen["A"], params_frozen["pi"],
        params_frozen["mu"], params_frozen["sigma2"],
    )
    elapsed_frozen = time.time() - t0

    signal_frozen = predictions_to_signal(preds_frozen, "sign")
    bt_frozen = backtest(test_obs_frozen, signal_frozen, transaction_cost_bps=TC_BPS)

    print(f"  Frozen Sharpe: {bt_frozen['metrics']['sharpe']:.3f} ({elapsed_frozen:.0f}s)")

    # Buy-and-hold on same test period as longest rolling window
    # Use H=20 test period (most test days) for the main comparison
    H_main = min(H_values)
    bh_returns = results[H_main]["test_returns"]
    bt_bh = backtest(bh_returns, np.ones_like(bh_returns), transaction_cost_bps=0)

    # ── Report ───────────────────────────────────────────────────────────
    report_lines.append(f"\nData: ES 1-min RTH, {n_days} trading days")
    report_lines.append(f"K={K}, n_restarts={N_RESTARTS}, max_iter={MAX_ITER}, TC={TC_BPS} bps")

    report_lines.append(f"\n{'Strategy':<25} {'Sharpe':>8} {'Ann. Ret':>10} {'Max DD':>10} {'Turnover':>10} {'Time':>8}")
    report_lines.append("-" * 75)

    for H in H_values:
        m = results[H]["metrics"]
        report_lines.append(
            f"Rolling H={H:<3}             {m['sharpe']:>8.3f} "
            f"{m['annualized_return']:>9.2%} {m['max_drawdown']:>9.2%} "
            f"{m['turnover']:>10.4f} {results[H]['elapsed']:>7.0f}s"
        )

    m_f = bt_frozen["metrics"]
    report_lines.append(
        f"{'Frozen (70/30)':<25} {m_f['sharpe']:>8.3f} "
        f"{m_f['annualized_return']:>9.2%} {m_f['max_drawdown']:>9.2%} "
        f"{m_f['turnover']:>10.4f} {elapsed_frozen:>7.0f}s"
    )

    m_bh = bt_bh["metrics"]
    report_lines.append(
        f"{'Buy-and-hold':<25} {m_bh['sharpe']:>8.3f} "
        f"{m_bh['annualized_return']:>9.2%} {m_bh['max_drawdown']:>9.2%} "
        f"{m_bh['turnover']:>10.4f} {'—':>8}"
    )

    # Best rolling
    best_H = max(H_values, key=lambda h: results[h]["metrics"]["sharpe"])
    best_sharpe = results[best_H]["metrics"]["sharpe"]
    frozen_sharpe = m_f["sharpe"]

    report_lines.append(f"\nBest rolling window: H={best_H} (Sharpe {best_sharpe:.3f})")
    report_lines.append(f"Improvement over frozen: {best_sharpe - frozen_sharpe:+.3f} Sharpe")

    if best_sharpe > frozen_sharpe:
        report_lines.append("VERDICT: Rolling retraining IMPROVES the 1-min HMM.")
    else:
        report_lines.append("VERDICT: Rolling retraining does NOT help — the problem is deeper than stale parameters.")

    if best_sharpe > m_bh["sharpe"]:
        report_lines.append(f"Rolling HMM BEATS buy-and-hold ({best_sharpe:.3f} vs {m_bh['sharpe']:.3f})")
    else:
        report_lines.append(f"Rolling HMM does NOT beat buy-and-hold ({best_sharpe:.3f} vs {m_bh['sharpe']:.3f})")

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")

    # Figure 1: Sharpe vs H
    fig, ax = plt.subplots(figsize=(8, 5))
    sharpes = [results[H]["metrics"]["sharpe"] for H in H_values]
    ax.bar(range(len(H_values)), sharpes, color="#1976d2", alpha=0.8, width=0.6)
    ax.axhline(frozen_sharpe, color="#d32f2f", linestyle="--", linewidth=2, label=f"Frozen (Sharpe {frozen_sharpe:.3f})")
    ax.axhline(m_bh["sharpe"], color="#4caf50", linestyle="--", linewidth=2, label=f"B&H (Sharpe {m_bh['sharpe']:.3f})")
    ax.set_xticks(range(len(H_values)))
    ax.set_xticklabels([f"H={H}" for H in H_values])
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Rolling HMM Sharpe vs Window Size — 1-min ES")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "22_rolling_sharpe_vs_H.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Cumulative returns — best rolling vs frozen vs B&H
    fig, ax = plt.subplots(figsize=(12, 6))
    best_bt = results[best_H]["backtest"]
    ax.plot(np.cumsum(best_bt["net_returns"]) * 100, label=f"Rolling H={best_H}", linewidth=1.2)
    # Frozen — different test period, plot separately
    ax.plot(np.cumsum(bt_frozen["net_returns"]) * 100, label="Frozen (70/30)",
            linewidth=1.2, linestyle="--", alpha=0.7)
    ax.plot(np.cumsum(results[best_H]["test_returns"]) * 100, label="Buy-and-hold",
            linewidth=1.2, color="gray", alpha=0.7)
    ax.set_xlabel("1-min bars")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"Rolling HMM (H={best_H}) vs Frozen vs B&H — 1-min ES")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "22_rolling_vs_frozen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Parameter evolution over time (best H)
    best_params_list = results[best_H]["result"]["daily_params"]
    if len(best_params_list) > 1:
        mus = np.array([p["mu"] for p in best_params_list])
        sigma2s = np.array([p["sigma2"] for p in best_params_list])
        days = np.arange(len(best_params_list))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for k in range(K):
            ax1.plot(days, mus[:, k], label=f"State {k}", linewidth=0.8)
        ax1.set_ylabel("μ_k (emission mean)")
        ax1.set_title(f"HMM Parameter Evolution — Rolling H={best_H}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for k in range(K):
            ax2.plot(days, np.sqrt(sigma2s[:, k]) * np.sqrt(252 * 390) * 100,
                     label=f"State {k}", linewidth=0.8)
        ax2.set_ylabel("σ_k (annualized vol %)")
        ax2.set_xlabel("Trading day")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "22_param_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Print and save report ────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")
    report_path = REPORTS_DIR / "22_rolling_hmm_1min.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
