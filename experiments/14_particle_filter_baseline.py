"""Experiment 14: Standard particle filter baseline on SPY (2015-2024).

Middle point in the comparison chain:
  HMM (no particles) → Standard PF (particles for everything) → RBPF (particles for jumps only)

Uses the same train/test split and backtest parameters as experiment 05
for direct comparison with HMM strategies.

Outputs:
  - figures/14_pf_cumulative_returns.png
  - figures/14_pf_trend_estimate.png
  - reports/14_particle_filter_baseline.txt

References: Christensen, Turner & Godsill (2012), §III.
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
from src.langevin.particle import run_particle_filter
from src.langevin.utils import estimate_langevin_params, trend_to_trading_signal
from src.strategy.backtest import backtest

# ── Parameters ────────────────────────────────────────────────────────
TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
N_PARTICLES = 200
TRANSACTION_COST_BPS = 5
SIGMA_DELTA = 0.001       # trading signal smoothing parameter
SEED = 42
DT = 1.0                  # daily timestep

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _print_metric_row(name, metrics):
    return (
        f"{name:<20}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_cumulative_figure(test_index, cumulative_pf, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cumulative_pf, label="Standard PF", linewidth=1.8)
    ax.plot(test_index, cumulative_bh, label="Buy-and-Hold", linewidth=1.8, linestyle="--")
    ax.set_title("Out-of-Sample: Standard PF vs Buy-and-Hold (SPY)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_pf_cumulative_returns.png", dpi=150)
    plt.close(fig)


def _save_trend_figure(test_index, filtered_trend, signals, log_prices):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Top: log prices
    ax = axes[0]
    ax.plot(test_index, log_prices, "k-", linewidth=1.0)
    ax.set_ylabel("Log price")
    ax.set_title("Standard PF: Trend Estimate and Trading Signal (SPY)")
    ax.grid(alpha=0.3)

    # Middle: filtered trend
    ax = axes[1]
    ax.plot(test_index, filtered_trend, "tab:blue", linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Filtered trend (x2)")
    ax.grid(alpha=0.3)

    # Bottom: trading signal
    ax = axes[2]
    # signals has length T-1 (from trend_to_trading_signal)
    ax.plot(test_index[1:], signals, "tab:orange", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Trading signal")
    ax.set_xlabel("Date")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_pf_trend_estimate.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 14: Standard PF Baseline on SPY ===")
    log(f"N_particles={N_PARTICLES}, transaction_cost={TRANSACTION_COST_BPS}bps")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close)
    log_prices = np.log(close.to_numpy())

    # Same 70/30 split as experiment 05
    split = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]
    test_log_prices = log_prices[split + 1:]  # +1 because returns loses 1 element

    log(f"Train: {train_returns.index.min().date()} to {train_returns.index.max().date()} "
        f"({len(train_returns)} days)")
    log(f"Test:  {test_returns.index.min().date()} to {test_returns.index.max().date()} "
        f"({len(test_returns)} days)")

    # ── 2. Estimate Langevin parameters from training data ────────────
    log("\n--- Estimating Langevin parameters from training data ---")
    params = estimate_langevin_params(train_returns.to_numpy(), dt=DT)
    log(f"theta:     {params['theta']:.4f}")
    log(f"sigma:     {params['sigma']:.6f}")
    log(f"sigma_obs: {params['sigma_obs']:.6f}")
    log(f"lambda_J:  {params['lambda_J']:.4f}")
    log(f"mu_J:      {params['mu_J']:.6f}")
    log(f"sigma_J:   {params['sigma_J']:.6f}")

    # ── 3. Run standard PF on test data ───────────────────────────────
    log("\n--- Running standard particle filter on test data ---")
    rng = np.random.default_rng(SEED)

    # Prior: centered at first observation, zero trend
    mu0 = np.array([test_log_prices[0], 0.0])
    C0 = np.diag([params['sigma_obs']**2, params['sigma']**2])

    sigma_obs_sq = params['sigma_obs']**2

    t_pf = time.time()
    pf_means, pf_stds, pf_lls, pf_total_ll = run_particle_filter(
        test_log_prices,
        N_PARTICLES,
        theta=params['theta'],
        sigma=params['sigma'],
        sigma_obs_sq=sigma_obs_sq,
        lambda_J=params['lambda_J'],
        mu_J=params['mu_J'],
        sigma_J=params['sigma_J'],
        mu0=mu0,
        C0=C0,
        dt=DT,
        rng=rng,
    )
    log(f"PF completed in {time.time() - t_pf:.1f}s")
    log(f"Total log-likelihood: {pf_total_ll:.2f}")

    # ── 4. Generate trading signals ───────────────────────────────────
    log("\n--- Generating trading signals ---")
    filtered_trend = pf_means[:, 1]  # trend component x2
    signals_raw = trend_to_trading_signal(filtered_trend, sigma_delta=SIGMA_DELTA)

    # Pad signals to match test_returns length (first signal is NaN → 0)
    T_test = len(test_returns)
    T_obs = len(test_log_prices)
    signals = np.zeros(T_test)
    # signals_raw has length T_obs - 1; map back to test period
    # test_log_prices[0] corresponds to test_returns.index[0]
    # signals_raw[t] = signal at time t+1 (based on trend change from t to t+1)
    signals[1:T_obs] = signals_raw[:T_test - 1] if T_obs >= T_test else signals_raw

    log(f"Signal range: [{signals.min():.4f}, {signals.max():.4f}]")
    log(f"Signal mean:  {signals.mean():.4f}")
    log(f"Frac long:    {np.mean(signals > 0):.2%}")
    log(f"Frac short:   {np.mean(signals < 0):.2%}")
    log(f"Frac flat:    {np.mean(signals == 0):.2%}")

    # ── 5. Backtest ───────────────────────────────────────────────────
    log("\n--- Backtest Results ---")
    result_pf = backtest(
        test_returns.to_numpy(),
        signals,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    bh_signals = np.ones(T_test)
    result_bh = backtest(test_returns.to_numpy(), bh_signals, transaction_cost_bps=0)

    log(f"{'Strategy':<20}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 64)
    log(_print_metric_row("Standard PF", result_pf["metrics"]))
    log(_print_metric_row("Buy-and-Hold", result_bh["metrics"]))

    log(f"\nFinal cumulative value:")
    log(f"  Standard PF:   {result_pf['cumulative'][-1]:.4f}")
    log(f"  Buy-and-Hold:  {result_bh['cumulative'][-1]:.4f}")

    # ── 6. Save figures ───────────────────────────────────────────────
    _save_cumulative_figure(
        test_returns.index,
        result_pf["cumulative"],
        result_bh["cumulative"],
    )
    _save_trend_figure(
        test_returns.index,
        filtered_trend[:T_test],
        signals_raw[:T_test - 1],
        test_log_prices[:T_test],
    )

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/14_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "14_particle_filter_baseline.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
