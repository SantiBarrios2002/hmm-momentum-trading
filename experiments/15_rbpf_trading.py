"""Experiment 15: RBPF trading on SPY (2015-2024).

Reproduces the 2012 paper's trading result (§IV-D):
  - RBPF on SPY daily data with same 70/30 split as experiment 05
  - Trading signals via nonlinear transfer function (Eq 46)
  - Comparison: jumps ON vs jumps OFF

Outputs:
  - figures/15_rbpf_cumulative_returns.png
  - figures/15_rbpf_signal.png
  - figures/15_jumps_on_vs_off.png
  - reports/15_rbpf_trading.txt

References: Christensen, Turner & Godsill (2012), §IV-D.
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
from src.langevin.rbpf import run_rbpf
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
DT = 1.0

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _print_metric_row(name, metrics):
    return (
        f"{name:<24}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_cumulative_figure(test_index, cumulative_rbpf, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cumulative_rbpf, label="RBPF", linewidth=1.8)
    ax.plot(test_index, cumulative_bh, label="Buy-and-Hold", linewidth=1.8, linestyle="--")
    ax.set_title("Out-of-Sample: RBPF vs Buy-and-Hold (SPY)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_rbpf_cumulative_returns.png", dpi=150)
    plt.close(fig)


def _save_signal_figure(test_index, filtered_trend, signals, log_prices):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(test_index, log_prices, "k-", linewidth=1.0)
    ax.set_ylabel("Log price")
    ax.set_title("RBPF: Trend Estimate and Trading Signal (SPY)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(test_index, filtered_trend, "tab:blue", linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Filtered trend (x2)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(test_index[1:], signals, "tab:orange", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Trading signal")
    ax.set_xlabel("Date")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_rbpf_signal.png", dpi=150)
    plt.close(fig)


def _save_jumps_comparison_figure(test_index, cumulative_on, cumulative_off, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cumulative_on, label="RBPF (jumps ON)", linewidth=1.8)
    ax.plot(test_index, cumulative_off, label="RBPF (jumps OFF)", linewidth=1.8, linestyle="-.")
    ax.plot(test_index, cumulative_bh, label="Buy-and-Hold", linewidth=1.8, linestyle="--")
    ax.set_title("Effect of Jump Modeling: Jumps ON vs OFF (SPY)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_jumps_on_vs_off.png", dpi=150)
    plt.close(fig)


def _run_rbpf_strategy(test_log_prices, params, n_particles, sigma_delta,
                        dt, seed, use_jumps=True):
    """Run RBPF and generate trading signals. Returns signals, filtered_trend, rbpf outputs."""
    rng = np.random.default_rng(seed)

    # Prior: stationary variance for trend
    trend_stationary_var = params['sigma']**2 / (2.0 * abs(params['theta']))
    mu0 = np.array([test_log_prices[0], 0.0])
    C0 = np.diag([params['sigma_obs']**2, trend_stationary_var])
    sigma_obs_sq = params['sigma_obs']**2

    # If jumps OFF, set lambda_J = 0, sigma_J = 0
    lambda_J = params['lambda_J'] if use_jumps else 0.0
    sigma_J = params['sigma_J'] if use_jumps else 0.0

    rbpf_means, rbpf_stds, rbpf_lls, rbpf_total_ll, rbpf_neff = run_rbpf(
        test_log_prices,
        n_particles,
        theta=params['theta'],
        sigma=params['sigma'],
        sigma_obs_sq=sigma_obs_sq,
        lambda_J=lambda_J,
        mu_J=params['mu_J'],
        sigma_J=sigma_J,
        mu0=mu0,
        C0=C0,
        dt=dt,
        rng=rng,
    )

    filtered_trend = rbpf_means[:, 1]
    signals_raw = trend_to_trading_signal(filtered_trend, sigma_delta=sigma_delta)

    T = len(test_log_prices)
    signals = np.zeros(T)
    signals[1:] = signals_raw

    return signals, filtered_trend, rbpf_total_ll, rbpf_neff


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 15: RBPF Trading on SPY ===")
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

    split = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]
    test_log_prices = log_prices[split + 1:]

    log(f"Train: {train_returns.index.min().date()} to {train_returns.index.max().date()} "
        f"({len(train_returns)} days)")
    log(f"Test:  {test_returns.index.min().date()} to {test_returns.index.max().date()} "
        f"({len(test_returns)} days)")

    # ── 2. Estimate parameters ────────────────────────────────────────
    log("\n--- Estimated Langevin parameters (training data) ---")
    params = estimate_langevin_params(train_returns.to_numpy(), dt=DT)
    for k, v in params.items():
        log(f"  {k:<12}: {v:.6f}")

    # ── 3. Run RBPF with jumps ON ─────────────────────────────────────
    log("\n--- Running RBPF (jumps ON) ---")
    t_rbpf = time.time()
    signals_on, trend_on, ll_on, neff_on = _run_rbpf_strategy(
        test_log_prices, params, N_PARTICLES, SIGMA_DELTA, DT, SEED, use_jumps=True,
    )
    log(f"RBPF (jumps ON) completed in {time.time() - t_rbpf:.1f}s")
    log(f"Total log-likelihood: {ll_on:.2f}")
    log(f"Mean N_eff: {neff_on.mean():.1f} / {N_PARTICLES}")

    # ── 4. Run RBPF with jumps OFF ────────────────────────────────────
    log("\n--- Running RBPF (jumps OFF) ---")
    t_rbpf2 = time.time()
    signals_off, trend_off, ll_off, neff_off = _run_rbpf_strategy(
        test_log_prices, params, N_PARTICLES, SIGMA_DELTA, DT, SEED, use_jumps=False,
    )
    log(f"RBPF (jumps OFF) completed in {time.time() - t_rbpf2:.1f}s")
    log(f"Total log-likelihood: {ll_off:.2f}")
    log(f"Mean N_eff: {neff_off.mean():.1f} / {N_PARTICLES}")

    # ── 5. Backtest all strategies ────────────────────────────────────
    log("\n--- Backtest Results ---")
    T_test = len(test_returns)
    test_ret = test_returns.to_numpy()

    result_on = backtest(test_ret, signals_on[:T_test], transaction_cost_bps=TRANSACTION_COST_BPS)
    result_off = backtest(test_ret, signals_off[:T_test], transaction_cost_bps=TRANSACTION_COST_BPS)

    bh_signals = np.ones(T_test)
    result_bh = backtest(test_ret, bh_signals, transaction_cost_bps=0)

    log(f"{'Strategy':<24}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 68)
    log(_print_metric_row("RBPF (jumps ON)", result_on["metrics"]))
    log(_print_metric_row("RBPF (jumps OFF)", result_off["metrics"]))
    log(_print_metric_row("Buy-and-Hold", result_bh["metrics"]))

    # ── 6. Jump effect analysis ───────────────────────────────────────
    log("\n--- Jump Effect ---")
    sharpe_on = result_on['metrics']['sharpe']
    sharpe_off = result_off['metrics']['sharpe']
    if sharpe_off != 0:
        improvement = (sharpe_on - sharpe_off) / abs(sharpe_off) * 100
        log(f"Sharpe improvement (jumps ON vs OFF): {improvement:+.1f}%")
    else:
        log(f"Sharpe ON: {sharpe_on:.2f}, Sharpe OFF: {sharpe_off:.2f}")

    log(f"\nLog-likelihood improvement: {ll_on - ll_off:.2f}")
    log(f"Turnover ON:  {result_on['metrics']['turnover']:.4f}")
    log(f"Turnover OFF: {result_off['metrics']['turnover']:.4f}")

    log(f"\nFinal cumulative values:")
    log(f"  RBPF (jumps ON):   {result_on['cumulative'][-1]:.4f}")
    log(f"  RBPF (jumps OFF):  {result_off['cumulative'][-1]:.4f}")
    log(f"  Buy-and-Hold:      {result_bh['cumulative'][-1]:.4f}")

    # ── 7. Signal statistics ──────────────────────────────────────────
    log("\n--- Signal Statistics (jumps ON) ---")
    log(f"  Range: [{signals_on.min():.4f}, {signals_on.max():.4f}]")
    log(f"  Mean:  {signals_on.mean():.4f}")
    log(f"  Frac long:  {np.mean(signals_on > 0):.2%}")
    log(f"  Frac short: {np.mean(signals_on < 0):.2%}")

    # ── 8. Save figures ───────────────────────────────────────────────
    _save_cumulative_figure(
        test_returns.index, result_on["cumulative"], result_bh["cumulative"],
    )
    _save_signal_figure(
        test_returns.index, trend_on[:T_test],
        signals_on[1:T_test], test_log_prices[:T_test],
    )
    _save_jumps_comparison_figure(
        test_returns.index,
        result_on["cumulative"],
        result_off["cumulative"],
        result_bh["cumulative"],
    )

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/15_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "15_rbpf_trading.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
