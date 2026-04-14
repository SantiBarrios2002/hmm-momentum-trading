"""Experiment 17: RBPF on intraday ES futures (raw prices, full signal pipeline).

Applies the 2012 paper's RBPF strategy to E-mini S&P 500 (ES) intraday data
from Databento, using the exact parameters from Table I of the paper.

Compares three signal variants:
  1. RBPF jumps ON  + FIR smoothing + IGARCH vol scaling  (full 2012 pipeline)
  2. RBPF jumps OFF + FIR smoothing + IGARCH vol scaling
  3. Buy-and-hold

And a frequency comparison against the daily RBPF result from Experiment 15:
  - Daily SPY (log-prices, exp 15):  Sharpe = -1.77
  - Intraday ES (raw prices, exp 17): this experiment

Parameters (Table I, Christensen et al. 2012):
  theta      = -0.2      (mean-reversion, per trading day)
  SF(sigma)  = 0.35%     (jumps ON) / 3.5% (jumps OFF)  → sigma = SF * P_0
  SF(sigma_obs) = 35.0%  → sigma_obs = 0.35 * P_0
  lambda_J   = 5.0       (jumps per trading day)
  SF(sigma_J) = 6.0%     → sigma_J = 0.06 * P_0
  mu_J       = 0.0

Note on time scaling: theta, sigma, sigma_obs are continuous-time parameters
passed to discretize_langevin(theta, sigma, dt) — the discretization handles
the dt scaling internally.  lambda_J is a Poisson rate per day; the RBPF
scales it by dt per bar automatically.

Key differences from Experiment 15:
  - Raw prices instead of log-prices  (2012 paper §IV-A)
  - Intraday frequency (5-min default, 1-min optional via FREQ)
  - FIR smoothing + IGARCH vol scaling  (2012 paper §IV-D)
  - Table I parameters — directly from the paper, no estimation needed
  - Futures transaction costs (2 bps, vs 5 bps for SPY ETF)

Data: data/databento/ES_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet

Outputs:
  - figures/17_cumulative_returns.png
  - figures/17_filtered_trend.png
  - figures/17_turnover_comparison.png
  - reports/17_rbpf_intraday_es.txt

References:
  Christensen, Murphy & Godsill (2012), IEEE JSTSP, §III-B, §IV-C, §IV-D, Table I.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.futures_loader import filter_rth, load_futures_1m, resample_bars
from src.langevin.rbpf_numba import run_rbpf_numba as run_rbpf
from src.langevin.utils import (
    fir_momentum_signal,
    igarch_volatility_scale,
    trend_to_trading_signal,
)
from src.strategy.backtest import backtest

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL = "ES"
START  = "2019-01-01"
END    = "2024-12-31"

# Resampling frequency.  "1min" is now feasible with the Numba backend (~3 min).
FREQ = "1min"

# dt in trading-day units: (minutes per bar) / (minutes per RTH day)
# RTH = 6.5 h = 390 min.  5-min bars → dt = 5/390.
_FREQ_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30}
DT = _FREQ_MINUTES[FREQ] / 390.0

TRAIN_FRAC   = 0.70   # 70 / 30 train-test split
N_PARTICLES  = 100    # fewer than exp 15 (200) to compensate for more bars
N_TAPS       = 4      # FIR window (paper §IV-D)
IGARCH_ALPHA = 0.06   # RiskMetrics IGARCH weight (paper §IV-D)
TRANSACTION_COST_BPS = 2   # ES futures cheaper than SPY ETF
SEED = 42

# ── Table I parameters (Christensen, Murphy & Godsill 2012, Table I) ──────────
# Scale factors (SF) are relative to P_0 (initial price level).
# theta and lambda_J are daily rates — discretize_langevin and run_rbpf
# handle the per-bar scaling via the dt argument.
TABLE_I = {
    "jumps_on": {
        "theta":    -0.2,
        "sf_sigma":  0.0035,   # SF(sigma) = 0.35%
        "sf_sigma_obs": 0.35,  # SF(sigma_obs) = 35.0%
        "lambda_J":  5.0,      # jumps per trading day
        "mu_J":      0.0,
        "sf_sigma_J": 0.06,    # SF(sigma_J) = 6.0%
    },
    "jumps_off": {
        "theta":    -0.2,
        "sf_sigma":  0.035,    # SF(sigma) = 3.5%
        "sf_sigma_obs": 0.35,  # SF(sigma_obs) = 35.0%
        "lambda_J":  0.0,
        "mu_J":      0.0,
        "sf_sigma_J": 0.0,
    },
}

# Known baseline from Experiment 15 (daily SPY, log-prices, N=200, 5 bps)
EXP15_SHARPE_RBPF    = -1.77
EXP15_SHARPE_BH      =  0.40
EXP15_TURNOVER_RBPF  =  1.3656

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_params(variant: str, P0: float) -> dict:
    """Convert Table I scale factors to absolute price-unit parameters.

    Parameters are in absolute price units (e.g. sigma = SF * P_0 = $15.75
    for P_0 = $4500 and SF = 0.35%).  theta and lambda_J are daily rates.

    Parameters
    ----------
    variant : "jumps_on" or "jumps_off"
    P0 : initial price level (first bar of the series)

    Returns
    -------
    dict with keys theta, sigma, sigma_obs, lambda_J, mu_J, sigma_J
    """
    t = TABLE_I[variant]
    return {
        "theta":     t["theta"],
        "sigma":     t["sf_sigma"] * P0,
        "sigma_obs": t["sf_sigma_obs"] * P0,
        "lambda_J":  t["lambda_J"],
        "mu_J":      t["mu_J"],
        "sigma_J":   t["sf_sigma_J"] * P0,
    }


def _metric_row(name: str, m: dict) -> str:
    return (
        f"{name:<36}"
        f"{m['sharpe']:>8.2f}"
        f"{m['annualized_return'] * 100:>12.1f}%"
        f"{m['max_drawdown'] * 100:>13.1f}%"
        f"{m['turnover']:>11.4f}"
    )


def _run_rbpf_pipeline(
    prices: np.ndarray,
    params: dict,
    use_fir: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run RBPF + signal pipeline on a raw price series.

    Pipeline (2012 paper §IV-D):
        1. RBPF → filtered trend x2_t                       (§III-B)
        2a. FIR momentum signal M_t = mean(sign(x2_{t-n:t})) (§IV-D Steps 1-2)
        2b. IGARCH vol scaling: position_t = M_t / sigma_t   (§IV-D Step 4)
        OR (ablation):
        2. Transfer function: Z_t = delta_t / sqrt(delta_t^2 + sigma_delta^2)

    Parameters
    ----------
    prices   : raw price series, shape (T,)
    params   : dict with theta, sigma, sigma_obs, lambda_J, mu_J, sigma_J
    use_fir  : True → full FIR+IGARCH pipeline; False → transfer function only
    seed     : RBPF random seed

    Returns
    -------
    signals  : trading signals in [-1, 1], shape (T,)
    trend    : filtered trend x2_t, shape (T,)
    total_ll : total RBPF log-likelihood
    """
    T = len(prices)
    sigma_obs_sq = params["sigma_obs"] ** 2
    trend_var    = params["sigma"] ** 2 / (2.0 * abs(params["theta"]))
    mu0 = np.array([prices[0], 0.0])
    C0  = np.diag([prices[0] ** 2 * 0.01, trend_var])

    filtered_means, _, _, total_ll, _ = run_rbpf(
        prices,
        N_particles=N_PARTICLES,
        theta=params["theta"],
        sigma=params["sigma"],
        sigma_obs_sq=sigma_obs_sq,
        lambda_J=params["lambda_J"],
        mu_J=params["mu_J"],
        sigma_J=params["sigma_J"],
        mu0=mu0,
        C0=C0,
        dt=DT,
        rng=np.random.default_rng(seed),
    )

    trend   = filtered_means[:, 1]  # x2_t
    signals = np.zeros(T)

    if use_fir:
        # Step 1: FIR momentum (§IV-D Steps 1-2)
        m = fir_momentum_signal(trend, n_taps=N_TAPS)
        signals[N_TAPS - 1:] = m

        # Step 2: IGARCH vol scaling (§IV-D Step 4)
        returns = np.diff(prices) / prices[:-1]          # length T-1
        sig_for_igarch = signals[1:]
        sigma2_init = returns[0] ** 2 if returns[0] != 0 else float(np.var(returns))
        scaled = igarch_volatility_scale(
            sig_for_igarch, returns,
            alpha=IGARCH_ALPHA,
            sigma2_init=sigma2_init,
        )
        # Normalise to [-1, 1]
        signals[1:] = np.clip(scaled / (np.std(scaled) + 1e-10), -1.0, 1.0)
    else:
        # Transfer function only (ablation)
        sigma_delta = 0.005 * prices[0]
        if len(trend) >= 2:
            signals[1:] = trend_to_trading_signal(trend, sigma_delta=sigma_delta)

    return signals, trend, total_ll


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"Experiment 17: RBPF on Intraday ES Futures ({FREQ} bars)")
    log("=" * 70)
    log(f"Symbol: {SYMBOL}  |  Period: {START} → {END}")
    log(f"Frequency: {FREQ}  |  dt = {DT:.6f} trading days per bar")
    log(f"N_particles: {N_PARTICLES}  |  Transaction cost: {TRANSACTION_COST_BPS} bps")
    log(f"Parameters: Table I (Christensen et al. 2012) — no estimation needed")

    # ── 1. Load and prepare data ──────────────────────────────────────────────
    log("\n--- 1. Loading data ---")
    df_raw  = load_futures_1m(SYMBOL, start=START, end=END)
    df_rth  = filter_rth(df_raw)
    df_bars = resample_bars(df_rth, FREQ) if FREQ != "1min" else df_rth

    prices = df_bars["close"].values.astype(float)
    index  = df_bars.index

    T       = len(prices)
    T_train = int(T * TRAIN_FRAC)
    T_test  = T - T_train

    train_prices = prices[:T_train]
    test_prices  = prices[T_train:]
    train_index  = index[:T_train]
    test_index   = index[T_train:]
    test_returns = np.diff(test_prices) / test_prices[:-1]

    log(f"Total {FREQ} RTH bars: {T:,}")
    log(f"Train: {str(train_index[0].date())} → {str(train_index[-1].date())} ({T_train:,} bars)")
    log(f"Test:  {str(test_index[0].date())} → {str(test_index[-1].date())} ({T_test:,} bars)")
    log(f"Price range (full): {prices.min():.2f} – {prices.max():.2f}")
    log(f"P_0 (test start):   {test_prices[0]:.2f}")

    # ── 2. Build Table I parameters ───────────────────────────────────────────
    log("\n--- 2. Table I parameters (Christensen et al. 2012) ---")
    P0 = test_prices[0]
    params_on  = _build_params("jumps_on",  P0)
    params_off = _build_params("jumps_off", P0)

    log(f"  {'Parameter':<14} {'Jumps ON':>12} {'Jumps OFF':>12}  {'Source'}")
    log(f"  {'-'*60}")
    log(f"  {'theta':<14} {params_on['theta']:>12.4f} {params_off['theta']:>12.4f}  Table I")
    log(f"  {'sigma':<14} {params_on['sigma']:>12.4f} {params_off['sigma']:>12.4f}  SF * P_0")
    log(f"  {'sigma_obs':<14} {params_on['sigma_obs']:>12.4f} {params_off['sigma_obs']:>12.4f}  SF * P_0")
    log(f"  {'lambda_J':<14} {params_on['lambda_J']:>12.4f} {params_off['lambda_J']:>12.4f}  Table I (per day)")
    log(f"  {'sigma_J':<14} {params_on['sigma_J']:>12.4f} {params_off['sigma_J']:>12.4f}  SF * P_0")
    log(f"  (dt = {DT:.6f} days/bar → lambda_J per bar ≈ {params_on['lambda_J'] * DT:.5f})")

    # ── 3. Run RBPF on test set ───────────────────────────────────────────────
    log(f"\n--- 3. Running RBPF (N={N_PARTICLES}) ---")

    log("  Jumps ON  — FIR + IGARCH pipeline ...")
    t1 = time.time()
    sig_on,  trend_on,  ll_on  = _run_rbpf_pipeline(test_prices, params_on,  use_fir=True, seed=SEED)
    log(f"  Done in {time.time()-t1:.1f}s  |  log-likelihood: {ll_on:.2f}")

    log("  Jumps OFF — FIR + IGARCH pipeline ...")
    t2 = time.time()
    sig_off, trend_off, ll_off = _run_rbpf_pipeline(test_prices, params_off, use_fir=True, seed=SEED)
    log(f"  Done in {time.time()-t2:.1f}s  |  log-likelihood: {ll_off:.2f}")

    log(f"  Jump effect on LL: {ll_on - ll_off:+.2f} nats")

    # ── 4. Backtests ──────────────────────────────────────────────────────────
    log("\n--- 4. Backtest results (test period) ---")

    def _bt(signals):
        return backtest(test_returns, signals[:-1],
                        transaction_cost_bps=TRANSACTION_COST_BPS)

    bh_signals = np.ones(T_test)
    res_on  = _bt(sig_on)
    res_off = _bt(sig_off)
    res_bh  = _bt(bh_signals)

    header = f"{'Strategy':<36}{'Sharpe':>8}{'Ann.Ret':>13}{'MaxDD':>14}{'Turnover':>11}"
    sep    = "-" * 82
    log(header)
    log(sep)
    log(_metric_row(f"RBPF {FREQ} jumps ON  (exp 17)",  res_on["metrics"]))
    log(_metric_row(f"RBPF {FREQ} jumps OFF (exp 17)",  res_off["metrics"]))
    log(_metric_row("Buy-and-Hold ES (exp 17)",          res_bh["metrics"]))
    log(sep)
    log(f"{'RBPF daily SPY (exp 15)':<36}{EXP15_SHARPE_RBPF:>8.2f}  (known, log-prices, 5bps)")
    log(f"{'Buy-and-hold SPY (exp 15)':<36}{EXP15_SHARPE_BH:>8.2f}  (known)")

    # ── 5. Key comparisons ────────────────────────────────────────────────────
    log("\n--- 5. Key comparisons ---")
    delta_freq  = res_on["metrics"]["sharpe"] - EXP15_SHARPE_RBPF
    delta_jumps = res_on["metrics"]["sharpe"] - res_off["metrics"]["sharpe"]
    delta_turn  = res_off["metrics"]["turnover"] - res_on["metrics"]["turnover"]
    log(f"  Frequency effect   (intraday {FREQ} vs daily):  ΔSharpe = {delta_freq:+.3f}")
    log(f"  Jump effect        (jumps ON vs OFF):           ΔSharpe = {delta_jumps:+.3f}")
    log(f"  Turnover reduction (jumps ON vs OFF):           ΔTurnover = {delta_turn:+.4f}")
    log(f"  LL improvement from jumps: {ll_on - ll_off:+.2f} nats")
    log("")
    if res_on["metrics"]["sharpe"] > EXP15_SHARPE_RBPF:
        log("  → Intraday frequency IMPROVES on daily RBPF.")
    else:
        log("  → Intraday frequency does NOT improve on daily RBPF.")
    if delta_jumps > 0:
        log("  → Jumps improve Sharpe.")
    else:
        log("  → Jumps do NOT improve Sharpe (but may improve LL).")

    # ── 6. Figures ────────────────────────────────────────────────────────────
    log("\n--- 6. Saving figures ---")

    # Figure 1: Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_index[1:], res_on["cumulative"],  label=f"RBPF {FREQ} jumps ON",  linewidth=1.8)
    ax.plot(test_index[1:], res_off["cumulative"], label=f"RBPF {FREQ} jumps OFF", linewidth=1.4,
            linestyle="--")
    ax.plot(test_index[1:], res_bh["cumulative"],  label="Buy-and-Hold", linewidth=1.4,
            linestyle=":", color="k")
    ax.set_title(f"Experiment 17: RBPF Intraday ES ({FREQ}, Table I params) — Out-of-Sample")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value ($1 invested)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_cumulative_returns.png", dpi=150)
    plt.close(fig)
    log("  figures/17_cumulative_returns.png")

    # Figure 2: Filtered trend + signal (jumps ON)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(test_index, test_prices, "k-", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("ES price ($)")
    axes[0].set_title(f"Experiment 17: RBPF Filtered Trend ({FREQ}, jumps ON)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(test_index, trend_on, color="steelblue", linewidth=0.7)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Filtered trend x₂ ($/bar)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(test_index, sig_on, color="darkorange", linewidth=0.6, alpha=0.9,
                 label="FIR+IGARCH signal")
    axes[2].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("Signal [-1, 1]")
    axes[2].set_xlabel("Date")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_filtered_trend.png", dpi=150)
    plt.close(fig)
    log("  figures/17_filtered_trend.png")

    # Figure 3: Turnover + Sharpe bar chart
    labels    = [f"RBPF {FREQ}\njumps ON", f"RBPF {FREQ}\njumps OFF", "RBPF daily\n(exp 15)"]
    turnovers = [res_on["metrics"]["turnover"], res_off["metrics"]["turnover"], EXP15_TURNOVER_RBPF]
    sharpes   = [res_on["metrics"]["sharpe"],   res_off["metrics"]["sharpe"],   EXP15_SHARPE_RBPF]
    colors    = ["steelblue", "darkorange", "gray"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(labels, turnovers, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax1.set_ylabel("Daily turnover")
    ax1.set_title("Turnover")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(labels, sharpes, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_ylabel("Out-of-sample Sharpe")
    ax2.set_title("Sharpe ratio")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 17: Intraday ES vs Daily SPY RBPF (Table I params)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_turnover_comparison.png", dpi=150)
    plt.close(fig)
    log("  figures/17_turnover_comparison.png")

    # ── 7. Save report ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    log(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    report_path = REPORTS_DIR / "17_rbpf_intraday_es.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
