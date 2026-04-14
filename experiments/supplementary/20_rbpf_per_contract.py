"""Experiment 20: Per-contract RBPF calibration + selective portfolio.

Diagnosis from Experiments 17-19:
  1. n_taps=4 at 1-min = 4-minute lookback, far too short for momentum.
  2. sf_obs and n_taps are substitute smoothers — exp 18 found sf_obs=35%
     only because n_taps=4 was fixed (the filter compensates by ignoring data).
  3. No single parameter set works across 26 diverse contracts.

Fix: calibrate EACH contract independently over (n_taps, sf_obs, theta),
then build a portfolio using ONLY contracts with positive training Sharpe.

Grid per contract (168 combinations)
-------------------------------------
  n_taps   : [4, 15, 30, 60, 120, 240, 390]   (7)  — 4 min to 1 day
  sf_obs   : [0.01, 0.1, 1.0, 5.0, 35.0] %    (5)  — tight to Table I
  theta    : [-0.1, -0.2, -0.5, -1.0]          (4)  — moderate to fast reversion
  IGARCH α : [0.01, 0.06]                       (2)  — slow vs RiskMetrics

  Fixed: sf_sigma = 0.5%, lambda_J = 5, sf_sigma_J = 6%, mu_J = 0

Total: 7 × 5 × 4 × 2 = 280 combos × 26 contracts × ~2 s (N=5) ≈ 255 min.
To keep runtime manageable: N=3, use last 100k training bars for search.

Selection protocol (no data leakage)
--------------------------------------
1. Per-contract grid search on training set → best params per contract.
2. Drop contracts with training Sharpe ≤ 0 (no momentum signal).
3. Re-run survivors with N=100 on test set.
4. Equal-weight portfolio of survivors.

Outputs
-------
  figures/20_per_contract_sharpes.png
  figures/20_portfolio_cumulative.png
  reports/20_rbpf_per_contract.txt
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.futures_loader import filter_rth, load_futures_1m
from src.langevin.rbpf_numba import run_rbpf_numba
from src.langevin.utils import fir_momentum_signal, igarch_volatility_scale
from src.strategy.backtest import backtest

# ── Configuration ─────────────────────────────────────────────────────────────

START = "2019-01-01"
END   = "2024-12-31"
DT    = 1.0 / 390.0

TRAIN_FRAC   = 0.70
N_GRID       = 3        # particles for grid search (fast)
N_FINAL      = 100      # particles for final evaluation
SEARCH_BARS  = 100_000  # use last N training bars for grid search (speed)
MIN_BARS     = 50_000
TC_BPS       = 2
SEED         = 42

# ── Grid ──────────────────────────────────────────────────────────────────────
NTAPS_GRID   = [4, 15, 30, 60, 120, 240, 390]
SF_OBS_GRID  = [0.01, 0.1, 1.0, 5.0, 35.0]     # % of P_0
THETA_GRID   = [-0.1, -0.2, -0.5, -1.0]
ALPHA_GRID   = [0.01, 0.06]

SF_SIGMA     = 0.5     # % of P_0, fixed
LAMBDA_J     = 5.0
SF_SIGMA_J   = 0.06

SYMBOLS = [
    "ES", "NQ", "YM", "RTY",
    "6A", "6B", "6C", "6E", "6J", "6N", "6S",
    "ZB", "ZF", "ZN", "ZT",
    "CL", "HO", "NG", "RB",
    "ZC", "ZL", "ZM", "ZS", "ZW",
    "HE", "LE",
]

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eval_combo(prices, theta, sf_obs, ntaps, alpha, N, seed):
    """Run RBPF + FIR(ntaps) + IGARCH(alpha) → Sharpe."""
    T   = len(prices)
    P0  = prices[0]
    sigma        = SF_SIGMA * P0 / 100.0
    sigma_obs_sq = (sf_obs * P0 / 100.0) ** 2
    sigma_J      = SF_SIGMA_J * P0
    trend_var    = sigma ** 2 / (2.0 * abs(theta))
    mu0 = np.array([P0, 0.0])
    C0  = np.diag([P0 ** 2 * 0.01, trend_var])

    fmeans, _, _, _, _ = run_rbpf_numba(
        prices, N_particles=N,
        theta=theta, sigma=sigma, sigma_obs_sq=sigma_obs_sq,
        lambda_J=LAMBDA_J, mu_J=0.0, sigma_J=sigma_J,
        mu0=mu0, C0=C0, dt=DT,
        rng=np.random.default_rng(seed),
    )

    trend   = fmeans[:, 1]
    if len(trend) < ntaps:
        return np.nan, 0.0

    signals = np.zeros(T)
    m       = fir_momentum_signal(trend, n_taps=ntaps)
    signals[ntaps - 1:] = m

    returns    = np.diff(prices) / prices[:-1]
    sig_ig     = signals[1:]
    s2_init    = returns[0] ** 2 if returns[0] != 0 else float(np.var(returns))
    if s2_init <= 0:
        s2_init = float(np.var(returns)) if np.var(returns) > 0 else 1e-10

    scaled = igarch_volatility_scale(sig_ig, returns, alpha=alpha, sigma2_init=s2_init)
    std_sc = np.std(scaled)
    if std_sc < 1e-12:
        return np.nan, 0.0
    signals[1:] = np.clip(scaled / std_sc, -1.0, 1.0)

    res = backtest(returns, signals[:-1], transaction_cost_bps=TC_BPS)
    return float(res["metrics"]["sharpe"]), float(res["metrics"]["turnover"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    report: list[str] = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(NTAPS_GRID, SF_OBS_GRID, THETA_GRID, ALPHA_GRID))
    n_combos = len(combos)

    log("=" * 72)
    log("Experiment 20: Per-contract RBPF calibration + selective portfolio")
    log("=" * 72)
    log(f"Grid per contract: {len(NTAPS_GRID)}×{len(SF_OBS_GRID)}×{len(THETA_GRID)}×{len(ALPHA_GRID)}"
        f" = {n_combos} combos  |  N_GRID={N_GRID}")
    log(f"Search window: last {SEARCH_BARS:,} training bars per contract")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log("\n--- 1. Loading contracts ---")
    data: dict[str, dict] = {}
    for sym in SYMBOLS:
        try:
            df = filter_rth(load_futures_1m(sym, start=START, end=END))
            px = df["close"].values.astype(float)
            T  = len(px)
            Ttr = int(T * TRAIN_FRAC)
            if Ttr < MIN_BARS:
                log(f"  {sym:6s} SKIP (only {Ttr:,} train bars)")
                continue
            data[sym] = {
                "train_full": px[:Ttr],
                "train_search": px[max(0, Ttr - SEARCH_BARS):Ttr],
                "test": px[Ttr:],
                "test_index": df.index[Ttr:],
            }
            log(f"  {sym:6s}  train={Ttr:,}  search={len(data[sym]['train_search']):,}  "
                f"test={T-Ttr:,}")
        except Exception as e:
            log(f"  {sym:6s} ERROR: {e}")

    active = list(data.keys())
    log(f"\n  Active: {len(active)} contracts")

    # ── 2. JIT warm-up ────────────────────────────────────────────────────────
    log("\n--- 2. JIT warm-up ---")
    px0 = data[active[0]]["train_search"][:500]
    P0 = px0[0]
    t_jit = time.time()
    _eval_combo(px0, -0.2, 1.0, 30, 0.06, N=N_GRID, seed=SEED)
    log(f"  JIT: {time.time()-t_jit:.1f}s")

    # Time one full search-window run
    px_bench = data[active[0]]["train_search"]
    t_b = time.time()
    _eval_combo(px_bench, -0.2, 1.0, 30, 0.06, N=N_GRID, seed=SEED)
    per_run = time.time() - t_b
    eta = n_combos * len(active) * per_run / 60.0
    log(f"  One run (N={N_GRID}, {len(px_bench):,} bars): {per_run:.2f}s")
    log(f"  ETA: {n_combos} combos × {len(active)} contracts ≈ {eta:.0f} min")

    # ── 3. Per-contract grid search ──────────────────────────────────────────
    log(f"\n--- 3. Per-contract grid search ---")

    best_per_contract: dict[str, dict] = {}
    t_grid = time.time()

    for si, sym in enumerate(active):
        px_search = data[sym]["train_search"]
        best_sh = -np.inf
        best_combo = combos[0]
        best_turn = 0.0

        for ntaps, sf_obs, theta, alpha in combos:
            sh, turn = _eval_combo(px_search, theta, sf_obs, ntaps, alpha,
                                   N=N_GRID, seed=SEED)
            if np.isfinite(sh) and sh > best_sh:
                best_sh = sh
                best_combo = (ntaps, sf_obs, theta, alpha)
                best_turn = turn

        ntaps_b, sfobs_b, theta_b, alpha_b = best_combo
        best_per_contract[sym] = {
            "ntaps": ntaps_b, "sf_obs": sfobs_b,
            "theta": theta_b, "alpha": alpha_b,
            "train_sharpe": best_sh, "train_turnover": best_turn,
        }

        elapsed = time.time() - t_grid
        eta_left = (len(active) - si - 1) * elapsed / (si + 1) / 60.0
        marker = " ✓" if best_sh > 0 else ""
        log(f"  {sym:6s}  ntaps={ntaps_b:>3}  sf_obs={sfobs_b:>5.2f}%  "
            f"θ={theta_b:>5.2f}  α={alpha_b:.2f}  "
            f"train_Sharpe={best_sh:>+7.3f}  turn={best_turn:.4f}{marker}"
            f"  [ETA {eta_left:.0f}m]")

    grid_time = (time.time() - t_grid) / 60.0
    log(f"\n  Grid search done in {grid_time:.1f} min")

    # ── 4. Select profitable contracts ────────────────────────────────────────
    survivors = {s: v for s, v in best_per_contract.items() if v["train_sharpe"] > 0}
    losers    = {s: v for s, v in best_per_contract.items() if v["train_sharpe"] <= 0}

    log(f"\n--- 4. Contract selection ---")
    log(f"  Positive training Sharpe: {len(survivors)} contracts")
    log(f"  Dropped (≤ 0 training Sharpe): {len(losers)} contracts")
    if survivors:
        log(f"  Survivors: {', '.join(sorted(survivors.keys()))}")
    else:
        log("  WARNING: No contracts have positive training Sharpe!")
        log("  Falling back to top-5 least-negative contracts.")
        sorted_all = sorted(best_per_contract.items(),
                            key=lambda x: x[1]["train_sharpe"], reverse=True)
        survivors = {s: v for s, v in sorted_all[:5]}
        log(f"  Using top-5: {', '.join(sorted(survivors.keys()))}")

    # ── 5. Test-set evaluation with N_FINAL ──────────────────────────────────
    log(f"\n--- 5. Test-set evaluation (N={N_FINAL}) ---")

    test_results: dict[str, dict] = {}
    for sym, params in sorted(survivors.items()):
        px_test = data[sym]["test"]
        sh_test, turn_test = _eval_combo(
            px_test, params["theta"], params["sf_obs"], params["ntaps"],
            params["alpha"], N=N_FINAL, seed=SEED,
        )
        test_results[sym] = {
            "test_sharpe": sh_test, "test_turnover": turn_test,
            **params,
        }
        log(f"  {sym:6s}  test_Sharpe={sh_test:>+7.3f}  "
            f"(train={params['train_sharpe']:>+7.3f}  "
            f"gap={params['train_sharpe']-sh_test:>+.3f})")

    # ── 6. Portfolio metrics ──────────────────────────────────────────────────
    log(f"\n--- 6. Portfolio results ---")

    # Compute portfolio: average net returns across survivor contracts
    surv_list = sorted(test_results.keys())
    ref_sym = surv_list[0]
    T_ref = len(data[ref_sym]["test"]) - 1  # returns length

    all_net_rets = []
    for sym in surv_list:
        px  = data[sym]["test"]
        p   = test_results[sym]
        T_s = len(px)
        P0  = px[0]

        sigma        = SF_SIGMA * P0 / 100.0
        sigma_obs_sq = (p["sf_obs"] * P0 / 100.0) ** 2
        sigma_J      = SF_SIGMA_J * P0
        trend_var    = sigma ** 2 / (2.0 * abs(p["theta"]))
        mu0 = np.array([P0, 0.0])
        C0  = np.diag([P0 ** 2 * 0.01, trend_var])

        fmeans, _, _, _, _ = run_rbpf_numba(
            px, N_particles=N_FINAL,
            theta=p["theta"], sigma=sigma, sigma_obs_sq=sigma_obs_sq,
            lambda_J=LAMBDA_J, mu_J=0.0, sigma_J=sigma_J,
            mu0=mu0, C0=C0, dt=DT,
            rng=np.random.default_rng(SEED),
        )

        trend   = fmeans[:, 1]
        signals = np.zeros(T_s)
        m       = fir_momentum_signal(trend, n_taps=p["ntaps"])
        signals[p["ntaps"] - 1:] = m

        returns  = np.diff(px) / px[:-1]
        sig_ig   = signals[1:]
        s2_init  = returns[0] ** 2 if returns[0] != 0 else float(np.var(returns))
        if s2_init <= 0:
            s2_init = float(np.var(returns)) if np.var(returns) > 0 else 1e-10
        scaled   = igarch_volatility_scale(sig_ig, returns,
                                           alpha=p["alpha"], sigma2_init=s2_init)
        std_sc = np.std(scaled)
        if std_sc > 1e-12:
            signals[1:] = np.clip(scaled / std_sc, -1.0, 1.0)

        s     = signals[:-1]
        delta = np.abs(np.diff(np.concatenate([[0.0], s])))
        tc    = (TC_BPS / 10000.0) * delta
        nr    = s * returns - tc

        # Truncate/pad to common length (T_ref) using the LAST T_ref bars
        T_use = min(len(nr), T_ref)
        padded = np.zeros(T_ref)
        padded[-T_use:] = nr[-T_use:]
        all_net_rets.append(padded)

    port_ret = np.mean(all_net_rets, axis=0)
    port_cum = np.cumprod(1.0 + port_ret)

    # Buy-and-hold equal weight
    bh_rets = []
    for sym in surv_list:
        ret = data[sym]["test"]
        ret_pct = np.diff(ret) / ret[:-1]
        T_use = min(len(ret_pct), T_ref)
        padded = np.zeros(T_ref)
        padded[-T_use:] = ret_pct[-T_use:]
        bh_rets.append(padded)
    bh_ret = np.mean(bh_rets, axis=0)
    bh_cum  = np.cumprod(1.0 + bh_ret)

    # Portfolio Sharpe
    bars_per_year = 252 * 390
    port_std = np.std(port_ret)
    port_sharpe = port_ret.mean() / (port_std + 1e-15) * np.sqrt(bars_per_year)
    bh_std = np.std(bh_ret)
    bh_sharpe = bh_ret.mean() / (bh_std + 1e-15) * np.sqrt(bars_per_year)

    port_cum_final = port_cum[-1]
    bh_cum_final   = bh_cum[-1]
    port_dd = np.min(port_cum / np.maximum.accumulate(port_cum) - 1.0)
    bh_dd   = np.min(bh_cum / np.maximum.accumulate(bh_cum) - 1.0)

    log(f"\n  Portfolio ({len(surv_list)} contracts, per-contract calibration):")
    log(f"    Sharpe:    {port_sharpe:+.4f}")
    log(f"    Final NAV: {port_cum_final:.4f}  (${port_cum_final:.2f} from $1)")
    log(f"    Max DD:    {port_dd*100:.1f}%")
    log(f"\n  Buy-and-Hold ({len(surv_list)} survivor contracts):")
    log(f"    Sharpe:    {bh_sharpe:+.4f}")
    log(f"    Final NAV: {bh_cum_final:.4f}")
    log(f"    Max DD:    {bh_dd*100:.1f}%")
    log(f"\n  Baselines (from Experiments 17-19):")
    log(f"    Single ES Table I (exp 17): Sharpe = -0.20")
    log(f"    Same-params portfolio (exp 19): Sharpe = -3.38")
    log(f"    Paper (75 contracts, 2006-2011): Sharpe = 1.82")

    mean_indiv_test = np.mean([v["test_sharpe"] for v in test_results.values()
                               if np.isfinite(v["test_sharpe"])])
    log(f"\n  Mean individual test Sharpe (survivors): {mean_indiv_test:+.4f}")
    log(f"  Diversification ratio: portfolio / mean = "
        f"{port_sharpe / mean_indiv_test:.2f}x" if abs(mean_indiv_test) > 1e-6
        else "  (undefined)")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    log("\n--- 7. Figures ---")

    # Per-contract bar chart: train vs test Sharpe
    all_syms_sorted = sorted(best_per_contract.keys(),
                             key=lambda s: best_per_contract[s]["train_sharpe"],
                             reverse=True)
    train_sharpes = [best_per_contract[s]["train_sharpe"] for s in all_syms_sorted]
    test_sharpes  = [test_results[s]["test_sharpe"]
                     if s in test_results else float("nan")
                     for s in all_syms_sorted]
    colors_train = ["steelblue" if v > 0 else "lightcoral" for v in train_sharpes]

    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(all_syms_sorted))
    width = 0.38
    ax.bar(x - width/2, train_sharpes, width, color=colors_train,
           edgecolor="k", linewidth=0.4, alpha=0.85, label="Train (best N=3)")
    # Plot test Sharpe for survivors only
    test_vals = []
    test_x    = []
    for i, s in enumerate(all_syms_sorted):
        if s in test_results and np.isfinite(test_results[s]["test_sharpe"]):
            test_vals.append(test_results[s]["test_sharpe"])
            test_x.append(i)
    if test_vals:
        colors_test = ["darkblue" if v > 0 else "darkred" for v in test_vals]
        ax.bar(np.array(test_x) + width/2, test_vals, width,
               color=colors_test, edgecolor="k", linewidth=0.4,
               alpha=0.65, label=f"Test (N={N_FINAL})")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(port_sharpe, color="blue", linewidth=1.5, linestyle="--",
               label=f"Portfolio Sharpe = {port_sharpe:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(all_syms_sorted, rotation=45, ha="right")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title(f"Exp 20: Per-contract calibration ({n_combos} combos/contract)\n"
                 f"Survivors: {len(surv_list)} | Portfolio Sharpe: {port_sharpe:.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "20_per_contract_sharpes.png", dpi=150)
    plt.close(fig)
    log("  figures/20_per_contract_sharpes.png")

    # Cumulative returns
    ref_idx = data[ref_sym]["test_index"]
    plot_idx = ref_idx[1:T_ref+1] if len(ref_idx) > T_ref else ref_idx[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_idx, port_cum,
            label=f"Portfolio ({len(surv_list)} contracts, Sharpe={port_sharpe:.3f})",
            linewidth=1.8, color="steelblue")
    ax.plot(plot_idx, bh_cum,
            label=f"Buy-and-Hold ({len(surv_list)} contracts, Sharpe={bh_sharpe:.3f})",
            linewidth=1.4, linestyle=":", color="k")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(f"Exp 20: Per-contract calibrated RBPF portfolio — Out-of-Sample")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value ($1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "20_portfolio_cumulative.png", dpi=150)
    plt.close(fig)
    log("  figures/20_portfolio_cumulative.png")

    total = time.time() - t_start
    log(f"\nTotal runtime: {total:.0f}s ({total/60:.1f} min)")

    rp = REPORTS_DIR / "20_rbpf_per_contract.txt"
    rp.write_text("\n".join(report))
    print(f"\nReport saved to {rp}")


if __name__ == "__main__":
    main()
