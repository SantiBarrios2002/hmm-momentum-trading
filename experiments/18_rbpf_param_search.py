"""Experiment 18: Parameter grid search for 1-min RBPF on ES futures.

The Table I parameters from Christensen et al. (2012) were calibrated on the
paper's own dataset and produce near-zero Kalman gain at 1-min frequency
(sigma_obs = 35% × P_0 is enormous for liquid futures).  This experiment
searches for parameters that actually work at 1-min resolution.

Grid search strategy
--------------------
Three parameters are varied:
  sf_sigma_obs  — observation noise as % of P_0  (9 values, log-spaced)
  sf_sigma      — trend diffusion as % of P_0    (7 values, log-spaced)
  theta         — mean-reversion coefficient      (6 values)

All other parameters are held at the paper values:
  lambda_J = 5  (jumps per day),  sf_sigma_J = 6%,  mu_J = 0,  N_TAPS = 4

Total grid: 9 × 7 × 6 = 378 combinations.

Selection protocol (no data leakage)
--------------------------------------
1.  Grid search is run on the TRAINING set only (first 70% of data).
    N_GRID = 20 particles — sufficient to rank combinations; fast with Numba.
2.  Top-10 training-Sharpe combinations are re-evaluated on the TRAINING set
    with N_FINAL = 100 particles to get a stable ranking.
3.  The single best combination is applied to the TEST set exactly once.
    Test results are reported only for the winner.

The test set is touched exactly once, at the end, with the best parameters.

Runtime estimate
----------------
378 × ~12 s/run (N=20, 409 k bars) ≈ 75 min total.
Progress is printed every 20 runs with a live ETA.

Outputs
-------
  figures/18_grid_heatmap.png          — training Sharpe heat-map (best theta)
  figures/18_cumulative_returns.png    — best params vs Table I vs buy-and-hold
  reports/18_rbpf_param_search.txt     — full results table

References
----------
Christensen, Murphy & Godsill (2012), IEEE JSTSP, Table I.
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

from src.data.futures_loader import filter_rth, load_futures_1m, resample_bars
from src.langevin.rbpf_numba import run_rbpf_numba
from src.langevin.utils import fir_momentum_signal, igarch_volatility_scale
from src.strategy.backtest import backtest

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL = "ES"
START  = "2019-01-01"
END    = "2024-12-31"
FREQ   = "1min"
DT     = 1.0 / 390.0          # 1 trading-min / 390 min per RTH day

TRAIN_FRAC = 0.70
N_GRID     = 20               # particles during grid search  (fast)
N_FINAL    = 100              # particles for final evaluation (accurate)
N_TOP      = 10               # top-N candidates re-evaluated before final pick

N_TAPS          = 4
IGARCH_ALPHA    = 0.06
TRANSACTION_COST_BPS = 2
SEED = 42

# ── Parameter grid ─────────────────────────────────────────────────────────────
# sf_sigma_obs: % of P_0 — log-spaced from 0.1 bp to 35 % (Table I)
SF_SIGMA_OBS_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 5.0, 35.0]

# sf_sigma: % of P_0 — log-spaced from tiny to Table-I-jumps-OFF value
SF_SIGMA_GRID = [0.005, 0.02, 0.05, 0.15, 0.5, 1.5, 3.5]

# theta: mean-reversion speed (per trading day)
# -0.05 → half-life 14 days;  -2.0 → half-life 0.35 days ≈ 8.5 h
THETA_GRID = [-0.05, -0.1, -0.2, -0.5, -1.0, -2.0]

# Fixed at paper values
LAMBDA_J  = 5.0    # jumps per trading day
SF_SIGMA_J = 0.06  # 6% of P_0

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Known baselines
EXP17_SHARPE_TABLE_I = -0.20   # jumps ON, 1-min, Table I params (exp 17)
EXP17_SHARPE_BH      =  0.10   # buy-and-hold, same test period


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_single(
    prices: np.ndarray,
    theta:  float,
    sigma:  float,
    sigma_obs_sq: float,
    sigma_J: float,
    N: int,
    seed: int,
) -> float:
    """Run RBPF + FIR + IGARCH pipeline and return Sharpe ratio.

    Returns np.nan if the backtest produces a degenerate signal (all-zero
    returns), which can happen when sigma_obs is so large that the Kalman
    gain collapses to zero and the filter outputs a flat trend.

    Parameters
    ----------
    prices       : raw price series, shape (T,)
    theta        : mean-reversion coefficient (< 0)
    sigma        : trend diffusion (absolute price units)
    sigma_obs_sq : observation noise variance
    sigma_J      : jump size std (absolute price units)
    N            : number of particles
    seed         : RNG seed

    Returns
    -------
    sharpe : float or np.nan
    """
    T            = len(prices)
    trend_var    = sigma ** 2 / (2.0 * abs(theta))
    mu0          = np.array([prices[0], 0.0])
    C0           = np.diag([prices[0] ** 2 * 0.01, trend_var])

    filtered_means, _, _, _, _ = run_rbpf_numba(
        prices,
        N_particles=N,
        theta=theta,
        sigma=sigma,
        sigma_obs_sq=sigma_obs_sq,
        lambda_J=LAMBDA_J,
        mu_J=0.0,
        sigma_J=sigma_J,
        mu0=mu0,
        C0=C0,
        dt=DT,
        rng=np.random.default_rng(seed),
    )

    trend   = filtered_means[:, 1]
    signals = np.zeros(T)

    # FIR momentum signal (§IV-D)
    m = fir_momentum_signal(trend, n_taps=N_TAPS)
    signals[N_TAPS - 1:] = m

    # IGARCH volatility scaling
    returns      = np.diff(prices) / prices[:-1]
    sig_for_igc  = signals[1:]
    sigma2_init  = returns[0] ** 2 if returns[0] != 0 else float(np.var(returns))
    scaled       = igarch_volatility_scale(
        sig_for_igc, returns,
        alpha=IGARCH_ALPHA,
        sigma2_init=sigma2_init,
    )
    std_scaled = np.std(scaled)
    if std_scaled < 1e-12:       # degenerate: flat signal
        return np.nan
    signals[1:] = np.clip(scaled / std_scaled, -1.0, 1.0)

    bt = backtest(returns, signals[:-1],
                  transaction_cost_bps=TRANSACTION_COST_BPS)
    return float(bt["metrics"]["sharpe"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("Experiment 18: RBPF 1-min ES — Parameter Grid Search")
    log("=" * 72)
    log(f"Symbol: {SYMBOL}  |  Period: {START} → {END}  |  Freq: {FREQ}")
    log(f"Grid: {len(SF_SIGMA_OBS_GRID)} × {len(SF_SIGMA_GRID)} × {len(THETA_GRID)} "
        f"= {len(SF_SIGMA_OBS_GRID)*len(SF_SIGMA_GRID)*len(THETA_GRID)} combinations")
    log(f"N_GRID={N_GRID} (search)  N_FINAL={N_FINAL} (evaluation)")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log("\n--- 1. Loading 1-min ES data ---")
    df_raw  = load_futures_1m(SYMBOL, start=START, end=END)
    df_rth  = filter_rth(df_raw)
    prices  = df_rth["close"].values.astype(float)
    index   = df_rth.index

    T       = len(prices)
    T_train = int(T * TRAIN_FRAC)

    train_prices = prices[:T_train]
    test_prices  = prices[T_train:]
    train_idx    = index[:T_train]
    test_idx     = index[T_train:]
    test_returns = np.diff(test_prices) / test_prices[:-1]

    log(f"Total 1-min RTH bars: {T:,}")
    log(f"Train: {str(train_idx[0].date())} → {str(train_idx[-1].date())} ({T_train:,} bars)")
    log(f"Test : {str(test_idx[0].date())} → {str(test_idx[-1].date())} ({T - T_train:,} bars)")

    P0 = test_prices[0]
    log(f"P_0 (test start): {P0:.2f}")

    # ── 2. Timing warm-up (trigger Numba JIT, measure per-run cost) ───────────
    log("\n--- 2. JIT warm-up ---")
    t_jit = time.time()
    _run_single(train_prices[:500], -0.2, 0.01 * P0, (0.01 * P0) ** 2,
                SF_SIGMA_J * P0, N=N_GRID, seed=SEED)
    jit_time = time.time() - t_jit
    log(f"  JIT compile + 500-bar run: {jit_time:.1f}s (one-time cost)")

    # Estimate per-run time with full training set
    t_bench = time.time()
    _run_single(train_prices, -0.2, 0.01 * P0, (0.01 * P0) ** 2,
                SF_SIGMA_J * P0, N=N_GRID, seed=SEED)
    per_run = time.time() - t_bench
    n_combos = len(SF_SIGMA_OBS_GRID) * len(SF_SIGMA_GRID) * len(THETA_GRID)
    eta_min  = n_combos * per_run / 60.0
    log(f"  Full training-set run (N={N_GRID}): {per_run:.1f}s")
    log(f"  ETA for {n_combos} grid points: {eta_min:.0f} min")

    # ── 3. Grid search on training set ────────────────────────────────────────
    log(f"\n--- 3. Grid search ({n_combos} combinations, N={N_GRID}) ---")

    all_combos = list(itertools.product(
        SF_SIGMA_OBS_GRID, SF_SIGMA_GRID, THETA_GRID
    ))
    results: list[dict] = []

    t_grid = time.time()
    for idx_c, (sf_obs, sf_sig, theta) in enumerate(all_combos):
        sigma        = sf_sig * P0 / 100.0
        sigma_obs_sq = (sf_obs * P0 / 100.0) ** 2
        sigma_J      = SF_SIGMA_J * P0

        sharpe = _run_single(
            train_prices, theta, sigma, sigma_obs_sq, sigma_J,
            N=N_GRID, seed=SEED,
        )
        results.append({
            "sf_obs": sf_obs, "sf_sig": sf_sig, "theta": theta,
            "sigma": sigma, "sigma_obs_sq": sigma_obs_sq,
            "train_sharpe_fast": sharpe,
        })

        # Progress
        if (idx_c + 1) % 20 == 0 or idx_c == n_combos - 1:
            elapsed  = time.time() - t_grid
            rate     = (idx_c + 1) / elapsed
            eta_left = (n_combos - idx_c - 1) / rate / 60.0
            valid    = sum(1 for r in results if not np.isnan(r["train_sharpe_fast"]))
            best_so_far = max(
                (r["train_sharpe_fast"] for r in results
                 if not np.isnan(r["train_sharpe_fast"])),
                default=float("nan"),
            )
            print(
                f"  [{idx_c+1:3d}/{n_combos}] "
                f"valid={valid}  best_train={best_so_far:.3f}  "
                f"ETA={eta_left:.1f} min"
            )

    log(f"  Grid search done in {(time.time()-t_grid)/60:.1f} min")

    # Sort by training Sharpe (descending), drop NaN
    results_valid = [r for r in results if not np.isnan(r["train_sharpe_fast"])]
    results_valid.sort(key=lambda r: r["train_sharpe_fast"], reverse=True)
    n_valid = len(results_valid)
    log(f"  Valid results: {n_valid}/{n_combos}  "
        f"(NaN = flat signal → sigma_obs too large)")

    log("\n  Top-20 by training Sharpe (fast, N=20):")
    log(f"  {'sf_obs%':>8} {'sf_sig%':>8} {'theta':>7} {'train_sharpe':>13}")
    log("  " + "-" * 40)
    for r in results_valid[:20]:
        log(f"  {r['sf_obs']:>8.3f} {r['sf_sig']:>8.3f} {r['theta']:>7.2f} "
            f"{r['train_sharpe_fast']:>13.4f}")

    # ── 4. Re-evaluate top-N candidates with N_FINAL particles ───────────────
    top_candidates = results_valid[:N_TOP]
    log(f"\n--- 4. Re-evaluating top-{N_TOP} with N={N_FINAL} on training set ---")

    for r in top_candidates:
        sharpe_final = _run_single(
            train_prices, r["theta"], r["sigma"], r["sigma_obs_sq"],
            SF_SIGMA_J * P0, N=N_FINAL, seed=SEED,
        )
        r["train_sharpe_final"] = sharpe_final
        print(f"  sf_obs={r['sf_obs']:.3f}% sf_sig={r['sf_sig']:.3f}% "
              f"theta={r['theta']:.2f}  train_sharpe(N=100)={sharpe_final:.4f}")

    top_candidates.sort(key=lambda r: r["train_sharpe_final"], reverse=True)
    best = top_candidates[0]

    log(f"\n  Best parameters (by N={N_FINAL} training Sharpe):")
    log(f"    sf_sigma_obs = {best['sf_obs']:.3f}%   "
        f"→ sigma_obs = {np.sqrt(best['sigma_obs_sq']):.4f} $/bar")
    log(f"    sf_sigma     = {best['sf_sig']:.3f}%   "
        f"→ sigma     = {best['sigma']:.4f} $/bar")
    log(f"    theta        = {best['theta']:.3f}")
    log(f"    lambda_J     = {LAMBDA_J}  (fixed, paper value)")
    log(f"    sf_sigma_J   = {SF_SIGMA_J*100:.1f}%  (fixed, paper value)")
    log(f"    Training Sharpe (N={N_FINAL}): {best['train_sharpe_final']:.4f}")

    # ── 5. Final test-set evaluation (touched exactly once) ───────────────────
    log(f"\n--- 5. Test-set evaluation (N={N_FINAL}, one shot) ---")

    test_sharpe_best = _run_single(
        test_prices, best["theta"], best["sigma"], best["sigma_obs_sq"],
        SF_SIGMA_J * P0, N=N_FINAL, seed=SEED,
    )
    log(f"  Best params  — test Sharpe: {test_sharpe_best:.4f}")
    log(f"  Table I      — test Sharpe: {EXP17_SHARPE_TABLE_I:.4f}  (from exp 17)")
    log(f"  Buy-and-Hold — test Sharpe: {EXP17_SHARPE_BH:.4f}  (from exp 17)")

    improvement = test_sharpe_best - EXP17_SHARPE_TABLE_I
    log(f"  Improvement over Table I: {improvement:+.4f} Sharpe points")

    # Full backtest for cumulative returns plot
    T_test  = len(test_prices)
    sigma   = best["sigma"]
    s_obs_sq = best["sigma_obs_sq"]
    theta   = best["theta"]
    sigma_J = SF_SIGMA_J * P0

    trend_var = sigma ** 2 / (2.0 * abs(theta))
    mu0_t = np.array([test_prices[0], 0.0])
    C0_t  = np.diag([test_prices[0] ** 2 * 0.01, trend_var])
    fmeans, _, _, _, _ = run_rbpf_numba(
        test_prices, N_particles=N_FINAL,
        theta=theta, sigma=sigma, sigma_obs_sq=s_obs_sq,
        lambda_J=LAMBDA_J, mu_J=0.0, sigma_J=sigma_J,
        mu0=mu0_t, C0=C0_t, dt=DT,
        rng=np.random.default_rng(SEED),
    )
    trend_best = fmeans[:, 1]
    sigs_best  = np.zeros(T_test)
    m_best     = fir_momentum_signal(trend_best, n_taps=N_TAPS)
    sigs_best[N_TAPS - 1:] = m_best
    ret_test   = np.diff(test_prices) / test_prices[:-1]
    s2_init    = ret_test[0] ** 2 if ret_test[0] != 0 else float(np.var(ret_test))
    sc = igarch_volatility_scale(sigs_best[1:], ret_test,
                                  alpha=IGARCH_ALPHA, sigma2_init=s2_init)
    std_sc = np.std(sc)
    if std_sc > 1e-12:
        sigs_best[1:] = np.clip(sc / std_sc, -1.0, 1.0)

    res_best = backtest(ret_test, sigs_best[:-1],
                        transaction_cost_bps=TRANSACTION_COST_BPS)
    res_bh   = backtest(ret_test, np.ones(T_test - 1),
                        transaction_cost_bps=TRANSACTION_COST_BPS)

    log(f"\n--- 6. Test-set backtest metrics ---")
    header = f"{'Strategy':<42}{'Sharpe':>8}{'Ann.Ret':>13}{'MaxDD':>14}{'Turnover':>11}"
    sep    = "-" * 88
    def row(name, m):
        return (f"{name:<42}{m['sharpe']:>8.2f}"
                f"{m['annualized_return']*100:>12.1f}%"
                f"{m['max_drawdown']*100:>13.1f}%"
                f"{m['turnover']:>11.4f}")
    log(header)
    log(sep)
    log(row(f"Best params (sf_obs={best['sf_obs']:.3f}%, θ={best['theta']:.2f})",
            res_best["metrics"]))
    log(row("Buy-and-Hold ES", res_bh["metrics"]))
    log(sep)
    log(f"  Table I (exp 17, 1-min, jumps ON):  Sharpe = {EXP17_SHARPE_TABLE_I:.2f}")

    # ── 6. Figures ─────────────────────────────────────────────────────────────
    log("\n--- 7. Saving figures ---")

    # Figure 1: heat-map — training Sharpe vs sf_obs × sf_sig (best theta)
    best_theta = best["theta"]
    hm_data = np.full((len(SF_SIGMA_GRID), len(SF_SIGMA_OBS_GRID)), np.nan)
    for r in results_valid:
        if r["theta"] == best_theta:
            i = SF_SIGMA_GRID.index(r["sf_sig"])
            j = SF_SIGMA_OBS_GRID.index(r["sf_obs"])
            hm_data[i, j] = r["train_sharpe_fast"]

    fig, ax = plt.subplots(figsize=(11, 5))
    vmax = np.nanmax(np.abs(hm_data))
    im = ax.imshow(hm_data, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, label="Training Sharpe")
    ax.set_xticks(range(len(SF_SIGMA_OBS_GRID)))
    ax.set_xticklabels([f"{v:.3g}%" for v in SF_SIGMA_OBS_GRID], rotation=45, ha="right")
    ax.set_yticks(range(len(SF_SIGMA_GRID)))
    ax.set_yticklabels([f"{v:.3g}%" for v in SF_SIGMA_GRID])
    ax.set_xlabel("SF(σ_obs)  [% of P₀]")
    ax.set_ylabel("SF(σ)  [% of P₀]")
    ax.set_title(
        f"Exp 18: Training Sharpe heat-map  (θ = {best_theta}, N={N_GRID})\n"
        f"Table I is at SF(σ_obs)=35%, SF(σ)=0.35%  |  optimal highlighted"
    )
    # Mark best cell
    bi = SF_SIGMA_GRID.index(best["sf_sig"])
    bj = SF_SIGMA_OBS_GRID.index(best["sf_obs"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=2.5))
    ax.text(bj, bi, "★", ha="center", va="center", fontsize=14, color="blue")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_grid_heatmap.png", dpi=150)
    plt.close(fig)
    log("  figures/18_grid_heatmap.png")

    # Figure 2: cumulative returns — best params vs buy-and-hold
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_idx[1:], res_best["cumulative"],
            label=f"Best params (Sharpe={res_best['metrics']['sharpe']:.2f})",
            linewidth=1.8, color="steelblue")
    ax.plot(test_idx[1:], res_bh["cumulative"],
            label=f"Buy-and-Hold (Sharpe={res_bh['metrics']['sharpe']:.2f})",
            linewidth=1.4, linestyle=":", color="k")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(
        f"Exp 18: Best RBPF params — Out-of-Sample 1-min ES\n"
        f"SF(σ_obs)={best['sf_obs']:.3f}%, SF(σ)={best['sf_sig']:.3f}%, "
        f"θ={best['theta']:.2f}  vs  Table I Sharpe = {EXP17_SHARPE_TABLE_I:.2f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value ($1 invested)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_cumulative_returns.png", dpi=150)
    plt.close(fig)
    log("  figures/18_cumulative_returns.png")

    # Figure 3: training Sharpe vs theta for best (sf_obs, sf_sig) pair
    theta_sharpes = []
    for r in results_valid:
        if r["sf_obs"] == best["sf_obs"] and r["sf_sig"] == best["sf_sig"]:
            theta_sharpes.append((r["theta"], r["train_sharpe_fast"]))
    theta_sharpes.sort(key=lambda x: x[0])
    thetas_plot, sharpes_plot = zip(*theta_sharpes)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thetas_plot, sharpes_plot, "o-", color="steelblue", linewidth=1.8)
    ax.axvline(best["theta"], color="red", linewidth=1.2, linestyle="--",
               label=f"Best θ = {best['theta']:.2f}")
    ax.axvline(-0.2, color="gray", linewidth=1.0, linestyle=":",
               label="Table I θ = −0.2")
    ax.set_xlabel("θ (mean-reversion coefficient)")
    ax.set_ylabel("Training Sharpe (N=20)")
    ax.set_title(f"Training Sharpe vs θ  |  SF(σ_obs)={best['sf_obs']:.3f}%, "
                 f"SF(σ)={best['sf_sig']:.3f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_theta_sensitivity.png", dpi=150)
    plt.close(fig)
    log("  figures/18_theta_sensitivity.png")

    # ── 7. Report ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

    report_path = REPORTS_DIR / "18_rbpf_param_search.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
