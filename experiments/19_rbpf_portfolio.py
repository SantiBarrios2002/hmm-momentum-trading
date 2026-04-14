"""Experiment 19: Multi-contract RBPF portfolio grid search.

Retries the parameter grid search from Experiment 18, but evaluates each
parameter combination on the EQUAL-WEIGHT PORTFOLIO of all 29 available
contracts instead of single-asset ES.

Motivation
----------
Experiment 18 showed that parameter tuning gives negligible improvement on a
single contract (best test Sharpe = -0.19 vs Table I -0.20).  The paper
(Christensen et al. 2012) achieves Sharpe 1.82 by diversifying across 75
contracts — the individual signal is noisy but averaging across uncorrelated
assets cancels idiosyncratic noise.  This experiment tests whether the same
effect holds for our 29-contract universe.

Grid (42 combinations, informed by Exp 18)
------------------------------------------
Experiment 18 showed sf_sigma_obs = 35% is consistently best for single-asset
ES.  We fix it here and search over the two remaining parameters:

  sf_sigma  : trend diffusion as % of P_0  (7 log-spaced values)
  theta     : mean-reversion coefficient   (6 values)
  sf_obs    : FIXED at 35% (Table I, validated by Exp 18)

Total: 7 × 6 = 42 combinations.
Each evaluated on all 29 contracts (N=5 particles, full training set).

Selection protocol
------------------
1. Grid search on TRAINING set (first 70% of bars per contract).
   Criterion: PORTFOLIO training Sharpe (equal-weight, same params for all).
2. Best training combo → test-set evaluation (N=100, touched once).

Contracts available (29)
------------------------
Equity:    ES, NQ, YM, RTY
FX:        6A, 6B, 6C, 6E, 6J, 6N, 6S
Rates:     ZB, ZF, ZN, ZT
Energy:    CL, HO, NG, RB
Metals:    GC, HG, SI
Grains:    ZC, ZL, ZM, ZS, ZW
Livestock: HE, LE

Runtime estimate
----------------
42 combos × 29 contracts × ~3.5 s/run (N=5, 409 k bars) ≈ 71 min.

Outputs
-------
  figures/19_portfolio_heatmap.png     — portfolio training Sharpe heat-map
  figures/19_individual_sharpes.png    — per-contract test Sharpe bar chart
  figures/19_cumulative_returns.png    — portfolio vs buy-and-hold (test)
  reports/19_rbpf_portfolio.txt        — full results table

References
----------
Christensen, Murphy & Godsill (2012), IEEE JSTSP, §IV-C (portfolio).
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

# ── Configuration ─────────────────────────────────────────────────────────────

START = "2019-01-01"
END   = "2024-12-31"
DT    = 1.0 / 390.0     # 1-min bar in trading-day units

TRAIN_FRAC = 0.70
N_GRID     = 5           # particles during grid search
N_FINAL    = 100         # particles for final test evaluation

N_TAPS          = 4
IGARCH_ALPHA    = 0.06
TC_BPS          = 2
SEED            = 42
MIN_BARS        = 50_000  # drop contracts with fewer training bars

# Fixed from Experiment 18 (optimal for single-asset ES and very likely
# optimal for portfolio too — heavy smoothing, FIR+IGARCH does the rest)
SF_OBS_FIXED = 35.0      # % of P_0

# Grid over the two free parameters
SF_SIGMA_GRID = [0.005, 0.02, 0.05, 0.15, 0.5, 1.5, 3.5]   # % of P_0
THETA_GRID    = [-0.05, -0.1, -0.2, -0.5, -1.0, -2.0]

# Fixed at paper values (Table I)
LAMBDA_J   = 5.0
SF_SIGMA_J = 0.06

# All 29 downloaded contracts
SYMBOLS = [
    "ES", "NQ", "YM", "RTY",          # equity index
    "6A", "6B", "6C", "6E", "6J", "6N", "6S",  # FX
    "ZB", "ZF", "ZN", "ZT",           # rates
    "CL", "HO", "NG", "RB",           # energy
    "GC", "HG", "SI",                 # metals
    "ZC", "ZL", "ZM", "ZS", "ZW",    # grains
    "HE", "LE",                       # livestock
]

# Known baseline from Experiment 17/18 (single-asset ES, 1-min)
EXP18_SHARPE_TABLE_I = -0.20

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_signal(prices: np.ndarray, theta: float, sigma: float,
                    sigma_obs_sq: float, N: int, seed: int) -> np.ndarray:
    """Run RBPF + FIR + IGARCH and return signal array in [-1, 1], shape (T,)."""
    T         = len(prices)
    trend_var = sigma ** 2 / (2.0 * abs(theta))
    mu0       = np.array([prices[0], 0.0])
    C0        = np.diag([prices[0] ** 2 * 0.01, trend_var])

    fmeans, _, _, _, _ = run_rbpf_numba(
        prices, N_particles=N,
        theta=theta, sigma=sigma, sigma_obs_sq=sigma_obs_sq,
        lambda_J=LAMBDA_J, mu_J=0.0, sigma_J=SF_SIGMA_J * prices[0],
        mu0=mu0, C0=C0, dt=DT,
        rng=np.random.default_rng(seed),
    )

    trend   = fmeans[:, 1]
    signals = np.zeros(T)
    m       = fir_momentum_signal(trend, n_taps=N_TAPS)
    signals[N_TAPS - 1:] = m

    returns    = np.diff(prices) / prices[:-1]
    sig_for_ig = signals[1:]
    s2_init    = returns[0] ** 2 if returns[0] != 0 else float(np.var(returns))
    scaled     = igarch_volatility_scale(sig_for_ig, returns,
                                         alpha=IGARCH_ALPHA, sigma2_init=s2_init)
    std_sc = np.std(scaled)
    if std_sc > 1e-12:
        signals[1:] = np.clip(scaled / std_sc, -1.0, 1.0)

    return signals


def _portfolio_sharpe(
    signals_list: list[np.ndarray],
    returns_list: list[np.ndarray],
    tc_bps: float,
) -> float:
    """Mean individual Sharpe across N contracts (portfolio proxy).

    Different contracts have different bar counts (FX/grains have missing
    1-min bars during RTH), so we cannot stack them into a single matrix.
    Instead we compute per-contract annualised Sharpe after transaction costs
    and return the mean — this equals the portfolio Sharpe in the limit of
    many uncorrelated strategies (ρ → 0):

        Sharpe_portfolio ≈ mean(Sharpe_i)  for uncorrelated strategies.

    For selection purposes, maximising mean individual Sharpe is equivalent
    to maximising portfolio Sharpe.

    TC per bar: (tc_bps / 10000) × |Δposition|

    Parameters
    ----------
    signals_list : list of (T_i,) arrays  — signal per contract (varying T_i)
    returns_list : list of (T_i-1,) arrays — pct returns per contract
    tc_bps       : transaction cost in basis points

    Returns
    -------
    float — mean annualised individual Sharpe, or np.nan if all flat.
    """
    bars_per_year = 252 * 390
    sharpes = []

    for sig, ret in zip(signals_list, returns_list):
        s      = sig[:-1]
        delta  = np.abs(np.diff(np.concatenate([[0.0], s])))
        tc     = (tc_bps / 10000.0) * delta
        nr     = s * ret - tc
        std    = np.std(nr)
        if std < 1e-12:
            continue
        sharpes.append(nr.mean() / std * np.sqrt(bars_per_year))

    if not sharpes:
        return np.nan
    return float(np.mean(sharpes))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    n_combos = len(SF_SIGMA_GRID) * len(THETA_GRID)

    log("=" * 72)
    log("Experiment 19: Multi-contract RBPF Portfolio — Parameter Grid Search")
    log("=" * 72)
    log(f"Contracts: {len(SYMBOLS)}  |  Period: {START} → {END}  |  Freq: 1min")
    log(f"Grid: {len(SF_SIGMA_GRID)} (sf_sigma) × {len(THETA_GRID)} (theta) "
        f"= {n_combos} combinations")
    log(f"sf_obs FIXED = {SF_OBS_FIXED}% (optimal from Exp 18)")
    log(f"N_GRID={N_GRID} (search)  N_FINAL={N_FINAL} (test evaluation)")

    # ── 1. Load all contracts ─────────────────────────────────────────────────
    log("\n--- 1. Loading all contracts ---")
    train_data: dict[str, dict] = {}   # sym → {prices, returns}
    test_data:  dict[str, dict] = {}

    skipped = []
    for sym in SYMBOLS:
        try:
            df   = load_futures_1m(sym, start=START, end=END)
            df   = filter_rth(df)
            px   = df["close"].values.astype(float)
            idx  = df.index
            T    = len(px)
            Ttr  = int(T * TRAIN_FRAC)

            if Ttr < MIN_BARS:
                skipped.append(f"{sym} (only {Ttr:,} train bars)")
                continue

            train_data[sym] = {
                "prices":  px[:Ttr],
                "returns": np.diff(px[:Ttr]) / px[:Ttr][:-1],
                "index":   idx[:Ttr],
                "P0_train": px[0],
            }
            test_data[sym] = {
                "prices":  px[Ttr:],
                "returns": np.diff(px[Ttr:]) / px[Ttr:][:-1],
                "index":   idx[Ttr:],
                "P0_test": px[Ttr],
            }
            log(f"  {sym:6s}  train={Ttr:>7,} bars  test={T-Ttr:>7,} bars  "
                f"P0_test={px[Ttr]:.4f}")
        except Exception as exc:
            skipped.append(f"{sym} ({exc})")

    if skipped:
        log(f"\n  Skipped: {', '.join(skipped)}")

    active_syms = list(train_data.keys())
    log(f"\n  Active contracts: {len(active_syms)}")

    # ── 2. JIT warm-up ────────────────────────────────────────────────────────
    log("\n--- 2. JIT warm-up ---")
    sym0   = active_syms[0]
    px_s   = train_data[sym0]["prices"][:500]
    P0_s   = px_s[0]
    t_jit  = time.time()
    _compute_signal(px_s, -0.2, 0.01 * P0_s, (0.35 * P0_s) ** 2,
                    N=N_GRID, seed=SEED)
    log(f"  JIT compile: {time.time()-t_jit:.1f}s")

    # Time one full-length contract run
    t_bench = time.time()
    _compute_signal(train_data[sym0]["prices"], -0.2, 0.01 * P0_s,
                    (0.35 * P0_s) ** 2, N=N_GRID, seed=SEED)
    per_run = time.time() - t_bench
    eta     = n_combos * len(active_syms) * per_run / 60.0
    log(f"  One contract, N={N_GRID}: {per_run:.1f}s")
    log(f"  ETA: {n_combos} combos × {len(active_syms)} contracts ≈ {eta:.0f} min")

    # ── 3. Grid search on training portfolio ──────────────────────────────────
    log(f"\n--- 3. Grid search (portfolio, N={N_GRID}) ---")

    all_combos = list(itertools.product(SF_SIGMA_GRID, THETA_GRID))
    results: list[dict] = []

    t_grid = time.time()
    for idx_c, (sf_sig, theta) in enumerate(all_combos):
        # Per-contract signals on training set
        sigs_train  = []
        rets_train  = []

        for sym in active_syms:
            P0    = train_data[sym]["P0_train"]
            sigma = sf_sig * P0 / 100.0
            s_obs = SF_OBS_FIXED * P0 / 100.0

            sig = _compute_signal(
                train_data[sym]["prices"],
                theta, sigma, s_obs ** 2,
                N=N_GRID, seed=SEED,
            )
            sigs_train.append(sig)
            rets_train.append(train_data[sym]["returns"])

        port_sh = _portfolio_sharpe(sigs_train, rets_train, TC_BPS)

        results.append({
            "sf_sig": sf_sig, "theta": theta,
            "train_sharpe": port_sh,
        })

        # Progress
        if (idx_c + 1) % 6 == 0 or idx_c == n_combos - 1:
            elapsed  = time.time() - t_grid
            rate     = (idx_c + 1) / elapsed
            eta_left = (n_combos - idx_c - 1) / rate / 60.0
            best_so_far = max(
                (r["train_sharpe"] for r in results
                 if not np.isnan(r["train_sharpe"])),
                default=float("nan"),
            )
            print(
                f"  [{idx_c+1:2d}/{n_combos}]  "
                f"best_portfolio_train={best_so_far:.4f}  ETA={eta_left:.1f} min"
            )

    log(f"  Grid search done in {(time.time()-t_grid)/60:.1f} min")

    # Sort by portfolio training Sharpe
    results_valid = [r for r in results if not np.isnan(r["train_sharpe"])]
    results_valid.sort(key=lambda r: r["train_sharpe"], reverse=True)

    log(f"\n  Full results — portfolio training Sharpe (N={N_GRID}):")
    log(f"  {'sf_sig%':>8} {'theta':>7} {'port_train_sharpe':>18}")
    log("  " + "-" * 38)
    for r in results_valid:
        marker = " ◄ BEST" if r is results_valid[0] else ""
        log(f"  {r['sf_sig']:>8.3f} {r['theta']:>7.2f} {r['train_sharpe']:>18.4f}{marker}")

    best = results_valid[0]
    log(f"\n  Best: sf_sigma={best['sf_sig']:.3f}%  theta={best['theta']:.2f}  "
        f"train_sharpe={best['train_sharpe']:.4f}")

    # ── 4. Test evaluation with N_FINAL ───────────────────────────────────────
    log(f"\n--- 4. Test-set evaluation (N={N_FINAL}, touched once) ---")

    sigs_test_best: dict[str, np.ndarray] = {}
    rets_test_all:  dict[str, np.ndarray] = {}

    for sym in active_syms:
        P0    = test_data[sym]["P0_test"]
        sigma = best["sf_sig"] * P0 / 100.0
        s_obs = SF_OBS_FIXED * P0 / 100.0

        sig = _compute_signal(
            test_data[sym]["prices"],
            best["theta"], sigma, s_obs ** 2,
            N=N_FINAL, seed=SEED,
        )
        sigs_test_best[sym] = sig
        rets_test_all[sym]  = test_data[sym]["returns"]
        print(f"  {sym:6s} done")

    port_sharpe_test = _portfolio_sharpe(
        list(sigs_test_best.values()),
        list(rets_test_all.values()),
        TC_BPS,
    )
    log(f"\n  Portfolio test Sharpe (best params): {port_sharpe_test:.4f}")
    log(f"  Table I single-asset ES (Exp 17/18): {EXP18_SHARPE_TABLE_I:.4f}")
    log(f"  Improvement from diversification: {port_sharpe_test - EXP18_SHARPE_TABLE_I:+.4f}")

    # Per-contract test Sharpes
    log(f"\n  Per-contract test Sharpe (best params, N={N_FINAL}):")
    indiv_sharpes: dict[str, float] = {}
    for sym in active_syms:
        sh = _portfolio_sharpe(
            [sigs_test_best[sym]],
            [rets_test_all[sym]],
            TC_BPS,
        )
        indiv_sharpes[sym] = sh
    indiv_sorted = sorted(indiv_sharpes.items(), key=lambda x: x[1], reverse=True)
    for sym, sh in indiv_sorted:
        log(f"    {sym:6s}  {sh:+.4f}")

    # Portfolio cumulative returns — use the longest contract (ES) as time axis;
    # average net returns across contracts using their own bar counts.
    # Since bar counts differ, we normalise each contract's cumulative return
    # independently and average the cumulative curves (equal-weight NAV).
    ref_sym = "ES" if "ES" in active_syms else active_syms[0]
    ref_ret = rets_test_all[ref_sym]
    T_ref   = len(ref_ret)

    cum_curves = []
    bh_curves  = []
    for sym in active_syms:
        s     = sigs_test_best[sym][:-1]
        ret   = rets_test_all[sym]
        delta = np.abs(np.diff(np.concatenate([[0.0], s])))
        tc    = (TC_BPS / 10000.0) * delta
        nr    = s * ret - tc
        # Resample/truncate to ref length by keeping last T_ref bars
        # (test sets all end at the same date; shorter contracts started later)
        T_use = min(len(nr), T_ref)
        cum_curves.append(np.cumprod(1.0 + nr[-T_use:]))
        bh_curves.append(np.cumprod(1.0 + ret[-T_use:]))

    # Pad shorter curves to T_ref with their final value
    def _pad(curves, length):
        out = []
        for c in curves:
            if len(c) < length:
                pad = np.full(length - len(c), c[-1])
                out.append(np.concatenate([np.ones(length - len(c)), c]))
            else:
                out.append(c)
        return np.array(out)

    cum_arr = _pad(cum_curves, T_ref)
    bh_arr  = _pad(bh_curves,  T_ref)
    port_cum = np.mean(cum_arr, axis=0)
    bh_cum   = np.mean(bh_arr,  axis=0)

    # ── 5. Figures ─────────────────────────────────────────────────────────────
    log("\n--- 5. Saving figures ---")

    # Figure 1: heat-map of portfolio training Sharpe
    hm = np.full((len(THETA_GRID), len(SF_SIGMA_GRID)), np.nan)
    for r in results:
        i = THETA_GRID.index(r["theta"])
        j = SF_SIGMA_GRID.index(r["sf_sig"])
        hm[i, j] = r["train_sharpe"]

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = max(abs(np.nanmin(hm)), abs(np.nanmax(hm)))
    im = ax.imshow(hm, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, label="Portfolio Training Sharpe")
    ax.set_xticks(range(len(SF_SIGMA_GRID)))
    ax.set_xticklabels([f"{v:.3g}%" for v in SF_SIGMA_GRID], rotation=45, ha="right")
    ax.set_yticks(range(len(THETA_GRID)))
    ax.set_yticklabels([f"{v}" for v in THETA_GRID])
    ax.set_xlabel("SF(σ)  [% of P₀]")
    ax.set_ylabel("θ  (mean-reversion)")
    ax.set_title(
        f"Exp 19: Portfolio Training Sharpe  ({len(active_syms)} contracts, "
        f"N={N_GRID})\nSF(σ_obs) fixed = {SF_OBS_FIXED}%"
    )
    bi = THETA_GRID.index(best["theta"])
    bj = SF_SIGMA_GRID.index(best["sf_sig"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=2.5))
    ax.text(bj, bi, "★", ha="center", va="center", fontsize=14, color="blue")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "19_portfolio_heatmap.png", dpi=150)
    plt.close(fig)
    log("  figures/19_portfolio_heatmap.png")

    # Figure 2: per-contract individual test Sharpes
    syms_sorted  = [s for s, _ in indiv_sorted]
    sharps_sorted = [v for _, v in indiv_sorted]
    colors = ["steelblue" if v >= 0 else "tomato" for v in sharps_sorted]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(syms_sorted, sharps_sorted, color=colors,
                  edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(port_sharpe_test, color="blue", linewidth=1.5, linestyle="--",
               label=f"Portfolio Sharpe = {port_sharpe_test:.3f}")
    ax.axhline(EXP18_SHARPE_TABLE_I, color="gray", linewidth=1.2, linestyle=":",
               label=f"Table I ES (Exp 18) = {EXP18_SHARPE_TABLE_I:.2f}")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.set_title(
        f"Exp 19: Per-contract test Sharpe  "
        f"(sf_σ={best['sf_sig']:.3f}%, θ={best['theta']:.2f}, "
        f"N={N_FINAL})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "19_individual_sharpes.png", dpi=150)
    plt.close(fig)
    log("  figures/19_individual_sharpes.png")

    # Figure 3: portfolio cumulative returns
    test_idx0 = test_data[active_syms[0]]["index"]
    plot_idx  = test_idx0[1:len(port_cum) + 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_idx, port_cum,
            label=f"Portfolio {len(active_syms)} contracts "
                  f"(Sharpe={port_sharpe_test:.3f})",
            linewidth=1.8, color="steelblue")
    ax.plot(plot_idx, bh_cum,
            label="Equal-weight Buy-and-Hold",
            linewidth=1.4, linestyle=":", color="k")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(
        f"Exp 19: Equal-weight Portfolio ({len(active_syms)} contracts, 1-min)\n"
        f"sf_σ={best['sf_sig']:.3f}%, θ={best['theta']:.2f}  |  "
        f"Table I single-asset Sharpe = {EXP18_SHARPE_TABLE_I:.2f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value ($1 invested)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "19_cumulative_returns.png", dpi=150)
    plt.close(fig)
    log("  figures/19_cumulative_returns.png")

    # ── 6. Report ──────────────────────────────────────────────────────────────
    total = time.time() - t_start
    log(f"\nTotal runtime: {total:.0f}s ({total/60:.1f} min)")
    rp = REPORTS_DIR / "19_rbpf_portfolio.txt"
    rp.write_text("\n".join(report_lines))
    print(f"\nReport saved to {rp}")


if __name__ == "__main__":
    main()
