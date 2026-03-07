"""Experiment 06: EM vs Gibbs-MCMC comparison for Gaussian HMM."""

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
from src.hmm.forward import forward
from src.hmm.gibbs import gibbs_sampler
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states, train_best_model
from src.strategy.backtest import backtest
from src.strategy.signals import states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
TRAIN_FRACTION = 0.7

EM_SUCCESSFUL_RESTARTS = 3
EM_MAX_ATTEMPTS = 40
EM_MAX_ITER = 120
EM_TOL = 1e-6
EM_RANDOM_STATE = 42

MCMC_SAMPLES = 150
MCMC_BURN_IN = 150
MCMC_THIN = 2
MCMC_RANDOM_STATE = 7

TRANSACTION_COST_BPS = 5
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _print_metric_row(name, metrics):
    return (
        f"{name:<18}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.4f}"
    )


def _save_mu_trace_figure(mu_samples):
    fig, axes = plt.subplots(mu_samples.shape[1], 1, figsize=(10, 2.6 * mu_samples.shape[1]), sharex=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.plot(mu_samples[:, k], linewidth=1.0)
        ax.set_ylabel(f"mu[{k}]")
        ax.grid(alpha=0.3)
    axes[0].set_title("Gibbs Trace: Emission Means")
    axes[-1].set_xlabel("Saved sample index")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_mcmc_trace_mu.png", dpi=150)
    plt.close(fig)


def _save_mu_posterior_figure(mu_samples, em_mu):
    fig, axes = plt.subplots(1, mu_samples.shape[1], figsize=(4.2 * mu_samples.shape[1], 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.hist(mu_samples[:, k], bins=25, alpha=0.75, density=True)
        ax.axvline(em_mu[k], color="tab:red", linestyle="--", linewidth=1.5, label="EM")
        ax.set_title(f"mu[{k}] posterior")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("Density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_mcmc_mu_posterior.png", dpi=150)
    plt.close(fig)


def _save_transition_heatmap_figure(A_em, A_mcmc):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    vmin = min(np.min(A_em), np.min(A_mcmc))
    vmax = max(np.max(A_em), np.max(A_mcmc))

    im0 = axes[0].imshow(A_em, vmin=vmin, vmax=vmax, cmap="Blues")
    axes[0].set_title("EM Transition Matrix")
    axes[0].set_xlabel("to state")
    axes[0].set_ylabel("from state")

    axes[1].imshow(A_mcmc, vmin=vmin, vmax=vmax, cmap="Blues")
    axes[1].set_title("MCMC Posterior Mean A")
    axes[1].set_xlabel("to state")
    axes[1].set_ylabel("from state")

    for ax, matrix in zip(axes, [A_em, A_mcmc]):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(FIGURES_DIR / "06_transition_matrices.png", dpi=150)
    plt.close(fig)


def _save_backtest_figure(test_index, cum_em, cum_mcmc, cum_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cum_em, linewidth=1.8, label="EM weighted vote")
    ax.plot(test_index, cum_mcmc, linewidth=1.8, label="MCMC weighted vote")
    ax.plot(test_index, cum_bh, linewidth=1.8, linestyle="--", label="Buy-and-hold")
    ax.set_title("Out-of-Sample Backtest: EM vs MCMC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_backtest_em_vs_mcmc.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 06: EM vs MCMC ===")
    log(
        f"Ticker: {TICKER} | Period: {START} to {END} | K={K} | "
        f"MCMC samples={MCMC_SAMPLES}, burn-in={MCMC_BURN_IN}, thin={MCMC_THIN}"
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
    values = returns.to_numpy()
    split = int(values.size * TRAIN_FRACTION)
    train_values = values[:split]
    test_values = values[split:]
    train_index = returns.index[:split]
    test_index = returns.index[split:]

    log(f"Observations: {values.size} | Train={train_values.size} | Test={test_values.size}")
    log(
        f"Train period: {train_index.min().date()} to {train_index.max().date()} | "
        f"Test period: {test_index.min().date()} to {test_index.max().date()}"
    )

    t_em = time.time()
    log("\nTraining EM model...")
    em_params, em_history, _ = train_best_model(
        train_values,
        K,
        successful_restarts=EM_SUCCESSFUL_RESTARTS,
        max_attempts=EM_MAX_ATTEMPTS,
        max_iter=EM_MAX_ITER,
        tol=EM_TOL,
        random_state=EM_RANDOM_STATE,
    )
    em_params = sort_states(em_params)
    em_train_ll = float(em_history[-1])
    log(
        f"EM done in {time.time() - t_em:.1f}s | "
        f"iterations={len(em_history)} | final train LL={em_train_ll:.2f}"
    )

    t_mcmc = time.time()
    log("\nRunning Gibbs sampler...")
    obs_var = float(np.var(train_values) + 1e-8)
    mcmc = gibbs_sampler(
        train_values,
        K=K,
        n_samples=MCMC_SAMPLES,
        burn_in=MCMC_BURN_IN,
        thin=MCMC_THIN,
        random_state=MCMC_RANDOM_STATE,
        alpha_pi=1.0,
        alpha_A=1.0,
        mu0=float(np.mean(train_values)),
        tau2=max(obs_var, 1e-6),
        sigma2_alpha0=2.0,
        sigma2_beta0=max(obs_var * 0.25, 1e-6),
        min_variance=1e-8,
    )
    log(f"Gibbs done in {time.time() - t_mcmc:.1f}s")

    mcmc_params = mcmc["posterior_mean"]

    _, em_test_ll = forward(
        test_values, em_params["A"], em_params["pi"], em_params["mu"], em_params["sigma2"]
    )
    _, mcmc_test_ll = forward(
        test_values, mcmc_params["A"], mcmc_params["pi"], mcmc_params["mu"], mcmc_params["sigma2"]
    )
    log(f"\nTest log-likelihood: EM={em_test_ll:.2f}, MCMC posterior-mean={mcmc_test_ll:.2f}")

    _, em_probs = run_inference(
        test_values, em_params["A"], em_params["pi"], em_params["mu"], em_params["sigma2"]
    )
    _, mcmc_probs = run_inference(
        test_values,
        mcmc_params["A"],
        mcmc_params["pi"],
        mcmc_params["mu"],
        mcmc_params["sigma2"],
    )

    # Weighted-vote signal from filtered probabilities.
    em_signals = states_to_signal(em_probs, em_params["mu"])
    mcmc_signals = states_to_signal(mcmc_probs, mcmc_params["mu"])

    em_bt = backtest(test_values, em_signals, transaction_cost_bps=TRANSACTION_COST_BPS)
    mcmc_bt = backtest(test_values, mcmc_signals, transaction_cost_bps=TRANSACTION_COST_BPS)
    bh_bt = backtest(test_values, np.ones_like(test_values), transaction_cost_bps=0)

    log(
        f"\n--- Out-of-sample metrics ({TRANSACTION_COST_BPS} bps transaction costs) ---"
    )
    log(f"{'Strategy':<18}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 63)
    log(_print_metric_row("EM weighted vote", em_bt["metrics"]))
    log(_print_metric_row("MCMC weighted vote", mcmc_bt["metrics"]))
    log(_print_metric_row("Buy-and-hold", bh_bt["metrics"]))

    log("\n--- Parameter comparison (means) ---")
    for k in range(K):
        log(
            f"state {k}: "
            f"mu_em={em_params['mu'][k]: .6f}, mu_mcmc={mcmc_params['mu'][k]: .6f}, "
            f"sigma2_em={em_params['sigma2'][k]: .6e}, sigma2_mcmc={mcmc_params['sigma2'][k]: .6e}"
        )

    log("\n--- MCMC posterior std (mu, sigma2) ---")
    mu_std = np.std(mcmc["mu_samples"], axis=0)
    sigma2_std = np.std(mcmc["sigma2_samples"], axis=0)
    for k in range(K):
        log(
            f"state {k}: std(mu)={mu_std[k]:.6f}, std(sigma2)={sigma2_std[k]:.6e}"
        )

    _save_mu_trace_figure(mcmc["mu_samples"])
    _save_mu_posterior_figure(mcmc["mu_samples"], em_params["mu"])
    _save_transition_heatmap_figure(em_params["A"], mcmc_params["A"])
    _save_backtest_figure(
        test_index,
        em_bt["cumulative"],
        mcmc_bt["cumulative"],
        bh_bt["cumulative"],
    )

    elapsed = time.time() - t_start
    log("\nFigures saved to:")
    log("  figures/06_mcmc_trace_mu.png")
    log("  figures/06_mcmc_mu_posterior.png")
    log("  figures/06_transition_matrices.png")
    log("  figures/06_backtest_em_vs_mcmc.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "06_em_vs_mcmc.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
