"""Experiment 12: Kalman filter on synthetic Langevin data (no jumps).

Pedagogical experiment — understand the Kalman filter before the RBPF.
Generates synthetic data from the Langevin jump-diffusion model (with no jumps),
runs the Kalman filter, and verifies that:
  1. The filtered trend tracks the truth within 2σ for >95% of timesteps.
  2. Standardized residuals are approximately N(0,1).

Outputs:
  - figures/12_kalman_tracking.png
  - figures/12_kalman_residuals.png
  - reports/12_kalman_filter_intro.txt

References: Christensen, Turner & Godsill (2012), §III-A.
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.langevin.kalman import kalman_filter
from src.langevin.model import discretize_langevin, observation_matrix

# ── Parameters (Paper §IV-C, Table I style) ──────────────────────────
THETA = -0.5        # mean-reversion rate (stable OU dynamics)
SIGMA = 0.02        # diffusion coefficient of the trend
SIGMA_OBS = 0.01    # observation noise std
DT = 1.0            # daily timestep
T = 500             # number of observations
SEED = 42           # reproducibility

FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _generate_langevin_data(theta, sigma, sigma_obs, dt, T, rng):
    """Generate synthetic Langevin data (no jumps) with known true states.

    Model:
        x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
        y_t     = G x_t + v_t,   v_t ~ N(0, sigma_obs^2)

    Returns true_states (T, 2), observations (T,).
    """
    F, Q = discretize_langevin(theta, sigma, dt)
    G = observation_matrix()

    # Cholesky of Q for sampling process noise
    L_Q = np.linalg.cholesky(Q + 1e-15 * np.eye(2))

    true_states = np.zeros((T, 2))
    observations = np.zeros(T)

    # Initial state: start at [0, 0] (no trend, zero price level)
    x = np.array([0.0, 0.0])

    for t in range(T):
        if t > 0:
            w = L_Q @ rng.standard_normal(2)
            x = F @ x + w
        true_states[t] = x
        observations[t] = (G @ x).item() + sigma_obs * rng.standard_normal()

    return true_states, observations


def _save_tracking_figure(true_states, filtered_means, filtered_covs, observations):
    """Plot true trend, filtered trend, and ±2σ confidence band."""
    T = len(observations)
    time_axis = np.arange(T)

    # Extract trend component (x2)
    true_trend = true_states[:, 1]
    filtered_trend = filtered_means[:, 1]
    filtered_trend_std = np.sqrt(filtered_covs[:, 1, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: price tracking
    ax = axes[0]
    ax.plot(time_axis, true_states[:, 0], "k-", linewidth=1.0, label="True price (x1)")
    ax.plot(time_axis, observations, ".", color="tab:gray", markersize=1.5,
            alpha=0.5, label="Observations")
    ax.plot(time_axis, filtered_means[:, 0], "tab:blue", linewidth=1.0,
            label="Filtered price")
    price_std = np.sqrt(filtered_covs[:, 0, 0])
    ax.fill_between(time_axis,
                     filtered_means[:, 0] - 2 * price_std,
                     filtered_means[:, 0] + 2 * price_std,
                     alpha=0.2, color="tab:blue", label=r"$\pm 2\sigma$")
    ax.set_ylabel("Price level (x1)")
    ax.set_title("Kalman Filter: Price Tracking (Synthetic Langevin, No Jumps)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom: trend tracking
    ax = axes[1]
    ax.plot(time_axis, true_trend, "k-", linewidth=1.0, label="True trend (x2)")
    ax.plot(time_axis, filtered_trend, "tab:red", linewidth=1.0,
            label="Filtered trend")
    ax.fill_between(time_axis,
                     filtered_trend - 2 * filtered_trend_std,
                     filtered_trend + 2 * filtered_trend_std,
                     alpha=0.2, color="tab:red", label=r"$\pm 2\sigma$")
    ax.set_ylabel("Trend (x2)")
    ax.set_xlabel("Time step")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_kalman_tracking.png", dpi=150)
    plt.close(fig)


def _save_residuals_figure(observations, predicted_means, predicted_covs, G, sigma_obs_sq):
    """Plot standardized innovation residuals and their histogram."""
    T = len(observations)

    # Compute innovations and their variances
    innovations = np.zeros(T)
    innovation_vars = np.zeros(T)
    for t in range(T):
        y_pred = (G @ predicted_means[t]).item()
        S = (G @ predicted_covs[t] @ G.T).item() + sigma_obs_sq
        innovations[t] = observations[t] - y_pred
        innovation_vars[t] = S

    standardized = innovations / np.sqrt(innovation_vars)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: standardized residuals over time
    ax = axes[0]
    ax.plot(np.arange(T), standardized, ".", markersize=2, color="tab:blue", alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(2, color="tab:red", linewidth=0.5, linestyle="--", label=r"$\pm 2$")
    ax.axhline(-2, color="tab:red", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Innovation Residuals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: histogram vs N(0,1)
    ax = axes[1]
    ax.hist(standardized, bins=30, density=True, alpha=0.6, color="tab:blue",
            label="Empirical")
    x = np.linspace(-4, 4, 200)
    gaussian_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    ax.plot(x, gaussian_pdf, "tab:red", linewidth=2.0, label="N(0,1)")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_kalman_residuals.png", dpi=150)
    plt.close(fig)

    return standardized


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 12: Kalman Filter Intro ===")
    log(f"Synthetic Langevin model (no jumps)")
    log(f"Parameters: theta={THETA}, sigma={SIGMA}, sigma_obs={SIGMA_OBS}, dt={DT}, T={T}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── 1. Generate synthetic data ────────────────────────────────────
    log("\n--- Generating synthetic Langevin data (no jumps) ---")
    true_states, observations = _generate_langevin_data(THETA, SIGMA, SIGMA_OBS, DT, T, rng)

    log(f"True price range:  [{true_states[:, 0].min():.4f}, {true_states[:, 0].max():.4f}]")
    log(f"True trend range:  [{true_states[:, 1].min():.6f}, {true_states[:, 1].max():.6f}]")
    log(f"Observation range: [{observations.min():.4f}, {observations.max():.4f}]")

    # ── 2. Run Kalman filter ──────────────────────────────────────────
    log("\n--- Running Kalman filter ---")
    F, Q = discretize_langevin(THETA, SIGMA, DT)
    G = observation_matrix()
    sigma_obs_sq = SIGMA_OBS ** 2

    # Prior: centered at zero with moderate uncertainty
    mu0 = np.array([0.0, 0.0])
    C0 = np.diag([1.0, 0.01])

    (predicted_means, predicted_covs,
     filtered_means, filtered_covs,
     log_likelihoods, total_log_likelihood) = kalman_filter(
        observations, F, Q, G, sigma_obs_sq, mu0, C0
    )

    log(f"Total log-likelihood: {total_log_likelihood:.2f}")
    log(f"Mean per-step LL:     {total_log_likelihood / T:.4f}")

    # ── 3. Tracking quality ───────────────────────────────────────────
    log("\n--- Tracking Quality ---")

    # Price tracking
    price_errors = filtered_means[:, 0] - true_states[:, 0]
    price_std = np.sqrt(filtered_covs[:, 0, 0])
    within_2sigma_price = np.mean(np.abs(price_errors) < 2 * price_std) * 100

    # Trend tracking
    trend_errors = filtered_means[:, 1] - true_states[:, 1]
    trend_std = np.sqrt(filtered_covs[:, 1, 1])
    within_2sigma_trend = np.mean(np.abs(trend_errors) < 2 * trend_std) * 100

    log(f"Price: {within_2sigma_price:.1f}% of true values within 2σ band")
    log(f"Trend: {within_2sigma_trend:.1f}% of true values within 2σ band")
    log(f"Price RMSE:  {np.sqrt(np.mean(price_errors**2)):.6f}")
    log(f"Trend RMSE:  {np.sqrt(np.mean(trend_errors**2)):.6f}")

    # Acceptance criterion: >95% within 2σ
    if within_2sigma_trend >= 95.0:
        log("PASS: Filtered trend tracks truth within 2σ for >95% of timesteps")
    else:
        log(f"WARN: Only {within_2sigma_trend:.1f}% within 2σ (expected >95%)")

    # ── 4. Residual analysis ──────────────────────────────────────────
    log("\n--- Residual Analysis ---")
    standardized = _save_residuals_figure(observations, predicted_means, predicted_covs, G, sigma_obs_sq)

    residual_mean = np.mean(standardized)
    residual_std = np.std(standardized)
    frac_outside_2 = np.mean(np.abs(standardized) > 2) * 100

    log(f"Standardized residuals: mean={residual_mean:.4f}, std={residual_std:.4f}")
    log(f"Fraction outside ±2:   {frac_outside_2:.1f}% (expected ~5%)")

    # ── 5. Save figures ───────────────────────────────────────────────
    _save_tracking_figure(true_states, filtered_means, filtered_covs, observations)

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/12_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "12_kalman_filter_intro.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
