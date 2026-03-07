"""Experiment 01: data exploration on SPY daily prices (2015-2024)."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import ewma_volatility, log_returns
from src.data.loader import extract_close_series, load_daily_prices

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
LAMBDA_PARAM = 0.94
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")


def _distribution_stats(values):
    """Compute mean/std/skewness/excess-kurtosis for a 1D numeric array."""
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))

    if std == 0.0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        centered = values - mean
        skewness = float(np.mean((centered / std) ** 3))
        excess_kurtosis = float(np.mean((centered / std) ** 4) - 3.0)

    return mean, std, skewness, excess_kurtosis


def _save_price_figure(close):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(close.index, close.to_numpy(), color="tab:blue", linewidth=1.2)
    ax.set_title("SPY Daily Close (2015-2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_price_series.png", dpi=150)
    plt.close(fig)


def _save_return_distribution_figure(returns, mean, std):
    values = returns.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        values,
        bins=50,
        density=True,
        alpha=0.6,
        color="tab:blue",
        label="Empirical",
    )

    if std > 0.0:
        x = np.linspace(values.min(), values.max(), 500)
        gaussian_pdf = (
            (1.0 / (std * np.sqrt(2.0 * np.pi)))
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
        )
        ax.plot(x, gaussian_pdf, color="tab:red", linewidth=2.0, label="Gaussian fit")

    ax.set_title("Distribution of SPY Daily Log-Returns")
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_return_distribution.png", dpi=150)
    plt.close(fig)


def _save_ewma_figure(returns, ewma_sigma2):
    annualized_vol = np.sqrt(np.asarray(ewma_sigma2, dtype=float)) * np.sqrt(252.0)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(returns.index, annualized_vol, color="tab:green", linewidth=1.2)
    ax.set_title("EWMA Conditional Volatility (lambda=0.94)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_ewma_volatility.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 01: Data Exploration ===")
    log(f"Ticker: {TICKER} | Period: {START} to {END}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        log(f"Data load failed: {exc}")
        return 1

    close = extract_close_series(prices)
    returns = log_returns(close)
    ewma_sigma2 = ewma_volatility(returns, lambda_param=LAMBDA_PARAM)

    values = returns.to_numpy()
    mean, std, skewness, excess_kurtosis = _distribution_stats(values)

    min_idx = returns.idxmin()
    max_idx = returns.idxmax()
    annualized_return = float(np.exp(mean * 252.0) - 1.0)
    annualized_vol = float(std * np.sqrt(252.0))

    log(f"Price observations: {len(close)} trading days")
    log(f"Return observations: {len(returns)} daily log-returns")
    log(f"Date range: {close.index.min().date()} to {close.index.max().date()}")

    log("\n--- Return Statistics ---")
    log(f"Mean daily return:      {mean: .6f}")
    log(f"Std daily return:       {std: .6f}")
    log(f"Skewness:               {skewness: .4f}")
    log(f"Excess kurtosis:        {excess_kurtosis: .4f}")
    log(f"Min:                    {returns.loc[min_idx]: .4f} ({min_idx.date()})")
    log(f"Max:                    {returns.loc[max_idx]: .4f} ({max_idx.date()})")

    log("\n--- Annualized ---")
    log(f"Annualized return:      {annualized_return * 100: .2f}%")
    log(f"Annualized volatility:  {annualized_vol * 100: .2f}%")

    log("\n--- EWMA Volatility ---")
    ewma_ann = np.sqrt(np.asarray(ewma_sigma2, dtype=float)) * np.sqrt(252.0)
    log(f"Min annualized vol:     {ewma_ann.min() * 100: .2f}%")
    log(f"Max annualized vol:     {ewma_ann.max() * 100: .2f}%")
    log(f"Mean annualized vol:    {ewma_ann.mean() * 100: .2f}%")

    log("\n--- Percentiles of daily log-returns ---")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        log(f"  {p:>2}th percentile:      {np.percentile(values, p): .6f}")

    _save_price_figure(close)
    _save_return_distribution_figure(returns, mean, std)
    _save_ewma_figure(returns, ewma_sigma2)

    elapsed = time.time() - t_start
    log(f"\nFigures saved to figures/01_*.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "01_data_exploration.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
