"""Experiment 11: Robustness test across tickers and time periods.

Runs the full HMM pipeline (train, infer, backtest) on 6 configurations:
  - 4 tickers (SPY, QQQ, IWM, EEM) on 2015-2024, 70/30 split
  - 2 alternative periods for SPY (2010-2019, 2018-2024), 70/30 split

Reports win rate (HMM weighted vote vs buy-and-hold) and mean Sharpe
improvement across all configurations.
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
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states, train_best_model
from src.strategy.backtest import backtest
from src.strategy.signals import states_to_signal

K = 3
TRANSACTION_COST_BPS = 5
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")

CONFIGS = [
    # (label, ticker, start, end)
    ("SPY 2015-2024", "SPY", "2015-01-01", "2024-12-31"),
    ("QQQ 2015-2024", "QQQ", "2015-01-01", "2024-12-31"),
    ("IWM 2015-2024", "IWM", "2015-01-01", "2024-12-31"),
    ("EEM 2015-2024", "EEM", "2015-01-01", "2024-12-31"),
    ("SPY 2010-2019", "SPY", "2010-01-01", "2019-12-31"),
    ("SPY 2018-2024", "SPY", "2018-01-01", "2024-12-31"),
]


def _run_pipeline(label, ticker, start, end, log_fn):
    """Run full pipeline for one configuration. Returns metrics dict or None."""
    log_fn(f"\n--- {label} ---")

    try:
        prices = load_daily_prices(ticker, start, end)
    except Exception as exc:
        log_fn(f"  Data load failed: {exc}")
        return None

    close = extract_close_series(prices)
    returns = log_returns(close)

    split = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]
    test_np = test_returns.to_numpy()

    log_fn(f"  Obs: {len(returns)} | Train: {len(train_returns)} | Test: {len(test_returns)}")
    log_fn(f"  Train: {train_returns.index.min().date()} to {train_returns.index.max().date()}")
    log_fn(f"  Test:  {test_returns.index.min().date()} to {test_returns.index.max().date()}")

    t0 = time.time()
    try:
        params, history, _ = train_best_model(
            train_returns.to_numpy(), K,
            successful_restarts=2, max_attempts=30,
            max_iter=100, tol=1e-6, random_state=42,
        )
    except RuntimeError as exc:
        log_fn(f"  Training failed: {exc}")
        return None

    params = sort_states(params)
    elapsed = time.time() - t0
    log_fn(f"  Training done in {elapsed:.1f}s | LL={history[-1]:.2f}")

    A, pi, mu, sigma2 = params["A"], params["pi"], params["mu"], params["sigma2"]
    log_fn(f"  mu={mu}")

    predictions, state_probs = run_inference(test_np, A, pi, mu, sigma2)

    signals_vote = states_to_signal(state_probs, mu)
    result_vote = backtest(test_np, signals_vote,
                           transaction_cost_bps=TRANSACTION_COST_BPS)

    bh_signals = np.ones_like(test_np)
    result_bh = backtest(test_np, bh_signals, transaction_cost_bps=0)

    m_vote = result_vote["metrics"]
    m_bh = result_bh["metrics"]

    log_fn(f"  HMM vote:  Sharpe={m_vote['sharpe']:.2f}, "
           f"Ann.Ret={m_vote['annualized_return'] * 100:.2f}%, "
           f"MaxDD={m_vote['max_drawdown'] * 100:.2f}%")
    log_fn(f"  Buy-hold:  Sharpe={m_bh['sharpe']:.2f}, "
           f"Ann.Ret={m_bh['annualized_return'] * 100:.2f}%, "
           f"MaxDD={m_bh['max_drawdown'] * 100:.2f}%")

    return {
        "label": label,
        "ticker": ticker,
        "sharpe_hmm": m_vote["sharpe"],
        "sharpe_bh": m_bh["sharpe"],
        "annret_hmm": m_vote["annualized_return"],
        "annret_bh": m_bh["annualized_return"],
        "maxdd_hmm": m_vote["max_drawdown"],
        "maxdd_bh": m_bh["max_drawdown"],
        "turnover": m_vote["turnover"],
    }


def _save_sharpe_comparison(results):
    """Bar chart comparing HMM vs buy-and-hold Sharpe across configs."""
    labels = [r["label"] for r in results]
    sharpe_hmm = [r["sharpe_hmm"] for r in results]
    sharpe_bh = [r["sharpe_bh"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, sharpe_hmm, width, label="HMM weighted vote")
    ax.bar(x + width / 2, sharpe_bh, width, label="Buy-and-hold")

    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Robustness Test: HMM vs Buy-and-Hold Across Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_robustness_sharpe.png", dpi=150)
    plt.close(fig)


def _save_drawdown_comparison(results):
    """Bar chart comparing max drawdown across configs."""
    labels = [r["label"] for r in results]
    dd_hmm = [r["maxdd_hmm"] * 100 for r in results]
    dd_bh = [r["maxdd_bh"] * 100 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, dd_hmm, width, label="HMM weighted vote")
    ax.bar(x + width / 2, dd_bh, width, label="Buy-and-hold")

    ax.set_ylabel("Max drawdown (%)")
    ax.set_title("Robustness Test: Max Drawdown Across Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_robustness_drawdown.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 11: Robustness Test ===")
    log(f"Configurations: {len(CONFIGS)} | K={K} | Cost={TRANSACTION_COST_BPS} bps")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for label, ticker, start, end in CONFIGS:
        r = _run_pipeline(label, ticker, start, end, log)
        if r is not None:
            results.append(r)

    if not results:
        log("\nNo successful configurations. Exiting.")
        return 1

    # --- Summary table ---
    log(f"\n{'=' * 90}")
    log("--- Summary ---")
    log(f"{'Config':<18}{'Sharpe(HMM)':>12}{'Sharpe(BH)':>11}{'Delta':>8}"
        f"{'AnnRet(HMM)':>12}{'AnnRet(BH)':>11}"
        f"{'MaxDD(HMM)':>11}{'MaxDD(BH)':>10}")
    log("-" * 93)
    for r in results:
        delta = r["sharpe_hmm"] - r["sharpe_bh"]
        log(f"{r['label']:<18}"
            f"{r['sharpe_hmm']:>12.2f}"
            f"{r['sharpe_bh']:>11.2f}"
            f"{delta:>+8.2f}"
            f"{r['annret_hmm'] * 100:>11.2f}%"
            f"{r['annret_bh'] * 100:>10.2f}%"
            f"{r['maxdd_hmm'] * 100:>10.2f}%"
            f"{r['maxdd_bh'] * 100:>9.2f}%")

    # --- Aggregate statistics ---
    sharpe_deltas = [r["sharpe_hmm"] - r["sharpe_bh"] for r in results]
    dd_deltas = [r["maxdd_hmm"] - r["maxdd_bh"] for r in results]
    wins = sum(1 for d in sharpe_deltas if d > 0)

    log(f"\n--- Aggregate Statistics ---")
    log(f"Configurations tested: {len(results)}")
    log(f"Win rate (Sharpe): {wins}/{len(results)} ({wins / len(results) * 100:.0f}%)")
    log(f"Mean Sharpe delta: {np.mean(sharpe_deltas):+.3f}")
    log(f"Mean max drawdown delta: {np.mean(dd_deltas) * 100:+.2f}%")
    log(f"HMM improves drawdown in {sum(1 for d in dd_deltas if d < 0)}/{len(results)} configs")

    # --- Save figures ---
    _save_sharpe_comparison(results)
    _save_drawdown_comparison(results)

    elapsed = time.time() - t_start
    log(f"\nFigures saved to:")
    log(f"  figures/11_robustness_sharpe.png")
    log(f"  figures/11_robustness_drawdown.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "11_robustness_test.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
