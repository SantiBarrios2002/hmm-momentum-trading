"""Experiment 05: out-of-sample backtest comparison against buy-and-hold."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import log_returns
from src.data.loader import load_daily_prices
from src.hmm.baum_welch import baum_welch
from src.hmm.inference import run_inference
from src.strategy.backtest import backtest
from src.strategy.signals import predictions_to_signal, states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K = 3
MAX_ITER = 200
TOL = 1e-6
N_RESTARTS = 10
RANDOM_STATE = 42
TRANSACTION_COST_BPS = 5
EPS = 1e-12
FIGURES_DIR = Path("figures")


def _extract_close_series(prices):
    if "Close" in prices.columns:
        close = prices["Close"]
    elif "Adj Close" in prices.columns:
        close = prices["Adj Close"]
    else:
        raise ValueError("Expected 'Close' or 'Adj Close' in downloaded data")

    if hasattr(close, "ndim") and close.ndim != 1:
        close = close.iloc[:, 0]

    return close.astype(float)


def _sort_states(params):
    order = np.argsort(params["mu"])
    sorted_params = {
        "A": params["A"][np.ix_(order, order)],
        "pi": params["pi"][order],
        "mu": params["mu"][order],
        "sigma2": params["sigma2"][order],
    }

    A = np.clip(sorted_params["A"], EPS, None)
    sorted_params["A"] = A / A.sum(axis=1, keepdims=True)

    pi = np.clip(sorted_params["pi"], EPS, None)
    sorted_params["pi"] = pi / pi.sum()

    sorted_params["sigma2"] = np.clip(sorted_params["sigma2"], EPS, None)
    return sorted_params


def _train_best_model(observations, *, successful_restarts, max_attempts=150):
    best = None
    best_ll = -np.inf
    successes = 0

    for attempt in range(max_attempts):
        if successes >= successful_restarts:
            break

        seed = RANDOM_STATE + attempt
        try:
            params, history, gamma = baum_welch(
                observations,
                K=K,
                max_iter=MAX_ITER,
                tol=TOL,
                n_restarts=1,
                random_state=seed,
            )
        except ValueError as exc:
            if "strictly positive entries" not in str(exc):
                raise
            continue

        successes += 1
        ll = float(history[-1])
        if ll > best_ll:
            best_ll = ll
            best = (params, history, gamma)

    if best is None:
        raise RuntimeError("Training failed: no successful restart produced valid parameters")

    if successes < successful_restarts:
        print(
            f"Warning: used {successes}/{successful_restarts} successful restarts "
            f"after {max_attempts} attempts"
        )

    return best


def _print_metric_row(name, metrics):
    print(
        f"{name:<16}"
        f"{metrics['sharpe']:>8.2f}"
        f"{metrics['annualized_return'] * 100:>12.2f}%"
        f"{metrics['max_drawdown'] * 100:>13.2f}%"
        f"{metrics['turnover']:>11.2f}"
    )


def _save_comparison_figure(test_index, cumulative_sign, cumulative_vote, cumulative_bh):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, cumulative_sign, label="Sign Signal", linewidth=1.8)
    ax.plot(test_index, cumulative_vote, label="Weighted Vote", linewidth=1.8)
    ax.plot(test_index, cumulative_bh, label="Buy-and-Hold", linewidth=1.8, linestyle="--")

    ax.set_title("Out-of-Sample Backtest: HMM Strategies vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_backtest_comparison.png", dpi=150)
    plt.close(fig)


def main():
    print("=== Experiment 05: Backtest Comparison ===")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prices = load_daily_prices(TICKER, START, END)
    except Exception as exc:
        print(f"Data load failed: {exc}")
        return 1

    close = _extract_close_series(prices)
    returns = log_returns(close)

    split = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]

    print(
        f"Train period: {train_returns.index.min().date()} to "
        f"{train_returns.index.max().date()} ({len(train_returns)} days)"
    )
    print(
        f"Test period:  {test_returns.index.min().date()} to "
        f"{test_returns.index.max().date()} ({len(test_returns)} days)"
    )

    print(f"\nTraining {K}-state HMM on training data...")
    params, _, _ = _train_best_model(
        train_returns.to_numpy(),
        successful_restarts=N_RESTARTS,
    )
    params = _sort_states(params)

    A = params["A"]
    pi = params["pi"]
    mu = params["mu"]
    sigma2 = params["sigma2"]

    print("Running inference on test data...")
    predictions, state_probs = run_inference(test_returns.to_numpy(), A, pi, mu, sigma2)

    signals_sign = predictions_to_signal(predictions, transfer_fn="sign")
    signals_vote = states_to_signal(state_probs, mu)

    result_sign = backtest(
        test_returns.to_numpy(),
        signals_sign,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    result_vote = backtest(
        test_returns.to_numpy(),
        signals_vote,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    bh_signals = np.ones_like(test_returns.to_numpy())
    result_bh = backtest(test_returns.to_numpy(), bh_signals, transaction_cost_bps=0)

    print(
        f"\n--- Out-of-Sample Performance ({TRANSACTION_COST_BPS} bps transaction costs) ---"
    )
    print("Strategy          Sharpe   Ann.Return   MaxDrawdown   Turnover")
    _print_metric_row("Sign signal", result_sign["metrics"])
    _print_metric_row("Weighted vote", result_vote["metrics"])
    _print_metric_row("Buy-and-hold", result_bh["metrics"])

    # Check one-period lag convention: first net return should not use a prior signal.
    print(
        f"\nFirst-day net return check: "
        f"sign={result_sign['net_returns'][0]:.6f}, "
        f"vote={result_vote['net_returns'][0]:.6f}, "
        f"buy-hold={result_bh['net_returns'][0]:.6f}"
    )

    _save_comparison_figure(
        test_returns.index,
        result_sign["cumulative"],
        result_vote["cumulative"],
        result_bh["cumulative"],
    )
    print("Figure saved to figures/05_backtest_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
