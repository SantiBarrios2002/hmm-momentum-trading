"""Experiment 08: K=3 vs K=4 comparison for regime interpretability."""

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
from src.hmm.viterbi import viterbi
from src.strategy.backtest import backtest
from src.strategy.signals import predictions_to_signal, states_to_signal

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
K_VALUES = (3, 4)
TRAIN_FRACTION = 0.7
TOL = 1e-6
TRAINING_CONFIG = {
    3: {"successful_restarts": 2, "max_attempts": 30, "max_iter": 100},
    4: {"successful_restarts": 1, "max_attempts": 12, "max_iter": 50},
}
RANDOM_STATE = 42
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


def _train_sorted_model(observations, k, *, seed):
    cfg = TRAINING_CONFIG.get(k, TRAINING_CONFIG[3])
    params, history, _ = train_best_model(
        observations,
        k,
        successful_restarts=cfg["successful_restarts"],
        max_attempts=cfg["max_attempts"],
        max_iter=cfg["max_iter"],
        tol=TOL,
        random_state=seed,
    )
    return sort_states(params), history


def _state_crosstab(states_k3, states_k4):
    table = np.zeros((3, 4), dtype=int)
    for s3, s4 in zip(states_k3, states_k4):
        table[int(s3), int(s4)] += 1
    return table


def _format_crosstab(table):
    lines = []
    lines.append("K=3 \\ K=4 state assignment counts")
    lines.append("            s4=0    s4=1    s4=2    s4=3")
    for s3 in range(table.shape[0]):
        lines.append(
            f"s3={s3:<2d}     "
            f"{table[s3, 0]:>6d}  "
            f"{table[s3, 1]:>6d}  "
            f"{table[s3, 2]:>6d}  "
            f"{table[s3, 3]:>6d}"
        )
    return "\n".join(lines)


def _save_crosstab_figure(table):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(table, cmap="Blues")
    ax.set_title("Viterbi State Cross-Tab: K=3 vs K=4")
    ax.set_xlabel("K=4 state")
    ax.set_ylabel("K=3 state")
    ax.set_xticks(range(4))
    ax.set_yticks(range(3))

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table[i, j]}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_k3_k4_crosstab.png", dpi=150)
    plt.close(fig)


def _save_backtest_figure(test_index, k3_vote_cumulative, k4_vote_cumulative, bh_cumulative):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_index, k3_vote_cumulative, linewidth=1.8, label="K=3 weighted vote")
    ax.plot(test_index, k4_vote_cumulative, linewidth=1.8, label="K=4 weighted vote")
    ax.plot(test_index, bh_cumulative, linewidth=1.6, linestyle="--", label="Buy-and-hold")
    ax.set_title("Out-of-Sample Backtest (70/30): K=3 vs K=4")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_k3_k4_backtest.png", dpi=150)
    plt.close(fig)


def _log_params(log, k, params):
    log(f"\n--- K={k} parameters (sorted by mu) ---")
    for i in range(k):
        ann_mu = params["mu"][i] * 252.0
        ann_sigma = np.sqrt(params["sigma2"][i]) * np.sqrt(252.0)
        log(
            f"  state {i}: "
            f"mu={params['mu'][i]: .6f} (ann. {ann_mu * 100: .2f}%), "
            f"sigma={np.sqrt(params['sigma2'][i]): .6f} (ann. {ann_sigma * 100: .2f}%)"
        )


def _run_split_backtest(train_values, test_values, k, *, seed):
    params, _ = _train_sorted_model(train_values, k, seed=seed)
    predictions, state_probs = run_inference(
        test_values, params["A"], params["pi"], params["mu"], params["sigma2"]
    )

    signals_sign = predictions_to_signal(predictions, transfer_fn="sign")
    signals_vote = states_to_signal(state_probs, params["mu"])

    result_sign = backtest(
        test_values, signals_sign, transaction_cost_bps=TRANSACTION_COST_BPS
    )
    result_vote = backtest(
        test_values, signals_vote, transaction_cost_bps=TRANSACTION_COST_BPS
    )
    return {
        "params": params,
        "sign": result_sign,
        "vote": result_vote,
    }


def main():
    t_start = time.time()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=== Experiment 08: K=3 vs K=4 Comparison ===")
    log(
        f"Ticker: {TICKER} | Period: {START} to {END} | "
        f"K=3 config={TRAINING_CONFIG[3]} | K=4 config={TRAINING_CONFIG[4]}"
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
    log(f"Observations: {values.size} daily log-returns")

    full_models = {}
    full_states = {}

    log("\n--- Full-sample training and Viterbi decoding ---")
    for idx, k in enumerate(K_VALUES):
        t_k = time.time()
        log(f"Training K={k} model on full sample...")
        params, history = _train_sorted_model(values, k, seed=RANDOM_STATE + idx * 1000)
        states, viterbi_log_prob = viterbi(
            values, params["A"], params["pi"], params["mu"], params["sigma2"]
        )

        full_models[k] = {"params": params, "history": history}
        full_states[k] = states

        state_counts = np.bincount(states, minlength=k)
        log(
            f"K={k}: final LL={history[-1]:.2f}, "
            f"iterations={len(history)}, "
            f"Viterbi log-prob={viterbi_log_prob:.2f}, "
            f"elapsed={time.time() - t_k:.1f}s"
        )
        log(f"K={k}: state counts={state_counts.tolist()}")
        _log_params(log, k, params)

    cross_tab = _state_crosstab(full_states[3], full_states[4])
    log("\n--- State assignment overlap ---")
    log(_format_crosstab(cross_tab))

    split = int(values.size * TRAIN_FRACTION)
    if split <= 1 or split >= values.size:
        log("Invalid split configuration; cannot run out-of-sample backtest.")
        return 1

    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]
    train_values = train_returns.to_numpy()
    test_values = test_returns.to_numpy()

    log(
        f"\n--- 70/30 split backtest ---\n"
        f"Train: {train_returns.index.min().date()} to {train_returns.index.max().date()} "
        f"({train_values.size} obs)\n"
        f"Test:  {test_returns.index.min().date()} to {test_returns.index.max().date()} "
        f"({test_values.size} obs)"
    )

    split_results = {}
    for idx, k in enumerate(K_VALUES):
        t_k = time.time()
        log(f"Training K={k} model on train split...")
        split_results[k] = _run_split_backtest(
            train_values,
            test_values,
            k,
            seed=RANDOM_STATE + 5000 + idx * 1000,
        )
        log(f"K={k} split training + inference finished in {time.time() - t_k:.1f}s")

    buy_hold = backtest(test_values, np.ones_like(test_values), transaction_cost_bps=0)

    log(
        f"\n--- Out-of-sample metrics ({TRANSACTION_COST_BPS} bps transaction costs) ---"
    )
    log(f"{'Strategy':<18}{'Sharpe':>8}{'Ann.Return':>12}{'MaxDrawdown':>13}{'Turnover':>11}")
    log("-" * 63)
    log(_print_metric_row("K=3 sign", split_results[3]["sign"]["metrics"]))
    log(_print_metric_row("K=3 vote", split_results[3]["vote"]["metrics"]))
    log(_print_metric_row("K=4 sign", split_results[4]["sign"]["metrics"]))
    log(_print_metric_row("K=4 vote", split_results[4]["vote"]["metrics"]))
    log(_print_metric_row("Buy-and-hold", buy_hold["metrics"]))

    delta_vote_sharpe = (
        split_results[4]["vote"]["metrics"]["sharpe"]
        - split_results[3]["vote"]["metrics"]["sharpe"]
    )
    delta_vote_return = (
        split_results[4]["vote"]["metrics"]["annualized_return"]
        - split_results[3]["vote"]["metrics"]["annualized_return"]
    )
    log(
        f"\nK=4 minus K=3 (weighted vote): "
        f"delta sharpe={delta_vote_sharpe:+.3f}, "
        f"delta ann.return={delta_vote_return * 100:+.2f}%"
    )

    _save_crosstab_figure(cross_tab)
    _save_backtest_figure(
        test_returns.index,
        split_results[3]["vote"]["cumulative"],
        split_results[4]["vote"]["cumulative"],
        buy_hold["cumulative"],
    )

    elapsed = time.time() - t_start
    log("\nFigures saved to:")
    log("  figures/08_k3_k4_crosstab.png")
    log("  figures/08_k3_k4_backtest.png")
    log(f"Elapsed time: {elapsed:.1f}s")

    report_path = REPORTS_DIR / "08_k3_vs_k4.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
