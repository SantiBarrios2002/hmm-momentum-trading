import numpy as np
import pytest

from src.strategy.backtest import backtest


def test_backtest_buy_and_hold():
    returns = np.array([0.01, 0.02, -0.01, 0.03], dtype=float)
    signals = np.ones_like(returns)

    result = backtest(returns, signals, transaction_cost_bps=0)

    np.testing.assert_allclose(result["net_returns"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["net_returns"][1:], returns[1:], atol=1e-12)


def test_backtest_zero_signal_gives_zero_returns():
    returns = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)
    signals = np.zeros_like(returns)

    result = backtest(returns, signals, transaction_cost_bps=5)

    np.testing.assert_allclose(result["net_returns"], np.zeros_like(returns), atol=1e-12)


def test_backtest_one_period_lag():
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
    signals = np.array([0.0, 0.0, 1.0, 1.0, -1.0], dtype=float)

    result = backtest(returns, signals, transaction_cost_bps=0)

    expected = np.array([0.0, 0.0, 0.0, 0.04, 0.05], dtype=float)
    np.testing.assert_allclose(result["net_returns"], expected, atol=1e-12)


def test_backtest_transaction_costs_reduce_returns():
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
    signals = np.array([0.0, 0.0, 1.0, 1.0, -1.0], dtype=float)

    no_cost = backtest(returns, signals, transaction_cost_bps=0)
    with_cost = backtest(returns, signals, transaction_cost_bps=10)

    assert np.all(with_cost["net_returns"] <= no_cost["net_returns"] + 1e-12)
    assert with_cost["cumulative"][-1] < no_cost["cumulative"][-1]


def test_backtest_transaction_cost_calculation():
    returns = np.zeros(3, dtype=float)
    signals = np.array([1.0, -1.0, 1.0], dtype=float)

    result = backtest(returns, signals, transaction_cost_bps=100)

    expected_tc = np.array([0.0, 0.01, 0.02], dtype=float)
    np.testing.assert_allclose(-result["net_returns"], expected_tc, atol=1e-12)


def test_backtest_metrics_keys():
    returns = np.array([0.01, -0.005, 0.002, 0.004], dtype=float)
    signals = np.array([1.0, 1.0, -1.0, 0.0], dtype=float)

    result = backtest(returns, signals, transaction_cost_bps=5)

    assert set(result.keys()) == {"net_returns", "cumulative", "metrics"}
    assert set(result["metrics"].keys()) == {
        "sharpe",
        "annualized_return",
        "max_drawdown",
        "turnover",
    }


def test_backtest_rejects_length_mismatch():
    returns = np.array([0.01, 0.02], dtype=float)
    signals = np.array([1.0], dtype=float)

    with pytest.raises(ValueError):
        backtest(returns, signals)


def test_backtest_rejects_empty():
    with pytest.raises(ValueError):
        backtest(np.array([], dtype=float), np.array([], dtype=float))


def test_backtest_rejects_negative_cost():
    returns = np.array([0.01, -0.02], dtype=float)
    signals = np.array([1.0, -1.0], dtype=float)

    with pytest.raises(ValueError):
        backtest(returns, signals, transaction_cost_bps=-1)


def test_backtest_single_observation():
    returns = np.array([0.02], dtype=float)
    signals = np.array([1.0], dtype=float)

    result = backtest(returns, signals, transaction_cost_bps=0)

    np.testing.assert_allclose(result["net_returns"][0], 0.0, atol=1e-12)
    assert result["metrics"]["turnover"] == 0.0


def test_backtest_short_only():
    returns = np.array([0.01, 0.02, -0.01, 0.03], dtype=float)
    signals = -np.ones_like(returns)

    result = backtest(returns, signals, transaction_cost_bps=0)

    # lagged = [0, -1, -1, -1], so strategy = [0, -0.02, 0.01, -0.03]
    expected = np.array([0.0, -0.02, 0.01, -0.03], dtype=float)
    np.testing.assert_allclose(result["net_returns"], expected, atol=1e-12)
