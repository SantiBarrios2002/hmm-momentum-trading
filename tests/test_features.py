import numpy as np
import pandas as pd
import pytest

from src.data.features import ewma_volatility, log_returns


def test_log_returns_numpy_values():
    prices = np.array([100.0, 110.0, 121.0])
    result = log_returns(prices)
    expected = np.array([np.log(1.1), np.log(1.1)])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_log_returns_preserves_series_index():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.Series([100.0, 105.0, 95.0], index=index, name="Close")
    result = log_returns(prices)

    assert isinstance(result, pd.Series)
    assert result.index.equals(index[1:])
    assert result.name == "Close"


def test_log_returns_rejects_non_positive_prices():
    with pytest.raises(ValueError, match="strictly positive"):
        log_returns([100.0, 0.0, 101.0])


def test_log_returns_rejects_short_input():
    with pytest.raises(ValueError, match="at least 2"):
        log_returns([100.0])


def test_ewma_volatility_recursion_values():
    returns = np.array([0.10, -0.20, 0.05], dtype=float)
    lam = 0.8
    result = ewma_volatility(returns, lambda_param=lam)

    expected = np.array(
        [
            0.10**2,
            lam * (0.10**2) + (1 - lam) * (0.10**2),
            lam * (0.10**2) + (1 - lam) * ((-0.20) ** 2),
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_ewma_volatility_preserves_series_index():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    returns = pd.Series([0.01, -0.02, 0.03], index=index, name="ret")
    result = ewma_volatility(returns, lambda_param=0.94)

    assert isinstance(result, pd.Series)
    assert result.index.equals(index)
    assert result.name == "ret"


def test_ewma_volatility_rejects_empty_input():
    with pytest.raises(ValueError, match="at least 1"):
        ewma_volatility([])


def test_ewma_volatility_rejects_invalid_lambda():
    with pytest.raises(ValueError, match="0 <= lambda_param < 1"):
        ewma_volatility([0.01, 0.02], lambda_param=1.0)
