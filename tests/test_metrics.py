import numpy as np
import pytest

from src.utils.metrics import max_drawdown, sharpe_ratio


def test_sharpe_ratio_matches_formula():
    returns = np.array([0.01, -0.02, 0.03, 0.00], dtype=float)
    expected = np.sqrt(252.0) * returns.mean() / returns.std(ddof=0)
    result = sharpe_ratio(returns)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sharpe_ratio_zero_std_returns_zero():
    result = sharpe_ratio([0.01, 0.01, 0.01])
    assert result == 0.0


def test_sharpe_ratio_rejects_empty_input():
    with pytest.raises(ValueError, match="at least 1"):
        sharpe_ratio([])


def test_max_drawdown_known_path():
    cumulative = np.array([1.0, 1.2, 1.1, 0.9, 1.3], dtype=float)
    result = max_drawdown(cumulative)
    expected = (1.2 - 0.9) / 1.2
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_max_drawdown_monotone_increase_is_zero():
    result = max_drawdown([1.0, 1.1, 1.2, 1.3])
    assert result == 0.0


def test_max_drawdown_rejects_empty_input():
    with pytest.raises(ValueError, match="at least 1"):
        max_drawdown([])
