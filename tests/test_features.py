import numpy as np
import pandas as pd
import pytest

from src.data.features import log_returns


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
