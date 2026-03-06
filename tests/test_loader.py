import pandas as pd
import pytest

from src.data.loader import load_daily_prices


def test_load_daily_prices_sorts_index_and_returns_data(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.5, 100.5],
            "Close": [100.5, 101.5],
            "Adj Close": [100.4, 101.4],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
    )

    def fake_download(*args, **kwargs):
        return raw

    monkeypatch.setattr("src.data.loader.yf.download", fake_download)

    result = load_daily_prices("SPY", "2024-01-01", "2024-01-31")

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]


def test_load_daily_prices_raises_for_empty_download(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("src.data.loader.yf.download", fake_download)

    with pytest.raises(ValueError, match="No data returned"):
        load_daily_prices("SPY", "2024-01-01", "2024-01-31")
