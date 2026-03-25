"""Tests for src/data/futures_loader.py.

Uses a 1-month ES parquet file (data/databento/ES_c_0_ohlcv-1m_2024-01-01_2024-02-01.parquet)
as the real-data fixture — this file is small (~0.11 MB) and already downloaded.
Synthetic DataFrame fixtures are used for edge-case tests that don't touch disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.futures_loader import filter_rth, load_futures_1m, resample_bars

# ---------------------------------------------------------------------------
# Path to the small 1-month ES parquet (already on disk)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "databento"
ES_1M_SHORT = DATA_DIR / "ES_c_0_ohlcv-1m_2024-01-01_2024-02-01.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_utc_df(n: int = 100, freq: str = "1min") -> pd.DataFrame:
    """Synthetic 1-min OHLCV DataFrame with UTC index."""
    idx = pd.date_range("2024-01-02 14:00", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 4500.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-1, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "volume": rng.integers(100, 1000, n),
        },
        index=idx,
    )


# ===========================================================================
# load_futures_1m
# ===========================================================================

class TestLoadFutures1m:
    def test_loads_es_short_file(self):
        """The 1-month ES parquet should load without error."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df = load_futures_1m("ES", data_dir=DATA_DIR)
        assert not df.empty
        assert "close" in df.columns
        assert df.index.tz is not None  # UTC-aware

    def test_returns_only_ohlcv_columns(self):
        """Metadata columns (rtype, publisher_id, …) must be dropped."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df = load_futures_1m("ES", data_dir=DATA_DIR)
        allowed = {"open", "high", "low", "close", "volume"}
        assert set(df.columns).issubset(allowed)

    def test_dot_notation_same_as_short(self):
        """'ES.c.0' and 'ES' must resolve to the same file."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df_short = load_futures_1m("ES", data_dir=DATA_DIR)
        df_dot = load_futures_1m("ES.c.0", data_dir=DATA_DIR)
        pd.testing.assert_frame_equal(df_short, df_dot)

    def test_start_end_slice(self):
        """start/end arguments must slice rows correctly."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df = load_futures_1m("ES", start="2024-01-10", end="2024-01-20", data_dir=DATA_DIR)
        assert df.index.min() >= pd.Timestamp("2024-01-10", tz="UTC")
        assert df.index.max() < pd.Timestamp("2024-01-20", tz="UTC")

    def test_close_prices_are_positive(self):
        """Raw ES prices are always positive."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df = load_futures_1m("ES", data_dir=DATA_DIR)
        assert (df["close"] > 0).all()

    def test_raises_on_empty_symbol(self):
        with pytest.raises(ValueError, match="non-empty"):
            load_futures_1m("", data_dir=DATA_DIR)

    def test_underscore_notation_accepted(self):
        """Already-normalised 'ES_c_0' must resolve the same file as 'ES'."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df_bare = load_futures_1m("ES", data_dir=DATA_DIR)
        df_norm = load_futures_1m("ES_c_0", data_dir=DATA_DIR)
        pd.testing.assert_frame_equal(df_bare, df_norm)

    def test_raises_on_missing_symbol(self, tmp_path):
        """Unknown symbol should raise ValueError with a helpful message."""
        with pytest.raises(ValueError, match="No parquet file found"):
            load_futures_1m("XX", data_dir=tmp_path)

    def test_raises_when_slice_leaves_no_rows(self):
        """start/end that exclude all rows must raise ValueError."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        with pytest.raises(ValueError, match="No rows remain"):
            load_futures_1m("ES", start="2030-01-01", end="2030-02-01", data_dir=DATA_DIR)


# ===========================================================================
# filter_rth
# ===========================================================================

class TestFilterRth:
    def test_removes_overnight_bars(self):
        """Bars outside 09:30-16:00 ET must be removed."""
        # 2024-01-02 is a Tuesday.  Build bars spanning midnight ET.
        # UTC midnight = 19:00 ET (winter, UTC-5) → outside RTH
        idx = pd.date_range("2024-01-02 00:00", periods=24 * 60, freq="1min", tz="UTC")
        df = pd.DataFrame({"close": 1.0}, index=idx)
        rth = filter_rth(df)
        # All surviving bars must be in [09:30, 16:00) ET
        eastern = rth.index.tz_convert("America/New_York")
        assert (eastern.time >= pd.Timestamp("09:30").time()).all()
        assert (eastern.time < pd.Timestamp("16:00").time()).all()

    def test_removes_weekend_bars(self):
        """Saturday and Sunday bars must be removed."""
        # 2024-01-06 is Saturday, 2024-01-07 is Sunday
        idx = pd.date_range("2024-01-06 14:30", periods=390, freq="1min", tz="UTC")
        df = pd.DataFrame({"close": 1.0}, index=idx)
        rth = filter_rth(df)
        assert rth.empty

    def test_rth_bar_count_typical_day(self):
        """A full Tuesday should yield exactly 390 RTH bars (09:30–15:59 ET)."""
        # 2024-01-02 is a Tuesday.  Create all 390 RTH minutes in UTC.
        # 09:30 ET = 14:30 UTC (winter, UTC-5)
        idx = pd.date_range("2024-01-02 14:30", periods=390, freq="1min", tz="UTC")
        df = pd.DataFrame({"close": 1.0}, index=idx)
        rth = filter_rth(df)
        assert len(rth) == 390

    def test_index_remains_utc(self):
        """Output index must stay in UTC (not converted to Eastern)."""
        df = _make_utc_df(60)
        rth = filter_rth(df)
        assert rth.index.tz is not None
        assert str(rth.index.tz) == "UTC"

    def test_raises_on_empty_df(self):
        df = pd.DataFrame(columns=["close"])
        df.index = pd.DatetimeIndex([], tz="UTC")
        with pytest.raises(ValueError, match="empty"):
            filter_rth(df)

    def test_raises_on_naive_index(self):
        """Timezone-naive index must raise ValueError."""
        idx = pd.date_range("2024-01-02 14:30", periods=10, freq="1min")
        df = pd.DataFrame({"close": 1.0}, index=idx)
        with pytest.raises(ValueError, match="timezone-aware"):
            filter_rth(df)

    def test_dst_spring_forward_day(self):
        """filter_rth must yield exactly 390 bars on the Monday after spring-forward (DST).

        2024-03-11 is the spring-forward Sunday (clocks go 02:00 → 03:00 ET).
        The following Monday 2024-03-11 opens at 09:30 ET = 13:30 UTC (UTC-4),
        NOT 14:30 UTC (UTC-5 as in winter).  Hardcoding UTC-5 would shift all
        bars by one hour and filter_rth would discard them — this test catches
        that regression.
        """
        # Use tz_localize so DST is handled by pytz/dateutil, not a manual offset.
        et_open = pd.Timestamp("2024-03-11 09:30", tz="America/New_York")
        utc_open = et_open.tz_convert("UTC")
        idx = pd.date_range(utc_open, periods=390, freq="1min")
        df = pd.DataFrame({"close": 1.0}, index=idx)
        rth = filter_rth(df)
        assert len(rth) == 390

    def test_on_real_es_data(self):
        """RTH bars from real ES data should be ≤ 390 per day."""
        pytest.importorskip("pyarrow")
        if not ES_1M_SHORT.exists():
            pytest.skip("ES short parquet not downloaded yet")

        df = load_futures_1m("ES", data_dir=DATA_DIR)
        rth = filter_rth(df)
        eastern = rth.index.tz_convert("America/New_York")
        bars_per_day = rth.groupby(eastern.date).size()
        assert (bars_per_day <= 390).all()
        # A normal trading day should have close to 390 bars
        assert bars_per_day.median() > 300


# ===========================================================================
# resample_bars
# ===========================================================================

class TestResampleBars:
    def test_5min_bar_count(self):
        """390 1-min RTH bars should collapse to 78 five-minute bars."""
        idx = pd.date_range("2024-01-02 14:30", periods=390, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "open": 100.0 + rng.standard_normal(390),
                "high": 101.0 + rng.standard_normal(390),
                "low": 99.0 + rng.standard_normal(390),
                "close": 100.0 + rng.standard_normal(390),
                "volume": rng.integers(100, 500, 390),
            },
            index=idx,
        )
        bars_5m = resample_bars(df, "5min")
        assert len(bars_5m) == 78

    def test_ohlcv_aggregation_correctness(self):
        """First 5 1-min bars → one 5-min bar with correct O/H/L/C/V."""
        idx = pd.date_range("2024-01-02 14:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open":   [100.0, 101.0, 102.0, 103.0, 104.0],
                "high":   [105.0, 106.0, 107.0, 108.0, 109.0],
                "low":    [ 99.0,  98.0,  97.0,  96.0,  95.0],
                "close":  [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [   10,    20,    30,    40,    50],
            },
            index=idx,
        )
        bar = resample_bars(df, "5min")
        assert len(bar) == 1
        np.testing.assert_almost_equal(bar["open"].iloc[0], 100.0)
        np.testing.assert_almost_equal(bar["high"].iloc[0], 109.0)
        np.testing.assert_almost_equal(bar["low"].iloc[0],   95.0)
        np.testing.assert_almost_equal(bar["close"].iloc[0], 105.0)
        assert bar["volume"].iloc[0] == 150

    def test_daily_resample(self):
        """5 days × 390 RTH bars → 5 daily bars."""
        days = pd.bdate_range("2024-01-02", periods=5, freq="B")
        frames = []
        for day in days:
            # Derive the UTC open from the ET open — correct across DST transitions.
            et_open = pd.Timestamp(day.date()).replace(hour=9, minute=30).tz_localize(
                "America/New_York"
            )
            utc_open = et_open.tz_convert("UTC")
            idx = pd.date_range(utc_open, periods=390, freq="1min")
            rng = np.random.default_rng(int(day.timestamp()))
            frames.append(pd.DataFrame({"close": 100.0 + rng.standard_normal(390)}, index=idx))
        df = pd.concat(frames)
        daily = resample_bars(df, "1D")
        assert len(daily) == 5

    def test_supported_freqs(self):
        """All documented frequencies must work without error."""
        df = _make_utc_df(200)
        for freq in ["5min", "15min", "30min", "1h", "1D"]:
            result = resample_bars(df, freq)
            assert not result.empty

    def test_raises_on_unsupported_freq(self):
        df = _make_utc_df(10)
        with pytest.raises(ValueError, match="freq must be one of"):
            resample_bars(df, "2min")

    def test_raises_on_empty_df(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz="UTC")
        with pytest.raises(ValueError, match="empty"):
            resample_bars(df, "5min")

    def test_drops_nan_close_periods(self):
        """Resampling windows with no data must be dropped, not kept as NaN."""
        # 5 bars then a 10-minute gap then 5 more bars
        idx1 = pd.date_range("2024-01-02 14:30", periods=5, freq="1min", tz="UTC")
        idx2 = pd.date_range("2024-01-02 14:50", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame({"close": 100.0}, index=idx1.append(idx2))
        result = resample_bars(df, "5min")
        assert result["close"].notna().all()
