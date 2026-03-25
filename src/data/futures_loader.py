"""Futures data loader for Databento 1-minute OHLCV parquet files.

Companion to src/data/loader.py (which handles yfinance daily data).
This module works with raw prices (not log-prices), matching the 2012
paper's setup for the RBPF strategy.

File naming convention (from scripts/databento_download.py):
    data/databento/{SYMBOL_c_0}_ohlcv-1m_{START}_{END}.parquet
    e.g. data/databento/ES_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet

Parquet schema (Databento GLBX.MDP3 ohlcv-1m):
    Index : ts_event — DatetimeIndex, UTC
    open, high, low, close : float64 — raw prices (e.g. 4500.25 for ES)
    volume                 : int64
    symbol, rtype, publisher_id, instrument_id — metadata (dropped on load)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Default data directory relative to this file: Project_2/data/databento/
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"

# Regular Trading Hours window (US/Eastern)
_RTH_START = "09:30"
_RTH_END = "16:00"

# Supported resample frequencies and human-readable labels
_VALID_FREQS = {"5min", "15min", "30min", "1h", "1D"}


def load_futures_1m(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data for a continuous futures contract from parquet.

    Searches data_dir for a parquet file whose name contains the normalised
    symbol (e.g. "ES_c_0").  The full date range stored in the file is
    returned; pass start/end to slice afterward if needed.

    The 2012 paper (§IV-A) operates on raw prices, so no log transform is
    applied here.  The caller decides whether to work in raw or log space.

    Parameters
    ----------
    symbol : str
        Continuous contract identifier.  Accepts either the Databento
        notation ("ES.c.0") or the short form ("ES").  The loader
        normalises both to the filename fragment "ES_c_0".
    start : str or None
        Optional ISO date string "YYYY-MM-DD".  Rows before this date
        are dropped (inclusive).
    end : str or None
        Optional ISO date string "YYYY-MM-DD".  Rows from this date
        onward are dropped (exclusive, matching pandas slicing convention).
    data_dir : str, Path, or None
        Directory containing the parquet files.  Defaults to
        ``Project_2/data/databento/``.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close (float64), volume (int64).
        Index: ts_event — UTC-aware DatetimeIndex, 1-minute frequency
        (gaps on weekends / maintenance windows are expected).

    Raises
    ------
    ValueError
        If symbol is empty or no matching parquet file is found.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    # Normalise to filename fragment: "ES.c.0" → "ES_c_0", "ES" → "ES_c_0"
    # Only append "_c_0" when the symbol has no underscore (bare root like "ES").
    # Symbols already containing underscores (e.g. "ES_c_0", "NQ_c_1") are
    # left as-is to avoid garbling back-month or already-normalised forms.
    sym_norm = symbol.strip().replace(".", "_")
    if "_" not in sym_norm:
        sym_norm = f"{sym_norm}_c_0"

    # Find matching parquet file (any date range)
    matches = sorted(data_dir.glob(f"{sym_norm}_ohlcv-1m_*.parquet"))
    if not matches:
        raise ValueError(
            f"No parquet file found for symbol '{symbol}' "
            f"(normalised '{sym_norm}') in {data_dir}.\n"
            f"Expected pattern: {sym_norm}_ohlcv-1m_*.parquet\n"
            f"Run: python scripts/databento_download.py --symbols {symbol.replace('_', '.')}.c.0 "
            f"--start YYYY-MM-DD --end YYYY-MM-DD"
        )

    # If multiple files exist for the same symbol, use the one with the
    # widest date range (last alphabetically by convention YYYY-MM-DD).
    path = matches[-1]

    df = pd.read_parquet(path)

    # Keep only OHLCV — drop Databento metadata columns
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[ohlcv_cols].copy()

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.sort_index()

    # Optional date slice
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]

    if df.empty:
        raise ValueError(
            f"No rows remain for symbol '{symbol}' after slicing "
            f"start={start}, end={end}."
        )

    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a 1-minute bar DataFrame to Regular Trading Hours (RTH) only.

    RTH for US equity-index and most CME futures: 09:30–16:00 US/Eastern,
    Monday–Friday.  This matches the ~390 bars/day used in the 2012 paper
    (§IV-A: "intraday 1-minute data during regular trading hours").

    Note: agricultural futures (ZC, ZW, ZS …) technically have different
    pit/electronic session hours, but the 09:30–16:00 ET window captures
    the most liquid period for all CME Globex contracts and is used
    uniformly here for simplicity.

    Parameters
    ----------
    df : pd.DataFrame
        1-minute OHLCV DataFrame with a UTC-aware DatetimeIndex, as
        returned by load_futures_1m().

    Returns
    -------
    pd.DataFrame
        Subset of df containing only bars whose timestamp (converted to
        US/Eastern) falls in [09:30, 16:00) on a weekday.
        Index remains UTC.

    Raises
    ------
    ValueError
        If df is empty or its index is not timezone-aware.
    """
    if df.empty:
        raise ValueError("df is empty — nothing to filter")
    if df.index.tz is None:
        raise ValueError(
            "df.index must be timezone-aware (UTC).  "
            "Call load_futures_1m() first, or df.index.tz_localize('UTC')."
        )

    # Convert to Eastern for time-of-day comparison (handles DST automatically)
    eastern = df.index.tz_convert("America/New_York")

    # Weekday filter (Monday=0 … Friday=4)
    is_weekday = eastern.dayofweek < 5

    # Time-of-day filter: [09:30, 16:00)
    time_of_day = eastern.time
    rth_start = pd.Timestamp(_RTH_START).time()
    rth_end = pd.Timestamp(_RTH_END).time()
    in_rth = (time_of_day >= rth_start) & (time_of_day < rth_end)

    mask = is_weekday & in_rth
    return df.loc[mask].copy()


def resample_bars(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV bars to a coarser frequency.

    Applies standard OHLCV aggregation:
        open   = first bar's open
        high   = max of highs
        low    = min of lows
        close  = last bar's close
        volume = sum of volumes

    Incomplete periods (e.g. the first/last bar of a session that does not
    fill the full resampling window) are kept if they contain at least one
    valid close price; otherwise they are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        1-minute OHLCV DataFrame with a UTC-aware DatetimeIndex.
        Typically the output of filter_rth().
    freq : str
        Resampling frequency.  Supported values: "5min", "15min", "30min",
        "1h", "1D".

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame.  Rows with NaN close are dropped.
        Index is UTC DatetimeIndex at the chosen frequency.

    Raises
    ------
    ValueError
        If df is empty, freq is not supported, or no valid bars remain
        after resampling.
    """
    if df.empty:
        raise ValueError("df is empty — nothing to resample")
    if freq not in _VALID_FREQS:
        raise ValueError(
            f"freq must be one of {sorted(_VALID_FREQS)}, got '{freq}'"
        )

    agg: dict[str, str] = {}
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    if "close" in df.columns:
        agg["close"] = "last"
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = df.resample(freq).agg(agg)

    # Drop periods where close is NaN (no trades in that window)
    resampled = resampled.dropna(subset=["close"])

    if resampled.empty:
        raise ValueError(
            f"No valid bars remain after resampling to '{freq}'. "
            "Check that df spans at least one full resampling window."
        )

    return resampled
