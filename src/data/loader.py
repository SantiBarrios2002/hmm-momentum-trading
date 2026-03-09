"""Data loading utilities for market price series."""

import tempfile
from pathlib import Path

import yfinance as yf


_YF_CACHE_CONFIGURED = False


def _configure_yfinance_cache():
    """Route yfinance timezone cache to a writable temp directory."""
    global _YF_CACHE_CONFIGURED
    if _YF_CACHE_CONFIGURED:
        return

    if hasattr(yf, "set_tz_cache_location"):
        cache_dir = Path(tempfile.gettempdir()) / "yfinance_tz_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

    _YF_CACHE_CONFIGURED = True


def extract_close_series(prices):
    """
    Return a 1-D float close-price Series from a yfinance DataFrame.

    Handles the column naming variation between yfinance versions (``Close``
    vs ``Adj Close``) and the occasional single-column DataFrame that
    ``yfinance`` returns instead of a Series.

    Parameters:
        prices: pandas DataFrame with OHLCV columns.

    Returns:
        pandas Series of float close prices with the original DatetimeIndex.

    Raises:
        ValueError: If neither 'Close' nor 'Adj Close' is found.
    """
    if "Close" in prices.columns:
        close = prices["Close"]
    elif "Adj Close" in prices.columns:
        close = prices["Adj Close"]
    else:
        raise ValueError("Expected 'Close' or 'Adj Close' column in downloaded prices")

    if hasattr(close, "ndim") and close.ndim != 1:
        close = close.iloc[:, 0]

    return close.astype(float)


def load_daily_prices(ticker: str, start: str, end: str):
    """
    Load daily OHLCV prices for one ticker from Yahoo Finance.

    Parameters:
        ticker: Asset symbol (for example, "SPY").
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        A pandas DataFrame indexed by date with OHLCV columns as returned by
        yfinance for daily data.

    Raises:
        ValueError: If no rows are returned for the query.
    """
    _configure_yfinance_cache()

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned for ticker={ticker}, start={start}, end={end}"
        )

    return data.sort_index()


def load_multiple(tickers, start: str, end: str):
    """
    Load daily OHLCV prices for multiple tickers.

    Parameters:
        tickers: Iterable of ticker symbols.
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        Dictionary mapping each ticker to its price DataFrame.

    Raises:
        ValueError: If tickers is empty.
    """
    tickers = list(tickers)
    if len(tickers) == 0:
        raise ValueError("tickers must contain at least one symbol")

    return {ticker: load_daily_prices(ticker, start, end) for ticker in tickers}
