"""Data loading utilities for market price series."""

import yfinance as yf


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
