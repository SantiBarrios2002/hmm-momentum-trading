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
