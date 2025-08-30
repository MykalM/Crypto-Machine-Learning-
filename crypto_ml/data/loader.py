from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional at test time
    yf = None  # type: ignore


def fetch_ohlcv(
    symbol: str,
    start: Optional[str | dt.date] = None,
    end: Optional[str | dt.date] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data using yfinance.

    Returns a dataframe indexed by datetime with columns: Open, High, Low, Close, Adj Close, Volume.
    """
    if yf is None:
        raise ImportError("yfinance is required to fetch data. Please install yfinance.")

    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No data returned for symbol {symbol}.")

    # Normalize index and columns
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    df.columns = [c.strip().title() for c in df.columns]
    expected_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = expected_cols.difference(set(df.columns))
    if missing:
        # Some intervals may omit Adj Close; fill if absent
        if "Adj Close" in missing and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            missing.remove("Adj Close")
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df.sort_index()

