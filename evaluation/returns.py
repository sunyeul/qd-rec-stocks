"""Realized returns after snapshot t, measured from t's close (total return, adjusted closes)."""

from datetime import date, timedelta

import pandas as pd

HORIZONS = {"1m": 21, "3m": 63}  # trading days


def fetch_closes(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(sorted(set(tickers)), start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                      auto_adjust=True, group_by="column", progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def window(closes: pd.DataFrame, t: date, days: int) -> pd.DataFrame:
    """Closes from t's close (the last trading day on or before t) through `days` sessions later.

    Raises if the horizon has not been reached yet, so a partial period is never scored.
    """
    start = closes.index[closes.index.date <= t]
    if start.empty:
        raise ValueError(f"no close on or before {t}")
    i = closes.index.get_loc(start[-1])
    if i + days >= len(closes.index):
        raise ValueError(f"{days} sessions after {t} are not available yet")
    return closes.iloc[i: i + days + 1]


def forward_returns(closes: pd.DataFrame, t: date, days: int) -> pd.Series:
    w = window(closes, t, days)
    return w.iloc[-1] / w.iloc[0] - 1
