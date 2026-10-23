"""Price features as of snapshot date t, from yfinance daily history.

Only closes dated on or before t are used. Prices are split- and
dividend-adjusted (auto_adjust=True); a dividend paid after t rescales all
earlier prices by the same factor, so the return ratios below do not change.
"""

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from . import config

# Trading-day windows
D_1M, D_3M, D_12M = 21, 63, 252


def _ret(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return math.nan
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def compute_features(close: pd.Series, volume: pd.Series, t: date) -> dict:
    """Features of one ticker from its daily closes and volumes up to t."""
    close = close.dropna()
    close = close[close.index.date <= t]
    volume = volume.reindex(close.index)
    if close.empty:
        return {"last_close": math.nan, "last_price_date": None}
    daily = close.pct_change().dropna()
    mom_12_1 = math.nan
    if len(close) > D_12M:
        mom_12_1 = float(close.iloc[-1 - D_1M] / close.iloc[-1 - D_12M] - 1)
    return {
        "last_close": float(close.iloc[-1]),
        "last_price_date": close.index[-1].date().isoformat(),
        "ret_1m": _ret(close, D_1M),
        "ret_3m": _ret(close, D_3M),
        "ret_12m": _ret(close, D_12M),
        "mom_12_1": mom_12_1,  # 12-month return skipping the last month
        "vol_60d": float(daily.tail(60).std() * np.sqrt(252)) if len(daily) >= 60 else math.nan,
        "avg_dollar_volume_20d": float((close * volume).tail(20).mean()) if len(close) >= 20 else math.nan,
        "n_price_days": int(len(close)),
    }


def fetch_history(tickers: list[str], t: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide frames (date x ticker) of adjusted closes and volumes ending at t."""
    import yfinance as yf

    start = t - timedelta(days=config.PRICE_LOOKBACK_DAYS)
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(t + timedelta(days=1)).isoformat(),  # end is exclusive
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data")
    close, volume = raw["Close"], raw["Volume"]
    if isinstance(close, pd.Series):  # a single ticker
        close, volume = close.to_frame(tickers[0]), volume.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    return close, volume


def collect(tickers: list[str], t: date) -> pd.DataFrame:
    close, volume = fetch_history(tickers, t)
    rows = []
    for tk in tickers:
        if tk not in close.columns:
            rows.append({"ticker": tk, "error": "no price data"})
            continue
        feats = compute_features(close[tk], volume[tk], t)
        feats["ticker"] = tk
        if feats["last_price_date"] is None:
            feats["error"] = "no closes on or before t"
        rows.append(feats)
    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = None
    df["is_benchmark"] = df["ticker"] == config.BENCHMARK_TICKER
    return df
