"""Point-in-time reader: the only way card building should read collected data.

Every function takes the snapshot date t and returns nothing from after
t's cutoff (the market close). The evaluation layer, which needs prices
after t, must read partitions directly and live in a separate module.
"""

from datetime import date, timedelta

import pandas as pd

from . import clock, config, snapshot


class NoSnapshotError(LookupError):
    pass


def latest_date_asof(source: str, t: date) -> date:
    """The newest partition date of `source` that is not after t."""
    dates = [d for d in snapshot.list_dates(source) if d <= t]
    if not dates:
        raise NoSnapshotError(f"no {source} snapshot on or before {t}")
    return dates[-1]


def load_asof(source: str, t: date) -> pd.DataFrame:
    """The latest partition of `source` dated on or before t.

    For fundamentals this may be a few days old (they are refreshed weekly);
    the returned frame keeps its own ``snapshot_date`` so the age is visible.
    """
    if source == "news":
        raise ValueError("use load_news_asof for news")
    d = latest_date_asof(source, t)
    df = snapshot.read(source, d)
    assert (pd.to_datetime(df["snapshot_date"]).dt.date <= t).all()
    return df


def load_news_asof(t: date, lookback_days: int = config.NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    """News published in (cutoff(t) - lookback, cutoff(t)), deduplicated.

    News is fetched daily and each fetch only returns recent articles, so the
    window is assembled from every news partition up to t. Articles are kept
    by publication time, not by the day they were fetched.
    """
    end = clock.cutoff(t)
    start = end - timedelta(days=lookback_days)
    frames = [snapshot.read("news", d) for d in snapshot.list_dates("news") if d <= t]
    if not frames:
        raise NoSnapshotError(f"no news snapshot on or before {t}")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    published = pd.to_datetime(df["published_at"], utc=True)
    df = df[(published < end) & (published >= start)]
    # The same article can be fetched on several days and for several tickers.
    df = df.sort_values("snapshot_date").drop_duplicates(["ticker", "id"], keep="first")
    return df.sort_values(["ticker", "published_at"], ascending=[True, False]).reset_index(drop=True)
