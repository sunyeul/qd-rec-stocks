"""Snapshot dates and information cutoffs, all in New York market time."""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config


def _ny() -> ZoneInfo:
    return ZoneInfo(config.MARKET_TZ)


def previous_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def latest_snapshot_date(now: datetime | None = None) -> date:
    """The most recent US trading date whose close has settled.

    Weekends roll back to Friday. Exchange holidays are not modelled: on a
    holiday the price features simply repeat the previous close, and
    ``last_price_date`` in the prices partition shows it.
    """
    now = now or datetime.now(timezone.utc)
    ny_now = now.astimezone(_ny())
    d = ny_now.date()
    if d.weekday() >= 5:
        return d - timedelta(days=d.weekday() - 4)
    if ny_now.time() < config.SNAPSHOT_READY:
        return previous_weekday(d)
    return d


def cutoff(t: date) -> datetime:
    """Information cutoff of snapshot t: t's market close, as an aware UTC time."""
    local = datetime.combine(t, config.MARKET_CLOSE, tzinfo=_ny())
    return local.astimezone(timezone.utc)


def next_open(t: date) -> datetime:
    """09:30 New York on the weekday after t, as an aware UTC time."""
    local = datetime.combine(next_weekday(t), config.MARKET_OPEN, tzinfo=_ny())
    return local.astimezone(timezone.utc)


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def in_fetch_window(t: date, now: datetime | None = None) -> bool:
    """True between t's close and the next session's open.

    Values that yfinance only gives "as of now" (fundamentals: market cap and
    P/E use the live price) are point-in-time for t only inside this window.
    """
    now = now or datetime.now(timezone.utc)
    return cutoff(t) <= now < next_open(t)


def parse_date(s: str) -> date:
    return date.fromisoformat(s)
