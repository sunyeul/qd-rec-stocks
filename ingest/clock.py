"""Information cutoffs of snapshot dates, in New York market time."""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from . import config


def cutoff(t: date) -> datetime:
    """Information cutoff of snapshot t: t's market close, as an aware UTC time."""
    local = datetime.combine(t, config.MARKET_CLOSE, tzinfo=ZoneInfo(config.MARKET_TZ))
    return local.astimezone(timezone.utc)


def parse_date(s: str) -> date:
    return date.fromisoformat(s)
