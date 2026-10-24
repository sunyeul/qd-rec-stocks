"""Past headlines for the backtest from Google News RSS search (no registration).

Yahoo only serves recent news and the GDELT API rate-limited this machine, so
past snapshots take headlines from Google News with a date range. RSS
gives a headline, publisher and publication date, but the time of day is not
reliable, so only articles dated strictly before t are kept: news from day t
itself could have come after the close.

shortcut: today's search ranking decides which past articles are returned,
which may favour stories that mattered later; note it as a limitation.
"""

import re
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import pandas as pd
import requests

from . import config

COLUMNS =["id", "ticker", "title", "summary", "publisher", "published_at", "url", "content_type", "source"]

RSS_URL = "https://news.google.com/rss/search"
_HEADERS = {"User-Agent": "qd-rec-stocks/0.1 (personal research)"}
_SUFFIX = re.compile(r"[\s,]+(inc|corp|corporation|companies|company|co|plc|ltd|holdings|group|incorporated)\b\.?.*$",
                     re.IGNORECASE)


def search_name(name: str) -> str:
    """"Walt Disney Company (The)" -> "Walt Disney"; "Apple Inc." -> "Apple"."""
    name = re.sub(r"\s*\(.*?\)", "", name).strip()
    return _SUFFIX.sub("", name).strip() or name


def parse_rss(xml: str, ticker: str, t: date) -> list[dict]:
    """Items dated strictly before t, newest first."""
    rows = []
    for item in ElementTree.fromstring(xml).iter("item"):
        title, pub = item.findtext("title") or "", item.findtext("pubDate")
        if not title or not pub:
            continue
        published = parsedate_to_datetime(pub).astimezone(timezone.utc)
        if published.date() >= t:
            continue
        source = item.find("source")
        publisher = source.text if source is not None else None
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        rows.append({"id": item.findtext("guid") or item.findtext("link"), "ticker": ticker, "title": title,
                     "summary": "", "publisher": publisher, "published_at": published.isoformat(),
                     "url": item.findtext("link"), "content_type": "rss", "source": "google_news"})
    return sorted(rows, key=lambda r: r["published_at"], reverse=True)


def fetch_one(ticker: str, name: str, t: date) -> list[dict]:
    start = t - timedelta(days=config.NEWS_LOOKBACK_DAYS)
    query = f'"{search_name(name)}" ({ticker.replace("-", ".")} OR stock OR shares) after:{start} before:{t}'
    for attempt in range(config.MAX_RETRIES):
        try:
            r = requests.get(RSS_URL, headers=_HEADERS, timeout=30,
                             params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
            r.raise_for_status()
            return parse_rss(r.text, ticker, t)[:20]
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                print(f"  [news_history] {ticker} {t}: {type(e).__name__}: {e}")
            time.sleep(2 ** (attempt + 1))
    return []


def collect(members: pd.DataFrame, t: date) -> pd.DataFrame:
    rows = []
    for r in members.itertuples():
        rows += fetch_one(r.ticker, r.name, t)
        time.sleep(max(config.REQUEST_PAUSE_SEC, 1.0))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    n_empty = len(set(members["ticker"]) - set(df["ticker"]))
    print(f"  [news_history] {t}: {len(df)} headlines, {n_empty}/{len(members)} tickers with none")
    return df.drop_duplicates(["ticker", "id"]).reset_index(drop=True)
