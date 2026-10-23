"""Recent news per ticker from yfinance.

Yahoo returns only the latest articles, so news has to be fetched every day
and accumulated; reader.load_news_asof assembles the window later. Each
article keeps its publication time, which is what the cutoff is checked on.
"""

from datetime import datetime, timezone
import time

import pandas as pd

from . import config

NEWS_COUNT = 20


def _to_utc_iso(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def parse_item(raw: dict, ticker: str) -> dict | None:
    """Normalise one yfinance news item. Handles both the current shape
    ({"id", "content": {...}}) and the older flat one ({"uuid", "title", ...})."""
    c = raw.get("content")
    if isinstance(c, dict):
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        item = {
            "id": raw.get("id") or c.get("id"),
            "title": c.get("title"),
            "summary": c.get("summary") or c.get("description") or "",
            "publisher": (c.get("provider") or {}).get("displayName"),
            "published_at": _to_utc_iso(c.get("pubDate") or c.get("displayTime")),
            "url": url,
            "content_type": c.get("contentType"),
        }
    else:
        item = {
            "id": raw.get("uuid") or raw.get("id"),
            "title": raw.get("title"),
            "summary": raw.get("summary", ""),
            "publisher": raw.get("publisher"),
            "published_at": _to_utc_iso(raw.get("providerPublishTime")),
            "url": raw.get("link"),
            "content_type": raw.get("type"),
        }
    if not item["id"] or not item["title"] or not item["published_at"]:
        return None
    item["ticker"] = ticker
    return item


def _with_retries(fn, label: str) -> list:
    """Call fn() up to MAX_RETRIES times; return [] (and say so) if all fail.

    yfinance hides many request errors (it logs them and returns an empty
    list), so an empty result is not proof that a ticker had no news.
    """
    for attempt in range(config.MAX_RETRIES):
        try:
            return list(fn() or [])
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                print(f"  [news] {label}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return []


def related_to(raw: dict, ticker: str) -> bool:
    """Search results are matched on text, so keep an article only when Yahoo
    tags it with the ticker (or gives no tags at all)."""
    tags = raw.get("relatedTickers")
    return not tags or ticker in tags


def fetch_one(ticker: str) -> tuple[list[dict], dict]:
    """Articles for one ticker, plus counts that show where they came from.

    First the ticker news stream (Ticker.get_news); if it returns nothing,
    Yahoo's search endpoint (yf.Search), whose items use the older flat shape.
    """
    import yfinance as yf

    stats = {"ticker": ticker, "stream_raw": 0, "search_raw": 0, "parsed": 0, "source": None}
    raw = _with_retries(lambda: yf.Ticker(ticker).get_news(count=NEWS_COUNT, tab="news"), f"{ticker} stream")
    stats["stream_raw"] = len(raw)
    items = [p for r in raw if (p := parse_item(r, ticker))]
    source = "stream"
    if not items:
        found = _with_retries(
            lambda: yf.Search(ticker, max_results=1, news_count=NEWS_COUNT).news, f"{ticker} search")
        stats["search_raw"] = len(found)
        items = [p for r in found if related_to(r, ticker) and (p := parse_item(r, ticker))]
        source = "search"
        if found and not items:
            print(f"  [news] {ticker}: {len(found)} search items, none parsed/related; "
                  f"first item keys: {sorted(found[0].keys())}")
    if raw and not items:
        print(f"  [news] {ticker}: {len(raw)} stream items, none parsed; first item keys: {sorted(raw[0].keys())}")
    for item in items:
        item["source"] = source
    stats["parsed"] = len(items)
    stats["source"] = source if items else None
    return items, stats


COLUMNS = ["id", "ticker", "title", "summary", "publisher", "published_at", "url", "content_type", "source"]


def collect(tickers: list[str]) -> pd.DataFrame:
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows, stats = [], []
    for tk in tickers:
        items, s = fetch_one(tk)
        rows.extend(items)
        stats.append(s)
        time.sleep(config.REQUEST_PAUSE_SEC)
    st = pd.DataFrame(stats)
    print(f"  [news] stream raw {int(st['stream_raw'].sum())}, search raw {int(st['search_raw'].sum())}, "
          f"kept {int(st['parsed'].sum())}; tickers with no news: {int((st['parsed'] == 0).sum())}/{len(st)}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["fetched_at"] = fetched_at
    return df.drop_duplicates(["ticker", "id"]).reset_index(drop=True)
