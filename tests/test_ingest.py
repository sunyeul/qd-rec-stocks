"""Leakage and parsing tests for ingest/. No network needed.

    uv run python -m pytest tests -q
"""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ingest import clock, config, prices, reader, snapshot


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")


# --- clock -----------------------------------------------------------------


def test_cutoff_is_new_york_close():
    assert clock.cutoff(date(2026, 10, 23)) == datetime(2026, 10, 23, 20, 0, tzinfo=timezone.utc)  # EDT
    assert clock.cutoff(date(2026, 12, 4)) == datetime(2026, 12, 4, 21, 0, tzinfo=timezone.utc)  # EST


# --- snapshot / reader -----------------------------------------------------

def test_reader_never_returns_a_later_partition():
    for d, v in [(date(2026, 10, 20), 1), (date(2026, 10, 21), 2), (date(2026, 10, 23), 3)]:
        snapshot.write(pd.DataFrame({"ticker": ["AAA"], "v": [v]}), "prices", d)
    assert reader.load_asof("prices", date(2026, 10, 22))["v"].iloc[0] == 2
    assert reader.load_asof("prices", date(2026, 10, 23))["v"].iloc[0] == 3
    with pytest.raises(reader.NoSnapshotError):
        reader.load_asof("prices", date(2026, 10, 19))


def test_write_refuses_to_rewrite_history():
    d = date(2026, 10, 20)
    snapshot.write(pd.DataFrame({"ticker": ["AAA"]}), "prices", d)
    with pytest.raises(FileExistsError):
        snapshot.write(pd.DataFrame({"ticker": ["BBB"]}), "prices", d)


def _news_row(id_, ticker, published_at):
    return {"id": id_, "ticker": ticker, "title": id_, "summary": "", "publisher": "x",
            "published_at": published_at, "url": "", "content_type": "STORY"}


def test_news_cutoff_and_dedup():
    t = date(2026, 10, 22)
    # Fetched on t: one article before the close, one after it (published 17:30 New York).
    snapshot.write(pd.DataFrame([
        _news_row("before", "AAA", "2026-10-22T19:00:00+00:00"),
        _news_row("after-close", "AAA", "2026-10-22T21:30:00+00:00"),
    ]), "news", t)
    # Fetched on the next day: repeats 'before' and adds a new article.
    snapshot.write(pd.DataFrame([
        _news_row("before", "AAA", "2026-10-22T19:00:00+00:00"),
        _news_row("next-day", "AAA", "2026-10-23T13:00:00+00:00"),
    ]), "news", date(2026, 10, 23))

    got = reader.load_news_asof(t)
    assert got["id"].tolist() == ["before"]

    got_next = reader.load_news_asof(date(2026, 10, 23))
    assert sorted(got_next["id"]) == ["after-close", "before", "next-day"]


def test_news_lookback_window():
    t = date(2026, 10, 22)
    snapshot.write(pd.DataFrame([
        _news_row("old", "AAA", "2026-09-01T12:00:00+00:00"),
        _news_row("recent", "AAA", "2026-10-15T12:00:00+00:00"),
    ]), "news", t)
    assert reader.load_news_asof(t, lookback_days=30)["id"].tolist() == ["recent"]


# --- prices ----------------------------------------------------------------

def test_price_features_ignore_rows_after_t():
    idx = pd.bdate_range("2025-01-01", "2026-10-30")
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    volume = pd.Series(1_000.0, index=idx)
    t = date(2026, 10, 22)
    feats = prices.compute_features(close, volume, t)
    assert feats["last_price_date"] == "2026-10-22"
    assert feats["last_close"] == pytest.approx(close.loc["2026-10-22"])
    # A spike after t must not change anything.
    spiked = close.copy()
    spiked.loc["2026-10-23":] = 10_000
    assert prices.compute_features(spiked, volume, t) == feats


def test_price_features_short_history_is_nan():
    idx = pd.bdate_range("2026-10-01", "2026-10-22")
    feats = prices.compute_features(pd.Series(100.0, index=idx), pd.Series(1.0, index=idx), date(2026, 10, 22))
    assert np.isnan(feats["ret_12m"]) and np.isnan(feats["vol_60d"])
    assert feats["ret_1m"] == pytest.approx(0.0) or np.isnan(feats["ret_1m"])


# --- backfill (reconstructed fundamentals) -----------------------------------

def _raw_history():
    from ingest import backfill  # noqa: F401
    st = pd.DataFrame({"revenue": [100.0, 120.0], "operating_income": [10.0, 18.0], "net_income": [5.0, 12.0],
                       "equity": [50.0, 60.0], "debt": [20.0, 30.0]},
                      index=pd.to_datetime(["2023-12-31", "2024-12-31"]))
    div = pd.Series([1.0, 1.0, 9.0], index=pd.to_datetime(["2024-06-01", "2025-02-01", "2025-04-15"]).tz_localize("UTC"))
    return {"info": {"name": "Acme", "industry": "Tools", "business_summary": "", "shares_now": 10.0},
            "statements": st, "dividends": div}


def test_backfill_uses_only_statements_filed_by_t():
    from ingest import backfill
    idx = pd.bdate_range("2024-01-01", "2025-06-30")
    close = pd.Series(np.linspace(10, 20, len(idx)), index=idx)
    before = backfill.fundamentals_asof(_raw_history(), date(2025, 2, 27), close, close)  # FY2024 + 58 days
    after = backfill.fundamentals_asof(_raw_history(), date(2025, 3, 3), close, close)
    assert before["fiscal_year_end"] == "2023-12-31" and before["revenue_growth_yoy"] is None
    assert after["fiscal_year_end"] == "2024-12-31"
    assert after["revenue_growth_yoy"] == pytest.approx(0.2)
    price_t = close[close.index.date <= date(2025, 3, 3)].iloc[-1]
    assert after["market_cap"] == pytest.approx(10 * price_t)  # no close after t
    assert after["pe_trailing"] == pytest.approx(10 * price_t / 12)
    # the 2025-04-15 dividend is after t; the 2024-06-01 one is inside the trailing year
    assert after["dividend_yield"] == pytest.approx(2.0 / price_t * 100)


# --- news_history (Google News RSS) -----------------------------------------

_RSS = """<rss><channel>
<item><title>Abbott beats estimates - Reuters</title><link>https://x/1</link><guid>g1</guid>
<pubDate>Sun, 30 Mar 2025 07:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>Abbott news on the day - Reuters</title><link>https://x/2</link><guid>g2</guid>
<pubDate>Mon, 31 Mar 2025 07:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
</channel></rss>"""


def test_news_history_keeps_only_days_before_t():
    from ingest import news_history
    rows = news_history.parse_rss(_RSS, "ABT", date(2025, 3, 31))
    assert [r["id"] for r in rows] == ["g1"]  # dated t itself: time of day unknown, could be after the close
    assert rows[0]["title"] == "Abbott beats estimates" and rows[0]["publisher"] == "Reuters"


def test_news_history_search_name():
    from ingest.news_history import search_name
    assert search_name("Walt Disney Company (The)") == "Walt Disney"
    assert search_name("Apple Inc.") == "Apple"
    assert search_name("Lowe's Companies, Inc.") == "Lowe's"
