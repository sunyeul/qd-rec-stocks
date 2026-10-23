"""Stock card tests: point-in-time joins, anonymization, number grounding. No network.

    uv run python -m pytest tests -q
"""

import asyncio
from datetime import date

import pandas as pd
import pytest

from apps.stocks import config as stocks_config
from apps.stocks import items, summaries
from ingest import config, snapshot

T = date(2026, 10, 22)


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(stocks_config, "CARDS_DIR", tmp_path / "cards")


def _write_universe_and_prices(d):
    snapshot.write(pd.DataFrame({"ticker": ["AAPL", "LOW", "XXX"],
                                 "name": ["Apple Inc.", "Lowe's Companies, Inc.", "Broken Co"],
                                 "sector": ["Information Technology", "Consumer Discretionary", "Energy"]}),
                   "universe", d)
    snapshot.write(pd.DataFrame({"ticker": ["AAPL", "LOW", "XXX", "SPY"],
                                 "ret_1m": [0.033, -0.01, None, 0.0], "ret_3m": [0.05, 0.0, None, 0.0],
                                 "ret_12m": [0.33, 0.1, None, 0.0], "mom_12_1": [0.29, 0.1, None, 0.0],
                                 "vol_60d": [0.25, 0.2, None, 0.1], "avg_dollar_volume_20d": [1.2e10, 1e9, None, 1e10],
                                 "error": [None, None, "HTTPError", None],
                                 "is_benchmark": [False, False, False, True]}),
                   "prices", d)


def _fundamentals(pe):
    return pd.DataFrame({"ticker": ["AAPL", "LOW"], "name": ["Apple Inc.", "Lowe's Companies"],
                         "sector": ["Technology", "Consumer Cyclical"], "industry": ["Consumer Electronics", "Home Improvement"],
                         "business_summary": ["Apple Inc. designs the iPhone. It sells services. It is big. Fourth.", None],
                         "market_cap": [3.2e12, 1.4e11], "pe_trailing": [pe, 20.0], "dividend_yield": [0.41, 1.9],
                         "error": [None, None], "fetched_at": ["x", "x"]})


def test_rows_use_latest_fundamentals_not_after_t():
    _write_universe_and_prices(T)
    snapshot.write(_fundamentals(30.0), "fundamentals", date(2026, 10, 19))
    snapshot.write(_fundamentals(99.0), "fundamentals", date(2026, 10, 23))  # after t
    rows = {r["ticker"]: r for r in items.load_rows(T)}
    assert set(rows) == {"AAPL", "LOW"}  # price error dropped, benchmark not in the universe
    assert rows["AAPL"]["pe_trailing"] == 30.0
    assert rows["AAPL"]["fundamentals_date"] == "2026-10-19"
    assert rows["AAPL"]["name"] == "Apple Inc." and rows["AAPL"]["short_name"] == "Apple Inc."


def test_rows_without_fundamentals_still_make_a_card():
    _write_universe_and_prices(T)
    row = {**items.load_rows(T)[0], "news": {}, "anonymized": False}
    assert "not available at this snapshot" in items.item_text(row)


def test_summary_build_sees_only_news_before_the_close(monkeypatch):
    _write_universe_and_prices(T)
    snapshot.write(pd.DataFrame({
        "id": ["a", "b"], "ticker": ["AAPL", "AAPL"], "title": ["before", "after"], "summary": ["", ""],
        "publisher": ["P", "P"], "url": ["", ""], "content_type": ["", ""], "source": ["stream", "stream"],
        "published_at": ["2026-10-22T19:59:00+00:00", "2026-10-22T20:01:00+00:00"],  # close is 20:00 UTC
    }), "news", T)
    seen = {}

    async def fake(row, articles, anonymize):
        seen[row["ticker"]] = list(articles["title"])
        return {"facts": [], "contested": [], "business": None, "n_articles": len(articles)}

    monkeypatch.setattr(summaries, "summarize_one", fake)
    rows = items.load_rows(T)
    asyncio.run(summaries.build_async(T, rows, anonymize=False))
    assert seen["AAPL"] == ["before"]
    assert len(items.attach_summaries(rows, T, False)) == 2  # cached for every stock


def test_load_items_requires_summaries(monkeypatch):
    _write_universe_and_prices(T)
    monkeypatch.setattr(stocks_config, "SNAPSHOT_DATE", T.isoformat())
    with pytest.raises(RuntimeError, match="create_embeddings"):
        items.load_items("")


def test_keep_grounded_drops_invented_numbers():
    source = "Revenue rose 8% to $1,200 million in Q3."
    bullets = ["Revenue rose 8%.", "Revenue was 1200 million.", "Margins hit 45%.", "Demand was strong.", 3]
    assert summaries.keep_grounded(bullets, source) == ["Revenue rose 8%.", "Revenue was 1200 million.",
                                                        "Demand was strong."]


def _card_row(anonymized, **news):
    return {"ticker": "LOW", "name": "Lowe's Companies, Inc.", "short_name": "Lowe's Companies",
            "anon_id": "C-321", "sector": "Consumer Discretionary", "industry": "Home Improvement",
            "snapshot_date": "2026-10-22", "market_cap": 1.4e11, "ret_1m": 0.01, "vol_60d": 0.2,
            "fundamentals_date": "2026-10-19", "pe_trailing": 20.0,
            "business_summary": "Lowe's Companies, Inc. operates home improvement stores.",
            "anonymized": anonymized, "news": {"n_articles": 2, **news}}


def test_anonymized_card_hides_names_and_exact_size():
    row = _card_row(True, business="The company runs home improvement stores.",
                    facts=["Lowe's (LOW) raised guidance.", "LOWE'S beat estimates with low costs."])
    card = items.item_text(row)
    assert "lowe" not in card.lower() and "LOW)" not in card
    assert "C-321 (C-321) raised guidance" in card
    assert "with low costs" in card  # the ticker is matched case-sensitively
    assert "large ($10B-200B)" in card and "$140" not in card and "Home Improvement" not in card


def test_named_card_shows_names_and_numbers():
    card = items.item_text(_card_row(False, facts=["Raised guidance."]))
    assert "[Stock Lowe's Companies, Inc. (LOW)]" in card and "$140.0B" in card
    assert "P/E trailing 20.0" in card and "Raised guidance." in card


def test_long_card_is_cut_news_first(monkeypatch):
    monkeypatch.setattr(stocks_config, "CARD_TOKEN_LIMIT", 200)
    row = _card_row(False, facts=["fact " * 40] * 4, contested=["open " * 40] * 3)
    card = items.item_text(row)
    assert len(card) // 4 <= 200 and "open open" not in card


def test_anon_ids_are_unique_and_stable():
    tickers = [f"T{i}" for i in range(100)]
    ids = items.anon_ids(tickers)
    assert len(set(ids.values())) == 100
    assert ids == items.anon_ids(list(reversed(tickers)))
