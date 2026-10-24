"""Thesis-side tests: number verifier, per-sector pool, thesis fields, loop gating. No network.

    uv run python -m pytest tests -q
"""

import asyncio

import pytest

from apps.stocks import config, items, profiles, proposer, qd_rec
from apps.stocks.verifier import verify

CARD = "Returns: 1M +3.3%, 12M +33.0%; Volatility (60d): 24.6%; P/E trailing 31.2; avg volume $12.9B"
PROFILE = "Beat the S&P 500 over 3 to 12 months."


def test_verifier_accepts_card_numbers_and_rounding():
    fields = {"thesis": "Up 33% over 12M with 25% volatility beats the S&P 500.",
              "evidence": ["P/E trailing 31.2", "1M return +3.3%", "$12.9B daily volume"],
              "catalyst": "Q3 results", "risks": [], "falsifier": "falls 15% below entry", "horizon": "6 months"}
    assert verify(fields, CARD, PROFILE) == {"passed": True, "unmatched": [], "checked": 8}


def test_verifier_rejects_invented_or_changed_numbers():
    fields = {"thesis": "Revenue grew 18% and P/E is 31.5.", "evidence": ["P/E 31.2"],
              "catalyst": "", "risks": ["Debt of $40B"], "falsifier": "", "horizon": ""}
    result = verify(fields, CARD, PROFILE)
    assert not result["passed"] and result["unmatched"] == ["18", "31.5", "40"]


def _row(ticker, name, sector, cap):
    return {"ticker": ticker, "name": name, "sector": sector, "market_cap": cap}


def test_select_rows_takes_largest_per_sector_and_one_share_class():
    rows = [_row("GOOGL", "Alphabet Inc. (Class A)", "Comm", 3e12), _row("GOOG", "Alphabet Inc. (Class C)", "Comm", 3e12),
            _row("META", "Meta Platforms", "Comm", 2e12), _row("T", "AT&T", "Comm", 1e11),
            _row("XOM", "Exxon Mobil", "Energy", 5e11)]
    picked = [r["ticker"] for r in items.select_rows(rows, 2)]
    assert picked == ["GOOGL", "META", "XOM"]
    assert items.select_rows(rows, 0) is rows


def test_profile_tags_never_prefix_each_other(monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DATE", "2026-10-23")
    tags = set()
    for anon in (False, True):
        for per in (0, 3, 30):
            monkeypatch.setattr(config, "ANONYMIZE", anon)
            monkeypatch.setattr(config, "PER_SECTOR", per)
            tags.add(profiles.profile_tag("base"))
    assert len(tags) == 6
    assert not any(a != b and b.startswith(a + "-") for a in tags for b in tags)


def test_thesis_fields_and_render():
    fields = proposer.thesis_fields({"thesis": " Buy. ", "evidence": "one fact", "risks": None, "horizon": 6})
    assert fields == {"thesis": "Buy.", "evidence": ["one fact"], "catalyst": "", "risks": [],
                      "falsifier": "", "horizon": "6"}
    assert "Evidence:\n- one fact" in proposer.render_hypothesis(fields)


def test_failed_number_check_is_discarded_before_novelty(monkeypatch):
    async def propose(profile, stock, context):
        return "text", {"thesis": "Up 99%."}, {"operation": "explore"}

    async def critique(stock, hypothesis):
        return "- weak"

    async def judge(profile, stock, hypothesis, fields, critique=""):
        return {"score": 0, "reasoning": "Number check failed: 99 not on the card", "verifier": {"passed": False}}

    async def embed(text):
        return [1.0, 0.0]

    async def never(*args):
        raise AssertionError("novelty check must not run")

    monkeypatch.setattr(qd_rec, "generate_hypothesis_fewshot_async", propose)
    monkeypatch.setattr(qd_rec, "judge_score_async", judge)
    monkeypatch.setattr(qd_rec, "critique_async", critique)
    monkeypatch.setattr(qd_rec, "get_embedding_async", embed)
    monkeypatch.setattr(qd_rec, "archive_text", lambda stock, h: h)
    monkeypatch.setattr(qd_rec, "check_novelty_async", never)
    seed = {"entry_id": "e1", "status": "active", "embedding": [1.0, 0.0]}
    result = asyncio.run(qd_rec.iteration_async("p", {"item_id": "AAPL"}, [seed], [seed], "qd_rec"))
    assert result["action"] == "discarded"
    assert result["new_entry"]["admit_path"] == "verifier_fail"


def test_verifier_ignores_holding_periods_but_not_data():
    fields = {"thesis": "Outperform over a 6 to 12 month horizon and the next 2 quarters; a 6-month view.",
              "evidence": ["Revenue grew 18% in 12 months"], "catalyst": "", "risks": []}
    assert verify(fields, CARD, PROFILE)["unmatched"] == ["18"]
