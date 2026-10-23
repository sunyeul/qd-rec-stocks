"""Backtest pieces: portfolio rules and forward-return windows. No network.

    uv run python -m pytest tests -q
"""

from datetime import date

import pandas as pd
import pytest

from evaluation.returns import forward_returns, window
from portfolio.build import build


def _entry(tk, sector, score, status="active"):
    return {"item_id": tk, "sector": sector, "status": status, "entry_id": f"{tk}{score}",
            "judgment": {"score": score}}


def test_build_takes_best_thesis_per_stock_and_caps_sectors():
    entries = [_entry("A", "Tech", 9), _entry("A", "Tech", 4), _entry("B", "Tech", 8), _entry("C", "Tech", 7),
               _entry("D", "Energy", 6), _entry("E", "Energy", 10, status="discarded")]
    port = build(entries, n=3, k=2)
    assert [p["ticker"] for p in port] == ["A", "B", "D"]
    assert port[0]["score"] == 9


def test_forward_window_starts_at_t_close_and_needs_the_full_horizon():
    idx = pd.bdate_range("2025-03-27", periods=10)
    closes = pd.DataFrame({"X": range(100, 110)}, index=idx, dtype=float)
    # 2025-03-29 is a Saturday: the window starts at Friday's close
    assert window(closes, date(2025, 3, 29), 2).index[0] == pd.Timestamp("2025-03-28")
    assert forward_returns(closes, date(2025, 3, 28), 3)["X"] == pytest.approx(104 / 101 - 1)
    with pytest.raises(ValueError, match="not available"):
        forward_returns(closes, date(2025, 4, 7), 3)
