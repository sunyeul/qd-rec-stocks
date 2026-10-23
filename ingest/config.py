"""Settings for data collection. Paths can be redirected with QD_STOCKS_DATA."""

import os
from datetime import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / os.getenv("QD_STOCKS_DATA", "data")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LOG_DIR = DATA_DIR / "logs"

# A snapshot date t is a US trading date. Its information cutoff is that day's
# close: news published after 16:00 New York time belongs to the next snapshot.
MARKET_TZ = "America/New_York"
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
# Wait a little after the close so the closing prices have settled.
SNAPSHOT_READY = time(16, 30)

# S&P 100 members (PoC universe) and the benchmark stored next to them.
UNIVERSE_URL = "https://en.wikipedia.org/wiki/S%26P_100"
BENCHMARK_TICKER = "SPY"

# Calendar days of price history fetched to compute 12-month features.
PRICE_LOOKBACK_DAYS = 400
# Fundamentals are refreshed when the latest partition is older than this.
FUNDAMENTALS_MAX_AGE_DAYS = 7
# News window used when cards are built (reader.load_news_asof default).
NEWS_LOOKBACK_DAYS = 30

# Politeness and robustness for per-ticker Yahoo requests.
REQUEST_PAUSE_SEC = float(os.getenv("INGEST_PAUSE_SEC", "0.4"))
MAX_RETRIES = 3
