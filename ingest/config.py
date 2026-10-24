"""Settings for the backtest data store. The path can be redirected with QD_STOCKS_DATA."""

import os
from datetime import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / os.getenv("QD_STOCKS_DATA", "data/backtest500")
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# A snapshot date t is a US trading date. Its information cutoff is that day's
# close: news published after 16:00 New York time belongs to the next snapshot.
MARKET_TZ = "America/New_York"
MARKET_CLOSE = time(16, 0)

# Benchmark stored next to the universe's prices.
BENCHMARK_TICKER = "SPY"

# Calendar days of price history fetched to compute 12-month features.
PRICE_LOOKBACK_DAYS = 400
# News window used when cards are built (reader.load_news_asof default).
NEWS_LOOKBACK_DAYS = 30

# Politeness and robustness for per-ticker requests.
REQUEST_PAUSE_SEC = float(os.getenv("INGEST_PAUSE_SEC", "0.4"))
MAX_RETRIES = 3
