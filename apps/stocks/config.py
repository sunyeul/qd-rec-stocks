"""
Configuration for the stock idea recommender.

The snapshot date t and the card mode are chosen with environment variables,
because the shared QD loop calls items.load_items(profile) with no other input:

    $env:SNAPSHOT_DATE = "2026-10-22"   # default: the latest universe snapshot
    $env:ANONYMIZE = "1"                # C-### IDs, market-cap buckets, masked names
"""

import os
from pathlib import Path

from ingest.config import DATA_DIR

SNAPSHOT_DATE = os.getenv("SNAPSHOT_DATE") or None
ANONYMIZE = os.getenv("ANONYMIZE", "0") == "1"
# Seeds the ticker -> C-### shuffle; keep it fixed for the whole experiment.
ANON_SEED = int(os.getenv("ANON_SEED", "127"))

# Summaries and card embeddings, one folder per snapshot and mode.
CARDS_DIR = DATA_DIR / "cards"


def cards_dir(t, anonymize: bool) -> Path:
    return CARDS_DIR / f"snapshot_date={t}" / ("anon" if anonymize else "named")


# News summary LLM: one call per stock and snapshot, cached in cards_dir.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-6-luna")
SUMMARY_REASONING_EFFORT = os.getenv("SUMMARY_REASONING_EFFORT", "low") or None
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "10"))
# Most recent articles per stock given to the summary LLM.
MAX_ARTICLES = 20

# Rough card budget (characters / 4); news bullets are cut first when over.
CARD_TOKEN_LIMIT = 1500

# Embeddings
EMBEDDING_DIMENSIONS = 512
EMBEDDING_CONCURRENCY = 50
