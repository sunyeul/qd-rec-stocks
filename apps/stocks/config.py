"""
Configuration for the stock idea recommender.

The snapshot date t and the card mode are chosen with environment variables,
because the shared QD loop calls items.load_items(profile) with no other input:

    $env:SNAPSHOT_DATE = "2026-10-22"   # default: the latest universe snapshot
    $env:ANONYMIZE = "1"                # C-### IDs, market-cap buckets, masked names
    $env:PER_SECTOR = "3"               # only the 3 largest stocks per sector (PoC stage B)

Archive filenames carry the snapshot date and mode in the profile tag, so a
run never picks up a seed archive built from other cards.
"""

import os
from pathlib import Path

from ingest.config import DATA_DIR
from shared.paths import model_tag

_PROJECT_ROOT = Path(__file__).parent.parent.parent

SNAPSHOT_DATE = os.getenv("SNAPSHOT_DATE") or None
ANONYMIZE = os.getenv("ANONYMIZE", "0") == "1"
# "0" leaves the news section off the cards (the numbers-only variant).
WITH_NEWS = os.getenv("WITH_NEWS", "1") == "1"
# 0 = every universe stock; N = the N largest by market cap in each sector.
PER_SECTOR = int(os.getenv("PER_SECTOR", "0"))
# Seeds the ticker -> C-### shuffle; keep it fixed for the whole experiment.
ANON_SEED = int(os.getenv("ANON_SEED", "127"))

# Summaries and card embeddings, one folder per snapshot and mode.
CARDS_DIR = DATA_DIR / "cards"


def cards_dir(t, anonymize: bool) -> Path:
    return CARDS_DIR / f"snapshot_date={t}" / ("anon" if anonymize else "named")


# Every default model's knowledge cutoff precedes the backtest snapshots
# (gemini-2.5-flash-lite 2025-01-31, gemini-3.5-flash-lite 2026-03 per Google's
# model card, the latest; for the Jev judge see JUDGE_MODEL), so no model can know
# what happened after t. Change them only together with scripts/backtest.py DATES.

# News summary LLM: one call per stock and snapshot, cached in cards_dir.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "google/gemini-2.5-flash-lite")
SUMMARY_REASONING_EFFORT = os.getenv("SUMMARY_REASONING_EFFORT", "low") or None
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "10"))
# Most recent articles per stock given to the summary LLM.
MAX_ARTICLES = 20

# Rough card budget (characters / 4); news bullets are cut first when over.
CARD_TOKEN_LIMIT = 1500

# Embeddings
EMBEDDING_DIMENSIONS = 512
EMBEDDING_CONCURRENCY = 50

# QD_ARCHIVES_ROOT redirects both seed lookup and run output.
ARCHIVES_DIR = str(_PROJECT_ROOT / (os.getenv("QD_ARCHIVES_ROOT") or "archives") / "stocks")

# OpenRouter model IDs. The proposer and the judge come from different model
# families so the judge does not share the proposer's blind spots.
PROPOSER_MODEL = os.getenv("PROPOSER_MODEL", "google/gemini-3.5-flash-lite")
# Jev (TypeSafe, via OpenRouter's Decisions API) answers the judge's questions with
# probabilities: the score is 10 x P(beats the S&P 500 in 3 months). Its cutoff is not
# published; asked about 60 real Mar-Sep 2026 moves it could not tell true from false
# (AUC 0.50, vs 0.57 on 2024 moves). Any OpenRouter chat model ID switches to the LLM judge.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "typesafe/jev-1.14")
# Set to an empty string to disable reasoning effort. "low" scored like "medium" at lower cost.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low") or None
# The proposer writes the ideas, so it thinks harder than the judge ("high" in the paper's apps).
PROPOSER_REASONING_EFFORT = os.getenv("PROPOSER_REASONING_EFFORT", "medium") or None
PROPOSER_SUFFIX = "_prop-" + model_tag(PROPOSER_MODEL)
JUDGE_SUFFIX = "_judge-" + model_tag(JUDGE_MODEL)

# Seeds Python and NumPy randomness.
SEED = int(os.getenv("SEED", "0"))

# Random active entries shown as few-shot examples (Novelty+Evo, QD-Rec)
NUM_CONTEXT_SAMPLES = int(os.getenv("NUM_CONTEXT_SAMPLES", "3"))
# Nearest neighbors the novelty check compares against
NUM_NEIGHBORS_NOVELTY = int(os.getenv("NUM_NEIGHBORS_NOVELTY", "3"))

# Seed / QD loop (PoC: 300 evaluations per run)
NUM_SEEDS = 10
NUM_ITERATIONS = 300
NUM_PARALLEL_WORKERS = int(os.getenv("NUM_PARALLEL_WORKERS", "10"))

# Cosine-similarity cutoff for deduplicating additions within a batch.
DEDUP_SIMILARITY_THRESHOLD = 0.9
