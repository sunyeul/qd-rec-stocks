"""Label each thesis with its main return driver (Jev Choice), cached by entry ID.

Readers cannot interpret an embedding distance, but "11 of 40 ideas are turnarounds"
they can. Runs in the evaluation layer only, after the fact, so the labeler's
knowledge cutoff does not matter here.
"""

import asyncio
import json
import math
from pathlib import Path

from apps.stocks import jev

MODEL = "typesafe/jev-1.14"
CACHE = Path("reports/drivers_cache.json")
DRIVERS = {
    "growth": "Revenue or earnings growth that the market underprices",
    "rerating": "Valuation too low relative to quality or peers; expects a re-rating",
    "turnaround": "Recovery from a recent setback, restructuring or new management",
    "defensive_income": "Stability, low volatility or dividends",
    "event": "A specific event: deal, spin-off, product launch, regulatory decision",
    "momentum": "Recent price or earnings momentum expected to continue",
    "quality": "High margins, returns on capital or balance-sheet strength",
}


async def _label(entry: dict, sem: asyncio.Semaphore) -> str:
    fields = entry.get("thesis_fields") or {}
    async with sem:
        answers = await jev.decide(MODEL, {"thesis": fields.get("thesis", ""), "catalyst": fields.get("catalyst", "")},
                                   {"driver": {"type": "choice", "instructions": "What is this thesis's main return driver?",
                                               "criteria": DRIVERS}})
    return answers["driver"]["choice"]


def labels(entries: list[dict]) -> dict[str, str]:
    """{entry_id: driver} for the given entries, calling Jev only for uncached ones."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [e for e in entries if e["entry_id"] not in cache]

    async def run():
        sem = asyncio.Semaphore(8)
        return await asyncio.gather(*(_label(e, sem) for e in todo))

    if todo:
        for e, d in zip(todo, asyncio.run(run())):
            cache[e["entry_id"]] = d
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {e["entry_id"]: cache[e["entry_id"]] for e in entries}


def entropy(values) -> float:
    """Normalized Shannon entropy of the label mix: 0 = all one driver, 1 = all drivers equally."""
    values = list(values)
    counts = [values.count(v) for v in set(values)]
    n = sum(counts)
    h = -sum(c / n * math.log(c / n) for c in counts)
    return h / math.log(len(DRIVERS))
