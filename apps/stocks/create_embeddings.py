"""Build the stock cards of snapshot t: news summaries (LLM, cached), then card embeddings.

    uv run python -m apps.stocks.create_embeddings                # latest snapshot, named cards
    uv run python -m apps.stocks.create_embeddings --limit 5      # trial on 5 stocks
    $env:ANONYMIZE = "1"; uv run python -m apps.stocks.create_embeddings

Summaries already cached for t are reused; embeddings are rebuilt each run
because they depend on the whole card.
"""

import argparse
import asyncio
import json

from shared.embeddings import get_embedding_async

from . import config, items, summaries


async def main_async(limit: int | None) -> None:
    t = items.snapshot_date()
    rows = items.select_rows(items.load_rows(t), config.PER_SECTOR)[:limit]
    print(f"snapshot {t}, {len(rows)} stocks, {'anonymized' if config.ANONYMIZE else 'named'} cards")

    await summaries.build_async(t, rows, config.ANONYMIZE)
    cards = items.attach_summaries(rows, t, config.ANONYMIZE)
    texts = {c["item_id"]: items.embedding_text(c) for c in cards}
    sizes = sorted(len(s) // 4 for s in texts.values())
    print(f"card tokens (approx.): median {sizes[len(sizes) // 2]}, max {sizes[-1]}")

    sem = asyncio.Semaphore(config.EMBEDDING_CONCURRENCY)

    async def embed(text):
        async with sem:
            return await get_embedding_async(text, dimensions=config.EMBEDDING_DIMENSIONS)

    vectors = await asyncio.gather(*(embed(s) for s in texts.values()))
    path = items.embeddings_file(t, config.ANONYMIZE)
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store.update(zip(texts, vectors))
    path.write_text(json.dumps(store), encoding="utf-8")
    print(f"{len(vectors)} embeddings -> {path}")
    print("\n--- sample card ---\n" + texts[cards[0]["item_id"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N stocks (for a trial run)")
    asyncio.run(main_async(ap.parse_args().limit))


if __name__ == "__main__":
    main()
