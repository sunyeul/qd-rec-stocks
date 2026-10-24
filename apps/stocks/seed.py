"""
Seed the archive for an investor profile: sample NUM_SEEDS stocks and write,
check, judge and embed a thesis for each.

Usage:
    uv run python -m apps.stocks.seed
    $env:PER_SECTOR = "3"; uv run python -m apps.stocks.seed --num-seeds 10
"""

import os
import random
import asyncio
import argparse
from datetime import datetime

from shared.archive_format import archive_results, seed_config, write_archive
from shared.archive_records import new_entry_id
from shared.embeddings import get_embedding_async
from shared.paths import seed_suffix, unique_archive_path
from shared.seed_loop import evaluate_seeds
from shared.seeding import seed_everything

from .config import (
    REASONING_EFFORT,
    ARCHIVES_DIR,
    JUDGE_SUFFIX,
    PROPOSER_SUFFIX,
    PROPOSER_MODEL,
    JUDGE_MODEL,
    NUM_SEEDS,
    NUM_PARALLEL_WORKERS,
    SEED,
)
from .evaluation import judge_score_async
from .items import archive_text, item_label, load_items
from .profiles import DEFAULT_PROFILE, get_profile_text, profile_tag
from .proposer import critique_async, generate_hypothesis_async


MAX_SWAPS = 2  # stocks tried in place of one whose theses keep failing the number check


def sample_stocks(profile: str, num_samples: int) -> tuple[list[dict], list[dict]]:
    """(seed stocks, spare stocks in random order)."""
    stocks = load_items(profile)
    if num_samples > len(stocks):
        raise ValueError(f"--num-seeds {num_samples} exceeds the {len(stocks)} stocks in the pool")
    picked = random.sample(stocks, num_samples)  # same draw as before spares existed
    spares = [s for s in stocks if s not in picked]
    random.shuffle(spares)
    return picked, spares


async def process_seed(
    stock: dict,
    investor_profile: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    spares: list[dict] | None = None,
) -> dict:
    """One seed entry: thesis, number check, judge score and archive embedding.

    A thesis that fails the number check is never seeded. Some stocks fail every
    time (the proposer keeps adding a number it knows but the card lacks, e.g. DUK
    on 2026-05-29), so after a failure the slot moves to a spare stock, up to
    MAX_SWAPS times; then it raises and evaluate_seeds retries the slot.
    """
    async with semaphore:
        for swap in range(MAX_SWAPS + 1):
            print(f"[Seed {index+1}/{total}] Processing: {item_label(stock)}")
            hypothesis, fields = await generate_hypothesis_async(investor_profile, stock)
            critique, embedding = await asyncio.gather(
                critique_async(stock, hypothesis), get_embedding_async(archive_text(stock, hypothesis)))
            judgment = await judge_score_async(investor_profile, stock, hypothesis, fields, critique)
            if judgment["verifier"]["passed"]:
                break
            if swap == MAX_SWAPS or not spares:
                raise ValueError(judgment["reasoning"])
            print(f"[Seed {index+1}/{total}] {item_label(stock)}: {judgment['reasoning']}; trying a spare stock")
            stock = spares.pop()

        entry = {
            "entry_id": new_entry_id(),
            "item_id": stock["item_id"],
            "status": "active",
            "method": "seed",
            **stock,
            "hypothesis": hypothesis,
            "thesis_fields": fields,
            "critique": critique,
            "judgment": judgment,
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
        }
        print(f"[Seed {index+1}/{total}] Done: {item_label(stock)} - Score: {judgment['score']}/10")
        return entry


async def main_async(profile: str = DEFAULT_PROFILE, num_seeds: int = NUM_SEEDS, seed: int = SEED):
    started_at = datetime.now().isoformat()
    investor_profile = get_profile_text(profile)
    tag = profile_tag(profile)
    print(f"Profile: {profile} ({tag})")
    print(f"Parallel workers: {NUM_PARALLEL_WORKERS}\n")

    semaphore = asyncio.Semaphore(NUM_PARALLEL_WORKERS)
    stocks, spares = sample_stocks(profile, num_seeds)
    print(f"Sampled {len(stocks)} stocks. Processing in parallel...\n")

    archive, num_attempted = await evaluate_seeds(
        stocks, lambda stock, index: process_seed(stock, investor_profile, semaphore, index, len(stocks), spares),
        num_parallel_workers=NUM_PARALLEL_WORKERS,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_file = unique_archive_path(os.path.join(
        ARCHIVES_DIR, f"{timestamp}_{tag}_seed{PROPOSER_SUFFIX}{JUDGE_SUFFIX}{seed_suffix(seed)}.json"))
    os.makedirs(ARCHIVES_DIR, exist_ok=True)

    write_archive(
        output_file,
        profile=profile,
        method="seed",
        created_at=started_at,
        completed=len(archive) == num_seeds,
        config=seed_config(
            proposer_model=PROPOSER_MODEL,
            judge_model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            num_seeds=num_seeds,
            num_parallel_workers=NUM_PARALLEL_WORKERS,
            seed=seed,
        ),
        results=archive_results(0, archive, num_attempted, quality=lambda e: e["judgment"]["score"]),
        entries=archive,
    )
    if len(archive) != num_seeds:
        raise RuntimeError(
            f"incomplete seeding: {len(archive)}/{num_seeds} seeds persisted; archive saved to {output_file}"
        )

    print(f"\nSaved {len(archive)} seeds to {output_file} ({num_attempted} attempts)")
    for entry in archive:
        print(f"  {item_label(entry)}: {entry['judgment']['score']}/10")


def main():
    parser = argparse.ArgumentParser(description="Generate the seed archive for stock experiments")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=f"investor profile (default: {DEFAULT_PROFILE})")
    parser.add_argument("--num-seeds", "-n", type=int, default=NUM_SEEDS,
                        help=f"number of seeds to generate (default: {NUM_SEEDS})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()
    seed_everything(args.seed)
    asyncio.run(main_async(args.profile, args.num_seeds, seed=args.seed))


if __name__ == "__main__":
    main()
