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
from .proposer import generate_hypothesis_async


def sample_stocks(profile: str, num_samples: int) -> list[dict]:
    stocks = load_items(profile)
    if num_samples > len(stocks):
        raise ValueError(f"--num-seeds {num_samples} exceeds the {len(stocks)} stocks in the pool")
    return random.sample(stocks, num_samples)


async def process_seed(
    stock: dict,
    investor_profile: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict:
    """One seed entry: thesis, number check, judge score and archive embedding.

    A thesis that fails the number check raises, so evaluate_seeds retries the
    stock instead of seeding the archive with invented numbers.
    """
    async with semaphore:
        print(f"[Seed {index+1}/{total}] Processing: {item_label(stock)}")

        hypothesis, fields = await generate_hypothesis_async(investor_profile, stock)
        judgment, embedding = await asyncio.gather(
            judge_score_async(investor_profile, stock, hypothesis, fields),
            get_embedding_async(archive_text(stock, hypothesis)),
        )
        if not judgment["verifier"]["passed"]:
            raise ValueError(judgment["reasoning"])

        entry = {
            "entry_id": new_entry_id(),
            "item_id": stock["item_id"],
            "status": "active",
            "method": "seed",
            **stock,
            "hypothesis": hypothesis,
            "thesis_fields": fields,
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
    stocks = sample_stocks(profile, num_seeds)
    print(f"Sampled {len(stocks)} stocks. Processing in parallel...\n")

    archive, num_attempted = await evaluate_seeds(
        stocks, lambda stock, index: process_seed(stock, investor_profile, semaphore, index, len(stocks)),
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
