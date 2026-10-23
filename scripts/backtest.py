"""Run the whole backtest: backfill snapshots, build cards, seed, run each method, report.

    uv run python -m scripts.backtest                          # default dates and methods
    uv run python -m scripts.backtest --iterations 30 --dates 2025-03-31 --variants qd_rec_news   # a quick trial

Every snapshot date is after the knowledge cutoff of the default proposer,
judge and summary models (config.py), so the models cannot know what followed.
Finished steps are skipped on a rerun. Data and archives go to their own folders
so the forward-test store is never touched.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from shared.archive_format import load_archive
from shared.paths import find_latest_archive

ROOT = Path(__file__).resolve().parent.parent
# Month ends after the judge's knowledge cutoff (gpt-6-luna, 2026-05-18), up to the last
# date with a full month of closes before 2026-10-22. Scored on 1-month returns.
DATES = ["2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-22"]
# The design's three variants: name -> (loop method, news on the cards)
VARIANTS = {"novelty": ("novelty", True), "qd_rec_numbers": ("qd_rec", False), "qd_rec_news": ("qd_rec", True)}
ENV = {"QD_STOCKS_DATA": "data/backtest", "QD_ARCHIVES_ROOT": "archives/backtest",
       "ANONYMIZE": "0", "PER_SECTOR": "0"}


def remaining_budget() -> float | None:
    """Dollars left under the OpenRouter key's limit, or None when it has no limit."""
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    data = requests.get("https://openrouter.ai/api/v1/key", timeout=30,
                        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()["data"]
    return data.get("limit_remaining")


def run(module: str, *args: str, **env: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **ENV, **env}, check=True)


def latest(tag: str, stage: str) -> str | None:
    """The newest completed archive for this tag and stage, or None."""
    from apps.stocks import config  # imported after ENV is set, for its model suffixes

    path = find_latest_archive(config.ARCHIVES_DIR, tag, f"{stage}{config.PROPOSER_SUFFIX}{config.JUDGE_SUFFIX}")
    return path if path and load_archive(path)["metadata"].get("completed_at") else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", default=",".join(DATES))
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"any of {', '.join(VARIANTS)}")
    # 5 dates x 3 variants x 150 evaluations fits a $9 key limit at luna's prices (with margin).
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--min-budget", type=float, default=0.5,
                    help="stop starting new LLM runs when the key has less than this many dollars left")
    ap.add_argument("--manifest", default="archives/backtest/runs.json")
    args = ap.parse_args()
    os.environ.update(ENV)
    dates = args.dates.split(",")

    run("ingest.backfill", "--dates", args.dates)
    runs = []
    out_of_budget = False
    for t in dates:
        for variant in args.variants.split(","):
            left = remaining_budget()
            if left is not None and left < args.min_budget:
                print(f"\n[budget] ${left:.2f} left, below --min-budget; reporting the {len(runs)} finished runs")
                out_of_budget = True
                break
            method, news = VARIANTS[variant]
            env = {"SNAPSHOT_DATE": t, "WITH_NEWS": "1" if news else "0"}
            tag = f"base-{t}-named-{'news' if news else 'nonews'}-all"
            run("apps.stocks.create_embeddings", **env)
            # Each card variant gets its own seed archive, shared by the methods that use it.
            if not latest(tag, "seed"):
                run("apps.stocks.seed", **env)
            if not latest(tag, method):
                run("apps.stocks.qd_rec", "--method", method, "--iterations", str(args.iterations), **env)
            runs.append({"t": t, "method": variant, "archive": latest(tag, method)})
        manifest = ROOT / args.manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(runs, indent=1), encoding="utf-8")
        if out_of_budget:
            break

    run("evaluation.report", "--manifest", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
