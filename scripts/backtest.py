"""Run the backtest: backfill snapshots, build cards, seed, run each variant, report.

    uv run python -m scripts.backtest                  # every date
    uv run python -m scripts.backtest --step 1         # one more unfinished date, then report and stop
    uv run python -m scripts.backtest --dates 2026-04-30 --iterations 10   # a quick trial

Every snapshot date is after the knowledge cutoff of the proposer, critique and
summary models (apps/stocks/config.py), so the models cannot know what followed.
Finished runs are skipped on a rerun.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from shared.archive_format import load_archive
from shared.paths import find_latest_archive

ROOT = Path(__file__).resolve().parent.parent
# Month ends after the proposer's knowledge cutoff (gemini-3.5-flash-lite, March 2026),
# up to the last date with a full month of closes before 2026-10-22. Scored on 1-month
# returns (3 months where passed).
DATES = ["2026-04-30", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-22"]
# name -> (loop method, anonymized cards)
VARIANTS = {"qd_rec": ("qd_rec", False), "novelty": ("novelty", False), "qd_rec_anon": ("qd_rec", True)}
# QD-Rec against the Novelty contrast (random stocks, no few-shot evolution) on every date.
DEFAULT_VARIANTS = ["qd_rec", "novelty"]
# Anonymized QD-Rec on two dates: how much the model's knowledge of company names
# shifts the picks (the "distraction effect" of Glasserman & Lin).
EXTRA = {"2026-05-29": ["qd_rec_anon"], "2026-08-31": ["qd_rec_anon"]}
ENV = {"QD_STOCKS_DATA": "data/backtest500", "QD_ARCHIVES_ROOT": "archives/backtest500",
       "PYTHONUNBUFFERED": "1"}  # stream each step's progress into the log as it happens


def key_status() -> dict:
    """The OpenRouter key's spend so far ("usage") and "limit_remaining" (None without a limit)."""
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return requests.get("https://openrouter.ai/api/v1/key", timeout=30,
                        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()["data"]


def remaining_budget() -> float | None:
    """Dollars still spendable: the smaller of the key's limit and the account's credit
    balance (a key limit above the balance does not help)."""
    import requests

    status = key_status()  # also loads .env
    data = requests.get("https://openrouter.ai/api/v1/credits", timeout=30,
                        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()["data"]
    left = [data["total_credits"] - data["total_usage"], status.get("limit_remaining")]
    return min(x for x in left if x is not None)


def settled_usage() -> float:
    """Spend so far, after a pause: OpenRouter books costs with a delay of some seconds.

    shortcut: per-run costs can still leak a little into the next run; the total is exact.
    """
    time.sleep(30)
    return key_status()["usage"]


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
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help=f"any of {', '.join(VARIANTS)}")
    # About $0.002 per evaluation (measured 2026-10-24) plus $0.15 per date for news summaries.
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--min-budget", type=float, default=0.5,
                    help="stop starting new LLM runs when less than this many dollars can be spent")
    ap.add_argument("--step", type=int, default=0,
                    help="run LLM work for at most this many unfinished dates, then report on every finished "
                         "date and stop (0 = all)")
    ap.add_argument("--no-extra", action="store_true", help="skip the extra runs in EXTRA")
    ap.add_argument("--manifest", default="archives/backtest500/runs.json")
    args = ap.parse_args()
    os.environ.update(ENV)
    dates = args.dates.split(",")

    run("ingest.backfill", "--dates", args.dates)
    from apps.stocks.profiles import DEFAULT_PROFILE

    runs = []
    out_of_budget = False
    started = 0
    # Spend and wall time per finished run, kept across calls for the report.
    costs_file = ROOT / args.manifest.replace(".json", "_costs.json")
    costs = json.loads(costs_file.read_text(encoding="utf-8")) if costs_file.exists() else {}

    def tag_of(t, variant):
        return f"{DEFAULT_PROFILE}-{t}-{'anon' if VARIANTS[variant][1] else 'named'}"

    for t in dates:
        variants = args.variants.split(",")
        if args.variants == ",".join(DEFAULT_VARIANTS) and not args.no_extra:
            variants += EXTRA.get(t, [])
        unfinished = [v for v in variants if not latest(tag_of(t, v), VARIANTS[v][0])]
        if unfinished and args.step and started >= args.step:
            continue  # left for the next --step call
        started += bool(unfinished)
        for variant in variants:
            method, anon = VARIANTS[variant]
            env = {"SNAPSHOT_DATE": t, "ANONYMIZE": "1" if anon else "0"}
            tag = tag_of(t, variant)
            if variant not in unfinished:
                runs.append({"t": t, "method": variant, "archive": latest(tag, method)})
                continue
            left = remaining_budget()
            if left is not None and left < args.min_budget:
                print(f"\n[budget] ${left:.2f} left, below --min-budget; reporting the {len(runs)} finished runs")
                out_of_budget = True
                break
            spent, started_at = settled_usage(), time.time()
            run("apps.stocks.create_embeddings", **env)
            # Each card mode gets its own seed archive, shared by the methods that use it.
            if not latest(tag, "seed"):
                run("apps.stocks.seed", **env)
            if not latest(tag, method):
                run("apps.stocks.qd_rec", "--method", method, "--iterations", str(args.iterations), **env)
            archive = latest(tag, method)
            # Includes this date's cards and seed when this run was the first to need them.
            costs[archive] = {"cost": round(settled_usage() - spent, 4),
                              "minutes": round((time.time() - started_at) / 60, 1)}
            costs_file.write_text(json.dumps(costs, indent=1), encoding="utf-8")
            runs.append({"t": t, "method": variant, "archive": archive})
        manifest = ROOT / args.manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(runs, indent=1), encoding="utf-8")
        if out_of_budget:
            break

    run("evaluation.report", "--manifest", args.manifest, "--out", "reports/backtest500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
