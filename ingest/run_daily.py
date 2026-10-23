"""Daily collection job: universe, prices, news, and weekly fundamentals.

    uv run python -m ingest.run_daily                 # latest settled US close
    uv run python -m ingest.run_daily --date 2026-10-23
    uv run python -m ingest.run_daily --limit 5       # quick trial on 5 tickers

Schedule it once a day after the US close (e.g. 07:00 JST). A partition
that already exists is left untouched unless --overwrite is given.
Exit code 1 if any step failed, so a scheduler can flag it.
"""

import argparse
import sys
import traceback
from datetime import date

import pandas as pd

from . import clock, config, fundamentals, news, prices, snapshot, universe


def _fundamentals_due(t: date, force: bool) -> bool:
    if force:
        return True
    past = [d for d in snapshot.list_dates("fundamentals") if d <= t]
    return not past or (t - past[-1]).days >= config.FUNDAMENTALS_MAX_AGE_DAYS


def _step(name: str, t: date, overwrite: bool, fn) -> bool:
    """Run one collector and write its partition. Returns True on success."""
    if snapshot.exists(name, t) and not overwrite:
        print(f"[{name}] {t} already collected, skipping")
        return True
    try:
        df: pd.DataFrame = fn()
        path = snapshot.write(df, name, t, overwrite=overwrite)
    except Exception:
        print(f"[{name}] FAILED")
        traceback.print_exc()
        return False
    n_err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    print(f"[{name}] {len(df)} rows -> {path}" + (f" ({n_err} with errors)" if n_err else ""))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="snapshot date YYYY-MM-DD (default: latest settled US close)")
    ap.add_argument("--universe-file", help="CSV with a 'ticker' column instead of the S&P 100")
    ap.add_argument("--limit", type=int, help="only the first N tickers (for a trial run)")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--force-fundamentals", action="store_true")
    ap.add_argument("--allow-late-fundamentals", action="store_true",
                    help="collect fundamentals outside t's close-to-next-open window (not point-in-time)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing partitions for this date")
    args = ap.parse_args(argv)

    t = clock.parse_date(args.date) if args.date else clock.latest_snapshot_date()
    print(f"snapshot date {t} (cutoff {clock.cutoff(t).isoformat()})")

    ok = _step("universe", t, args.overwrite, lambda: universe.collect(args.universe_file))
    if not ok and not snapshot.exists("universe", t):
        return 1
    tickers = snapshot.read("universe", t)["ticker"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"{len(tickers)} tickers")

    results = [ok]
    results.append(_step("prices", t, args.overwrite,
                         lambda: prices.collect(tickers + [config.BENCHMARK_TICKER], t)))
    if not args.skip_news:
        results.append(_step("news", t, args.overwrite, lambda: news.collect(tickers)))
    if not clock.in_fetch_window(t) and not args.allow_late_fundamentals:
        # Market cap and P/E would reflect prices after t's close.
        print(f"[fundamentals] skipped: only point-in-time for {t} between its close and "
              f"{clock.next_open(t).isoformat()} (use --allow-late-fundamentals to override)")
    elif _fundamentals_due(t, args.force_fundamentals):
        results.append(_step("fundamentals", t, args.overwrite, lambda: fundamentals.collect(tickers)))
    else:
        print(f"[fundamentals] collected within {config.FUNDAMENTALS_MAX_AGE_DAYS} days, skipping")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
