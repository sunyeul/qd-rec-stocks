"""Rebuild past snapshots for the backtest from data that is dated.

    $env:QD_STOCKS_DATA = "data/backtest500"
    uv run python -m ingest.backfill --dates 2026-04-30,2026-05-29 --index sp500

Each t gets the index members Wikipedia listed at t's close. Prices are
point-in-time by construction (closes dated on or before t). Fundamentals are
recomputed from annual statements that were filed by t: a fiscal year counts
only from FILING_LAG_DAYS after its end (the 10-K deadline for large filers).
News comes from Google News RSS search (news_history.py).
"""

import argparse
import math
import pickle
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import clock, config, news_history, prices, snapshot, universe

# Columns of a fundamentals partition, the fields stock cards read.
COLUMNS = ["name", "sector", "industry", "business_summary", "market_cap", "pe_trailing", "pe_forward", "pb",
           "dividend_yield", "revenue_growth_yoy", "earnings_growth_yoy", "operating_margin", "profit_margin",
           "roe", "debt_to_equity", "beta"]
FILING_LAG_DAYS = 60
# Name and description do not change with t; sector and summary text are taken from today.
STATIC_INFO = {"shortName": "name", "industry": "industry", "longBusinessSummary": "business_summary",
               "sharesOutstanding": "shares_now"}
_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income": ["Net Income Common Stockholders", "Net Income"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "debt": ["Total Debt"],
}


def _pick(stmt: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in stmt.index:
            return stmt.loc[n]
    return pd.Series(dtype=float)


def fetch_raw(ticker: str) -> dict:
    """Everything one ticker needs for any past t; fetched once and cached."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    info = tk.info or {}
    income, balance = tk.income_stmt, tk.balance_sheet
    statements = pd.DataFrame({key: _pick(income if key in ("revenue", "operating_income", "net_income")
                                          else balance, names) for key, names in _ROWS.items()})
    statements.index = pd.to_datetime(statements.index)
    return {
        "info": {col: info.get(key) for key, col in STATIC_INFO.items()},
        "statements": statements.sort_index(),  # fiscal year end x line item
        "dividends": tk.dividends,
    }


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _ratio(a, b):
    a, b = _num(a), _num(b)
    return None if a is None or not b else a / b


def fundamentals_asof(raw: dict, t: date, close: pd.Series, spy: pd.Series) -> dict:
    """The fundamentals columns as they could have been known at t's close.

    `close` and `spy` are split-adjusted, not dividend-adjusted, daily closes.

    shortcut: market cap = today's share count x close at t, so buybacks or
    issuance since t shift P/E and P/B by a few percent; use dated share counts
    if valuation levels start to matter.
    """
    st = raw["statements"]
    filed = st[st.index + pd.Timedelta(days=FILING_LAG_DAYS) <= pd.Timestamp(t)]
    close = close[close.index.date <= t].dropna()
    row = {col: None for col in COLUMNS}
    row.update({k: v for k, v in raw["info"].items() if k != "shares_now"})
    row["sector"] = None  # the universe's GICS sector is used
    if close.empty:
        return {**row, "error": "no closes on or before t"}
    price = float(close.iloc[-1])
    cap = _num(raw["info"].get("shares_now")) and raw["info"]["shares_now"] * price
    row["market_cap"] = cap
    if not filed.empty:
        fy = filed.iloc[-1]
        row["pe_trailing"] = _ratio(cap, fy["net_income"]) if (_num(fy["net_income"]) or 0) > 0 else None
        row["pb"] = _ratio(cap, fy["equity"]) if (_num(fy["equity"]) or 0) > 0 else None
        row["operating_margin"] = _ratio(fy["operating_income"], fy["revenue"])
        row["profit_margin"] = _ratio(fy["net_income"], fy["revenue"])
        row["roe"] = _ratio(fy["net_income"], fy["equity"])
        de = _ratio(fy["debt"], fy["equity"])
        row["debt_to_equity"] = None if de is None else de * 100  # yfinance reports percent
        row["fiscal_year_end"] = filed.index[-1].date().isoformat()
        if len(filed) >= 2:
            prev = filed.iloc[-2]
            for col, key in (("revenue_growth_yoy", "revenue"), ("earnings_growth_yoy", "net_income")):
                g = _ratio(fy[key], prev[key])
                row[col] = None if g is None or (_num(prev[key]) or 0) <= 0 else g - 1
    div = raw["dividends"]
    if div is not None and len(div):
        idx = pd.to_datetime(div.index)
        div = pd.Series(div.values, index=idx.tz_convert(None) if idx.tz is not None else idx)
        paid = div[(div.index.date <= t) & (div.index.date > t - timedelta(days=365))].sum()
        row["dividend_yield"] = float(paid / price * 100)  # percent, like yfinance 1.x
    r = pd.concat([close.pct_change(), spy[spy.index.date <= t].pct_change()], axis=1).dropna().tail(252)
    if len(r) >= 120:
        row["beta"] = float(np.cov(r.iloc[:, 0], r.iloc[:, 1])[0, 1] / r.iloc[:, 1].var())
    row["error"] = None
    return row


def _raw_cache(tickers: list[str]) -> dict:
    path = config.DATA_DIR / "history_raw.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    for i, tk in enumerate(tk for tk in tickers if tk not in cache):
        try:
            cache[tk] = fetch_raw(tk)
        except Exception as e:  # keep going; the ticker gets an error row
            print(f"  [backfill] {tk}: {type(e).__name__}: {e}")
        time.sleep(config.REQUEST_PAUSE_SEC)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache))
    return cache


def _raw_closes(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                      auto_adjust=False, group_by="column", progress=False, threads=True)
    close = raw["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True, help="comma-separated snapshot dates YYYY-MM-DD")
    ap.add_argument("--universe-file", help="CSV with a 'ticker' column, used for every date")
    ap.add_argument("--index", choices=sorted(universe.INDEXES), default="sp500",
                    help="index whose point-in-time members form each date's universe (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    dates = sorted(clock.parse_date(s) for s in args.dates.split(","))

    # Each t gets the S&P 100 as Wikipedia listed it at t's close (no survivorship bias),
    # unless a fixed universe file is given.
    members_at = {}
    for t in dates:
        if args.universe_file:
            members_at[t] = universe.collect(args.universe_file)
        else:
            members_at[t], rev = universe.fetch_index_asof(args.index, clock.cutoff(t))
            members_at[t]["source"] = f"Wikipedia {universe.INDEXES[args.index][0]} (revision {rev})"
    all_tickers = sorted({tk for m in members_at.values() for tk in m["ticker"]})
    cache = _raw_cache(all_tickers)
    closes = _raw_closes(all_tickers + [config.BENCHMARK_TICKER], dates[0] - timedelta(days=400), dates[-1])

    for t in dates:
        members = members_at[t]
        tickers = members["ticker"].tolist()
        if not snapshot.exists("news", t) or args.overwrite:
            snapshot.write(news_history.collect(members, t), "news", t, overwrite=args.overwrite)
        if snapshot.exists("fundamentals", t) and not args.overwrite:
            print(f"[backfill] {t} prices and fundamentals exist, skipping")
            continue
        snapshot.write(members, "universe", t, overwrite=args.overwrite)
        snapshot.write(prices.collect(tickers + [config.BENCHMARK_TICKER], t), "prices", t, overwrite=args.overwrite)
        rows = []
        for tk in tickers:
            row = (fundamentals_asof(cache[tk], t, closes[tk], closes[config.BENCHMARK_TICKER])
                   if tk in cache and tk in closes.columns else {"error": "no history"})
            rows.append({**row, "ticker": tk, "fetched_at": f"reconstructed as of {t}"})
        df = pd.DataFrame(rows)
        snapshot.write(df, "fundamentals", t, overwrite=args.overwrite)
        print(f"[backfill] {t}: {len(df)} fundamentals rows, {int(df['error'].notna().sum())} with errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
