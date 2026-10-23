"""Fundamentals and company profile from yfinance ``Ticker.info``.

yfinance only gives today's values, so these are point-in-time only if
they are saved on the day they are fetched. That is why they cannot be
backfilled and why collection starts now.
"""

from datetime import datetime, timezone
import time

import pandas as pd

from . import config

# yfinance info key -> stored column
FIELDS = {
    "shortName": "name",
    "sector": "sector",
    "industry": "industry",
    "longBusinessSummary": "business_summary",
    "marketCap": "market_cap",
    "trailingPE": "pe_trailing",
    "forwardPE": "pe_forward",
    "priceToBook": "pb",
    # Units have changed across yfinance versions (fraction vs percent);
    # stored as returned, normalise when building cards.
    "dividendYield": "dividend_yield",
    "revenueGrowth": "revenue_growth_yoy",
    "earningsGrowth": "earnings_growth_yoy",
    "operatingMargins": "operating_margin",
    "profitMargins": "profit_margin",
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
    "beta": "beta",
}


def extract(info: dict) -> dict:
    return {col: info.get(key) for key, col in FIELDS.items()}


def fetch_one(ticker: str) -> dict:
    import yfinance as yf

    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).info or {}
            if not info:
                raise RuntimeError("empty info")
            row = extract(info)
            row["error"] = None
            return row
        except Exception as e:  # network hiccups, rate limits, delisted symbols
            last_err = e
            time.sleep(2 ** attempt)
    return {"error": f"{type(last_err).__name__}: {last_err}"}


def collect(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, tk in enumerate(tickers, 1):
        row = fetch_one(tk)
        row["ticker"] = tk
        row["fetched_at"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        if row["error"]:
            print(f"  [fundamentals {i}/{len(tickers)}] {tk}: {row['error']}")
        time.sleep(config.REQUEST_PAUSE_SEC)
    return pd.DataFrame(rows)
