"""Universe: S&P 100 members from Wikipedia, or a CSV you provide.

The membership is stored with every snapshot, so later evaluation knows
exactly which tickers were investable on each date (no survivorship bias
from re-reading today's list).
"""

from io import StringIO

import pandas as pd
import requests

from . import config

_HEADERS = {"User-Agent": "qd-rec-stocks/0.1 (personal research)"}


def to_yahoo_symbol(symbol: str) -> str:
    """BRK.B -> BRK-B (Yahoo uses a dash for share classes)."""
    return symbol.strip().upper().replace(".", "-")


def fetch_sp100() -> pd.DataFrame:
    resp = requests.get(config.UNIVERSE_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return _parse_sp100(resp.text)


def fetch_sp100_asof(when) -> tuple[pd.DataFrame, str]:
    """Members from the last Wikipedia revision at or before `when` (an aware datetime),
    so a backfilled snapshot uses the membership of its own date. Returns (members, revision time)."""
    api = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "revisions", "titles": "S&P 100", "rvprop": "ids|timestamp",
              "rvlimit": 1, "rvstart": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "json"}
    page = next(iter(requests.get(api, params=params, headers=_HEADERS, timeout=30).json()["query"]["pages"].values()))
    rev = page["revisions"][0]
    resp = requests.get("https://en.wikipedia.org/w/index.php", params={"oldid": rev["revid"]},
                        headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return _parse_sp100(resp.text), rev["timestamp"]


def _parse_sp100(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), match="Symbol")
    table = next(t for t in tables if {"Symbol", "Name"} <= set(t.columns))
    out = pd.DataFrame({
        "ticker": table["Symbol"].map(to_yahoo_symbol),
        "name": table["Name"].astype(str),
        "sector": table["Sector"].astype(str) if "Sector" in table else "",
    })
    if not 90 <= len(out) <= 110:
        raise ValueError(f"S&P 100 table has {len(out)} rows; the page layout may have changed")
    return out.drop_duplicates("ticker").reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    """A custom universe: a CSV with a 'ticker' column (name, sector optional)."""
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} needs a 'ticker' column")
    df["ticker"] = df["ticker"].map(to_yahoo_symbol)
    for col in ("name", "sector"):
        if col not in df.columns:
            df[col] = ""
    return df[["ticker", "name", "sector"]].drop_duplicates("ticker").reset_index(drop=True)


def collect(universe_file: str | None = None) -> pd.DataFrame:
    df = load_csv(universe_file) if universe_file else fetch_sp100()
    df["source"] = universe_file or config.UNIVERSE_URL
    return df
