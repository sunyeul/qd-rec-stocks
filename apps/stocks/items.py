"""Stock cards at snapshot t: records, card text and item embeddings.

Collected data is read only through ingest.reader, so nothing published after
t's close reaches a card. The snapshot date and the anonymized mode come from
config (SNAPSHOT_DATE, ANONYMIZE); each record keeps both, so an archive entry
renders the same card later.
"""

import json
import random
import re
from datetime import date

import pandas as pd

from ingest import reader

from . import config, summaries


def snapshot_date() -> date:
    if config.SNAPSHOT_DATE:
        return date.fromisoformat(config.SNAPSHOT_DATE)
    return reader.latest_date_asof("universe", date.today())


def anon_ids(tickers) -> dict[str, str]:
    """{ticker: "C-###"}, a fixed-seed shuffle so the number says nothing about the ticker.

    shortcut: IDs change if the universe changes; fine within one experiment's snapshots.
    """
    order = sorted(set(tickers))
    nums = random.Random(config.ANON_SEED).sample(range(100, 1000), len(order))
    return {tk: f"C-{n}" for tk, n in zip(order, nums)}


def _records(df: pd.DataFrame) -> list[dict]:
    rows = df.astype(object).where(df.notna(), None).to_dict("records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, date):  # archives are JSON
                row[k] = v.isoformat()[:10]
    return rows


def load_rows(t: date) -> list[dict]:
    """Universe members at t with a price row, joined with the latest fundamentals
    on or before t (if any). No summaries yet."""
    uni = reader.load_asof("universe", t)[["ticker", "name", "sector"]]
    px = reader.load_asof("prices", t)
    px = px[px["error"].isna()].drop(columns=["error", "is_benchmark"])
    df = uni.merge(px, on="ticker", how="inner")
    if len(df) < len(uni):
        print(f"[items] {len(uni) - len(df)} universe members without prices at {t} left out")
    try:
        fu = reader.load_asof("fundamentals", t)
        fu = fu[fu["error"].isna()].drop(columns=["error", "sector", "fetched_at"], errors="ignore")
        df = df.merge(fu.rename(columns={"name": "short_name", "snapshot_date": "fundamentals_date"}),
                      on="ticker", how="left")
    except reader.NoSnapshotError:
        print(f"[items] no fundamentals on or before {t}; cards show prices only")
    df["anon_id"] = df["ticker"].map(anon_ids(df["ticker"]))
    return _records(df)


def select_rows(rows: list[dict], per_sector: int) -> list[dict]:
    """The `per_sector` largest stocks of each sector (all rows when 0), one
    share class per company.

    Size is market cap, or 20-day dollar volume when fundamentals are missing,
    so the pick is a rule and not a hand choice.
    """
    if not per_sector:
        return rows

    def size(r):
        return r.get("market_cap") or r.get("avg_dollar_volume_20d") or 0

    by_company: dict[str, dict] = {}
    for r in sorted(rows, key=size, reverse=True):
        by_company.setdefault(re.sub(r"\s*\(.*?\)", "", r["name"]).strip(), r)  # GOOGL / GOOG
    picked, count = [], {}
    for r in sorted(by_company.values(), key=size, reverse=True):
        if count.get(r["sector"], 0) < per_sector:
            count[r["sector"]] = count.get(r["sector"], 0) + 1
            picked.append(r)
    return sorted(picked, key=lambda r: r["ticker"])


def attach_summaries(rows: list[dict], t: date, anonymize: bool) -> list[dict]:
    cache = summaries.load(t, anonymize)
    missing = [r["ticker"] for r in rows if r["ticker"] not in cache]
    if missing:
        raise RuntimeError(
            f"{len(missing)} stocks have no summary for {t} ({missing[:5]}...). "
            "Run `uv run python -m apps.stocks.create_embeddings` first."
        )
    return [{**r, "item_id": r["ticker"], "anonymized": anonymize, "with_news": config.WITH_NEWS,
             "news": cache[r["ticker"]]} for r in rows]


def load_items(profile: str) -> list[dict]:
    """The universe stocks at the configured snapshot, narrowed by PER_SECTOR
    (the same pool for every profile)."""
    t = snapshot_date()
    return attach_summaries(select_rows(load_rows(t), config.PER_SECTOR), t, config.ANONYMIZE)


def item_label(row: dict) -> str:
    return row["anon_id"] if row.get("anonymized") else f"{row['name']} ({row['ticker']})"


# --- card text ---------------------------------------------------------------

def _pct(x, signed=True) -> str:
    return "N/A" if x is None else f"{x * 100:{'+' if signed else ''}.1f}%"


def _num(x, fmt=".1f") -> str:
    return "N/A" if x is None else format(x, fmt)


def _money(x) -> str:
    if x is None:
        return "N/A"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:,.0f}"


def cap_bucket(cap) -> str:
    if cap is None:
        return "unknown"
    if cap >= 200e9:
        return "mega (>= $200B)"
    if cap >= 10e9:
        return "large ($10B-200B)"
    if cap >= 2e9:
        return "mid ($2B-10B)"
    return "small (< $2B)"


def _first_sentences(text: str | None, n: int = 3) -> str:
    if not text:
        return ""
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:n])


_NAME_SUFFIX = re.compile(
    r"[\s,]+(inc|corp|corporation|companies|company|co|plc|ltd|limited|holdings|group|incorporated|n\.?v|s\.?a)\b\.?.*$",
    re.IGNORECASE)


def mask_names(text: str, row: dict) -> str:
    """Rule-based safety net for anonymized cards: ticker and company names -> anon ID.

    shortcut: brand and product names are only removed by the summary LLM; a
    name list per company would be needed to catch every one.
    """
    names = set()
    for name in (row.get("name"), row.get("short_name")):
        if name:
            name = re.sub(r"\s*\(.*?\)", "", name).strip()
            names |= {name, _NAME_SUFFIX.sub("", name).strip()}
    # Tickers match case-sensitively so that LOW or MA do not eat ordinary words.
    tickers = {row["ticker"], row["ticker"].replace(".", "-"), row["ticker"].replace("-", ".")}
    patterns = [(n, re.IGNORECASE) for n in names] + [(tk, 0) for tk in tickers]
    for name, flags in sorted(patterns, key=lambda p: len(p[0]), reverse=True):
        if len(name) >= 2:
            text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", row["anon_id"], text, flags=flags)
    return text


def _tokens(text: str) -> int:
    return len(text) // 4


def item_text(row: dict) -> str:
    """The stock card: header, numbers at t, business summary and news summary.

    Over CARD_TOKEN_LIMIT, news bullets are cut first (open points, then facts).
    """
    anon = row.get("anonymized", False)
    news = row.get("news") or {}
    cap = row.get("market_cap")
    header = (f"[Stock {row['anon_id']}] Sector: {row.get('sector') or 'N/A'} / "
              f"Market cap: {cap_bucket(cap)} / Snapshot: {row['snapshot_date']}")
    if not anon:
        header = (f"[Stock {row['name']} ({row['ticker']})] Sector: {row.get('sector') or 'N/A'}"
                  f" / Industry: {row.get('industry') or 'N/A'} / Market cap: {_money(cap)} ({cap_bucket(cap)})"
                  f" / Snapshot: {row['snapshot_date']}")

    numbers = [
        f"- Returns: 1M {_pct(row.get('ret_1m'))}, 3M {_pct(row.get('ret_3m'))}, "
        f"12M {_pct(row.get('ret_12m'))}, 12-1M momentum {_pct(row.get('mom_12_1'))}",
        f"- Volatility (60d, annualized): {_pct(row.get('vol_60d'), signed=False)}; "
        f"avg daily dollar volume (20d): {_money(row.get('avg_dollar_volume_20d'))}",
    ]
    if row.get("fundamentals_date"):
        # yfinance 1.x returns dividendYield in percent (0.41 = 0.41%); see ingest/fundamentals.py.
        dy = row.get("dividend_yield")
        numbers += [
            f"- Valuation (as of {row['fundamentals_date']}): P/E trailing {_num(row.get('pe_trailing'))}, "
            f"forward {_num(row.get('pe_forward'))}; P/B {_num(row.get('pb'))}; "
            f"dividend yield {'N/A' if dy is None else f'{dy:.2f}%'}; beta {_num(row.get('beta'), '.2f')}",
            f"- Growth (YoY): revenue {_pct(row.get('revenue_growth_yoy'))}, "
            f"earnings {_pct(row.get('earnings_growth_yoy'))}",
            f"- Profitability: operating margin {_pct(row.get('operating_margin'), signed=False)}, "
            f"net margin {_pct(row.get('profit_margin'), signed=False)}, ROE {_pct(row.get('roe'), signed=False)}; "
            f"debt/equity {_num(row.get('debt_to_equity'))}%",
        ]
    else:
        numbers.append("- Valuation and fundamentals: not available at this snapshot")

    business = news.get("business") if anon else _first_sentences(row.get("business_summary"))
    facts, contested = list(news.get("facts", [])), list(news.get("contested", []))

    def render() -> str:
        news_lines = [f"Articles: {news.get('n_articles', 0)}"]
        if facts:
            news_lines += ["Facts:"] + [f"- {b}" for b in facts]
        if contested:
            news_lines += ["Where readings differ:"] + [f"- {b}" for b in contested]
        news_block = (["", "## Recent news (last 30 days, published before the close)", *news_lines]
                      if row.get("with_news", True) else [])
        return "\n".join([
            header, "",
            f"## Numbers (as of the close on {row['snapshot_date']})", *numbers, "",
            "## Business", business or "(not available)", *news_block,
        ])

    text = render()
    while _tokens(text) > config.CARD_TOKEN_LIMIT and (contested or facts):
        (contested or facts).pop()
        text = render()
    return mask_names(text, row) if anon else text


def archive_text(row: dict, hypothesis) -> str:
    """Text embedded as entry["embedding"]: the card plus the investment thesis."""
    if not isinstance(hypothesis, str):
        hypothesis = json.dumps(hypothesis, ensure_ascii=False)
    return f"{item_text(row)}\n\nInvestment thesis:\n{hypothesis}"


def embedding_text(row: dict) -> str:
    """Text of the per-item embedding (create_embeddings.py): the whole card."""
    return item_text(row)


def embeddings_file(t: date, anonymize: bool, with_news: bool = True):
    return config.cards_dir(t, anonymize) / ("embeddings.json" if with_news else "embeddings_nonews.json")


def load_item_embeddings(profile: str) -> dict[str, list[float]]:
    """{item_id: embedding} for the configured snapshot and mode (the same for every profile)."""
    path = embeddings_file(snapshot_date(), config.ANONYMIZE, config.WITH_NEWS)
    if not path.exists():
        return {}  # shared.embeddings.require_item_embeddings explains what to run
    return json.loads(path.read_text(encoding="utf-8"))
