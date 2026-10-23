"""News summaries for stock cards: one LLM call per stock and snapshot, cached.

The summary splits news into facts and points where readings differ. Bullets
that state a number the source text does not contain are dropped, so the
summary cannot introduce numbers. In anonymized mode the same call also
rewrites the business description without company, brand or product names.
"""

import asyncio
import json
import re
from datetime import date

import pandas as pd

from ingest import reader
from shared.llm import acreate, client, loads_lenient

from . import config

SYSTEM_PROMPT = """You summarize recent news about one listed company for an equity research card.

Rules:
- Use only the articles given. Do not add outside knowledge.
- Do not write any number that does not appear in the articles. Prefer words over numbers.
- "facts": up to 4 short bullets of reported events (results, guidance, deals, products, legal, management).
- "contested": up to 3 short bullets on points where readings differ or the impact is uncertain.
- Return JSON: {"facts": [...], "contested": [...]}"""

ANON_RULES = """
This card is anonymized. Never name the company, its tickers, brands, products, people or
competitors: write "the company", "a flagship product", "a large competitor" and so on."""

BUSINESS_RULE = """
Also rewrite the business description given below in 2-3 sentences under the same rule,
with no numbers, as "business". Return JSON: {"business": "...", "facts": [...], "contested": [...]}"""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUMBER.findall(text)}


def keep_grounded(bullets, source: str) -> list[str]:
    """The bullets whose numbers all appear in `source`."""
    allowed = numbers(source)
    return [b.strip() for b in bullets if isinstance(b, str) and b.strip() and numbers(b) <= allowed]


def cache_file(t: date, anonymize: bool):
    return config.cards_dir(t, anonymize) / "summaries.json"


def load(t: date, anonymize: bool) -> dict:
    """{ticker: summary} already built for t, or {}."""
    path = cache_file(t, anonymize)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def article_text(articles: pd.DataFrame) -> str:
    return "\n".join(
        f"- [{str(a.published_at)[:10]}, {a.publisher}] {a.title}. {a.summary or ''}".strip()
        for a in articles.itertuples()
    )


async def summarize_one(row: dict, articles: pd.DataFrame, anonymize: bool) -> dict:
    business = row.get("business_summary") or ""
    out = {"facts": [], "contested": [], "business": None, "n_articles": len(articles)}
    if articles.empty and not (anonymize and business):
        return out
    source = article_text(articles)
    user = f"Company: {row['name']} ({row['ticker']})\n\nArticles (newest first):\n{source or '(none)'}"
    system = SYSTEM_PROMPT
    if anonymize:
        system += ANON_RULES
    if anonymize and business:
        system += BUSINESS_RULE
        user += f"\n\nBusiness description:\n{business}"
    response = await acreate(client(),
        model=config.SUMMARY_MODEL,
        reasoning_effort=config.SUMMARY_REASONING_EFFORT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    result = loads_lenient(response.choices[0].message.content)
    out["facts"] = keep_grounded(result.get("facts", []), source)[:4]
    out["contested"] = keep_grounded(result.get("contested", []), source)[:3]
    if anonymize and business:
        out["business"] = next(iter(keep_grounded([result.get("business")], business)), None)
    out["model"] = config.SUMMARY_MODEL
    return out


async def build_async(t: date, rows: list[dict], anonymize: bool) -> dict:
    """Summaries for every row not yet cached, or cached from a different number
    of articles (news added later); failures are printed and left out, so
    running again retries only them."""
    cache = load(t, anonymize)
    try:
        news = reader.load_news_asof(t)
    except reader.NoSnapshotError:
        news = pd.DataFrame(columns=["ticker"])
    articles_of = {r["ticker"]: news[news["ticker"] == r["ticker"]].head(config.MAX_ARTICLES) for r in rows}
    todo = [r for r in rows
            if r["ticker"] not in cache or cache[r["ticker"]].get("n_articles") != len(articles_of[r["ticker"]])]
    sem = asyncio.Semaphore(config.SUMMARY_CONCURRENCY)

    async def one(row):
        async with sem:
            articles = articles_of[row["ticker"]]
            try:
                cache[row["ticker"]] = await summarize_one(row, articles, anonymize)
                print(f"[summary] {row['ticker']}: {len(articles)} articles")
            except Exception as e:
                print(f"[summary] ERROR {row['ticker']}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(r) for r in todo))
    path = cache_file(t, anonymize)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache
