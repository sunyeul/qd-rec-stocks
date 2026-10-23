"""Turn an idea archive into an equal-weight portfolio.

Each stock is scored by its best active thesis; the top N stocks are taken,
at most k per theme. Uses only the archive, so it is safe at time t.

shortcut: the theme is the stock's sector; switch to clustering thesis
embeddings if sector caps prove too coarse.
"""

N_DEFAULT = 20
K_DEFAULT = 3


def stock_scores(entries: list[dict]) -> dict[str, dict]:
    """{ticker: best active entry} by judge score."""
    best: dict[str, dict] = {}
    for e in entries:
        if e.get("status") != "active":
            continue
        cur = best.get(e["item_id"])
        if cur is None or e["judgment"]["score"] > cur["judgment"]["score"]:
            best[e["item_id"]] = e
    return best


def build(entries: list[dict], n: int = N_DEFAULT, k: int = K_DEFAULT) -> list[dict]:
    """[{"ticker", "sector", "score", "entry_id"}] for up to n stocks, k per sector.

    Ties keep archive order, so a rerun on the same archive gives the same portfolio.
    """
    ranked = sorted(stock_scores(entries).values(), key=lambda e: -e["judgment"]["score"])
    picked, per_theme = [], {}
    for e in ranked:
        theme = e.get("sector") or "unknown"
        if per_theme.get(theme, 0) >= k:
            continue
        per_theme[theme] = per_theme.get(theme, 0) + 1
        picked.append({"ticker": e["item_id"], "sector": theme, "score": e["judgment"]["score"],
                       "entry_id": e["entry_id"]})
        if len(picked) == n:
            break
    return picked
