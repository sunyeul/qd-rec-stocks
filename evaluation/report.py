"""Backtest report in the order the blog tells it: A picks and reasons, B returns,
C feasibility, D impact (IC), E what made it QD-Rec.

    uv run python -m evaluation.report --manifest archives/backtest500/runs.json

The manifest (written by scripts/backtest.py) lists {"t", "method", "archive"}; a
sibling *_costs.json gives each run's spend and minutes. Baselines per t: the
RSP and QQQ ETFs and the equal-weight universe (SPY is the zero line). Portfolio
returns are net of a round-trip cost. Return drivers are labeled by Jev (cached).
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import reader
from ingest.config import BENCHMARK_TICKER
from portfolio.build import K_DEFAULT, N_DEFAULT, build, stock_scores
from shared.archive_format import load_archive

from . import drivers
from .returns import HORIZONS, fetch_closes, forward_returns, window

ETFS = ["RSP", "QQQ"]  # equal-weight S&P 500 and Nasdaq 100, as excess over SPY

ROUND_TRIP_COST = 0.002  # 10 bp each way, the whole book turned over each period
# The horizon every snapshot can be scored on (monthly snapshots up to a month ago);
# 3m columns are filled where three months have passed.
PRIMARY = "1m"


def _mean_offdiag(m: np.ndarray) -> float:
    n = len(m)
    return float((m.sum() - np.trace(m)) / (n * (n - 1))) if n > 1 else float("nan")


def _markdown(df: pd.DataFrame) -> str:
    """A pipe table (pandas' to_markdown needs the tabulate package)."""
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _forward(closes: pd.DataFrame, t: date, horizon: str) -> pd.Series | None:
    """Forward returns, or None while the horizon has not passed yet."""
    try:
        return forward_returns(closes, t, HORIZONS[horizon])
    except ValueError:
        return None


def portfolio_metrics(tickers: list[str], closes: pd.DataFrame, t: date) -> dict:
    out = {"n": len(tickers)}
    for name in HORIZONS:
        fwd = _forward(closes, t, name)
        out[f"excess_{name}"] = (float("nan") if fwd is None else
                                 float(fwd[tickers].mean() - ROUND_TRIP_COST - fwd[BENCHMARK_TICKER]))
    daily = window(closes, t, HORIZONS[PRIMARY])[tickers].pct_change().dropna()
    out["holding_corr"] = _mean_offdiag(daily.corr().to_numpy())
    return out


def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def parent_influence(entries: list[dict]) -> float:
    """Mean of (similarity to the parent a thesis named) - (mean similarity to the other examples
    it was shown). Near 0: the named parent shaped it no more than the other examples did."""
    by_id = {e["entry_id"]: e for e in entries}
    gaps = []
    for e in entries:
        named = (e.get("evolution") or {}).get("evo_parent_entry_ids") or []
        shown = [by_id[i] for i in e.get("parent_entry_ids", []) if i in by_id]
        others = [s for s in shown if s["entry_id"] not in named]
        if named and others and all(i in by_id for i in named):
            gaps.append(np.mean([_cos(e["embedding"], by_id[i]["embedding"]) for i in named])
                        - np.mean([_cos(e["embedding"], o["embedding"]) for o in others]))
    return float(np.mean(gaps)) if gaps else float("nan")


def active_quality_curve(entries: list[dict], batch: int = 10, top: int | None = None) -> list[float]:
    """Mean judge score of the active archive (or of its `top` best ideas) after each
    batch, rebuilt from timestamps. The whole-archive mean can fall while QD-Rec works,
    because novel ideas are kept even when their score is low; the top-k mean cannot."""
    evaluated = sorted((e for e in entries if e.get("method") != "seed"), key=lambda e: e["created_at"])
    ends = [evaluated[min(i + batch, len(evaluated)) - 1]["created_at"] for i in range(0, len(evaluated), batch)]
    curve = []
    for end in ends:
        alive = [e for e in entries
                 if e["created_at"] <= end and e.get("admit_path") not in ("duel_loss", "verifier_fail")
                 and not e.get("discarded_at") and (not e.get("replaced_at") or e["replaced_at"] > end)]
        scores = sorted((e["judgment"]["score"] for e in alive), reverse=True)
        curve.append(float(np.mean(scores[:top] if top else scores)))
    return curve


def proposal_trend(entries: list[dict]) -> float:
    """Mean score of the second half of new theses minus the first half (learning signal)."""
    s = [e["judgment"]["score"] for e in sorted(entries, key=lambda e: e["created_at"]) if e.get("method") != "seed"]
    return float(np.mean(s[len(s) // 2:]) - np.mean(s[:len(s) // 2])) if len(s) >= 2 else float("nan")


def archive_metrics(entries: list[dict], closes: pd.DataFrame, t: date, driver_of: dict | None = None) -> dict:
    active = [e for e in entries if e["status"] == "active"]
    emb = np.array([e["embedding"] for e in active], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    evaluated = [e for e in entries if e.get("method") != "seed"]
    top = active_quality_curve(entries, top=N_DEFAULT)
    extra = {"ideas_per_stock": len(active) / max(1, len({e["item_id"] for e in active})),
             "parent_influence": parent_influence(entries),
             "top20_quality_gain": top[-1] - top[0] if top else float("nan"),
             "proposal_trend": proposal_trend(entries)}
    if driver_of:
        extra["driver_entropy"] = drivers.entropy(driver_of[e["entry_id"]] for e in active)
    scores = pd.Series({tk: e["judgment"]["score"] for tk, e in stock_scores(entries).items()})
    ic = {}
    for name in HORIZONS:
        fwd = _forward(closes, t, name)
        ok = fwd is not None and len(scores) >= 5
        ic[f"ic_{name}"] = (float(scores.rank().corr((fwd[scores.index] - fwd[BENCHMARK_TICKER]).rank()))
                            if ok else float("nan"))
    return {
        "active": len(active),
        "mean_score": float(np.mean([e["judgment"]["score"] for e in active])),
        "diversity": 1 - _mean_offdiag(emb @ emb.T),  # mean pairwise cosine distance
        "stocks_covered": len(scores),
        "verifier_fail_rate": float(np.mean([e.get("admit_path") == "verifier_fail" for e in evaluated]))
        if evaluated else float("nan"),
        **ic,
        **extra,
    }


def _first_sentence(text: str) -> str:
    # A sentence ends after a word of 3+ lowercase letters, so "U.S." or "& Co." do not cut it.
    return re.split(r"(?<=[a-z]{3}[.!?])\s", (text or "").strip(), maxsplit=1)[0].replace("|", "/")


def pick_rows(entries: list[dict], port: list[dict], closes: pd.DataFrame, t: date) -> list[dict]:
    """One row per portfolio stock: why it was picked and what it did afterwards."""
    by_id = {e["entry_id"]: e for e in entries}
    fwd = {name: _forward(closes, t, name) for name in HORIZONS}
    rows = []
    for p in port:
        e = by_id[p["entry_id"]]
        f = e.get("thesis_fields", {})
        row = {"ticker": p["ticker"], "sector": p["sector"], "score": p["score"],
               "thesis": _first_sentence(f.get("thesis")), "catalyst": _first_sentence(f.get("catalyst")),
               "falsifier": _first_sentence(f.get("falsifier")),
               "critique": _first_sentence((e.get("critique") or "").lstrip("- "))}
        for name, r in fwd.items():
            ok = r is not None and p["ticker"] in r
            row[f"ret_{name}"] = round(float(r[p["ticker"]]), 4) if ok else float("nan")
            row[f"vs_spy_{name}"] = round(float(r[p["ticker"]] - r[BENCHMARK_TICKER]), 4) if ok else float("nan")
        rows.append(row)
    return rows


def qd_traits(entries: list[dict], driver_of: dict | None = None) -> list[str]:
    """Markdown showing what makes the run QD-Rec: evolution, novelty gating, the
    archive's quality over time, item selection, and the spread of ideas."""
    by_id = {e["entry_id"]: e for e in entries}
    evaluated = sorted((e for e in entries if e.get("method") != "seed"), key=lambda e: e["created_at"])
    active = [e for e in entries if e["status"] == "active"]
    count = lambda xs: ", ".join(f"{k} {v}" for k, v in pd.Series(xs).value_counts().items())
    tag = lambda e: f", {driver_of[e['entry_id']]}" if driver_of and e["entry_id"] in driver_of else ""
    say = lambda e: (f"{e['item_id']} ({e['judgment']['score']}{tag(e)}): "
                     f"{_first_sentence(e.get('thesis_fields', {}).get('thesis'))}")
    batches = [evaluated[i:i + 10] for i in range(0, len(evaluated), 10)]
    curve = " → ".join(f"{np.mean([e['judgment']['score'] for e in b]):.1f}" for b in batches)
    visits = pd.Series([e["item_id"] for e in evaluated]).value_counts()
    out = [
        f"- Evaluations: {len(evaluated)}; archive at the end: {len(active)} active ideas on "
        f"{len({e['item_id'] for e in active})} stocks",
        f"- How each new thesis was made: {count([(e.get('evolution') or {}).get('operation', 'zero-shot') for e in evaluated])}",
        f"- What happened to it: {count([e.get('admit_path') for e in evaluated])}",
        f"- Mean judge score of new theses per batch of 10: {curve}",
        "- Mean judge score of the active archive after each batch: "
        + " → ".join(f"{q:.1f}" for q in active_quality_curve(entries)),
        f"- Mean of the archive's top {N_DEFAULT} after each batch: "
        + " → ".join(f"{q:.2f}" for q in active_quality_curve(entries, top=N_DEFAULT)),
        f"- Parent influence (similarity to the named parent minus to the other examples): "
        f"{parent_influence(entries):+.3f}",
        f"- Stocks evaluated: {len(visits)} distinct; most visited: "
        + ", ".join(f"{tk} ×{n}" for tk, n in visits.head(5).items()),
        f"- Ideas per stock in the archive: {len(active) / max(1, len({e['item_id'] for e in active})):.2f} "
        f"(max {max(pd.Series([e['item_id'] for e in active]).value_counts())})",
    ]
    if driver_of:
        out.append(f"- Return drivers of the active ideas: {count([driver_of[e['entry_id']] for e in active])} "
                   f"(entropy {drivers.entropy(driver_of[e['entry_id']] for e in active):.2f})")
    child = max((e for e in active if (e.get("evolution") or {}).get("operation") in ("mutate", "crossover")
                 and e["evolution"].get("evo_parent_entry_ids")), key=lambda e: e["judgment"]["score"], default=None)
    if child:
        parents = [by_id[p] for p in child["evolution"]["evo_parent_entry_ids"] if p in by_id]
        out += ["", f"**Evolution example ({child['evolution']['operation']}):**"]
        out += [f"- parent {say(p)}" for p in parents]
        out += [f"- child {say(child)}", f"- note: {child['evolution'].get('evolution_note', '')}"]
    swap = next((e for e in evaluated if e.get("replaces_entry_id") in by_id), None)
    if swap:
        out += ["", "**Replacement example (a similar idea; the blind duel preferred the new one, "
                    "which need not have the higher judge score):**",
                f"- old {say(by_id[swap['replaces_entry_id']])}", f"- new {say(swap)}"]
    out += ["", "**All active ideas, by sector:**"]
    by_sector: dict[str, list[dict]] = {}
    for e in sorted(active, key=lambda e: -e["judgment"]["score"]):
        by_sector.setdefault(e.get("sector") or "unknown", []).append(e)
    out += [f"- {sector}: " + "; ".join(say(e) for e in group) for sector, group in sorted(by_sector.items())]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="reports/backtest500")
    ap.add_argument("--no-drivers", action="store_true", help="skip labeling return drivers with Jev")
    args = ap.parse_args(argv)
    runs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dates = sorted({date.fromisoformat(r["t"]) for r in runs})

    universes = {t: reader.load_asof("universe", t)["ticker"].tolist() for t in dates}
    all_tickers = sorted({tk for u in universes.values() for tk in u} | {BENCHMARK_TICKER, *ETFS})
    closes = fetch_closes(all_tickers, dates[0], date.today())

    costs_file = Path(args.manifest.replace(".json", "_costs.json"))
    costs = json.loads(costs_file.read_text(encoding="utf-8")) if costs_file.exists() else {}

    rows = []
    for t in dates:
        tickers = [tk for tk in universes[t] if tk in closes.columns]
        rows.append({"t": t, "method": "equal_weight", **portfolio_metrics(tickers, closes, t)})
        for etf in ETFS:  # one-stock "portfolios" with no trading cost
            out = {"t": t, "method": etf, "n": 1}
            for name in HORIZONS:
                fwd = _forward(closes, t, name)
                out[f"excess_{name}"] = float("nan") if fwd is None else float(fwd[etf] - fwd[BENCHMARK_TICKER])
            rows.append(out)
    picks_md, picks, traits_md, ranked = [], [], [], []
    for run in runs:
        t = date.fromisoformat(run["t"])
        entries = load_archive(run["archive"])["entries"]
        driver_of = None if args.no_drivers else drivers.labels([e for e in entries if e["status"] == "active"])
        built = [p for p in build(entries) if p["ticker"] in closes.columns]
        port = [p["ticker"] for p in built]
        rows.append({"t": t, "method": run["method"], **portfolio_metrics(port, closes, t),
                     **archive_metrics(entries, closes, t, driver_of), **costs.get(run["archive"], {}),
                     "portfolio": " ".join(port)})
        scores = pd.Series({tk: e["judgment"]["score"] for tk, e in stock_scores(entries).items()})
        for name in HORIZONS:
            fwd = _forward(closes, t, name)
            if fwd is not None:
                ex = fwd.reindex(scores.index) - fwd[BENCHMARK_TICKER]
                ranked.append(pd.DataFrame({"method": run["method"], "horizon": name,
                                            "s": scores.rank(pct=True), "r": ex.rank(pct=True)}).dropna())
        stock_rows = pick_rows(entries, built, closes, t)
        picks += [{"t": t, "method": run["method"], **r} for r in stock_rows]
        picks_md += ["", f"### {t} · {run['method']}", "", _markdown(pd.DataFrame(stock_rows))]
        traits_md += ["", f"### {t} · {run['method']}", "", *qd_traits(entries, driver_of)]

    df = pd.DataFrame(rows)
    num = df.drop(columns=["t", "portfolio"], errors="ignore")
    summary = num.groupby("method").mean(numeric_only=True)
    summary[f"hit_rate_{PRIMARY}"] = df.groupby("method")[f"excess_{PRIMARY}"].apply(lambda s: float((s > 0).mean()))
    summary["periods"] = df.groupby("method").size()
    # Pooled IC: within-date percentile ranks of score and of excess return, stacked over
    # dates, so the standard error shrinks with the total number of stocks scored.
    pooled = []
    for (method, horizon), g in (pd.concat(ranked).groupby(["method", "horizon"]) if ranked else []):
        pooled.append({"method": method, "horizon": horizon, "n": len(g), "pooled_ic": round(g.s.corr(g.r), 4),
                       "se": round(1 / np.sqrt(max(len(g) - 1, 1)), 4)})
    pooled = pd.DataFrame(pooled)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    pd.DataFrame(picks).to_csv(out.with_name(out.name + "_picks.csv"), index=False)

    def table(cols, runs_only=False):
        have = [c for c in cols if c in summary.columns]
        s = summary.drop(index=["equal_weight", *ETFS], errors="ignore") if runs_only else summary
        return _markdown(s[have].round(4).reset_index())

    md = ["# QD-Rec stock ideas: backtest report", "",
          f"Snapshots: {', '.join(map(str, dates))}. Excess = portfolio minus {BENCHMARK_TICKER}, net of "
          f"{ROUND_TRIP_COST:.1%} round trip (ETFs without cost). Portfolio: the best idea per stock, top "
          f"{N_DEFAULT}, at most {K_DEFAULT} per sector, equal weight. Means are over snapshots.", "",
          "## A. What was recommended, and why", *picks_md, "",
          "## B. What happened next", "",
          table(["n", "excess_1m", "excess_3m", f"hit_rate_{PRIMARY}", "holding_corr", "periods"]), "",
          "## C. Feasibility", "",
          table(["cost", "minutes", "verifier_fail_rate", "stocks_covered", "active"], runs_only=True), "",
          "## D. Impact: did higher scores pick better stocks?", "",
          "Pooled IC = rank correlation of judge score and excess return, within-date ranks stacked over dates.", "",
          _markdown(pooled) if len(pooled) else "(none)", "",
          "## E. What made it QD-Rec (against the Novelty contrast)", "",
          table(["stocks_covered", "active", "ideas_per_stock", "driver_entropy", "diversity", "mean_score",
                 "top20_quality_gain", "proposal_trend", "parent_influence"], runs_only=True), "", *traits_md, "",
          "## Appendix: every metric per snapshot", "", _markdown(df.drop(columns=["portfolio"], errors="ignore").round(4))]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:7]))
    print(f"\n-> {out.with_suffix('.md')}, {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
