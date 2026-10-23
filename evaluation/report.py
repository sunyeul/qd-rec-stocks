"""Score backtest runs: portfolio excess returns, IC and diversity per snapshot and method.

    uv run python -m evaluation.report --manifest archives/backtest/runs.json

The manifest (written by scripts/backtest.py) lists {"t", "method", "archive"}.
Baselines per t: the equal-weight universe and a 12-1 momentum portfolio with
the same N and sector cap. Portfolio returns are net of a round-trip cost.
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

from .returns import HORIZONS, fetch_closes, forward_returns, window

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


def archive_metrics(entries: list[dict], closes: pd.DataFrame, t: date) -> dict:
    active = [e for e in entries if e["status"] == "active"]
    emb = np.array([e["embedding"] for e in active], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    evaluated = [e for e in entries if e.get("method") != "seed"]
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
        f = by_id[p["entry_id"]].get("thesis_fields", {})
        row = {"ticker": p["ticker"], "sector": p["sector"], "score": p["score"],
               "thesis": _first_sentence(f.get("thesis")), "catalyst": _first_sentence(f.get("catalyst")),
               "falsifier": _first_sentence(f.get("falsifier"))}
        for name, r in fwd.items():
            ok = r is not None and p["ticker"] in r
            row[f"ret_{name}"] = round(float(r[p["ticker"]]), 4) if ok else float("nan")
            row[f"vs_spy_{name}"] = round(float(r[p["ticker"]] - r[BENCHMARK_TICKER]), 4) if ok else float("nan")
        rows.append(row)
    return rows


def momentum_portfolio(t: date, n: int = N_DEFAULT, k: int = K_DEFAULT) -> list[str]:
    px = reader.load_asof("prices", t)
    uni = reader.load_asof("universe", t)[["ticker", "sector"]]
    df = uni.merge(px[["ticker", "mom_12_1"]], on="ticker").dropna().sort_values("mom_12_1", ascending=False)
    picked, per = [], {}
    for r in df.itertuples():
        if per.get(r.sector, 0) < k:
            per[r.sector] = per.get(r.sector, 0) + 1
            picked.append(r.ticker)
        if len(picked) == n:
            break
    return picked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="reports/backtest")
    args = ap.parse_args(argv)
    runs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dates = sorted({date.fromisoformat(r["t"]) for r in runs})

    universes = {t: reader.load_asof("universe", t)["ticker"].tolist() for t in dates}
    all_tickers = sorted({tk for u in universes.values() for tk in u} | {BENCHMARK_TICKER})
    closes = fetch_closes(all_tickers, dates[0], date.today())

    rows = []
    for t in dates:
        for name, tickers in (("equal_weight", universes[t]), ("momentum_12_1", momentum_portfolio(t))):
            tickers = [tk for tk in tickers if tk in closes.columns]
            rows.append({"t": t, "method": name, **portfolio_metrics(tickers, closes, t)})
    picks_md, picks = [], []
    for run in runs:
        t = date.fromisoformat(run["t"])
        entries = load_archive(run["archive"])["entries"]
        built = [p for p in build(entries) if p["ticker"] in closes.columns]
        port = [p["ticker"] for p in built]
        rows.append({"t": t, "method": run["method"], **portfolio_metrics(port, closes, t),
                     **archive_metrics(entries, closes, t), "portfolio": " ".join(port)})
        stock_rows = pick_rows(entries, built, closes, t)
        picks += [{"t": t, "method": run["method"], **r} for r in stock_rows]
        picks_md += ["", f"### {t} · {run['method']}", "", _markdown(pd.DataFrame(stock_rows))]

    df = pd.DataFrame(rows)
    num = df.drop(columns=["t", "portfolio"], errors="ignore")
    summary = num.groupby("method").mean(numeric_only=True)
    summary[f"hit_rate_{PRIMARY}"] = df.groupby("method")[f"excess_{PRIMARY}"].apply(lambda s: float((s > 0).mean()))
    summary["periods"] = df.groupby("method").size()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    pd.DataFrame(picks).to_csv(out.with_name(out.name + "_picks.csv"), index=False)
    md = ["# Backtest", "", f"Snapshots: {', '.join(map(str, dates))}. Excess = portfolio minus {BENCHMARK_TICKER}, "
          f"net of {ROUND_TRIP_COST:.1%} round trip. N={N_DEFAULT}, at most {K_DEFAULT} per sector.", "",
          "## Mean over snapshots", "", _markdown(summary.round(4).reset_index()), "",
          "## Per snapshot", "", _markdown(df.round(4)), "",
          "## Picks: why each stock was chosen and what it did next", *picks_md]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:7]))
    print(f"\n-> {out.with_suffix('.md')}, {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
