"""Partitioned storage: data/snapshots/<source>/snapshot_date=YYYY-MM-DD/data.parquet."""

import os
from datetime import date
from pathlib import Path

import pandas as pd

from . import config

SOURCES = ("universe", "prices", "fundamentals", "news")
_PREFIX = "snapshot_date="
_FILE = "data.parquet"


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def partition_path(source: str, t: date) -> Path:
    _check_source(source)
    return config.SNAPSHOT_DIR / source / f"{_PREFIX}{t.isoformat()}" / _FILE


def exists(source: str, t: date) -> bool:
    return partition_path(source, t).exists()


def write(df: pd.DataFrame, source: str, t: date, overwrite: bool = False) -> Path:
    """Write one partition atomically. Refuses to replace an existing one
    unless overwrite=True, so a rerun cannot silently rewrite history."""
    path = partition_path(source, t)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["snapshot_date"] = t.isoformat()
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read(source: str, t: date) -> pd.DataFrame:
    return pd.read_parquet(partition_path(source, t))


def list_dates(source: str) -> list[date]:
    _check_source(source)
    root = config.SNAPSHOT_DIR / source
    if not root.exists():
        return []
    dates = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(_PREFIX) and (child / _FILE).exists():
            dates.append(date.fromisoformat(child.name[len(_PREFIX):]))
    return sorted(dates)
