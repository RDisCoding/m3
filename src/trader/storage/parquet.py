"""Parquet storage for daily bars: one file per instrument, atomic writes, idempotent upserts.

File schema (BAR_SCHEMA):
    date date32 | open, high, low, close float64 | volume int64 | source string |
    ingested_at timestamp[us, UTC]
File-level metadata key b"trader" holds JSON: name, instrument_key, isin, kind, provider, adjustment.

Upserting identical data does not touch the file, so reruns leave it byte-identical.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trader.data.providers.base import BAR_COLUMNS, PRICE_COLUMNS, normalize_bars

BAR_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
STORED_COLUMNS = [field.name for field in BAR_SCHEMA]
_VALUE_COLUMNS = PRICE_COLUMNS + ["volume"]
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_INVALID = re.compile(r'[<>:"/\\|?*%\x00-\x1f]')


def safe_stem(name: str) -> str:
    """File stem for an instrument name that is valid on Windows ("Nifty 50" -> "NIFTY_50")."""
    stem = name.strip().upper().replace(" ", "_")
    stem = _INVALID.sub(lambda m: f"%{ord(m.group()):02X}", stem).rstrip(". ")
    if not stem:
        raise ValueError(f"cannot build a file name from {name!r}")
    if stem.split(".")[0] in _RESERVED:
        stem += "_"
    return stem


def empty_stored() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS})
    frame.insert(0, "date", pd.Series(dtype="datetime64[ns]"))
    frame["volume"] = pd.Series(dtype="int64")
    frame["source"] = pd.Series(dtype="object")
    frame["ingested_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame[STORED_COLUMNS]


@dataclass(frozen=True, slots=True)
class UpsertResult:
    name: str
    path: Path
    rows_before: int
    rows_after: int
    added: int
    updated: int
    max_close_rel_change: float  # largest |new/old - 1| among overlapping closes that changed
    written: bool
    max_confirmed_close_change: float = 0.0  # same, ignoring rows that were only provisional before


class BarStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / f"{safe_stem(name)}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def stems(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet")) if self.root.is_dir() else []

    def read(self, name: str) -> pd.DataFrame:
        """Stored frame with provenance columns (source, ingested_at)."""
        df = pq.read_table(self.path(name)).to_pandas(date_as_object=False)
        df["date"] = df["date"].astype("datetime64[ns]")
        df["ingested_at"] = df["ingested_at"].astype("datetime64[ns, UTC]")
        return df[STORED_COLUMNS]

    def read_bars(self, name: str) -> pd.DataFrame:
        """Canonical bar frame (see trader.data.providers.base). Files were validated when written.
        Reads only the bar columns (provenance columns are skipped, which halves the read time)."""
        return self.read_columns(name, list(BAR_COLUMNS))

    def read_columns(self, name: str, columns: list[str]) -> pd.DataFrame:
        """Selected stored columns only (much faster than a full read when scanning every file)."""
        df = pq.read_table(self.path(name), columns=columns).to_pandas(date_as_object=False)
        if "date" in df:
            df["date"] = df["date"].astype("datetime64[ns]")
        return df

    def read_meta(self, name: str) -> dict:
        metadata = pq.read_schema(self.path(name)).metadata or {}
        return json.loads(metadata.get(b"trader", b"{}"))

    def write(self, name: str, frame: pd.DataFrame, *, meta: dict) -> Path:
        """Replace the file atomically. `frame` must contain all STORED_COLUMNS."""
        table = _to_table(frame, meta)
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
        return path

    def upsert(
        self, name: str, bars: pd.DataFrame, *, source: str, meta: dict, now: datetime | None = None,
        provisional_sources: tuple[str, ...] = (),
    ) -> UpsertResult:
        """Merge bars into the stored file; new values win on overlapping dates."""
        return self._upsert(name, bars, source=source, meta=meta, now=now, provisional_sources=provisional_sources, write=True)

    def preview(self, name: str, bars: pd.DataFrame, *, source: str, provisional_sources: tuple[str, ...] = ()) -> UpsertResult:
        """What an upsert would change, without writing anything."""
        return self._upsert(name, bars, source=source, meta=None, now=None, provisional_sources=provisional_sources, write=False)

    def _upsert(self, name, bars, *, source, meta, now, provisional_sources, write) -> UpsertResult:
        stamp = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC").floor("us")
        new = normalize_bars(bars)
        path = self.path(name)
        existed = path.is_file()
        old = self.read(name) if existed else empty_stored()
        merged, added, updated, max_rel, max_confirmed = _merge(old, new, source, stamp, provisional_sources)
        if not write:
            return UpsertResult(name, path, len(old), len(merged), added, updated, max_rel, False, max_confirmed)
        if existed and added == 0 and updated == 0 and self.read_meta(name) == meta:
            return UpsertResult(name, path, len(old), len(old), 0, 0, 0.0, written=False)
        if merged.empty:
            return UpsertResult(name, path, 0, 0, 0, 0, 0.0, written=False)
        self.write(name, merged, meta=meta)
        return UpsertResult(name, path, len(old), len(merged), added, updated, max_rel, True, max_confirmed)


def _merge(
    old: pd.DataFrame, new: pd.DataFrame, source: str, stamp: pd.Timestamp, provisional_sources: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, int, int, float, float]:
    if new.empty:
        return old, 0, 0, 0.0, 0.0
    new_i = new.set_index("date")
    additions = new_i.copy()
    additions["source"] = source
    additions["ingested_at"] = stamp
    if old.empty:
        return additions.reset_index()[STORED_COLUMNS], len(additions), 0, 0.0, 0.0

    old_i = old.set_index("date")
    common = old_i.index.intersection(new_i.index)
    o = old_i.loc[common, _VALUE_COLUMNS]
    n = new_i.loc[common, _VALUE_COLUMNS]
    price_changed = ~np.isclose(o[PRICE_COLUMNS].to_numpy(), n[PRICE_COLUMNS].to_numpy(), rtol=0.0, atol=1e-9).all(axis=1)
    volume_changed = o["volume"].to_numpy() != n["volume"].to_numpy()
    # A provenance change counts too, so a provisional row is relabelled once history confirms it.
    source_changed = old_i.loc[common, "source"].to_numpy() != source
    changed = common[price_changed | volume_changed | source_changed]
    rel = (n.loc[changed, "close"] / o.loc[changed, "close"] - 1.0).abs()
    max_rel = float(rel.max()) if len(rel) else 0.0
    confirmed = ~old_i.loc[changed, "source"].isin(provisional_sources)
    max_confirmed = float(rel[confirmed.to_numpy()].max()) if confirmed.any() else 0.0

    result = old_i.copy()
    if len(changed):
        result.loc[changed, _VALUE_COLUMNS] = n.loc[changed, _VALUE_COLUMNS]
        result.loc[changed, "source"] = source
        result.loc[changed, "ingested_at"] = stamp
    only_new = new_i.index.difference(old_i.index)
    if len(only_new):
        result = pd.concat([result, additions.loc[only_new]])
    result = result.sort_index().reset_index()
    result["volume"] = result["volume"].astype("int64")
    return result[STORED_COLUMNS], len(only_new), len(changed), max_rel, max_confirmed


def _to_table(frame: pd.DataFrame, meta: dict) -> pa.Table:
    missing = set(STORED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns {sorted(missing)}")
    df = frame[STORED_COLUMNS].copy()
    if df[PRICE_COLUMNS].isna().any().any() or df["volume"].isna().any():
        raise ValueError("refusing to store bars with missing prices or volume")
    if pd.DatetimeIndex(df["date"]).has_duplicates:
        raise ValueError("refusing to store duplicate dates")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).dt.floor("us")
    table = pa.Table.from_pandas(df, schema=BAR_SCHEMA, preserve_index=False)
    return table.replace_schema_metadata({b"trader": json.dumps(meta, sort_keys=True).encode()})


__all__ = ["BAR_COLUMNS", "BAR_SCHEMA", "STORED_COLUMNS", "BarStore", "UpsertResult", "safe_stem"]
