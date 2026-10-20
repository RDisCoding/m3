"""Backfill and daily refresh of stored daily bars.

Backfill pulls full history per instrument and is resumable: an instrument whose last successful
backfill started at or before the requested start date is skipped.

Refresh, for each instrument:
  1. re-fetches the last `overlap` stored sessions through `as_of` from the historical endpoint
     and upserts them. History replaces provisional bars and relabels their source.
  2. if any overlapping close moved by more than `repull_threshold`, the vendor has re-adjusted
     history (a new split or bonus), so the full history is fetched again.
Then, when `as_of` is today, today's candles for all instruments come from the batch quote API
(the historical endpoint only covers completed sessions up to yesterday) and are stored with
source "upstox_live" until the next refresh confirms them.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from trader.data.providers.base import AuthError, Instrument, ProviderError
from trader.storage.db import utc_now
from trader.storage.parquet import BarStore

log = logging.getLogger(__name__)

SOURCE_HISTORY = "upstox"
SOURCE_LIVE = "upstox_live"


def file_meta(inst: Instrument) -> dict:
    return {
        "name": inst.symbol,
        "instrument_key": inst.instrument_key,
        "isin": inst.isin,
        "kind": inst.kind,
        "provider": "upstox",
        "adjustment": "splits_and_bonuses",
    }


@dataclass
class SyncResult:
    name: str
    kind: str
    status: str  # ok | skipped | error
    added: int = 0
    updated: int = 0
    repulled: bool = False
    detail: str = ""


class BarSync:
    def __init__(self, provider, daily: BarStore, index: BarStore, conn: sqlite3.Connection):
        self.provider = provider
        self.daily, self.index = daily, index
        self.conn = conn

    def store(self, inst: Instrument) -> BarStore:
        return self.index if inst.kind == "INDEX" else self.daily

    # Metadata -------------------------------------------------------------------------
    def meta(self, inst: Instrument) -> dict | None:
        row = self.conn.execute("SELECT * FROM bar_meta WHERE name = ? AND kind = ?", (inst.symbol, inst.kind)).fetchone()
        return dict(row) if row else None

    def record(self, inst: Instrument, status: str, error: str | None = None, backfill_from: date | None = None) -> None:
        store = self.store(inst)
        first = last = None
        rows, provisional = 0, 0
        if store.exists(inst.symbol):
            stored = store.read(inst.symbol)
            if not stored.empty:
                first, last = stored["date"].iloc[0].date().isoformat(), stored["date"].iloc[-1].date().isoformat()
                rows, provisional = len(stored), int(stored["source"].iloc[-1] == SOURCE_LIVE)
        self.conn.execute(
            "INSERT INTO bar_meta (name, kind, instrument_key, isin, first_date, last_date, rows, provisional, backfill_from, "
            "status, last_error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name, kind) DO UPDATE SET instrument_key=excluded.instrument_key, isin=excluded.isin, "
            "first_date=excluded.first_date, last_date=excluded.last_date, rows=excluded.rows, provisional=excluded.provisional, "
            "backfill_from=COALESCE(excluded.backfill_from, bar_meta.backfill_from), status=excluded.status, "
            "last_error=excluded.last_error, updated_at=excluded.updated_at",
            (inst.symbol, inst.kind, inst.instrument_key, inst.isin, first, last, rows, provisional,
             backfill_from.isoformat() if backfill_from else None, status, error, utc_now()),
        )
        self.conn.commit()

    # Backfill -------------------------------------------------------------------------
    def backfill(self, insts: Sequence[Instrument], start: date, end: date, *, force: bool = False,
                 progress: Callable[[Instrument], None] | None = None) -> list[SyncResult]:
        results = []
        for inst in insts:
            results.append(self._backfill_one(inst, start, end, force))
            if progress:
                progress(inst)
        return results

    def _backfill_one(self, inst: Instrument, start: date, end: date, force: bool) -> SyncResult:
        meta = self.meta(inst)
        store = self.store(inst)
        if (not force and meta and meta.get("backfill_from") and meta["backfill_from"] <= start.isoformat()
                and store.exists(inst.symbol)):
            return SyncResult(inst.symbol, inst.kind, "skipped", detail=f"backfilled from {meta['backfill_from']}")
        try:
            bars = self.provider.fetch_daily(inst, start, end)
        except AuthError:
            raise
        except ProviderError as exc:
            self.record(inst, "error", str(exc))
            return SyncResult(inst.symbol, inst.kind, "error", detail=str(exc))
        if bars.empty:
            self.record(inst, "error", "no bars returned")
            return SyncResult(inst.symbol, inst.kind, "error", detail="no bars returned")
        result = store.upsert(inst.symbol, bars, source=SOURCE_HISTORY, meta=file_meta(inst))
        self.record(inst, "ok", backfill_from=start)
        return SyncResult(inst.symbol, inst.kind, "ok", result.added, result.updated)

    # Renames ---------------------------------------------------------------------------
    def reconcile_renames(self, insts: Sequence[Instrument]) -> list[tuple[str, str]]:
        """Move a stock's file when NSE renames it (same instrument key, new trading symbol)."""
        renamed = []
        for inst in insts:
            store = self.store(inst)
            if store.exists(inst.symbol):
                continue
            row = self.conn.execute("SELECT name FROM bar_meta WHERE instrument_key = ? AND kind = ? AND name != ?",
                                    (inst.instrument_key, inst.kind, inst.symbol)).fetchone()
            if row is None or not store.exists(row["name"]):
                continue
            old = row["name"]
            os.replace(store.path(old), store.path(inst.symbol))
            self.conn.execute("UPDATE bar_meta SET name = ? WHERE name = ? AND kind = ?", (inst.symbol, old, inst.kind))
            self.conn.execute("UPDATE adjustment_factors SET symbol = ? WHERE symbol = ?", (inst.symbol, old))
            self.conn.commit()
            log.warning("renamed %s -> %s (same instrument key %s)", old, inst.symbol, inst.instrument_key)
            renamed.append((old, inst.symbol))
        return renamed

    # Refresh --------------------------------------------------------------------------
    def refresh(self, insts: Sequence[Instrument], as_of: date, *, today: date, new_start: date,
                overlap: int = 10, repull_threshold: float = 0.005,
                progress: Callable[[Instrument], None] | None = None) -> list[SyncResult]:
        results = []
        for inst in insts:
            results.append(self._refresh_one(inst, as_of, new_start, overlap, repull_threshold))
            if progress:
                progress(inst)
        if as_of == today:
            self._apply_live(insts, as_of, {r.name: r for r in results})
        return results

    def _refresh_one(self, inst: Instrument, as_of: date, new_start: date, overlap: int, threshold: float) -> SyncResult:
        store = self.store(inst)
        if not store.exists(inst.symbol):
            return self._backfill_one(inst, new_start, as_of, force=True)
        stored = store.read(inst.symbol)
        since = min(stored["date"].iloc[-min(overlap, len(stored))].date(), as_of)
        try:
            bars = self.provider.fetch_daily(inst, since, as_of)
            # Check before writing: if the vendor re-adjusted history, write only the full re-pull,
            # so a failed re-pull leaves the file untouched and the next refresh tries again.
            check = store.preview(inst.symbol, bars, source=SOURCE_HISTORY, provisional_sources=(SOURCE_LIVE,))
            repulled = False
            if check.max_confirmed_close_change > threshold:
                log.warning("%s: overlapping closes moved %.2f%%; re-pulling full history",
                            inst.symbol, 100 * check.max_confirmed_close_change)
                full = self.provider.fetch_daily(inst, stored["date"].iloc[0].date(), as_of)
                result = store.upsert(inst.symbol, full, source=SOURCE_HISTORY, meta=file_meta(inst))
                self.conn.execute("INSERT INTO repull_events (name, detected_at, max_close_rel_change, rows_updated) "
                                  "VALUES (?, ?, ?, ?)", (inst.symbol, utc_now(), check.max_confirmed_close_change, result.updated))
                repulled = True
            else:
                result = store.upsert(inst.symbol, bars, source=SOURCE_HISTORY, meta=file_meta(inst))
        except AuthError:
            raise
        except (ProviderError, ValueError) as exc:
            self.record(inst, "error", str(exc))
            return SyncResult(inst.symbol, inst.kind, "error", detail=str(exc))
        self.record(inst, "ok")
        return SyncResult(inst.symbol, inst.kind, "ok", result.added, result.updated, repulled)

    def _apply_live(self, insts: Sequence[Instrument], as_of: date, results: dict[str, SyncResult]) -> None:
        eligible = [i for i in insts if results.get(i.symbol) and results[i.symbol].status in ("ok", "skipped")]
        try:
            latest = self.provider.fetch_latest_daily(eligible)
        except ProviderError as exc:
            log.warning("live quotes unavailable, today's bars not stored: %s", exc)
            return
        today_rows = latest[latest["date"] == pd.Timestamp(as_of)].set_index("instrument_key")
        for inst in eligible:
            if inst.instrument_key not in today_rows.index:
                results[inst.symbol].detail = (results[inst.symbol].detail + "; no live bar for today").strip("; ")
                continue
            row = today_rows.loc[[inst.instrument_key]].reset_index(drop=True)
            store = self.store(inst)
            existing = store.read(inst.symbol)
            if (existing["date"] == pd.Timestamp(as_of)).any() and \
                    existing.loc[existing["date"] == pd.Timestamp(as_of), "source"].iloc[0] == SOURCE_HISTORY:
                continue  # history already has today; never downgrade it to a live bar
            res = store.upsert(inst.symbol, row[["date", "open", "high", "low", "close", "volume"]],
                               source=SOURCE_LIVE, meta=file_meta(inst))
            results[inst.symbol].added += res.added
            results[inst.symbol].updated += res.updated
            self.record(inst, "ok")
