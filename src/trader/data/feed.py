"""Read side of the data layer: what strategies and the backtester consume.

`LocalDataFeed` reads the Parquet bar files and the SQLite tables and hands out:
  * daily bars and wide panels (date x symbol) with corporate-action adjustments applied,
  * the trading calendar,
  * point-in-time index membership, sectors and trading-segment flags.
Stored files are never modified here; adjustments are applied on the fly.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from trader.data.calendar import TradingCalendar, load_holidays
from trader.data.corporate_actions import OfficialBars, adjust_bars, load_factors
from trader.data.instruments import REF_FILENAME, SME_TYPES, T2T_TYPES
from trader.data.quality import repair_volume, session_flags, trim_fragments
from trader.data.universe import members
from trader.market import INDIA_NSE, MarketProfile, get_profile
from trader.settings import Settings, get_settings
from trader.storage.db import connect
from trader.storage.parquet import BarStore, safe_stem

PANEL_FIELDS = ("open", "high", "low", "close", "volume", "unadj_close", "traded_value")
CALENDAR_INDEX = INDIA_NSE.calendar_index  # kept for existing callers; new code uses feed.profile
_FACTOR_COLUMNS = ["symbol", "ex_date", "kind", "factor", "applied", "method", "detail", "subject",
                   "vendor_factor", "restore_amount", "evidence"]


class LocalDataFeed:
    def __init__(self, settings: Settings | None = None, profile: MarketProfile | None = None):
        self.settings = settings or get_settings()
        self.profile = profile or get_profile(self.settings.market)
        self.daily = BarStore(self.settings.bars_dir / "daily")
        self.index_store = BarStore(self.settings.bars_dir / "index")
        self.reload()

    def reload(self) -> None:
        """Forget cached factors, official prices and session flags (call after the data changes)."""
        self._factors: pd.DataFrame | None = None
        self._by_symbol: dict[str, pd.DataFrame] | None = None
        self._as_traded: dict[str, OfficialBars] | None = None
        self._sessions: pd.DataFrame | None = None

    def _conn(self) -> sqlite3.Connection:
        return connect(self.settings.db_path)

    def factors(self) -> pd.DataFrame:
        if self._factors is None:
            conn = self._conn()
            try:
                frame = load_factors(conn)
            finally:
                conn.close()
            frame["ex_date"] = pd.to_datetime(frame["ex_date"])
            self._factors = frame if not frame.empty else pd.DataFrame(columns=_FACTOR_COLUMNS)
            if "restore_amount" in self._factors:
                self._factors["restore_amount"] = self._factors["restore_amount"].fillna(0.0)
        return self._factors

    # Calendar -------------------------------------------------------------------------
    def calendar(self) -> TradingCalendar:
        observed: list[date] = []
        if self.index_store.exists(self.profile.calendar_index):
            observed = [d.date() for d in self.index_store.read_bars(self.profile.calendar_index)["date"]]
        conn = self._conn()
        try:
            holidays = load_holidays(conn)
        finally:
            conn.close()
        return TradingCalendar(observed, holidays)

    def trading_days(self, start: date, end: date) -> list[date]:
        return self.calendar().sessions(start, end)

    def session_flags(self) -> pd.DataFrame:
        """Sparse and short sessions judged from every stored stock (see quality.session_flags),
        indexed by the calendar index's sessions plus any date a stock has that the index lacks."""
        if self._sessions is None:
            stems = self.daily.stems()
            frames = [self.daily.read_columns(s, ["date", "close", "volume"]) for s in stems]
            dates = [f["date"].to_numpy() for f in frames]
            if self.index_store.exists(self.profile.calendar_index):
                dates.append(self.index_store.read_columns(self.profile.calendar_index, ["date"])["date"].to_numpy())
            index = pd.DatetimeIndex(np.unique(np.concatenate(dates))) if dates else pd.DatetimeIndex([])
            values = np.full((len(index), len(stems)), np.nan)
            for column, frame in enumerate(frames):
                values[index.get_indexer(frame["date"]), column] = frame["close"].to_numpy() * frame["volume"].to_numpy()
            self._sessions = session_flags(pd.DataFrame(values, index=index, columns=stems))
        return self._sessions

    # Bars -----------------------------------------------------------------------------
    def symbols(self) -> list[str]:
        return self.daily.stems()

    def load_bars(self, symbol: str, start: date | None = None, end: date | None = None,
                  adjustment: str = "economic", *, official: bool = True) -> pd.DataFrame:
        """One stock's bars. With `official`, prices on days NSE published a file are NSE's own
        (on the corrected basis); the vendor's fill the rest (see corporate_actions.adjust_bars)."""
        bars = self.daily.read_bars(symbol)
        if adjustment != "none":
            bars, _ = trim_fragments(bars)
            bars, _ = repair_volume(bars)
        adjusted = adjust_bars(bars, self._factors_for(symbol), adjustment, self._as_traded_for(symbol), anchor=official)
        return _slice(adjusted, start, end)

    def _as_traded_for(self, symbol: str) -> OfficialBars | None:
        """NSE's official bars and turnover for the stock (what it actually traded at), if downloaded."""
        if self._as_traded is None:
            from trader.data.bhavcopy import load_by_stock

            frame = load_by_stock(self.settings.official_dir)
            self._as_traded = {}
            if frame is not None and len(frame):
                frame = frame.sort_values(["stock", "date"], kind="stable")
                stocks = frame["stock"].to_numpy()
                cuts = np.flatnonzero(stocks[1:] != stocks[:-1]) + 1
                dates = frame["date"].to_numpy(dtype="datetime64[ns]")
                fields = {k: frame[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close", "volume", "turnover")}
                for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(frame)]):
                    self._as_traded[stocks[lo]] = OfficialBars(dates[lo:hi], {k: v[lo:hi] for k, v in fields.items()})
        return self._as_traded.get(symbol)

    def _factors_for(self, symbol: str) -> pd.DataFrame:
        if self._by_symbol is None:
            self._by_symbol = {s: g for s, g in self.factors().groupby("symbol")}
        return self._by_symbol.get(symbol, self.factors().iloc[0:0])

    def load_index(self, name: str | None = None, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        return _slice(self.index_store.read_bars(name or self.profile.calendar_index), start, end)

    def load_panel(self, symbols: list[str] | None = None, start: date | None = None, end: date | None = None,
                   adjustment: str = "economic") -> dict[str, pd.DataFrame]:
        """Wide frames keyed by field (open, high, low, close, volume, unadj_close): rows = dates, columns = symbols.

        A missing bar (not listed yet, suspended, or a special session the vendor lacks) is NaN.
        """
        names = sorted(symbols) if symbols is not None else self.symbols()
        frames = [self.load_bars(name, start, end, adjustment) for name in names]
        if not frames:
            return {field_name: pd.DataFrame() for field_name in PANEL_FIELDS}
        # Place each symbol's rows into preallocated arrays on the union of dates (much faster
        # than aligning 500 Series with pd.concat).
        index = pd.DatetimeIndex(np.unique(np.concatenate([f["date"].to_numpy() for f in frames])))
        positions = [index.get_indexer(f["date"]) for f in frames]
        panel = {}
        for field_name in PANEL_FIELDS:
            values = np.full((len(index), len(names)), np.nan)
            for column, (frame, rows) in enumerate(zip(frames, positions)):
                values[rows, column] = frame[field_name].to_numpy(dtype="float64")
            panel[field_name] = pd.DataFrame(values, index=index, columns=names)
        return panel

    # Universe -------------------------------------------------------------------------
    def universe_at(self, as_of: date | None = None, index: str = "nifty500") -> list[str]:
        conn = self._conn()
        try:
            return members(conn, index, as_of)["symbol"].tolist()
        finally:
            conn.close()

    def sector_map(self, as_of: date | None = None, index: str = "nifty500") -> dict[str, str]:
        conn = self._conn()
        try:
            frame = members(conn, index, as_of)
        finally:
            conn.close()
        return dict(zip(frame["symbol"], frame["industry"]))

    def symbol_flags(self) -> dict[str, set[str]]:
        """Current trading-segment flags from the instrument master: T2T (trade-for-trade) and SME."""
        path = self.settings.ref_dir / REF_FILENAME
        if not path.is_file():
            return {}
        ref = pd.read_parquet(path, columns=["segment", "trading_symbol", "instrument_type"])
        ref = ref[ref["segment"] == "NSE_EQ"]
        flags: dict[str, set[str]] = {}
        for symbol, series in zip(ref["trading_symbol"], ref["instrument_type"]):
            tags = ({"T2T"} if series in T2T_TYPES else set()) | ({"SME"} if series in SME_TYPES else set())
            if tags:
                flags[str(symbol)] = tags
        return flags

    def fingerprint(self) -> str:
        """Changes whenever any bar file or adjustment factor changes; used to key backtest runs."""
        digest = hashlib.sha256()
        for store in (self.daily, self.index_store):
            for stem in store.stems():
                stat = (store.root / f"{stem}.parquet").stat()
                digest.update(f"{store.root.name}/{stem}:{stat.st_size}:{stat.st_mtime_ns};".encode())
        from trader.data.bhavcopy import by_stock_path

        official = by_stock_path(self.settings.official_dir)
        if official.is_file():  # official closes and turnover feed unadj_close and traded_value
            stat = official.stat()
            digest.update(f"official:{stat.st_size}:{stat.st_mtime_ns};".encode())
        f = self.factors()
        if not f.empty:
            columns = ["symbol", "ex_date", "kind", "factor", "applied", "restore_amount"]
            digest.update(pd.util.hash_pandas_object(f[[c for c in columns if c in f]], index=False).values.tobytes())
        return digest.hexdigest()[:16]


def _slice(frame: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    return frame.loc[mask].reset_index(drop=True)


def panel_contract_check(panel: dict[str, pd.DataFrame]) -> list[str]:
    """Structural checks every consumer can rely on. Returns a list of problems (empty = OK)."""
    problems: list[str] = []
    close = panel.get("close")
    if close is None or close.empty:
        return ["panel is empty"]
    index = close.index
    if not index.is_monotonic_increasing or index.has_duplicates:
        problems.append("dates are not strictly increasing")
    for name in PANEL_FIELDS:
        frame = panel[name]
        if not frame.index.equals(index) or list(frame.columns) != list(close.columns):
            problems.append(f"{name} is not aligned with close")
    o, h, l, c = (panel[k] for k in ("open", "high", "low", "close"))
    tolerance = 1e-6 * c.abs()
    bad = ((h + tolerance) < np.maximum(o, c)) | ((l - tolerance) > np.minimum(o, c))
    if bad.to_numpy().any():
        cols = list(bad.columns[bad.any()])[:5]
        problems.append(f"high/low do not bound open/close for {cols}")
    if (c <= 0).to_numpy().any():
        problems.append("non-positive closes")
    if ((o <= 0) | (h <= 0) | (l <= 0) | (panel["unadj_close"] <= 0)).to_numpy().any():
        problems.append("non-positive open/high/low or traded price")
    if (panel["volume"] < 0).to_numpy().any():
        problems.append("negative volume")
    incomplete = (o.isna() | h.isna() | l.isna()) & c.notna()
    if incomplete.to_numpy().any():
        problems.append(f"open/high/low missing where a close exists for {list(incomplete.columns[incomplete.any()])[:5]}")
    if c.columns.has_duplicates:
        problems.append("duplicate symbols")
    stems = [safe_stem(s) for s in c.columns]
    if len(set(stems)) != len(stems):
        problems.append("symbols collide on disk")
    return problems
