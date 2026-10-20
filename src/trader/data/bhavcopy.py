"""NSE's official daily price files ("bhavcopy"): the exchange's own record of every equity's day.

Used as the reference to measure exactly what the vendor did to each corporate action, and to
recover the price each stock actually traded at. Files are public:
  before 2024-07-08  https://archives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
  from 2024-07-08    https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
Each downloaded day is stored as data/official/bhavcopy/YYYY/YYYYMMDD.parquet (equity series only),
and recorded in the `official_days` table so downloads resume and never repeat.
"""

from __future__ import annotations

import io
import logging
import os
import sqlite3
import time
import zipfile
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from trader.storage.db import utc_now

log = logging.getLogger(__name__)

UDIFF_FROM = date(2024, 7, 8)
EQUITY_SERIES = ("EQ", "BE", "BZ", "SM", "ST")
COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low", "close", "last", "prev_close", "volume", "turnover"]
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
_OLD = {"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin", "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
        "LAST": "last", "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover"}
_UDIFF = {"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open", "HghPric": "high", "LwPric": "low",
          "ClsPric": "close", "LastPric": "last", "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume", "TtlTrfVal": "turnover"}


def urls_for(day: date) -> list[str]:
    mon = day.strftime("%b").upper()
    old = f"https://archives.nseindia.com/content/historical/EQUITIES/{day:%Y}/{mon}/cm{day:%d}{mon}{day:%Y}bhav.csv.zip"
    new = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    return [new, old] if day >= UDIFF_FROM else [old, new]


def parse_bhavcopy(content: bytes, day: date) -> pd.DataFrame:
    """Normalize either file format to COLUMNS, equity series only."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]))
    raw.columns = [c.strip() for c in raw.columns]
    mapping = _UDIFF if "TckrSymb" in raw.columns else _OLD
    df = raw.rename(columns=mapping)[[c for c in mapping.values() if c in raw.rename(columns=mapping).columns]].copy()
    for column in ("symbol", "series", "isin"):
        df[column] = df[column].astype(str).str.strip()
    df = df[df["series"].isin(EQUITY_SERIES)].copy()
    for column in ("open", "high", "low", "close", "last", "prev_close", "turnover"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).round().astype("int64")
    df.insert(0, "date", pd.Timestamp(day))
    return df[COLUMNS].reset_index(drop=True)


class BhavcopyArchive:
    def __init__(self, root: Path, conn: sqlite3.Connection, *, client: httpx.Client | None = None,
                 pace_s: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.root = Path(root)
        self.conn = conn
        self._client = client or httpx.Client(headers=_HEADERS, timeout=httpx.Timeout(60.0, connect=5.0), follow_redirects=True)
        self._pace_s, self._sleep = pace_s, sleep
        self._last = 0.0

    def path(self, day: date) -> Path:
        return self.root / f"{day:%Y}" / f"{day:%Y%m%d}.parquet"

    def status(self, day: date) -> str | None:
        row = self.conn.execute("SELECT status FROM official_days WHERE date = ?", (day.isoformat(),)).fetchone()
        return row[0] if row else None

    def _get(self, url: str) -> httpx.Response | None:
        wait = self._pace_s - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        try:
            return self._client.get(url)
        except httpx.HTTPError as exc:
            log.warning("bhavcopy request failed: %s (%s)", url, type(exc).__name__)
            return None

    def fetch(self, day: date) -> str:
        """Download and store one day. Returns 'ok', 'missing' (no file published) or 'error'."""
        status, rows, source = "missing", 0, None
        for url in urls_for(day):
            resp = self._get(url)
            if resp is None:
                status = "error"
                continue
            if resp.status_code == 200 and resp.content[:2] == b"PK":
                frame = parse_bhavcopy(resp.content, day)
                path = self.path(day)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                frame.to_parquet(tmp, index=False)
                os.replace(tmp, path)
                status, rows, source = "ok", len(frame), url
                break
        self.conn.execute(
            "INSERT INTO official_days (date, status, rows, source, fetched_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET status=excluded.status, rows=excluded.rows, source=excluded.source, "
            "fetched_at=excluded.fetched_at",
            (day.isoformat(), status, rows, source, utc_now()),
        )
        self.conn.commit()
        return status

    def sync(self, days: Iterable[date], *, retry_missing: bool = False,
             progress: Callable[[date, str], None] | None = None, max_consecutive_errors: int = 10) -> dict[str, int]:
        counts = {"ok": 0, "missing": 0, "error": 0, "skipped": 0}
        errors_in_a_row = 0
        for day in days:
            current = self.status(day)
            if current == "ok" or (current == "missing" and not retry_missing):
                counts["skipped"] += 1
                if progress:
                    progress(day, "skipped")
                continue
            result = self.fetch(day)
            counts[result] += 1
            errors_in_a_row = errors_in_a_row + 1 if result == "error" else 0
            if progress:
                progress(day, result)
            if errors_in_a_row >= max_consecutive_errors:
                log.error("stopping: %d consecutive download errors (NSE may be blocking); rerun to resume", errors_in_a_row)
                break
        return counts

    def load(self, days: Iterable[date] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        """Stored official rows for the given days (default: all stored days)."""
        paths = [self.path(d) for d in days] if days is not None else sorted(self.root.glob("*/*.parquet"))
        frames = [pd.read_parquet(p, columns=columns) for p in paths if p.is_file()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or COLUMNS)


BY_STOCK_FILE = "by_stock.parquet"


def build_by_stock(archive_root: Path, stocks: dict[str, str | None]) -> pd.DataFrame:
    """Official daily rows for each of our stocks, following renames and ISIN changes.

    `stocks` maps our symbol to its current ISIN. A stock's history is the official rows whose symbol
    or ISIN belongs to it: a rename keeps the ISIN, a face-value split keeps the symbol, so the two
    are expanded alternately until stable. When several rows share a date (e.g. old and new ISIN
    around a split), the row matching the current symbol wins.
    """
    import pyarrow.dataset as ds

    files = sorted(Path(archive_root).glob("*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["stock", *COLUMNS])
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    keys = dataset.to_table(columns=["symbol", "isin"]).to_pandas().drop_duplicates()
    by_symbol = keys.groupby("symbol")["isin"].apply(set).to_dict()
    by_isin = keys.groupby("isin")["symbol"].apply(set).to_dict()
    owner: dict[tuple[str, str], str] = {}
    for stock, isin in stocks.items():
        symbols, isins = {stock}, {isin} if isin else set()
        while True:
            more_isins = set().union(*(by_symbol.get(s, set()) for s in symbols)) - isins
            more_symbols = set().union(*(by_isin.get(i, set()) for i in isins | more_isins)) - symbols
            if not more_isins and not more_symbols:
                break
            isins |= more_isins
            symbols |= more_symbols
        for s in symbols:
            for i in by_symbol.get(s, set()):
                if i in isins:
                    owner.setdefault((s, i), stock)
    wanted = sorted({s for s, _ in owner})
    table = dataset.to_table(filter=ds.field("symbol").isin(wanted)).to_pandas()
    table["stock"] = [owner.get((s, i)) for s, i in zip(table["symbol"], table["isin"])]
    table = table.dropna(subset=["stock"])
    table["exact"] = table["symbol"] == table["stock"]
    table = table.sort_values(["stock", "date", "exact"]).drop_duplicates(["stock", "date"], keep="last")
    return table.drop(columns="exact")[["stock", *COLUMNS]].reset_index(drop=True)


def by_stock_path(official_root: Path) -> Path:
    return Path(official_root).parent / BY_STOCK_FILE


def save_by_stock(frame: pd.DataFrame, official_root: Path) -> Path:
    path = by_stock_path(official_root)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def load_by_stock(official_root: Path) -> pd.DataFrame | None:
    path = by_stock_path(official_root)
    return pd.read_parquet(path) if path.is_file() else None
