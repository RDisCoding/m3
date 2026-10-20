"""Index constituent lists from NSE Indices and their mapping to Upstox instruments.

Constituents are the *current* members only. Backtests that use them carry survivorship
bias; the point-in-time membership table that fixes this comes with the backfill pipeline.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from trader.data.instruments import T2T_TYPES, SymbolMap

CONSTITUENT_FILES = {
    "nifty50": "ind_nifty50list.csv",
    "nifty500": "ind_nifty500list.csv",
}
_SOURCES = (
    "https://www.niftyindices.com/IndexConstituent/{file}",
    "https://archives.nseindia.com/content/indices/{file}",
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0 Safari/537.36",
    "Accept": "text/csv,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
_COLUMN_MAP = {
    "Company Name": "company_name",
    "Industry": "industry",
    "Symbol": "symbol",
    "Series": "series",
    "ISIN Code": "isin",
}


def download_constituents(
    index: str, dest_dir: Path, *, client: httpx.Client | None = None, today: date | None = None
) -> Path:
    """Download an index constituent CSV, trying each official source in turn."""
    if index not in CONSTITUENT_FILES:
        raise ValueError(f"unknown index {index!r}; choose from {sorted(CONSTITUENT_FILES)}")
    today = today or date.today()
    file = CONSTITUENT_FILES[index]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{Path(file).stem}_{today:%Y%m%d}.csv"
    owns_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0), follow_redirects=True, headers=_HEADERS)
    errors: list[str] = []
    try:
        for template in _SOURCES:
            url = template.format(file=file)
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                continue
            if not resp.content.lstrip().startswith(b"Company Name"):
                errors.append(f"{url}: unexpected content")
                continue
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(resp.content)
            os.replace(tmp, dest)
            return dest
    finally:
        if owns_client:
            client.close()
    raise RuntimeError("could not download constituents: " + " | ".join(errors))


def is_placeholder(symbol: str, isin: str | None) -> bool:
    """True for index rows that stand in for a security that is not listed yet."""
    return symbol.upper().startswith("DUMMY") or not (isin or "").upper().startswith("IN")


def mapping_coverage(mapping: pd.DataFrame) -> tuple[int, int]:
    """(mapped, eligible) where placeholders are excluded from the eligible count."""
    eligible = mapping[mapping["status"] != "placeholder"]
    return int(eligible["instrument_key"].notna().sum()), len(eligible)


def load_constituents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).rename(columns=lambda c: c.strip())
    missing = set(_COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    df = df.rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
    for column in df.columns:
        df[column] = df[column].str.strip()
    return df.dropna(subset=["symbol"]).reset_index(drop=True)


def map_constituents(constituents: pd.DataFrame, symbols: SymbolMap) -> pd.DataFrame:
    """Map each constituent to an Upstox instrument, by ISIN first, then by symbol.

    status values:
      ok               ISIN found and the Upstox symbol matches the NSE symbol
      symbol_mismatch  ISIN found but Upstox lists a different trading symbol (rename)
      isin_mismatch    ISIN not found but the symbol is (ISIN changed, e.g. after a split)
      placeholder      not a tradable security: NSE Indices inserts rows such as
                       "Dummy HEG Ltd." (symbol DUMMY*, ISIN not starting with IN) while a
                       demerged entity awaits listing
      missing          neither ISIN nor symbol found
    """
    records = []
    for row in constituents.to_dict("records"):
        isin = (row.get("isin") or "").upper()
        by_isin = symbols.by_isin(isin) if isin else None
        by_symbol = symbols.equity(row["symbol"])
        inst = by_isin or by_symbol
        if is_placeholder(row["symbol"], isin):
            inst, status = None, "placeholder"
        elif by_isin is not None:
            status = "ok" if by_isin.symbol == row["symbol"] else "symbol_mismatch"
        elif by_symbol is not None:
            status = "isin_mismatch"
        else:
            status = "missing"
        records.append(
            {
                "symbol": row["symbol"],
                "isin": isin or None,
                "company_name": row.get("company_name"),
                "industry": row.get("industry"),
                "series": row.get("series"),
                "instrument_key": inst.instrument_key if inst else None,
                "upstox_symbol": inst.symbol if inst else None,
                "upstox_type": inst.instrument_type if inst else None,
                "tick_size": inst.tick_size if inst else None,
                "is_t2t": bool(inst and inst.instrument_type in T2T_TYPES),
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


# Point-in-time membership ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MembershipDiff:
    index_name: str
    as_of: date
    initial_load: bool
    added: list[str]
    removed: list[str]
    unchanged: int


def update_membership(conn: sqlite3.Connection, index_name: str, mapping: pd.DataFrame, as_of: date) -> MembershipDiff:
    """Record today's constituent list in the point-in-time membership table.

    The first load cannot know when existing members joined, so their start_date stays NULL
    ("member since before tracking began"). Later loads open rows for additions and close rows for
    removals, so membership becomes point-in-time correct from the first load onward. Identity is
    the ISIN; placeholders and unmapped rows are skipped.
    """
    stamp = as_of.isoformat()
    listed = mapping[mapping["status"] != "placeholder"]
    present = {str(i) for i in listed["isin"].dropna()}  # still in NSE's list, even if not mappable today
    current = listed[listed["instrument_key"].notna()]
    initial = conn.execute("SELECT COUNT(*) FROM universe_membership WHERE index_name = ?", (index_name,)).fetchone()[0] == 0
    open_rows = {r["isin"]: r for r in conn.execute(
        "SELECT isin, symbol, first_seen FROM universe_membership WHERE index_name = ? AND end_date IS NULL", (index_name,))}
    added: list[str] = []
    unchanged = 0
    seen: set[str] = set()
    for row in current.to_dict("records"):
        isin = row["isin"] or ""
        seen.add(isin)
        if isin in open_rows:
            conn.execute(
                "UPDATE universe_membership SET last_seen = ?, symbol = ?, instrument_key = ?, company_name = ?, industry = ? "
                "WHERE index_name = ? AND isin = ? AND first_seen = ?",
                (stamp, row["symbol"], row["instrument_key"], row.get("company_name"), row.get("industry"),
                 index_name, isin, open_rows[isin]["first_seen"]),
            )
            unchanged += 1
            continue
        conn.execute(
            "INSERT OR REPLACE INTO universe_membership (index_name, isin, symbol, instrument_key, company_name, industry, "
            "start_date, end_date, first_seen, last_seen, note) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
            (index_name, isin, row["symbol"], row["instrument_key"], row.get("company_name"), row.get("industry"),
             None if initial else stamp, stamp, stamp, "initial_load_assumed" if initial else "added"),
        )
        if not initial:
            added.append(row["symbol"])
    removed: list[str] = []
    for isin, row in open_rows.items():
        if isin not in seen and isin not in present:
            conn.execute(
                "UPDATE universe_membership SET end_date = ?, note = 'removed' WHERE index_name = ? AND isin = ? AND first_seen = ?",
                (stamp, index_name, isin, row["first_seen"]),
            )
            removed.append(row["symbol"])
    return MembershipDiff(index_name, as_of, initial, sorted(added), sorted(removed), unchanged)


def members(conn: sqlite3.Connection, index_name: str, as_of: date | None = None) -> pd.DataFrame:
    """Members on `as_of` (default: current members): symbol, isin, instrument_key, company_name, industry."""
    if as_of is None:
        query = ("SELECT symbol, isin, instrument_key, company_name, industry FROM universe_membership "
                 "WHERE index_name = ? AND end_date IS NULL ORDER BY symbol")
        params: tuple = (index_name,)
    else:
        stamp = as_of.isoformat()
        query = ("SELECT symbol, isin, instrument_key, company_name, industry FROM universe_membership "
                 "WHERE index_name = ? AND (start_date IS NULL OR start_date <= ?) AND (end_date IS NULL OR end_date > ?) "
                 "ORDER BY symbol")
        params = (index_name, stamp, stamp)
    return pd.DataFrame([dict(r) for r in conn.execute(query, params)],
                        columns=["symbol", "isin", "instrument_key", "company_name", "industry"])
