"""SQLite store for the relational side of the data layer.

Bars live in Parquet files; everything small and relational lives here: index membership,
per-file bar metadata, exchange holidays, corporate actions, adjustment factors, the run log
and data-quality findings. WAL mode lets a reader (the dashboard, later) query while the
pipeline writes. Dates are ISO strings (YYYY-MM-DD); timestamps are ISO UTC strings.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_membership (
    index_name     TEXT NOT NULL,
    isin           TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    instrument_key TEXT,
    company_name   TEXT,
    industry       TEXT,
    start_date     TEXT,            -- NULL: member since before tracking began (survivorship-biased)
    end_date       TEXT,            -- NULL: still a member
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    note           TEXT,
    PRIMARY KEY (index_name, isin, first_seen)
);

CREATE TABLE IF NOT EXISTS bar_meta (
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,   -- EQ or INDEX
    instrument_key  TEXT,
    isin            TEXT,
    first_date      TEXT,
    last_date       TEXT,
    rows            INTEGER,
    provisional     INTEGER DEFAULT 0,  -- 1 if the last bar came from the live quote, not history
    backfill_from   TEXT,            -- start date of the last successful full backfill
    status          TEXT,            -- ok | error
    last_error      TEXT,
    updated_at      TEXT,
    PRIMARY KEY (name, kind)
);

CREATE TABLE IF NOT EXISTS repull_events (
    name                 TEXT NOT NULL,
    detected_at          TEXT NOT NULL,
    max_close_rel_change REAL,
    rows_updated         INTEGER
);

CREATE TABLE IF NOT EXISTS market_holidays (
    date          TEXT PRIMARY KEY,
    description   TEXT,
    holiday_type  TEXT,
    nse_closed    INTEGER NOT NULL,
    nse_special   INTEGER NOT NULL,  -- 1 if NSE holds a special session that day (e.g. Muhurat, Budget Sunday)
    fetched_at    TEXT
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol       TEXT NOT NULL,
    isin         TEXT,
    series       TEXT,
    ex_date      TEXT NOT NULL,
    record_date  TEXT,
    subject      TEXT NOT NULL,
    face_value   REAL,
    company      TEXT,
    fetched_at   TEXT,
    PRIMARY KEY (symbol, ex_date, subject)
);

CREATE TABLE IF NOT EXISTS adjustment_factors (
    symbol       TEXT NOT NULL,
    ex_date      TEXT NOT NULL,
    kind         TEXT NOT NULL,     -- split | bonus | consolidation | demerger | rights | dividend
    factor       REAL NOT NULL,     -- multiply prices before ex_date by this
    applied      INTEGER NOT NULL,  -- 1: applied by us; 0: already in the vendor data or not applicable
    method       TEXT,
    detail       TEXT,
    subject      TEXT,
    computed_at  TEXT,
    PRIMARY KEY (symbol, ex_date, kind)
);

CREATE TABLE IF NOT EXISTS data_runs (
    run_id       TEXT PRIMARY KEY,
    job          TEXT NOT NULL,
    as_of        TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS official_days (
    date        TEXT PRIMARY KEY,
    status      TEXT NOT NULL,      -- ok | missing (no file published) | error
    rows        INTEGER,
    source      TEXT,
    fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS dq_findings (
    run_id      TEXT,
    name        TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL,
    detail      TEXT,
    checked_at  TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Columns added after a table was first created: (table, column, type).
MIGRATIONS = (
    ("adjustment_factors", "vendor_factor", "REAL"),
    ("adjustment_factors", "restore_amount", "REAL DEFAULT 0"),
    ("adjustment_factors", "evidence", "TEXT DEFAULT 'gaps'"),
)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    for table, column, kind in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()
    return conn


@contextmanager
def open_db(path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
