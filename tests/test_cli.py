"""End-to-end tests of the command-line workflows on a temporary data folder, without network."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from trader.cli import app
from trader.data.bars import file_meta
from trader.data.instruments import SymbolMap, parse_instruments
from trader.data.universe import update_membership
from trader.settings import get_settings
from trader.storage.db import connect
from trader.storage.parquet import BarStore

runner = CliRunner()
SYMBOLS = ["RELIANCE", "TCS", "INFY"]
HISTORY = pd.bdate_range(end="2026-10-16", periods=320)
STORED_UNTIL = HISTORY[-3]  # the stores are two sessions behind the vendor


def synthetic(seed: int, start: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, len(HISTORY))))
    open_ = close * (1 + rng.normal(0, 0.002, len(HISTORY)))
    return pd.DataFrame({"date": HISTORY, "open": open_, "high": np.maximum(open_, close) * 1.01,
                         "low": np.minimum(open_, close) * 0.99, "close": close, "volume": rng.integers(1e5, 1e6, len(HISTORY))})


class FakeProvider:
    request_count = 0

    def __init__(self, history):
        self.history = history

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def fetch_daily(self, inst, start, end):
        FakeProvider.request_count += 1
        frame = self.history[inst.symbol]
        return frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    def fetch_latest_daily(self, insts):
        return pd.DataFrame(columns=["instrument_key", "date", "open", "high", "low", "close", "volume"])

    def market_holidays(self):
        return [{"date": "2026-10-20", "description": "Dussehra", "holiday_type": "TRADING_HOLIDAY",
                 "closed_exchanges": ["NSE"], "open_exchanges": []}]


@pytest.fixture
def env(tmp_path, monkeypatch, fixtures):
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "test-token-not-real")
    get_settings.cache_clear()
    settings = get_settings()
    ref = parse_instruments(fixtures / "instruments_sample.json.gz")
    symbols = SymbolMap(ref)
    history = {s: synthetic(i, 100.0 * (i + 1)) for i, s in enumerate(SYMBOLS)}
    history["Nifty 50"] = synthetic(9, 23000.0)
    history["Nifty 500"] = synthetic(10, 21000.0)
    daily, index = BarStore(settings.bars_dir / "daily"), BarStore(settings.bars_dir / "index")
    for name, frame in history.items():
        inst = symbols.resolve(name)
        store = index if inst.kind == "INDEX" else daily
        store.upsert(inst.symbol, frame[frame["date"] <= STORED_UNTIL], source="upstox", meta=file_meta(inst))
    conn = connect(settings.db_path)
    mapping = pd.DataFrame([{"symbol": s, "isin": symbols.equity(s).isin, "instrument_key": symbols.equity(s).instrument_key,
                             "company_name": s, "industry": "Test", "status": "ok"} for s in SYMBOLS])
    update_membership(conn, "nifty500", mapping, date(2026, 10, 1))
    conn.commit()
    conn.close()

    import trader.cli.data_pipeline as pipeline

    monkeypatch.setattr(pipeline, "load_reference", lambda refresh=False: (ref, date(2026, 10, 16), False))
    monkeypatch.setattr(pipeline, "make_provider", lambda: FakeProvider(history))
    monkeypatch.setattr(pipeline, "fetch_corporate_actions", lambda client, start, end: [])
    constituents = tmp_path / "ind_nifty500list.csv"
    lines = ["Company Name,Industry,Symbol,Series,ISIN Code"] + [f"{s} Ltd.,Test,{s},EQ,{symbols.equity(s).isin}" for s in SYMBOLS]
    constituents.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "download_constituents", lambda index, dest, today=None: constituents)
    yield settings, history
    get_settings.cache_clear()


def run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_refresh_brings_every_instrument_current_and_logs_the_run(env):
    settings, _ = env
    out = run("data", "refresh", "--as-of", "2026-10-16")
    assert "Refresh through 2026-10-16" in out
    daily = BarStore(settings.bars_dir / "daily")
    for symbol in SYMBOLS:
        assert daily.read_bars(symbol)["date"].iloc[-1] == pd.Timestamp("2026-10-16")
    conn = connect(settings.db_path)
    status, summary = conn.execute("SELECT status, summary_json FROM data_runs WHERE job = 'refresh'").fetchone()
    assert status == "ok" and '"errors": 0' in summary
    assert conn.execute("SELECT COUNT(*) FROM bar_meta WHERE last_date = '2026-10-16'").fetchone()[0] == 5


def test_second_refresh_changes_nothing(env):
    settings, _ = env
    run("data", "refresh", "--as-of", "2026-10-16")
    daily = BarStore(settings.bars_dir / "daily")
    before = {s: daily.path(s).stat().st_mtime_ns for s in SYMBOLS}
    run("data", "refresh", "--as-of", "2026-10-16")
    assert {s: daily.path(s).stat().st_mtime_ns for s in SYMBOLS} == before


def test_backfill_status_panel_and_verify_commands(env):
    settings, _ = env
    assert "instruments" in run("data", "backfill", "--only", "TCS", "--start", "2025-01-01")
    out = run("data", "status")
    assert "current index members: 3" in out
    assert "Contract check: OK" in run("data", "panel-check")
    assert "Checked 3 symbols" in run("data", "verify")


def test_market_and_features_commands(env):
    assert "IN_NSE" in run("market", "show")
    out = run("features", "check", "--days", "2", "--window", "300")
    assert "features identical with data ending that day" in out
