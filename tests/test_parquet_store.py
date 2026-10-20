from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import pytest

from trader.data.providers.base import normalize_bars
from trader.storage.parquet import BAR_SCHEMA, BarStore, safe_stem

META = {"name": "TEST", "instrument_key": "NSE_EQ|INE000TEST01", "isin": "INE000TEST01", "kind": "EQ",
        "provider": "upstox", "adjustment": "splits_and_bonuses"}
T0 = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def bars(dates, closes, volume=1000):
    return normalize_bars(pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [volume] * len(closes),
    }))


def test_round_trip_keeps_values_schema_and_metadata(tmp_path):
    store = BarStore(tmp_path)
    frame = bars(["2026-09-28", "2026-09-29", "2026-09-30"], [100.0, 101.5, 99.25])
    result = store.upsert("TEST", frame, source="upstox", meta=META, now=T0)
    assert result.written and (result.added, result.updated, result.rows_after) == (3, 0, 3)
    assert store.read_bars("TEST").equals(frame)
    assert pq.read_schema(store.path("TEST")).remove_metadata().equals(BAR_SCHEMA)
    assert store.read_meta("TEST") == META
    stored = store.read("TEST")
    assert (stored["source"] == "upstox").all() and (stored["ingested_at"] == pd.Timestamp(T0)).all()
    assert not list(tmp_path.glob("*.tmp"))


def test_identical_upsert_leaves_the_file_byte_for_byte_unchanged(tmp_path):
    store = BarStore(tmp_path)
    frame = bars(["2026-09-29", "2026-09-30"], [100.0, 101.0])
    store.upsert("TEST", frame, source="upstox", meta=META, now=T0)
    before = (store.path("TEST").read_bytes(), store.path("TEST").stat().st_mtime_ns)
    result = store.upsert("TEST", frame, source="upstox", meta=META, now=T1)
    assert not result.written and (result.added, result.updated) == (0, 0)
    assert (store.path("TEST").read_bytes(), store.path("TEST").stat().st_mtime_ns) == before


def test_changed_overlap_is_updated_and_the_change_is_measured(tmp_path):
    store = BarStore(tmp_path)
    store.upsert("TEST", bars(["2026-09-28", "2026-09-29", "2026-09-30"], [100.0, 101.0, 102.0]), source="upstox", meta=META, now=T0)
    # A retroactive 1:1 bonus adjustment halves history; one new session arrives.
    update = bars(["2026-09-29", "2026-09-30", "2026-10-01"], [101.0, 51.0, 52.0])
    result = store.upsert("TEST", update, source="upstox", meta=META, now=T1)
    assert (result.added, result.updated, result.rows_after) == (1, 1, 4)
    assert result.max_close_rel_change == pytest.approx(0.5)
    stored = store.read("TEST").set_index("date")
    assert stored.loc["2026-09-30", "close"] == 51.0
    assert stored.loc["2026-09-28", "ingested_at"] == pd.Timestamp(T0)  # untouched rows keep provenance
    assert stored.loc["2026-09-30", "ingested_at"] == pd.Timestamp(T1)


def test_source_change_relabels_a_provisional_row(tmp_path):
    store = BarStore(tmp_path)
    frame = bars(["2026-10-05"], [1186.4])
    store.upsert("TEST", frame, source="upstox_intraday", meta=META, now=T0)
    result = store.upsert("TEST", frame, source="upstox", meta=META, now=T1)
    assert result.written and result.updated == 1 and result.max_close_rel_change == 0.0
    assert store.read("TEST")["source"].tolist() == ["upstox"]


def test_metadata_change_alone_rewrites_the_file(tmp_path):
    store = BarStore(tmp_path)
    frame = bars(["2026-09-30"], [100.0])
    store.upsert("TEST", frame, source="upstox", meta=META, now=T0)
    result = store.upsert("TEST", frame, source="upstox", meta={**META, "isin": "INE000TEST02"}, now=T1)
    assert result.written and store.read_meta("TEST")["isin"] == "INE000TEST02"


def test_refuses_to_store_missing_prices(tmp_path):
    frame = bars(["2026-09-30"], [100.0]).assign(source="upstox", ingested_at=pd.Timestamp(T0))
    frame.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError):
        BarStore(tmp_path).write("TEST", frame, meta=META)


@pytest.mark.parametrize("name, stem", [
    ("Nifty 50", "NIFTY_50"), ("M&M", "M&M"), ("BAJAJ-AUTO", "BAJAJ-AUTO"), ("CON", "CON_"), ("nul.x", "NUL.X_"), ("A/B", "A%2FB"),
])
def test_safe_stem_produces_valid_windows_file_names(name, stem):
    assert safe_stem(name) == stem
