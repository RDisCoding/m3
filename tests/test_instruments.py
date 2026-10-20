import gzip
import shutil
from datetime import date

import pytest

from trader.data import instruments
from trader.data.instruments import SymbolMap, ensure_reference, parse_instruments


@pytest.fixture
def ref(fixtures):
    return parse_instruments(fixtures / "instruments_sample.json.gz")


def test_parse_keeps_equity_and_index_segments_and_converts_paise_ticks(ref):
    assert set(ref["segment"]) == {"NSE_EQ", "NSE_INDEX"}  # the F&O row is dropped
    reliance = ref[ref["trading_symbol"] == "RELIANCE"].iloc[0]
    assert reliance["isin"] == "INE002A01018"
    assert reliance["tick_size"] == pytest.approx(0.10)  # 10 paise
    bajaj = ref[ref["trading_symbol"] == "BAJAJ-AUTO"].iloc[0]
    assert bajaj["tick_size"] == pytest.approx(1.00)  # 100 paise
    assert ref[ref["segment"] == "NSE_INDEX"]["tick_size"].isna().all()


def test_parse_accepts_an_uncompressed_file(tmp_path, fixtures, ref):
    plain = tmp_path / "NSE.json"
    plain.write_bytes(gzip.decompress((fixtures / "instruments_sample.json.gz").read_bytes()))
    assert parse_instruments(plain).equals(ref)


def test_resolves_symbols_isins_keys_and_index_names(ref):
    symbols = SymbolMap(ref)
    assert symbols.resolve("reliance").instrument_key == "NSE_EQ|INE002A01018"
    assert symbols.resolve("INE002A01018").symbol == "RELIANCE"
    assert symbols.resolve("NSE_EQ|INE040A01034").symbol == "HDFCBANK"
    assert symbols.resolve("M&M").isin == "INE101A01026"
    nifty = symbols.resolve("nifty 50")
    assert (nifty.kind, nifty.instrument_key) == ("INDEX", "NSE_INDEX|Nifty 50")
    assert symbols.index("NIFTY").instrument_key == "NSE_INDEX|Nifty 50"  # trading-symbol alias
    assert symbols.equity("RELIANCE").yf_ticker == "RELIANCE.NS"


def test_symbol_lookup_uses_share_series_not_warrants(ref):
    # ELECTCAST exists both as an EQ share and as W1 warrants in the master.
    inst = SymbolMap(ref).resolve("ELECTCAST")
    assert inst.instrument_type == "EQ" and inst.isin == "INE086A01029"


def test_unknown_symbol_error_suggests_close_matches(ref):
    with pytest.raises(KeyError, match="RELIANCE"):
        SymbolMap(ref).resolve("RELIANC")


def test_reference_is_downloaded_at_most_once_per_day(tmp_path, monkeypatch, fixtures):
    calls = []

    def fake_download(dest_dir, *, client=None, today=None):
        calls.append(today)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"NSE_{today:%Y%m%d}.json.gz"
        shutil.copy(fixtures / "instruments_sample.json.gz", dest)
        return dest

    monkeypatch.setattr(instruments, "download_instruments", fake_download)
    raw, refdir = tmp_path / "raw", tmp_path / "ref"
    _, as_of, downloaded = ensure_reference(raw, refdir, today=date(2026, 10, 5))
    assert downloaded and as_of == date(2026, 10, 5)
    df, _, downloaded = ensure_reference(raw, refdir, today=date(2026, 10, 5))
    assert not downloaded and len(df) > 0
    _, as_of, downloaded = ensure_reference(raw, refdir, today=date(2026, 10, 6))
    assert downloaded and as_of == date(2026, 10, 6)
    assert calls == [date(2026, 10, 5), date(2026, 10, 6)]
