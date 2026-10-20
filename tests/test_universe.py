from datetime import date

import httpx
import pandas as pd

from trader.data.instruments import SymbolMap, parse_instruments
from trader.data.universe import (
    download_constituents,
    load_constituents,
    map_constituents,
    mapping_coverage,
)


def test_mapping_statuses_and_coverage(fixtures):
    symbols = SymbolMap(parse_instruments(fixtures / "instruments_sample.json.gz"))
    mapping = map_constituents(load_constituents(fixtures / "nifty500_sample.csv"), symbols)
    assert dict(zip(mapping["symbol"], mapping["status"])) == {
        "DUMMYHEG": "placeholder",  # demerger stand-in, not tradable
        "HDFCBANK": "ok",
        "HEGAM": "ok",
        "RELIANCE": "ok",
        "TCS": "ok",
        "OLDNAME": "symbol_mismatch",  # ISIN belongs to M&M
        "GHOST": "missing",
    }
    assert mapping_coverage(mapping) == (5, 6)
    renamed = mapping.set_index("symbol").loc["OLDNAME"]
    assert renamed["upstox_symbol"] == "M&M" and renamed["instrument_key"] == "NSE_EQ|INE101A01026"


def test_isin_change_is_flagged(fixtures):
    symbols = SymbolMap(parse_instruments(fixtures / "instruments_sample.json.gz"))
    frame = pd.DataFrame([{"company_name": "Infosys", "industry": "IT", "symbol": "INFY", "series": "EQ", "isin": "INE000000001"}])
    row = map_constituents(frame, symbols).iloc[0]
    assert row["status"] == "isin_mismatch" and row["instrument_key"] == "NSE_EQ|INE009A01021"


def test_constituent_columns_are_normalized(fixtures):
    df = load_constituents(fixtures / "nifty500_sample.csv")
    assert list(df.columns) == ["company_name", "industry", "symbol", "series", "isin"]


def test_download_falls_back_to_the_nse_mirror(tmp_path, fixtures):
    csv = (fixtures / "nifty500_sample.csv").read_bytes()
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "www.niftyindices.com":
            return httpx.Response(503, text="busy")
        return httpx.Response(200, content=csv)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path = download_constituents("nifty500", tmp_path, client=client, today=date(2026, 10, 5))
    assert hosts == ["www.niftyindices.com", "archives.nseindia.com"]
    assert path.name == "ind_nifty500list_20261005.csv" and path.read_bytes() == csv
