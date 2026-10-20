import io
import zipfile
from datetime import date

import pandas as pd

from trader.data.bhavcopy import COLUMNS, build_by_stock, parse_bhavcopy, urls_for


def zipped(csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("bhav.csv", csv_text)
    return buffer.getvalue()


def test_parses_both_file_formats():
    old = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
           "ITC,EQ,289,292,285.15,290.95,290.25,286.2,8004124,2309725000,26-FEB-2016,1,INE154A01025,\n"
           "SOMEBOND,N1,100,100,100,100,100,100,1,100,26-FEB-2016,1,INE000X01011,\n")
    new = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal\n"
           "2025-01-06,2025-01-06,CM,NSE,STK,1660,INE154A01025,ITC,EQ,455.6,461.75,441.25,442.65,445.35,481.6,22009958,9907771000\n")
    a, b = parse_bhavcopy(zipped(old), date(2016, 2, 26)), parse_bhavcopy(zipped(new), date(2025, 1, 6))
    assert list(a.columns) == COLUMNS and list(b.columns) == COLUMNS
    assert a["symbol"].tolist() == ["ITC"]  # bond series dropped
    assert (a["close"].iloc[0], b["open"].iloc[0], b["prev_close"].iloc[0]) == (290.95, 455.6, 481.6)


def test_url_format_switches_in_july_2024():
    assert "historical/EQUITIES/2016/FEB/cm26FEB2016bhav.csv.zip" in urls_for(date(2016, 2, 26))[0]
    assert "BhavCopy_NSE_CM_0_0_0_20250106_F_0000.csv.zip" in urls_for(date(2025, 1, 6))[0]


def test_per_stock_history_follows_renames_and_isin_changes(tmp_path):
    def day(d, rows):
        frame = pd.DataFrame(rows, columns=["symbol", "isin", "close"])
        frame.insert(0, "date", pd.Timestamp(d))
        for column in ("series", "open", "high", "low", "last", "prev_close", "volume", "turnover"):
            frame[column] = "EQ" if column == "series" else 1.0
        path = tmp_path / d[:4] / f"{d.replace('-', '')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[COLUMNS].to_parquet(path, index=False)

    day("2016-01-04", [("OLDNAME", "INE111A01010", 10.0), ("SPLITCO", "INE222A01010", 1000.0)])
    day("2017-01-04", [("NEWNAME", "INE111A01010", 11.0), ("SPLITCO", "INE222A01028", 100.0)])  # rename; split -> new ISIN
    by_stock = build_by_stock(tmp_path, {"NEWNAME": "INE111A01010", "SPLITCO": "INE222A01028"})
    history = {s: g["close"].tolist() for s, g in by_stock.groupby("stock")}
    assert history == {"NEWNAME": [10.0, 11.0], "SPLITCO": [1000.0, 100.0]}
