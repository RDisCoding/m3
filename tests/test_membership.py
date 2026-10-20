from datetime import date

import pandas as pd

from trader.data.universe import members, update_membership
from trader.storage.db import connect


def mapping(*rows) -> pd.DataFrame:
    records = []
    for symbol, isin, *status in rows:
        records.append({"symbol": symbol, "isin": isin, "instrument_key": f"NSE_EQ|{isin}", "company_name": symbol,
                        "industry": "Capital Goods", "status": status[0] if status else "ok"})
    return pd.DataFrame(records)


def symbols(conn, as_of=None):
    return members(conn, "nifty500", as_of)["symbol"].tolist()


def test_membership_history_is_point_in_time_after_the_first_load(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    first = update_membership(conn, "nifty500",
                              mapping(("AAA", "INE000A01011"), ("BBB", "INE000B01011"), ("DUMMYX", "DUM000X01011", "placeholder")),
                              date(2026, 10, 20))
    assert first.initial_load and first.added == [] and first.removed == []
    assert symbols(conn) == ["AAA", "BBB"]
    assert symbols(conn, date(2015, 1, 1)) == ["AAA", "BBB"]  # unknown start date: assumed a member

    later = update_membership(conn, "nifty500", mapping(("AAA", "INE000A01011"), ("CCC", "INE000C01011")), date(2027, 3, 31))
    assert (later.added, later.removed, later.unchanged) == (["CCC"], ["BBB"], 1)
    assert symbols(conn, date(2027, 3, 30)) == ["AAA", "BBB"]
    assert symbols(conn, date(2027, 3, 31)) == ["AAA", "CCC"]
    assert symbols(conn) == ["AAA", "CCC"]

    rerun = update_membership(conn, "nifty500", mapping(("AAA", "INE000A01011"), ("CCC", "INE000C01011")), date(2027, 3, 31))
    assert (rerun.added, rerun.removed, rerun.unchanged) == ([], [], 2)


def test_symbol_rename_keeps_membership_by_isin(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    update_membership(conn, "nifty500", mapping(("OLDNAME", "INE000A01011")), date(2026, 10, 20))
    diff = update_membership(conn, "nifty500", mapping(("NEWNAME", "INE000A01011")), date(2026, 11, 2))
    assert (diff.added, diff.removed) == ([], [])
    assert symbols(conn) == ["NEWNAME"]


def test_constituent_that_is_temporarily_unmapped_stays_a_member(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    update_membership(conn, "nifty500", mapping(("AAA", "INE000A01011"), ("BBB", "INE000B01011")), date(2026, 10, 20))
    unmapped = mapping(("AAA", "INE000A01011"), ("BBB", "INE000B01011"))
    unmapped.loc[1, ["instrument_key", "status"]] = [None, "missing"]
    diff = update_membership(conn, "nifty500", unmapped, date(2026, 10, 21))
    assert diff.removed == [] and symbols(conn, date(2026, 10, 22)) == ["AAA", "BBB"]
