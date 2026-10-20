import json

import numpy as np
import pandas as pd
import pytest
from factories import action_rows, bars

from trader.data.corporate_actions import (
    EQUITY_SERIES,
    ParsedAction,
    adjust_bars,
    compute_factors,
    load_actions,
    parse_subject,
    store_actions,
    unparsed_price_events,
)
from trader.storage.db import connect


def factor_frame(factors) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": f.symbol, "ex_date": pd.Timestamp(f.ex_date), "kind": f.kind, "factor": f.factor,
                          "applied": int(f.applied), "method": f.method} for f in factors],
                        columns=["symbol", "ex_date", "kind", "factor", "applied", "method"])


@pytest.mark.parametrize("subject, face_value, expected", [
    ("Bonus 1:2", None, [ParsedAction("bonus", ratio_new=1, ratio_old=2)]),
    ("Demerger", None, [ParsedAction("demerger")]),
    ("Rights 1:15 @ Premium Rs 1247", 10, [ParsedAction("rights", ratio_new=1, ratio_old=15, issue_price=1257, premium=1247)]),
    ("Rights 3:19 @ Premium Rs 1090 Per Share", 1, [ParsedAction("rights", ratio_new=3, ratio_old=19, issue_price=1091, premium=1090)]),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", None,
     [ParsedAction("split", ratio_old=10, ratio_new=1)]),
    ("Consolidation Of Shares From Rs 1 To Rs 10", None, [ParsedAction("consolidation", ratio_old=1, ratio_new=10)]),
    ("Interim Dividend - Rs 10 Per Share Special Dividend - Rs 66 Per Share", None, [ParsedAction("dividend", cash=76)]),
    ("Dividend-Rs.8.50 Per Share (Including Special Dividend Of Rs.2/- Per Share).", None, [ParsedAction("dividend", cash=8.5)]),
    ("Dividend - Re 1 Per Share", None, [ParsedAction("dividend", cash=1)]),
    ("Interim Dividend - Rs 18/- Per Share (Purpose Revised)", None, [ParsedAction("dividend", cash=18)]),
    ("Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", None,
     [ParsedAction("bonus", ratio_new=1, ratio_old=1), ParsedAction("split", ratio_old=10, ratio_new=2)]),
    ("Annual General Meeting", None, []),
    # Spellings seen in the real NSE feed
    ("Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share", None, [ParsedAction("split", ratio_old=10, ratio_new=1)]),
    ("Face Value Split Rs 10 To Rs 1", None, [ParsedAction("split", ratio_old=10, ratio_new=1)]),
    ("Bonus- 1:2", None, [ParsedAction("bonus", ratio_new=1, ratio_old=2)]),
    ("Rights 1:10 @ Prem Rs 197/- Per Share", 10, [ParsedAction("rights", ratio_new=1, ratio_old=10, issue_price=207, premium=197)]),
    ("Composite Scheme Of Arrangement", None, [ParsedAction("arrangement")]),
    ("Sheme Of Arrangement", None, [ParsedAction("arrangement")]),
    ("Fv Splt Frm Rs 10 To Re 1", None, [ParsedAction("split", ratio_old=10, ratio_new=1)]),
    ("Interim Dividend - Rs 5.25 Per Sh", None, [ParsedAction("dividend", cash=5.25)]),
    ("Dividend Rs - 6 Per Share", None, [ParsedAction("dividend", cash=6)]),
    ("Int Div - Rs 4/- Per Share (Purpose Revised)", None, [ParsedAction("dividend", cash=4)]),
    ("Agm/Spl Div- Rs.5/- Div-7.5/-", None, [ParsedAction("dividend", cash=12.5)]),
    ("Annual General Meeting/Dividend Rs 2.5/-", None, [ParsedAction("dividend", cash=2.5)]),
    ("Interim Dividend", None, []),
    ("Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held", None, [ParsedAction("arrangement")]),
])
def test_parse_real_subject_lines(subject, face_value, expected):
    assert parse_subject(subject, face_value) == expected


def test_split_factors():
    assert ParsedAction("bonus", ratio_new=1, ratio_old=2).split_factor() == pytest.approx(2 / 3)
    assert ParsedAction("bonus", ratio_new=4, ratio_old=1).split_factor() == pytest.approx(0.2)
    assert ParsedAction("split", ratio_old=10, ratio_new=2).split_factor() == pytest.approx(0.2)
    assert ParsedAction("consolidation", ratio_old=1, ratio_new=10).split_factor() == pytest.approx(10)


def test_store_keeps_equity_series_and_parses_nse_dates(tmp_path, fixtures):
    rows = json.loads((fixtures / "nse_corporate_actions_sample.json").read_text())
    rows.append({"symbol": "759GS2026", "series": "GS", "exDate": "10-Jan-2025", "subject": "Interest Payment", "faceVal": "100"})
    conn = connect(tmp_path / "t.sqlite")
    store_actions(conn, rows, "now")
    store_actions(conn, rows, "later")  # idempotent upsert
    loaded = load_actions(conn)
    assert set(loaded["series"]) <= EQUITY_SERIES and "759GS2026" not in set(loaded["symbol"])
    assert not loaded.duplicated(["symbol", "ex_date", "subject"]).any()
    demerger = loaded[(loaded["symbol"] == "ITC") & (loaded["subject"] == "Demerger")]
    assert demerger["ex_date"].tolist() == ["2025-01-06"]


ITC_DATES = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def test_demerger_factor_is_ex_open_over_previous_close():
    # ITC around the hotels demerger: raw NSE closes 489.05, 481.60, then 442.65 on the ex-date.
    itc = bars(ITC_DATES, [492.0, 489.05, 481.6, 442.65, 441.1], opens=[490.0, 490.0, 488.0, 455.0, 443.0])
    factors = compute_factors("ITC", itc, action_rows(("ITC", "2025-01-06", "Demerger", 1.0)))
    assert [(f.kind, f.applied, f.method) for f in factors] == [("demerger", True, "ex_open_gap")]
    assert factors[0].factor == pytest.approx(455.0 / 481.6)
    adjusted = adjust_bars(itc, factor_frame(factors)).set_index("date")["close"]
    assert adjusted["2025-01-03"] == pytest.approx(455.0)
    assert adjusted["2025-01-06"] / adjusted["2025-01-03"] - 1 == pytest.approx(442.65 / 455.0 - 1)  # about -2.7%, not -8.1%


def test_demerger_without_gap_is_assumed_already_adjusted():
    smooth = bars(ITC_DATES, [100.0, 100.5, 101.0, 101.2, 101.5])
    factors = compute_factors("ITC", smooth, action_rows(("ITC", "2025-01-06", "Demerger")))
    assert [(f.applied, f.method, f.factor) for f in factors] == [(False, "no_gap", 1.0)]


DATES = ["2025-06-11", "2025-06-12", "2025-06-13", "2025-06-16", "2025-06-17"]


def test_bonus_in_vendor_data_is_verified_not_reapplied():
    smooth = bars(DATES, [930.0, 925.0, 933.0, 938.0, 923.0])
    factors = compute_factors("X", smooth, action_rows(("X", "2025-06-16", "Bonus 1:1")))
    assert [(f.applied, f.method) for f in factors] == [(False, "verified_upstream")]
    assert adjust_bars(smooth, factor_frame(factors))["close"].tolist() == smooth["close"].tolist()


def test_bonus_missing_from_vendor_data_is_applied_with_volume():
    raw = bars(DATES, [1860.0, 1850.0, 1866.0, 938.0, 923.0])
    factors = compute_factors("X", raw, action_rows(("X", "2025-06-16", "Bonus 1:1")))
    assert [(f.applied, f.method, f.factor) for f in factors] == [(True, "vendor_unadjusted", 0.5)]
    adjusted = adjust_bars(raw, factor_frame(factors))
    assert adjusted["close"].tolist() == [930.0, 925.0, 933.0, 938.0, 923.0]
    assert adjusted["volume"].tolist() == [2000, 2000, 2000, 1000, 1000]


def test_same_day_bonus_and_split_are_judged_together():
    rows = action_rows(("X", "2025-06-16", "Bonus 4:1"),
                       ("X", "2025-06-16", "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"))
    raw = bars(DATES, [9300.0, 9250.0, 9330.0, 938.0, 923.0])  # combined 1:10 move, unadjusted
    factors = compute_factors("X", raw, rows)
    assert {(f.kind, f.applied) for f in factors} == {("bonus", True), ("split", True)}
    assert adjust_bars(raw, factor_frame(factors))["close"].iloc[2] == pytest.approx(933.0)
    smooth = bars(DATES, [930.0, 925.0, 933.0, 938.0, 923.0])
    assert all(f.method == "verified_upstream" for f in compute_factors("X", smooth, rows))


def test_rights_and_dividends_are_converted_to_the_stored_price_basis():
    # Stored prices are halved by a later 1:1 bonus the vendor already applied, so rupee amounts
    # announced before that bonus must be halved too.
    dates = ["2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2024-10-25", "2024-10-28", "2024-10-29"]
    stored = bars(dates, [750.0, 748.0, 718.0, 720.0, 1330.0, 1335.0, 1340.0])
    rows = action_rows(("R", "2020-05-13", "Rights 1:15 @ Premium Rs 1247", 10.0),
                       ("R", "2020-05-12", "Dividend - Rs 10 Per Share", 10.0),
                       ("R", "2024-10-28", "Bonus 1:1", 10.0))
    by_kind = {f.kind: f for f in compute_factors("R", stored, rows)}
    assert by_kind["bonus"].method == "verified_upstream"
    issue = 1257 / 2
    assert by_kind["rights"].factor == pytest.approx((15 * 748.0 + issue) / 16 / 748.0)
    assert by_kind["dividend"].factor == pytest.approx((750.0 - 5.0) / 750.0)


def test_modes_and_unadjusted_close():
    dates = ["2024-10-24", "2024-10-25", "2024-10-28", "2024-10-29"]
    stored = bars(dates, [1300.0, 1330.0, 1335.0, 1340.0])
    rows = action_rows(("R", "2024-10-28", "Bonus 1:1"), ("R", "2024-10-25", "Dividend - Rs 20 Per Share"))
    f = factor_frame(compute_factors("R", stored, rows))
    economic = adjust_bars(stored, f, "economic")
    total = adjust_bars(stored, f, "total_return")
    assert economic["close"].tolist() == stored["close"].tolist()  # dividend excluded, bonus already in data
    assert total["close"].iloc[0] == pytest.approx(1300.0 * (1300.0 - 10.0) / 1300.0)
    assert economic["unadj_close"].tolist() == [2600.0, 2660.0, 1335.0, 1340.0]  # as actually traded
    assert adjust_bars(stored, f, "none")["close"].tolist() == stored["close"].tolist()


def test_events_outside_the_stored_history_are_ignored():
    stored = bars(DATES, [930.0, 925.0, 933.0, 938.0, 923.0])
    rows = action_rows(("X", "2025-06-11", "Bonus 1:1"), ("X", "2026-01-01", "Dividend - Rs 5 Per Share"))
    assert compute_factors("X", stored, rows) == []


def test_unparsed_price_relevant_subjects_are_flagged():
    actions = action_rows(("A", "2025-01-01", "Capital Reduction"), ("B", "2025-01-01", "Bonus 1:1"),
                          ("C", "2025-01-01", "Annual General Meeting"), ("D", "2025-01-01", "Scheme Of Arrangement"))
    assert unparsed_price_events(actions)["symbol"].tolist() == ["A"]


def test_multi_tranche_rights_combine_into_one_issue():
    subject = ("Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share / "
               "2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share")
    [rights] = parse_subject(subject, 10)
    assert rights.ratio_new == pytest.approx(6 / 25) and rights.ratio_old == 1.0
    assert rights.issue_price == pytest.approx((4 * 510 + 2 * 615) / 6)


def test_over_adjusted_demerger_is_corrected_upwards():
    # NMDC 2022: the vendor scaled pre-demerger prices too far down, leaving a fake +71% jump.
    stored = bars(ITC_DATES, [16.0, 16.2, 16.3, 31.0, 31.5], opens=[16.0, 16.1, 16.2, 27.88, 31.2])
    [f] = compute_factors("NMDC", stored, action_rows(("NMDC", "2025-01-06", "Demerger")))
    assert (f.kind, f.applied, f.factor) == ("demerger", True, pytest.approx(27.88 / 16.3))
    adjusted = adjust_bars(stored, factor_frame([f])).set_index("date")
    assert adjusted.loc["2025-01-03", "close"] == pytest.approx(27.88)


def test_arrangement_needs_a_material_gap():
    rows = action_rows(("X", "2025-01-06", "Scheme Of Arrangement"))
    small = bars(ITC_DATES, [100.0, 100.0, 100.0, 98.5, 98.0], opens=[100.0, 100.0, 100.0, 98.0, 98.0])
    assert [(f.applied, f.method) for f in compute_factors("X", small, rows)] == [(False, "no_gap")]
    spin_off = bars(ITC_DATES, [100.0, 100.0, 100.0, 61.0, 62.0], opens=[100.0, 100.0, 100.0, 60.0, 61.0])
    assert [(f.kind, f.applied, f.factor) for f in compute_factors("X", spin_off, rows)] == [("arrangement", True, 0.6)]


def test_history_break_drops_bars_from_a_different_security():
    # PRIVISCL: bars before the 2020 listing trade at Rs 0.15 and belong to something else.
    stored = bars(ITC_DATES, [0.15, 0.15, 0.15, 553.35, 563.7], opens=[0.15, 0.15, 0.15, 527.0, 555.0])
    [f] = compute_factors("PRIVISCL", stored, action_rows(("PRIVISCL", "2025-01-06", "Demerger")))
    assert (f.kind, f.method, f.applied) == ("history_break", "truncate", True)
    adjusted = adjust_bars(stored, factor_frame([f]))
    assert adjusted["date"].min() == pd.Timestamp("2025-01-06") and len(adjusted) == 2
    assert len(adjust_bars(stored, factor_frame([f]), "none")) == 5  # raw view keeps everything


def test_rights_already_adjusted_by_the_vendor_are_not_applied_twice():
    rows = action_rows(("X", "2025-01-06", "Rights 1:1 @ Premium Rs 40", 10.0))  # issue 50 vs close 140: TERP 0.68
    raw = bars(ITC_DATES, [140.0, 139.0, 138.95, 95.0, 96.0], opens=[140.0, 139.0, 139.0, 94.5, 95.0])
    [f] = compute_factors("X", raw, rows)
    assert (f.applied, f.method) == (True, "terp") and f.factor == pytest.approx((138.95 + 50) / 2 / 138.95)
    vendor_adjusted = bars(ITC_DATES, [95.0, 94.5, 94.5, 96.0, 97.0], opens=[95.0, 94.5, 94.5, 95.5, 96.0])
    [g] = compute_factors("X", vendor_adjusted, rows)
    assert (g.applied, g.method) == (False, "verified_upstream")


def test_vendor_adjusting_only_one_of_two_same_day_events():
    # MINDACORP 2015: bonus 1:1 plus split 10 -> 2 on one day; the vendor applied only the split.
    rows = action_rows(("M", "2025-06-16", "Bonus 1:1 / Face Value Split From Rs 10/- Per Share To Rs 2/- Per Share"))
    stored = bars(DATES, [180.0, 182.0, 182.25, 95.7, 100.4], opens=[180.0, 182.0, 182.0, 95.65, 100.45])
    by_kind = {f.kind: f for f in compute_factors("M", stored, rows)}
    assert (by_kind["bonus"].applied, by_kind["bonus"].method) == (True, "vendor_unadjusted")
    assert (by_kind["split"].applied, by_kind["split"].method) == (False, "verified_upstream")
    assert adjust_bars(stored, factor_frame(list(by_kind.values())))["close"].iloc[2] == pytest.approx(91.125)


def test_stale_ex_date_open_falls_back_to_the_close():
    # ADANIENT 2015: the vendor's ex-date open 573 is stale; the stock traded near 110 all day.
    stored = bars(ITC_DATES, [633.0, 637.0, 637.0, 109.75, 116.55], opens=[709.0, 638.0, 638.0, 573.29, 112.0])
    factors = compute_factors("ADANIENT", stored, action_rows(("ADANIENT", "2025-01-06", "Scheme Of Arrangement")))
    by_kind = {f.kind: f for f in factors}
    f = by_kind["arrangement"]
    assert f.applied and f.factor == pytest.approx(109.75 / 637.0) and "stale" in f.detail
    assert by_kind["stale_open"].applied  # and the stale print itself is repaired
    ex_day = adjust_bars(stored, factor_frame(factors)).set_index("date").loc["2025-01-06"]
    assert ex_day["open"] == ex_day["close"] == ex_day["high"] == pytest.approx(109.75)  # no fake 81% intraday fall


def test_vendor_placeholder_history_becomes_a_history_break():
    # PRIVISCL: the vendor stores a flat 0.15 for five years before a 2020 scheme; NSE has real prices.
    stored = bars(ITC_DATES, [0.15, 0.15, 0.15, 553.35, 563.7], opens=[0.15, 0.15, 0.15, 527.0, 578.0])
    off = official(ITC_DATES, [700.5, 684.8, 684.8, 553.35, 563.7], opens=[709.0, 686.95, 686.95, 527.0, 578.0])
    [f] = compute_factors("PRIVISCL", stored, action_rows(("PRIVISCL", "2025-01-06", "Demerger")), off)
    assert (f.kind, f.applied, f.method) == ("history_break", True, "truncate")
    assert adjust_bars(stored, factor_frame([f]))["date"].min() == pd.Timestamp("2025-01-06")


# Official-evidence path: vendor adjustments measured against NSE's own prices ------------------
def official(dates, closes, opens=None):
    opens = list(closes) if opens is None else list(opens)
    return pd.DataFrame({"open": opens, "close": list(closes)}, index=pd.to_datetime(list(dates)))


D5 = ["2024-02-15", "2024-02-16", "2024-02-20", "2024-02-21", "2024-02-22"]


def test_dividend_removed_by_the_vendor_is_detected_and_restored():
    # HEROMOTOCO 2024: the vendor subtracted the Rs 100 dividend from every earlier price.
    raw = [4700.0, 4720.0, 4750.0, 4660.0, 4680.0]
    stored = bars(D5, [c - 100 if i < 3 else c for i, c in enumerate(raw)])
    off = official(D5, raw)
    [f] = compute_factors("HERO", stored, action_rows(("HERO", "2024-02-21", "Interim Dividend - Rs 75 Per Share/ Special Dividend - Rs 25 Per Share")), off)
    assert (f.method, f.evidence, f.restore_amount) == ("vendor_removed", "official", pytest.approx(100.0))
    assert f.factor == pytest.approx(1 - 100 / 4750)
    frame = factor_frame([f]).assign(restore_amount=[f.restore_amount])
    economic = adjust_bars(stored, frame, "economic", as_traded=off["close"])
    assert economic["close"].tolist() == pytest.approx(raw)  # back to the prices that actually traded
    assert economic["unadj_close"].tolist() == pytest.approx(raw)
    total = adjust_bars(stored, frame, "total_return", as_traded=off["close"])
    assert total["close"].iloc[2] == pytest.approx(4750 * (1 - 100 / 4750))


def test_dividend_left_in_the_prices_is_not_restored():
    raw = [470.0, 472.0, 475.0, 466.0, 468.0]
    [f] = compute_factors("X", bars(D5, raw), action_rows(("X", "2024-02-21", "Dividend - Rs 10 Per Share")), official(D5, raw))
    assert (f.method, f.restore_amount, f.applied) == ("cash", 0.0, True)


def test_split_measured_against_official_prices():
    raw = [1860.0, 1850.0, 1866.0, 938.0, 923.0]
    rows = action_rows(("X", "2024-02-21", "Bonus 1:1"))
    adjusted = bars(D5, [930.0, 925.0, 933.0, 938.0, 923.0])
    [f] = compute_factors("X", adjusted, rows, official(D5, raw))
    assert (f.method, f.applied, f.evidence) == ("verified_upstream", False, "official") and f.vendor_factor == pytest.approx(0.5, rel=1e-3)
    [g] = compute_factors("X", bars(D5, raw), rows, official(D5, raw))  # vendor never adjusted it
    assert (g.method, g.applied) == ("vendor_unadjusted", True)


def test_rights_measured_against_official_prices():
    rows = action_rows(("X", "2024-02-21", "Rights 1:1 @ Premium Rs 40", 10.0))  # issue 50 vs 140
    raw = [139.0, 139.5, 140.0, 95.0, 96.0]
    terp = (140.0 + 50.0) / 2 / 140.0
    [unadjusted] = compute_factors("X", bars(D5, raw), rows, official(D5, raw))
    assert (unadjusted.method, unadjusted.applied, unadjusted.factor) == ("terp", True, pytest.approx(terp))
    vendor_adjusted = bars(D5, [c * terp if i < 3 else c for i, c in enumerate(raw)])
    [upstream] = compute_factors("X", vendor_adjusted, rows, official(D5, raw))
    assert (upstream.method, upstream.applied) == ("verified_upstream", False)


def test_partial_vendor_demerger_adjustment_is_completed_exactly():
    # RELIANCE 2023: official prev close 2841.85, discovery open 2580; vendor scaled history by 0.9532
    # and later halved everything for a 1:1 bonus.
    dates = ["2023-07-17", "2023-07-18", "2023-07-19", "2023-07-20", "2023-07-21"]
    off = official(dates, [2800.0, 2815.0, 2841.85, 2619.85, 2540.0], opens=[2790.0, 2805.0, 2820.0, 2580.0, 2600.0])
    stored = bars(dates, [c / 2 * 0.9532 for c in (2800.0, 2815.0, 2841.85)] + [1309.925, 1270.0],
                  opens=[o / 2 * 0.9532 for o in (2790.0, 2805.0, 2820.0)] + [1290.0, 1300.0])
    [f] = compute_factors("RELIANCE", stored, action_rows(("RELIANCE", "2023-07-20", "Demerger")), off)
    assert (f.method, f.evidence) == ("official_correction", "official")
    assert f.vendor_factor == pytest.approx(0.9532, rel=1e-4)
    assert f.factor * f.vendor_factor == pytest.approx(2580 / 2841.85, rel=1e-4)  # vendor + ours = true factor
    adjusted = adjust_bars(stored, factor_frame([f]))
    assert adjusted["close"].iloc[2] == pytest.approx(adjusted["open"].iloc[3], rel=1e-4)  # continuous at the open


def test_over_adjusted_vendor_demerger_is_corrected_upwards_exactly():
    off = official(D5, [99.0, 99.5, 100.0, 72.0, 73.0], opens=[99.0, 99.5, 100.0, 70.0, 72.5])
    stored = bars(D5, [c * 0.5 * 0.45 for c in (99.0, 99.5, 100.0)] + [36.0, 36.5],
                  opens=[o * 0.5 * 0.45 for o in (99.0, 99.5, 100.0)] + [35.0, 36.25])
    [f] = compute_factors("NMDC", stored, action_rows(("NMDC", "2024-02-21", "Demerger")), off)
    assert f.vendor_factor == pytest.approx(0.45) and f.factor == pytest.approx(0.70 / 0.45)


def test_missing_official_rows_fall_back_to_price_gaps():
    stored = bars(D5, [930.0, 925.0, 933.0, 938.0, 923.0])
    [f] = compute_factors("X", stored, action_rows(("X", "2024-02-21", "Bonus 1:1")), official(D5[:2], [930.0, 925.0]))
    assert (f.evidence, f.method) == ("gaps", "verified_upstream")


# Regressions from the audit (2026-10-20) ------------------------------------------------------
@pytest.mark.parametrize("subject, cash", [
    ("Dividend - Rs 19/ Special Dividend - Rs 3 Per Share", 22.0),  # CDSL 2024
    ("Annual General Meeting/ Final Dividend - Rs 7/- + Special Dividend -Rs 3/- Per Share", 10.0),  # CRISIL 2016
    ("Dividend - Rs 20 And Special Dividend - Rs 20 Per Share", 40.0),  # GLAXO 2020
    ("Interim Dividend Rs 11/- + Special Dividend Rs 7.50 Per Share", 18.5),  # NAVINFLUOR 2016
    ("Final Dividend Rs 6/- Plus Special Dividend Rs 4/- Per Share", 10.0),  # SIEMENS 2016
    ("Annual General Meeting/ Dividend - Rs 29.50/- Per Share And Bonus 1:1", 29.5),  # INFY 2015
    ("Dividend Of 150%", None),  # a percentage of face value, not rupees
])
def test_multi_amount_dividend_subjects(subject, cash):
    found = [p.cash for p in parse_subject(subject) if p.kind == "dividend"]
    assert found == ([] if cash is None else [pytest.approx(cash)])


def test_split_ratio_comes_from_the_split_clause_not_an_earlier_dividend():
    # WELSPUNLIV 2016-03-21
    parsed = parse_subject("Interim Div - Rs 6/- Per Share + Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share")
    assert parsed == [ParsedAction("split", ratio_old=10, ratio_new=1), ParsedAction("dividend", cash=6)]


def test_duplicate_dividend_rows_count_once():
    # POLYMED 2017-09-20 is listed twice in different words.
    rows = action_rows(("P", "2025-06-16", "Annual General Meeting/Dividend - Re 0.50 Per Share"),
                       ("P", "2025-06-16", "Dividend- Re 0.50 Per Share"))
    [f] = compute_factors("P", bars(DATES, [100.0] * 5), rows)
    assert f.factor == pytest.approx((100.0 - 0.5) / 100.0)


def test_dividend_with_a_same_day_bonus_uses_the_pre_bonus_share_basis():
    # INFY 2015-06-15: Rs 29.50 per pre-bonus share, ex-date also the 1:1 bonus (already in vendor data).
    stored = bars(DATES, [1000.0, 1001.0, 1000.0, 990.0, 995.0])
    rows = action_rows(("INFY", "2025-06-16", "Annual General Meeting/ Dividend - Rs 29.50/- Per Share And Bonus 1:1"))
    by_kind = {f.kind: f for f in compute_factors("INFY", stored, rows)}
    assert by_kind["bonus"].method == "verified_upstream"
    assert 1 - by_kind["dividend"].factor == pytest.approx(29.5 * 0.5 / 1000.0)  # 1.475%, not 2.95%


def test_same_day_bonus_does_not_make_a_dividend_look_vendor_removed():
    raw = [2000.0, 2002.0, 2000.0, 985.0, 990.0]  # official: bonus halves the price, dividend of Rs 30 visible
    stored = bars(DATES, [1000.0, 1001.0, 1000.0, 985.0, 990.0])
    off = pd.DataFrame({"open": raw, "close": raw}, index=pd.to_datetime(DATES))
    rows = action_rows(("X", "2025-06-16", "Dividend - Rs 30 Per Share And Bonus 1:1"))
    by_kind = {f.kind: f for f in compute_factors("X", stored, rows, off)}
    assert by_kind["bonus"].method == "verified_upstream"
    assert (by_kind["dividend"].method, by_kind["dividend"].restore_amount) == ("cash", 0.0)


def test_demerger_and_arrangement_rows_on_one_date_apply_once():
    rows = action_rows(("X", "2025-01-06", "Demerger"), ("X", "2025-01-06", "Scheme Of Arrangement"))
    spin_off = bars(ITC_DATES, [100.0, 100.0, 100.0, 61.0, 62.0], opens=[100.0, 100.0, 100.0, 60.0, 61.0])
    assert [(f.kind, f.factor) for f in compute_factors("X", spin_off, rows)] == [("demerger", 0.6)]


def test_rights_premium_uses_the_face_value_at_the_time():
    # TATASTEEL 2018: premium Rs 500 on a Rs 10 share; the feed now reports face value Re 1 after a 2024 split.
    dates = ["2018-01-29", "2018-01-30", "2018-01-31", "2018-02-01", "2024-07-26", "2024-07-29", "2024-07-30"]
    stored = bars(dates, [74.0, 74.5, 72.0, 72.5, 160.0, 161.0, 162.0])
    rows = action_rows(("TATASTEEL", "2018-01-31", "Rights 4:25 @ Premium Rs 500", 1.0),
                       ("TATASTEEL", "2024-07-29", "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 1.0))
    rights = next(f for f in compute_factors("TATASTEEL", stored, rows) if f.kind == "rights")
    assert "at 51.00" in rights.detail  # (500 + 10) in today's 1:10 basis, not (500 + 1)


def test_genuine_rally_on_an_arrangement_date_is_not_mistaken_for_a_stale_open():
    rows = action_rows(("X", "2025-01-06", "Scheme Of Arrangement"))
    rally = bars(ITC_DATES, [100.0, 100.0, 100.0, 112.0, 113.0], opens=[100.0, 100.0, 100.0, 100.5, 112.0])
    assert [(f.applied, f.method) for f in compute_factors("X", rally, rows)] == [(False, "no_gap")]


def test_traded_value_comes_from_official_turnover():
    stored = bars(DATES, [100.0] * 5, volume=10)  # vendor volume left unadjusted after a split
    official = pd.DataFrame({"close": [100.0] * 5, "turnover": [1e7] * 5}, index=pd.to_datetime(DATES))
    out = adjust_bars(stored, factor_frame([]), "economic", as_traded=official)
    assert out["traded_value"].tolist() == [1e7] * 5


def ohlcv(dates, closes, opens=None, volume=1000.0):
    frame = bars(dates, closes, opens, volume).set_index("date")[["open", "high", "low", "close", "volume"]]
    return frame.assign(turnover=frame["close"] * frame["volume"])


def test_official_prices_replace_rounded_vendor_prices_on_the_same_basis():
    # TATAINVEST 2016: after a later 1:10 split the vendor stores 494.45 as 49.0 (rounded).
    exact = [494.45, 499.65, 497.95, 496.50, 498.10]
    stored = bars(DATES, [round(c / 10) for c in exact], volume=10_000)
    official = ohlcv(DATES, exact, volume=1000.0)
    out = adjust_bars(stored, factor_frame([]), "economic", as_traded=official)
    basis = pd.Series([round(c / 10) / c for c in exact]).median()
    assert out["close"].tolist() == pytest.approx([c * basis for c in exact])  # official moves, vendor level
    assert out["close"].pct_change().iloc[1:].tolist() == pytest.approx(pd.Series(exact).pct_change().iloc[1:].tolist())
    assert out["volume"].tolist() == [10_000] * 5 and out["unadj_close"].tolist() == exact
    raw = adjust_bars(stored, factor_frame([]), "economic", as_traded=official, anchor=False)
    assert raw["close"].tolist() == stored["close"].tolist()  # the vendor's own view stays available


def test_anchoring_keeps_each_stretch_between_events_on_its_own_basis():
    # A 1:2 split the vendor applied: stored prices halve before the ex-date, official ones do not.
    exact = [1000.0, 1010.0, 1020.0, 505.0, 510.0]
    stored = bars(DATES, [c / 2 if i < 3 else c for i, c in enumerate(exact)])
    official = ohlcv(DATES, exact)
    factors = compute_factors("X", stored, action_rows(("X", DATES[3], "Face Value Split From Rs 10 To Rs 5")),
                              official[["open", "close"]])
    out = adjust_bars(stored, factor_frame(factors), "economic", as_traded=official)
    assert out["close"].tolist() == pytest.approx([500.0, 505.0, 510.0, 505.0, 510.0])


LONG = list(pd.bdate_range("2025-06-02", periods=14))


def test_unexplained_vendor_rebasing_does_not_create_a_fake_gap():
    # COHANCE 2020-03-24: the vendor's prices halve and stay halved; NSE's do not move.
    exact = [200.0 + i for i in range(14)]
    stored = bars(LONG, [c if i < 7 else c / 2 for i, c in enumerate(exact)])
    out = adjust_bars(stored, factor_frame([]), "economic", as_traded=ohlcv(LONG, exact))
    assert out["close"].tolist() == pytest.approx([c / 2 for c in exact])  # official moves on the latest basis
    assert out["close"].pct_change().abs().max() < 0.01


def test_one_bad_vendor_day_is_replaced_by_the_official_bar():
    # EMAMILTD 2023-05-26: one vendor close 2.7% off NSE's; it is not a change of basis.
    exact = [390.0 + i for i in range(14)]
    vendor = list(exact)
    vendor[6] = exact[6] * 0.97
    out = adjust_bars(bars(LONG, vendor), factor_frame([]), "economic", as_traded=ohlcv(LONG, exact))
    assert out["close"].tolist() == pytest.approx(exact)


def test_rounding_noise_is_not_mistaken_for_a_change_of_basis():
    # CUPID 2019: prices stored at 1/100 scale rounded to 0.05, so the stored/official ratio swings ~4%.
    exact = [114.6, 115.15, 117.35, 118.5, 123.1, 122.1, 120.5, 118.55, 116.2, 118.0, 115.45, 114.25, 113.05, 112.25]
    stored = bars(LONG, [round(c / 100 / 0.05) * 0.05 + 0.0004 for c in exact])
    out = adjust_bars(stored, factor_frame([]), "economic", as_traded=ohlcv(LONG, exact))
    np.testing.assert_allclose(out["close"].pct_change().iloc[1:], pd.Series(exact).pct_change().iloc[1:], rtol=1e-9)


def test_days_without_an_official_bar_keep_the_vendor_values():
    exact = [494.45, 499.65, 497.95, 496.50, 498.10]
    stored = bars(DATES, [round(c / 10) for c in exact])
    official = ohlcv(DATES, exact).drop(pd.Timestamp(DATES[2]))
    out = adjust_bars(stored, factor_frame([]), "economic", as_traded=official)
    assert out["close"].iloc[2] == stored["close"].iloc[2] and out["close"].iloc[1] != stored["close"].iloc[1]
