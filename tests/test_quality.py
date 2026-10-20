import numpy as np
import pandas as pd

from trader.data.providers.base import normalize_bars
from trader.data.quality import run_checks


def make_bars(n=80, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-05-04", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return normalize_bars(pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": np.maximum(open_, close) * 1.005,
        "low": np.minimum(open_, close) * 0.995,
        "close": close,
        "volume": rng.integers(10_000, 50_000, n),
    }))


def checks(findings):
    return {(f.check, f.severity) for f in findings}


def test_clean_series_has_no_findings():
    bars = make_bars()
    assert run_checks("X", bars, reference_dates=pd.DatetimeIndex(bars["date"])) == []


def test_duplicate_dates_and_ohlc_violations_are_errors():
    bars = make_bars()
    broken = pd.concat([bars, bars.iloc[[10]]]).sort_values("date", kind="stable").reset_index(drop=True)
    broken.loc[20, "high"] = broken.loc[20, "close"] * 0.9
    found = checks(run_checks("X", broken))
    assert ("duplicate_dates", "error") in found and ("ohlc_inconsistent", "error") in found


def test_unadjusted_bonus_is_flagged_as_split_like():
    bars = make_bars()
    bars.loc[40:, ["open", "high", "low", "close"]] /= 2.0  # a 1:1 bonus that was never adjusted
    assert ("split_like_jump", "error") in checks(run_checks("X", bars))


def test_crash_day_is_a_warning_not_a_split():
    bars = make_bars()
    bars.loc[30:, ["open", "high", "low", "close"]] *= 0.77  # -23% day, like 2020-03-23
    found = checks(run_checks("X", bars))
    assert ("extreme_move", "warn") in found and ("split_like_jump", "error") not in found


def test_missing_sessions_warn_and_recent_gaps_are_errors():
    bars = make_bars()
    ref = pd.DatetimeIndex(bars["date"])
    old_gaps = bars.drop(index=[5, 6]).reset_index(drop=True)
    assert ("missing_sessions", "warn") in checks(run_checks("X", old_gaps, reference_dates=ref))
    recent_gaps = bars.drop(index=[70, 71, 72, 73]).reset_index(drop=True)
    assert ("missing_sessions", "error") in checks(run_checks("X", recent_gaps, reference_dates=ref))


def test_stale_and_extra_sessions_are_reported():
    bars = make_bars()
    ref = pd.DatetimeIndex(bars["date"][:-1])  # reference lacks the last day
    stale = bars.iloc[:-5].reset_index(drop=True)
    assert ("stale", "warn") in checks(run_checks("X", stale, reference_dates=ref))
    with_holiday = make_bars()
    ref_without_day = pd.DatetimeIndex(with_holiday["date"]).delete(30)
    assert ("extra_sessions", "warn") in checks(run_checks("X", with_holiday, reference_dates=ref_without_day))


def test_zero_volume_is_ignored_for_indices():
    bars = make_bars().assign(volume=0)
    assert ("zero_volume", "warn") in checks(run_checks("X", bars))
    assert ("zero_volume", "warn") not in checks(run_checks("Nifty 50", bars, expect_volume=False))


def test_empty_frame_is_an_error():
    assert checks(run_checks("X", make_bars().iloc[0:0])) == {("empty", "error")}


def activity_frame(ref, present):
    return pd.DataFrame({name: pd.Series(1e8, index=dates) for name, dates in present.items()}).reindex(ref)


def test_sparse_sessions_are_market_wide_vendor_gaps():
    from trader.data.quality import session_flags

    ref = pd.bdate_range("2026-01-05", periods=10)
    special = ref[4]
    dates = {f"S{i}": ref.delete(4) for i in range(8)}  # 8 of 10 stocks lack the special session
    dates["S8"] = ref
    dates["S9"] = ref
    dates["LATE"] = ref[6:]  # listed after the special session: not counted against it
    flags = session_flags(activity_frame(ref, dates))
    assert list(flags.index[flags["sparse"]]) == [special]


def test_sparse_session_is_flagged_even_when_it_is_the_last_day_loaded():
    """Judged against the previous session, so cutting the data at the special day changes nothing."""
    from trader.data.quality import session_flags

    ref = pd.bdate_range("2026-01-05", periods=10)
    dates = {f"S{i}": ref.delete(4) for i in range(8)} | {"S8": ref, "S9": ref}
    cut = activity_frame(ref[:5], {k: v[v <= ref[4]] for k, v in dates.items()})
    assert bool(session_flags(cut)["sparse"].iloc[-1])


def test_short_sessions_are_detected_from_market_wide_activity():
    from trader.data.quality import session_flags

    rng = np.random.default_rng(0)
    ref = pd.bdate_range("2026-01-05", periods=60)
    values = np.exp(rng.normal(18, 0.3, (60, 80)))
    values[40] *= 0.15  # a one-hour Muhurat session: everyone trades about 15% of normal
    values[50] *= 0.6  # a quiet day before a holiday is still a session
    flags = session_flags(pd.DataFrame(values, index=ref))
    assert list(flags.index[flags["short"]]) == [ref[40]]
    assert flags["activity"].iloc[:10].isna().all()  # no baseline yet: not judged
    few = session_flags(pd.DataFrame(values[:, :5], index=ref))  # too few stocks to judge
    assert not few["short"].any()


def test_genuine_crash_is_not_mistaken_for_a_split():
    # RPOWER 2019-02-04: opened 5% down, fell 35% during the day on 11x volume.
    bars = make_bars()
    prev = bars.loc[39, "close"]
    bars.loc[40, ["open", "high", "low", "close"]] = [prev * 0.945, prev * 0.95, prev * 0.63, prev * 0.6515]
    bars.loc[41:, ["open", "high", "low", "close"]] *= 0.6515 / (bars.loc[41, "close"] / prev)
    assert ("split_like_jump", "error") not in checks(run_checks("X", bars))


def test_pre_listing_fragments_and_zero_prices_are_trimmed():
    from trader.data.quality import trim_fragments

    listed = make_bars(20)
    stray = pd.DataFrame({"date": pd.to_datetime(["2017-12-04", "2018-01-01"]), "open": [0.0, 0.0], "high": [0.0, 0.0],
                          "low": [0.0, 0.0], "close": [0.0, 0.0], "volume": [7154666, 670998]})
    old_trades = normalize_bars(pd.DataFrame({"date": pd.to_datetime(["2016-01-18", "2016-02-01"]), "open": [11.2, 10.65],
                                              "high": [11.2, 10.65], "low": [11.2, 10.65], "close": [11.2, 10.65], "volume": [200, 150]}))
    combined = normalize_bars(pd.concat([old_trades, stray, listed]))
    trimmed, cut = trim_fragments(combined)
    assert trimmed["date"].tolist() == listed["date"].tolist() and cut == listed["date"].iloc[0]
    untouched, no_cut = trim_fragments(listed)
    assert no_cut is None and untouched.equals(listed)
    assert ("long_gap", "warn") in checks(run_checks("X", combined[combined["close"] > 0]))


def test_negative_volume_is_flagged_and_overflow_repaired():
    from trader.data.quality import repair_volume

    bars = make_bars(5)
    bars.loc[3, "volume"] = -81259413
    assert ("negative_volume", "error") in checks(run_checks("IDEA", bars))
    fixed, n = repair_volume(bars)
    assert n == 1 and fixed.loc[3, "volume"] == 4213707883
