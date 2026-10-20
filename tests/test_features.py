import numpy as np
import pytest

from trader.data.synthetic import random_walk_panel
from trader.features import features as features_module
from trader.features.features import (
    FeatureConfig,
    build_features,
    columns_with_gaps,
    lookahead_differences,
)
from trader.market import INDIA_NSE


@pytest.fixture(scope="module")
def panel():
    p = random_walk_panel(25, 650, seed=11)
    for f in p.fields:  # symbol S003 suspended for three sessions mid-history
        p.fields[f].iloc[320:323, 3] = np.nan
    for f in p.fields:  # symbol S005 lists late
        p.fields[f].iloc[:200, 5] = np.nan
    return p


@pytest.fixture(scope="module")
def store(panel):
    return build_features(panel)


def test_expected_features_exist(store):
    expected = {"sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "rsi_2", "rsi_14", "macd", "macd_signal", "macd_hist",
                "atr_14", "atr_pct", "adx_14", "dc_high_20", "dc_high_55", "dc_low_10", "dc_low_20", "ret_21", "ret_63",
                "ret_126", "ret_252", "mom_12_1", "vol_20", "vol_60", "rs_63", "rs_126", "vol_z_20", "dist_52w_high",
                "adv_20", "mom_rank", "ret_21_rank"}
    assert set(store.names) == expected
    assert {"index_close", "index_sma_200", "index_rv_20", "index_ret_21", "index_dd_from_high"} <= set(store.index_features.columns)


def test_no_feature_changes_when_the_future_is_removed(panel, store):
    days = list(panel.dates[[205, 300, 320, 321, 322, 323, 330, 500, 649]])
    assert lookahead_differences(panel, days, full=store) == []


def test_slices_are_independent_copies(panel):
    piece = panel.slice_to(panel.dates[100])
    piece.fields["close"].iloc[:, 0] = -1.0
    assert (panel.close.iloc[:101, 0] > 0).all()


def test_changing_the_future_never_changes_the_past(panel, store):
    cut = panel.dates[400]
    mutated = panel.slice_to(panel.dates[-1])
    for f in mutated.fields:
        mutated.fields[f].loc[mutated.dates > cut] *= 3.0
    other = build_features(mutated)
    for name in store.names:
        a, b = store[name].loc[:cut], other[name].loc[:cut]
        assert a.equals(b), name


def test_live_window_matches_full_history(panel, store):
    assert lookahead_differences(panel, list(panel.dates[[640, 649]]), window=600, full=store) == []


def test_gap_symbols_use_their_own_bars(panel, store):
    assert columns_with_gaps(panel.close) == ["S003"]  # late listing is not a gap
    close = panel.close["S003"].dropna()
    assert store["sma_20"]["S003"].loc[close.index[340]] == pytest.approx(close.iloc[321:341].mean())
    assert store["sma_20"]["S003"].iloc[320:323].isna().all()  # no bar, no value


def test_late_listing_waits_for_enough_bars(panel, store):
    assert store["sma_200"]["S005"].first_valid_index() == panel.dates[399]
    assert store["mom_rank"]["S005"].iloc[:451].isna().all()


def test_view_never_reaches_past_its_day(panel, store):
    view = store.view(panel.dates[300])
    assert view.history("ret_21", "S000", 1000).index[-1] == panel.dates[300]
    assert view.at().loc["S000", "sma_20"] == store["sma_20"].loc[panel.dates[300], "S000"]
    assert view.index()["index_close"] == panel.index["Nifty 50"]["close"].loc[panel.dates[300]]


def test_config_follows_the_market_profile():
    cfg = FeatureConfig.for_profile(INDIA_NSE)
    assert cfg.annualization_days == 248 and cfg.warmup_bars == 253


def test_view_hands_out_copies_and_hides_the_store(panel, store):
    view = store.view(panel.dates[300])
    history = view.history("atr_14", "S000", 5)
    before = store["atr_14"].loc[history.index[-2], "S000"]
    history.iloc[-2] = -999.0
    index_row, cross_section = view.index(), view.at()
    index_row.iloc[0] = -999.0
    cross_section.iloc[0, 0] = -999.0
    assert store["atr_14"].loc[history.index[-2], "S000"] == before
    assert store.index_features.iloc[300, 0] != -999.0 and store.at(panel.dates[300]).iloc[0, 0] != -999.0
    assert not hasattr(view, "store") and "atr_14" in view.names
    with pytest.raises(AttributeError):
        view.extra = 1  # slotted: nothing else can be attached


def test_lookahead_checker_catches_a_planted_leak(panel, monkeypatch):
    honest = features_module.ind.sma

    def leaky(close, n):
        return honest(close, n).shift(-1)  # uses tomorrow's average today

    monkeypatch.setattr(features_module.ind, "sma", leaky)
    problems = lookahead_differences(panel, [panel.dates[400]])
    assert problems and "sma_20" in problems[0]


def test_lookahead_checker_compares_index_features(panel, store, monkeypatch):
    real = features_module.index_feature_frame

    def leaky(index_close, dates, cfg):
        out = real(index_close, dates, cfg)
        out["index_ret_21"] = out["index_ret_21"].shift(-1)
        return out

    monkeypatch.setattr(features_module, "index_feature_frame", leaky)
    problems = lookahead_differences(panel, [panel.dates[400]])
    assert problems and "index_ret_21" in problems[0]


def test_index_features_do_not_depend_on_how_much_history_is_loaded():
    """Drawdown is measured from the 52-week high, so a 600-session live window equals full history
    even in a long bear market whose high is older than the window (auditor's case: -0.712 vs -0.502)."""
    bear = random_walk_panel(10, 1200, seed=4, annual_drift=-0.6)
    full = build_features(bear)
    assert full.index_features["index_dd_from_high"].min() < -0.3
    assert lookahead_differences(bear, list(bear.dates[[900, 1199]]), window=600, full=full) == []


def test_one_missing_index_close_does_not_blank_the_regime_average(panel):
    holed = panel.slice_to(panel.dates[-1])
    holed.index["Nifty 50"].iloc[450, holed.index["Nifty 50"].columns.get_loc("close")] = np.nan
    features = build_features(holed).index_features
    assert np.isnan(features["index_sma_200"].iloc[450])  # no index bar, no value that day
    assert features["index_sma_200"].iloc[451:].notna().all()  # but the next days are fine
