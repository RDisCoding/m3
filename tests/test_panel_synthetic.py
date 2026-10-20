from datetime import date

import numpy as np
import pandas as pd
import pytest
from factories import bars

from trader.core.panel import Panel, PanelError
from trader.data.feed import LocalDataFeed
from trader.data.synthetic import breakout_panel, momentum_panel, random_walk_panel
from trader.settings import Settings
from trader.storage.parquet import BarStore


@pytest.fixture
def feed(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    daily, index = BarStore(settings.bars_dir / "daily"), BarStore(settings.bars_dir / "index")
    special = pd.Timestamp("2025-01-12")  # a Sunday session in the middle of everyone's history
    dates = sorted(list(pd.bdate_range("2025-01-01", periods=12)) + [special])
    for i, symbol in enumerate(["AAA", "BBB", "CCC"]):
        stock_dates = dates if symbol == "AAA" else [d for d in dates if d != special]  # only one stock has it
        daily.upsert(symbol, bars(stock_dates, [100.0 + i + d for d in range(len(stock_dates))]), source="upstox", meta={"name": symbol})
    index.upsert("Nifty 50", bars(dates, [23000.0 + 10 * d for d in range(len(dates))]), source="upstox", meta={"name": "Nifty 50"})
    return LocalDataFeed(settings)


def test_panel_from_feed_drops_market_wide_special_sessions(feed):
    panel = Panel.from_feed(feed)
    assert panel.symbols == ["AAA", "BBB", "CCC"] and len(panel.dates) == 12
    assert pd.Timestamp("2025-01-12") not in panel.dates
    assert list(panel.index) == ["Nifty 50"] and panel.index["Nifty 50"].index.equals(panel.dates)
    panel.validate()
    assert set(panel.fields) >= {"open", "high", "low", "close", "volume", "unadj_close", "traded_value"}


def test_slicing_and_bars(feed):
    panel = Panel.from_feed(feed)
    day = panel.dates[5]
    assert panel.slice_to(day).dates[-1] == day and len(panel.slice_to(day).dates) == 6
    assert len(panel.window(day, 3).dates) == 3 and panel.window(day, 3).dates[0] == panel.dates[3]
    assert len(panel.between(panel.dates[2], panel.dates[4]).dates) == 3
    bar = panel.bar(day, "BBB")
    assert bar.close == panel.close.loc[day, "BBB"] and bar.prev_close == panel.close.loc[panel.dates[4], "BBB"]
    assert panel.bar(date(2025, 1, 4), "BBB") is None  # Saturday


def test_validate_rejects_broken_panels():
    p = random_walk_panel(3, 30, seed=2)
    p.fields["high"].iloc[10, 0] = 0.0
    with pytest.raises(PanelError):
        p.validate()
    with pytest.raises(PanelError):
        Panel({"close": p.close})
    q = random_walk_panel(3, 30, seed=2)
    q.fields["open"].iloc[5, 1] = np.nan  # a close without an open
    with pytest.raises(PanelError, match="missing where a close exists"):
        q.validate()


def test_panel_construction_and_bar_lookups_are_safe():
    p = random_walk_panel(3, 30, seed=2)
    fields = {k: p.fields[k] for k in ("open", "high", "low", "close", "volume")}
    built = Panel(fields)
    assert set(fields) == {"open", "high", "low", "close", "volume"}  # the caller's dict is untouched
    assert built != p and built == built  # identity comparison, never an ambiguous-DataFrame error
    assert built.bar(built.dates[3], "NOPE") is None and built.bar(pd.Timestamp("1999-01-01"), "S000") is None
    built.fields["close"].iloc[10:13, 0] = np.nan  # suspended for three sessions
    for f in ("open", "high", "low"):
        built.fields[f].iloc[10:13, 0] = np.nan
    built._prev_close = None
    bar = built.bar(built.dates[13], "S000")
    assert bar.prev_close == built.close.iloc[9, 0]  # the last close before the gap, not NaN
    assert built.bar(built.dates[0], "S000").prev_close is None


def test_traded_scale_converts_adjusted_prices_to_traded_prices():
    from trader.core.types import Bar

    bar = Bar(date(2015, 1, 1), "BAJFINANCE", 34.5, 35.5, 34.0, 35.0, 1e6, None, 3500.0)
    assert bar.traded_scale == pytest.approx(100.0) and bar.open * bar.traded_scale == pytest.approx(3450.0)
    assert Bar(date(2015, 1, 1), "X", 1, 1, 1, 1, 1).traded_scale == 1.0


def test_synthetic_panels_are_deterministic_and_valid():
    a, b = random_walk_panel(10, 300, seed=5), random_walk_panel(10, 300, seed=5)
    assert a.close.equals(b.close) and not a.close.equals(random_walk_panel(10, 300, seed=6).close)
    for p in (a, momentum_panel(10, 300, seed=1), breakout_panel(10, 300, seed=1)):
        p.validate()


def test_random_walk_has_no_drift():
    p = random_walk_panel(200, 1000, seed=0)
    r = p.close.pct_change().stack()
    assert abs(r.mean()) < 3 * r.std() / np.sqrt(len(r))


def test_random_walk_gaps_do_not_predict_the_rest_of_the_day():
    p = random_walk_panel(200, 1000, seed=0)
    gap = np.log(p.open / p.close.shift(1)).iloc[1:].stack()
    intraday = np.log(p.close / p.open).iloc[1:].stack()
    assert abs(np.corrcoef(gap, intraday)[0, 1]) < 0.02  # was -0.21 when the open was independent noise
    daily = np.log(p.close).diff().iloc[1:].stack()
    assert daily.std() == pytest.approx(0.30 / np.sqrt(252), rel=0.02)  # the close path is unchanged


def test_momentum_panel_has_a_planted_momentum_edge():
    p = momentum_panel(80, 1200, seed=0)
    past = p.close.pct_change(126)
    future = p.close.shift(-21) / p.close - 1
    ic = past.iloc[126:-21:21].rank(axis=1).corrwith(future.iloc[126:-21:21].rank(axis=1), axis=1)  # rank correlation
    assert ic.mean() > 0.1  # past winners keep winning by construction


def test_breakout_panel_has_a_planted_breakout_edge():
    p = breakout_panel(80, 1200, seed=0)
    breakout = p.close > p.close.rolling(55).max().shift(1)
    forward = p.close.shift(-10) / p.close - 1
    assert forward[breakout].stack().mean() > forward.stack().mean() + 0.01
