import numpy as np
import pandas as pd
import pytest

from trader.features import indicators as ind


def frame(values, name="X"):
    return pd.DataFrame({name: np.asarray(values, dtype=float)}, index=pd.bdate_range("2025-01-01", periods=len(values)))


def ref_seeded(x, n, alpha):
    """Reference recursion: y = mean(x[:n]) on the n-th value, then y_t = (1 - alpha) y_{t-1} + alpha x_t."""
    out = np.full(len(x), np.nan)
    out[n - 1] = np.mean(x[:n])
    for t in range(n, len(x)):
        out[t] = (1 - alpha) * out[t - 1] + alpha * x[t]
    return out


@pytest.fixture
def prices():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    high = close * (1 + np.abs(rng.normal(0, 0.01, 300)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, 300)))
    return frame(high), frame(low), frame(close)


def test_rsi_matches_the_wilder_recursion(prices):
    _, _, c = prices
    delta = np.diff(c["X"].to_numpy())
    gain, loss = ref_seeded(np.clip(delta, 0, None), 14, 1 / 14), ref_seeded(np.clip(-delta, 0, None), 14, 1 / 14)
    got = ind.rsi(c, 14)["X"].to_numpy()
    assert np.isnan(got[:14]).all()  # the first value needs 14 changes
    np.testing.assert_allclose(got[1:], 100 - 100 / (1 + gain / loss), rtol=1e-12)


def test_rsi_edge_cases():
    assert ind.rsi(frame(np.arange(1, 40)), 14)["X"].iloc[-1] == 100.0  # only gains
    assert ind.rsi(frame([5.0] * 40), 14)["X"].iloc[-1] == 50.0  # no movement


def test_atr_uses_true_range_including_gaps(prices):
    h, l, c = prices
    hh, ll, cc = h["X"].to_numpy(), l["X"].to_numpy(), c["X"].to_numpy()
    tr = np.maximum.reduce([hh[1:] - ll[1:], np.abs(hh[1:] - cc[:-1]), np.abs(ll[1:] - cc[:-1])])
    got = ind.atr(h, l, c, 14)["X"].to_numpy()
    assert np.isnan(got[:14]).all()  # no true range on the first bar
    np.testing.assert_allclose(got[1:], ref_seeded(tr, 14, 1 / 14), rtol=1e-12)


def test_ema_sma_and_macd(prices):
    _, _, c = prices
    x = c["X"].to_numpy()
    np.testing.assert_allclose(ind.ema(c, 20)["X"].to_numpy(), ref_seeded(x, 20, 2 / 21), rtol=1e-12)
    assert ind.sma(c, 5)["X"].iloc[4] == pytest.approx(x[:5].mean())
    line, sig, hist = ind.macd(c)
    np.testing.assert_allclose(hist.dropna(), (line - sig).dropna())
    assert line["X"].first_valid_index() == c.index[25] and sig["X"].first_valid_index() == c.index[33]


@pytest.mark.parametrize("lead", [0, 37])  # 37: the stock lists after the first rows of the frame
def test_indicators_equal_ta_lib_from_the_first_value(lead):
    talib = pytest.importorskip("talib")
    rng = np.random.default_rng(5)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))

    def wide(x):
        return frame(np.r_[np.full(lead, np.nan), x])

    h, l, c = wide(high), wide(low), wide(close)
    line, signal, hist = ind.macd(c)
    m_line, m_signal, m_hist = talib.MACD(close, 12, 26, 9)
    pairs = {
        "rsi_14": (ind.rsi(c, 14), talib.RSI(close, 14)), "rsi_2": (ind.rsi(c, 2), talib.RSI(close, 2)),
        "atr_14": (ind.atr(h, l, c, 14), talib.ATR(high, low, close, 14)),
        "adx_14": (ind.adx(h, l, c, 14), talib.ADX(high, low, close, 14)),
        "ema_20": (ind.ema(c, 20), talib.EMA(close, 20)), "ema_50": (ind.ema(c, 50), talib.EMA(close, 50)),
        "sma_200": (ind.sma(c, 200), talib.SMA(close, 200)),
        "macd_signal": (signal, m_signal), "macd_hist": (hist, m_hist),
    }
    for name, (ours, reference) in pairs.items():
        got = ours["X"].to_numpy()[lead:]
        assert np.array_equal(np.isnan(got), np.isnan(reference)), name  # same first value
        np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-10, err_msg=name)
    both = ~np.isnan(m_line)
    np.testing.assert_allclose(line["X"].to_numpy()[lead:][both], m_line[both], rtol=1e-10)  # TA-Lib hides bars 25-32


def test_adx_is_bounded_and_rises_in_a_steady_trend():
    trend = np.linspace(100, 200, 120)
    a = ind.adx(frame(trend * 1.01), frame(trend * 0.99), frame(trend), 14)["X"]
    assert a.dropna().between(0, 100).all() and a.iloc[-1] > 50


def test_donchian_excludes_today():
    high = frame([1, 2, 3, 4, 10, 5])
    assert ind.donchian_high(high, 3)["X"].tolist()[3:] == [3.0, 4.0, 10.0]
    assert np.isnan(ind.donchian_high(high, 3)["X"].iloc[2])


def test_return_windows():
    c = frame(np.arange(1.0, 301.0))
    assert ind.rolling_return(c, 21)["X"].iloc[-1] == pytest.approx(300 / 279 - 1)
    assert ind.momentum_12_1(c)["X"].iloc[-1] == pytest.approx(279 / 48 - 1)  # 21 to 252 sessions back
    assert np.isnan(ind.momentum_12_1(c)["X"].iloc[251])


def test_volatility_relative_strength_and_volume_z():
    c = frame(100 * np.exp(np.cumsum(np.full(80, 0.01))))
    assert ind.realized_vol(c, 20)["X"].iloc[-1] == pytest.approx(0.0, abs=1e-9)  # constant return: zero volatility
    alternating = frame(100 * np.exp(np.cumsum(np.tile([0.01, -0.01], 40))))
    daily = np.std(np.tile([0.01, -0.01], 10), ddof=1)
    assert ind.realized_vol(alternating, 20, 248)["X"].iloc[-1] == pytest.approx(daily * np.sqrt(248))  # annualized
    rising_index = pd.Series(100 * np.exp(np.cumsum(np.full(80, 0.004))), index=c.index)
    rs = ind.relative_strength(c, rising_index, 21)
    assert rs["X"].iloc[-1] == pytest.approx((np.exp(0.21) - 1) - (np.exp(0.084) - 1))  # stock minus index
    base = 1000.0 * np.exp(np.random.default_rng(0).normal(0, 0.1, 30))
    v = frame(np.r_[base, 10 * base[-20:].mean()])
    logs = np.log1p(np.r_[base, 10 * base[-20:].mean()])
    expected = (logs[-1] - logs[-21:-1].mean()) / logs[-21:-1].std(ddof=1)  # baseline = the 20 days before today
    assert ind.volume_zscore(v, 20)["X"].iloc[-1] == pytest.approx(expected) and expected > 4.25
    assert np.isnan(ind.volume_zscore(frame([1000.0] * 30), 20)["X"].iloc[-1])  # no variation, no score


def test_distance_from_high_and_traded_value():
    c, h = frame([10.0, 12.0, 9.0]), frame([10.5, 13.0, 9.5])
    assert ind.dist_from_high(c, h, 3)["X"].iloc[-1] == pytest.approx(9 / 13 - 1)  # measured from the intraday high
    tv = frame([1.0, 1.0, 100.0, 1.0, 1.0])
    assert ind.median_traded_value(tv, 5)["X"].iloc[-1] == 1.0  # one block deal does not move the median


def test_cross_sectional_rank_ignores_missing():
    f = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, np.nan], "C": [3.0, 1.0]})
    ranks = ind.cs_rank(f)
    assert ranks.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert ranks.iloc[1]["A"] == 1.0 and np.isnan(ranks.iloc[1]["B"]) and ranks.iloc[1]["C"] == 0.5
    tied = ind.cs_rank(pd.DataFrame({"A": [5.0], "B": [5.0], "C": [1.0]}))
    assert tied.iloc[0].tolist() == pytest.approx([1.0, 1.0, 1 / 3])  # tied leaders both rank 1.0


def test_adx_directional_movement_rule():
    """Only the larger of the up-move and down-move counts, so a symmetric expansion has no direction."""
    rng = np.random.default_rng(1)
    mid = 100 + np.cumsum(rng.normal(0, 1, 120))
    width = np.abs(rng.normal(1, 0.2, 120))
    symmetric = ind.adx(frame(mid + width), frame(mid - width), frame(mid), 14)["X"]
    up = np.linspace(100, 160, 120)
    trending = ind.adx(frame(up + 1), frame(up - 1), frame(up), 14)["X"]
    assert trending.iloc[-1] > 90 and symmetric.iloc[-1] < trending.iloc[-1] / 2
