"""Synthetic markets for testing: one with no edge, and ones with a planted, known edge.

* `random_walk_panel`: geometric Brownian motion. Any strategy that "makes money" here after costs
  is finding look-ahead or a bug, not an edge.
* `momentum_panel`: each stock has a hidden drift that persists for months, so past returns
  predict future returns (cross-sectional momentum exists by construction).
* `breakout_panel`: after a close above the prior 55-day high, the stock drifts up for a while
  (a breakout edge exists by construction).
All are deterministic for a given seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.panel import Panel

TRADING_DAYS = 252


def _ohlcv(close: np.ndarray, rng: np.random.Generator, daily_vol: float, gap_vol: float) -> dict[str, np.ndarray]:
    n_days, n_sym = close.shape
    prev = np.vstack([close[:1], close[:-1]])
    # The open is a point on the path from the previous close to today's close (a Brownian bridge at
    # the overnight share of the day's variance), so the overnight gap and the open-to-close move are
    # independent, as in a real random walk. An open drawn independently of the close would make
    # gaps predict the rest of the day.
    share = min((gap_vol / daily_vol) ** 2, 1.0) if daily_vol > 0 else 0.0
    log_prev, log_close = np.log(prev), np.log(close)
    noise = rng.normal(0.0, np.sqrt(share * (1 - share)) * daily_vol, close.shape)
    open_ = np.exp((1 - share) * log_prev + share * log_close + noise)
    open_[0] = close[0]
    body_hi, body_lo = np.maximum(open_, close), np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0.0, daily_vol / 2, close.shape)))
    low = body_lo * np.exp(-np.abs(rng.normal(0.0, daily_vol / 2, close.shape)))
    volume = np.round(rng.lognormal(12.0, 0.5, close.shape))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _to_panel(arrays: dict[str, np.ndarray], start: str, index_name: str) -> Panel:
    n_days, n_sym = arrays["close"].shape
    dates = pd.bdate_range(start, periods=n_days)
    symbols = [f"S{i:03d}" for i in range(n_sym)]
    fields = {k: pd.DataFrame(v, index=dates, columns=symbols) for k, v in arrays.items()}
    log_ret = np.log(fields["close"]).diff().mean(axis=1).fillna(0.0)
    index_close = 10_000 * np.exp(log_ret.cumsum())
    prev = index_close.shift(1).fillna(index_close.iloc[0])
    index = pd.DataFrame({"open": prev, "high": np.maximum(prev, index_close) * 1.002,
                          "low": np.minimum(prev, index_close) * 0.998, "close": index_close, "volume": 0.0}, index=dates)
    return Panel(fields, {index_name: index})


def random_walk_panel(n_symbols: int = 40, n_days: int = 800, *, seed: int = 0, annual_vol: float = 0.30,
                      annual_drift: float = 0.0, gap_vol: float = 0.004, start: str = "2018-01-01",
                      index_name: str = "Nifty 50") -> Panel:
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS)
    mu = annual_drift / TRADING_DAYS - 0.5 * daily_vol**2  # zero expected arithmetic return when drift = 0
    start_prices = rng.uniform(50, 1500, n_symbols)
    close = start_prices * np.exp(np.cumsum(rng.normal(mu, daily_vol, (n_days, n_symbols)), axis=0))
    return _to_panel(_ohlcv(close, rng, daily_vol, gap_vol), start, index_name)


def momentum_panel(n_symbols: int = 60, n_days: int = 1000, *, seed: int = 0, annual_vol: float = 0.30,
                   drift_spread: float = 0.40, switch_prob: float = 1 / 250, gap_vol: float = 0.004,
                   start: str = "2018-01-01", index_name: str = "Nifty 50") -> Panel:
    """Hidden per-stock annual drifts ~ N(0, drift_spread) that switch with `switch_prob` per day."""
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS)
    drift = rng.normal(0.0, drift_spread, n_symbols)
    log_ret = np.empty((n_days, n_symbols))
    for t in range(n_days):
        switch = rng.random(n_symbols) < switch_prob
        drift = np.where(switch, rng.normal(0.0, drift_spread, n_symbols), drift)
        log_ret[t] = rng.normal(drift / TRADING_DAYS - 0.5 * daily_vol**2, daily_vol)
    close = rng.uniform(50, 1500, n_symbols) * np.exp(np.cumsum(log_ret, axis=0))
    return _to_panel(_ohlcv(close, rng, daily_vol, gap_vol), start, index_name)


def breakout_panel(n_symbols: int = 60, n_days: int = 1000, *, seed: int = 0, annual_vol: float = 0.30,
                   channel: int = 55, boost: float = 0.004, hold: int = 20, gap_vol: float = 0.004,
                   start: str = "2018-01-01", index_name: str = "Nifty 50") -> Panel:
    """After a close above the prior `channel`-day high, add `boost` log return per day for `hold` days."""
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS)
    mu = -0.5 * daily_vol**2
    close = np.empty((n_days, n_symbols))
    close[0] = rng.uniform(50, 1500, n_symbols)
    active = np.zeros(n_symbols, dtype=int)
    for t in range(1, n_days):
        step = rng.normal(mu, daily_vol, n_symbols) + np.where(active > 0, boost, 0.0)
        close[t] = close[t - 1] * np.exp(step)
        active = np.maximum(active - 1, 0)
        if t > channel:
            broke = close[t] > close[t - channel:t].max(axis=0)
            active = np.where(broke & (active == 0), hold, active)
    return _to_panel(_ohlcv(close, rng, daily_vol, gap_vol), start, index_name)
