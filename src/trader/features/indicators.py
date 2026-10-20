"""Technical indicators on wide frames (rows = sessions, columns = symbols).

Every function uses only data up to and including each row, so values on day t never depend on
later days. Windows count a symbol's own bars. RSI, ATR and ADX use Wilder smoothing (an exponential
average with alpha = 1/n), computed with pandas for all symbols at once. Exponential averages are
seeded the way TA-Lib seeds them (the first value is the plain mean of the first n values), so the
results equal TA-Lib's from each stock's first value on, not only after a long warm-up (tested).
Seeds assume a column has no missing values inside its history; the feature store recomputes
stocks with such gaps on their own bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.rolling(n, min_periods=n).mean()


def seeded_ewm(values: pd.DataFrame, n: int, alpha: float, *, seed_at: int | None = None) -> pd.DataFrame:
    """Exponential average whose first value is the mean of the n values ending at a column's
    `seed_at`-th value (default n, i.e. its first n values); y_t = y_{t-1} + alpha (x_t - y_{t-1}) after.
    If those n values are not consecutive rows (a gap), the average starts from the next value instead."""
    seed_at = seed_at or n
    x = values.to_numpy(dtype="float64")
    valid = ~np.isnan(x)
    count = np.cumsum(valid, axis=0)
    start = np.where(count > seed_at, x, np.nan)
    rows, cols = np.nonzero(valid & (count == seed_at))
    for r, c in zip(rows, cols, strict=True):  # one seed per column
        window = x[r - n + 1: r + 1, c] if r + 1 >= n else x[:0, c]
        if len(window) == n and not np.isnan(window).any():
            start[r, c] = window.mean()
    out = pd.DataFrame(start, index=values.index, columns=values.columns).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    return out.where(count >= seed_at)


def ema(close: pd.DataFrame, n: int, *, seed_at: int | None = None) -> pd.DataFrame:
    return seeded_ewm(close, n, 2.0 / (n + 1), seed_at=seed_at)


def wilder(values: pd.DataFrame, n: int) -> pd.DataFrame:
    return seeded_ewm(values, n, 1.0 / n)


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0.0), n)
    avg_loss = wilder((-delta).clip(lower=0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Undefined on a stock's first bar (no previous close), as in TA-Lib."""
    prev = close.shift(1)
    ranges = np.stack([(high - low).to_numpy(), (high - prev).abs().to_numpy(), (low - prev).abs().to_numpy()])
    out = pd.DataFrame(np.fmax.reduce(ranges, axis=0), index=close.index, columns=close.columns)
    return out.where(prev.notna())


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's ADX with TA-Lib's seeding: directional movement and true range are summed over bars
    1..n-1 (the mean of the first n bars with bar 0 counted as zero), DX starts on bar n, and the
    first ADX (bar 2n-1) is the mean of the first n DX values."""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    bars = high.notna().cumsum().where(high.notna())
    first = bars == 1
    tr = wilder(true_range(high, low, close).mask(first, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder(plus_dm.mask(first, 0.0), n) / tr
        minus_di = 100.0 * wilder(minus_dm.mask(first, 0.0), n) / tr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder(dx.replace([np.inf, -np.inf], np.nan).where(bars > n), n)


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MACD as TA-Lib computes it: the fast average is seeded on the same bar as the slow one."""
    line = ema(close, fast, seed_at=slow) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def donchian_high(high: pd.DataFrame, n: int) -> pd.DataFrame:
    """Highest high of the n sessions *before* today (today excluded, so a breakout is detectable)."""
    return high.rolling(n, min_periods=n).max().shift(1)


def donchian_low(low: pd.DataFrame, n: int) -> pd.DataFrame:
    return low.rolling(n, min_periods=n).min().shift(1)


def rolling_return(close: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    return close / close.shift(n) - 1.0


def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Return from `lookback` sessions ago to `skip` sessions ago (skips the latest month's reversal)."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def realized_vol(close: pd.DataFrame | pd.Series, n: int, periods_per_year: int = 252) -> pd.DataFrame | pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std() * np.sqrt(periods_per_year)


def relative_strength(close: pd.DataFrame, index_close: pd.Series, n: int) -> pd.DataFrame:
    return rolling_return(close, n).sub(rolling_return(index_close, n), axis=0)


def volume_zscore(activity: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """How unusual today's activity is against the previous n sessions (today is not in its own
    baseline), in standard deviations of log activity. Fed with traded value, which splits do not change."""
    logv = np.log1p(activity.clip(lower=0))
    prior = logv.shift(1)
    mean, std = prior.rolling(n, min_periods=n).mean(), prior.rolling(n, min_periods=n).std()
    return (logv - mean) / std.where(std > 0)


def dist_from_high(close: pd.DataFrame, high: pd.DataFrame, n: int = 252) -> pd.DataFrame:
    """Close relative to the highest high of the last n sessions (0 = at the high, -0.1 = 10% below)."""
    return close / high.rolling(n, min_periods=n).max() - 1.0


def median_traded_value(traded_value: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Typical daily turnover; the median ignores one-off block deals."""
    return traded_value.rolling(n, min_periods=n).median()


def cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per day (largest value = 1.0, ties share the higher rank);
    missing values stay missing."""
    return frame.rank(axis=1, pct=True, method="max")
