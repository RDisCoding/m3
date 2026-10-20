"""Feature store: every indicator the strategies use, computed once for the whole panel.

Point-in-time guarantees (tested): the value of any feature on day t is identical whether the
panel ends at t or continues past it, and changing data after t never changes values up to t.
Windows count each stock's own bars: stocks with gaps inside their history (suspensions, missing
vendor days) are recomputed on their own bar sequence so a gap cannot leak into a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from trader.core.panel import Panel
from trader.features import indicators as ind


@dataclass(frozen=True)
class FeatureConfig:
    sma: tuple[int, ...] = (20, 50, 200)
    ema: tuple[int, ...] = (20, 50)
    rsi: tuple[int, ...] = (2, 14)
    atr: int = 14
    adx: int = 14
    donchian_high: tuple[int, ...] = (20, 55)
    donchian_low: tuple[int, ...] = (10, 20)
    returns: tuple[int, ...] = (21, 63, 126, 252)
    vol: tuple[int, ...] = (20, 60)
    relative_strength: tuple[int, ...] = (63, 126)
    volume_z: int = 20
    high_window: int = 252
    adv: int = 20
    annualization_days: int = 252
    regime_sma: int = 200
    regime_vol: int = 20

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every feature exists (longest window plus the momentum skip)."""
        return max(max(self.returns), self.high_window, max(self.sma), 252) + 1

    @classmethod
    def for_profile(cls, profile) -> FeatureConfig:
        return cls(annualization_days=profile.trading_days_per_year)


def _per_symbol(panel_fields: dict[str, pd.DataFrame], index_close: pd.Series | None, cfg: FeatureConfig) -> dict[str, pd.DataFrame]:
    h, l, c = (panel_fields[k] for k in ("high", "low", "close"))
    out: dict[str, pd.DataFrame] = {}
    for n in cfg.sma:
        out[f"sma_{n}"] = ind.sma(c, n)
    for n in cfg.ema:
        out[f"ema_{n}"] = ind.ema(c, n)
    for n in cfg.rsi:
        out[f"rsi_{n}"] = ind.rsi(c, n)
    out["macd"], out["macd_signal"], out["macd_hist"] = ind.macd(c)
    out[f"atr_{cfg.atr}"] = ind.atr(h, l, c, cfg.atr)
    out["atr_pct"] = out[f"atr_{cfg.atr}"] / c
    out[f"adx_{cfg.adx}"] = ind.adx(h, l, c, cfg.adx)
    for n in cfg.donchian_high:
        out[f"dc_high_{n}"] = ind.donchian_high(h, n)
    for n in cfg.donchian_low:
        out[f"dc_low_{n}"] = ind.donchian_low(l, n)
    for n in cfg.returns:
        out[f"ret_{n}"] = ind.rolling_return(c, n)
    out["mom_12_1"] = ind.momentum_12_1(c)
    for n in cfg.vol:
        out[f"vol_{n}"] = ind.realized_vol(c, n, cfg.annualization_days)
    if index_close is not None:
        for n in cfg.relative_strength:
            out[f"rs_{n}"] = ind.relative_strength(c, index_close, n)
    out[f"vol_z_{cfg.volume_z}"] = ind.volume_zscore(panel_fields["traded_value"], cfg.volume_z)
    out["dist_52w_high"] = ind.dist_from_high(c, h, cfg.high_window)
    out[f"adv_{cfg.adv}"] = ind.median_traded_value(panel_fields["traded_value"], cfg.adv)
    return out


def columns_with_gaps(close: pd.DataFrame) -> list[str]:
    """Symbols with a missing bar somewhere between their first and last bar."""
    present = close.notna()
    started = present.cumsum() > 0
    not_ended = present[::-1].cumsum()[::-1] > 0
    inside_gap = (~present) & started & not_ended
    return list(close.columns[inside_gap.any().to_numpy()])


@dataclass
class FeatureStore:
    frames: dict[str, pd.DataFrame]
    index_features: pd.DataFrame
    gap_symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        first = next(iter(self.frames.values()))
        self.dates: pd.DatetimeIndex = first.index
        self.symbols: list[str] = list(first.columns)
        self.names: list[str] = list(self.frames)
        self._arrays = {k: v.to_numpy() for k, v in self.frames.items()}

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]

    def at(self, day: date | pd.Timestamp) -> pd.DataFrame:
        """Cross-section on `day`: rows = symbols, columns = features."""
        i = self.dates.get_loc(pd.Timestamp(day))
        return pd.DataFrame({k: a[i] for k, a in self._arrays.items()}, index=self.symbols)

    def view(self, day: date | pd.Timestamp) -> FeatureView:
        return FeatureView(self, pd.Timestamp(day))


class FeatureView:
    """Point-in-time accessor handed to strategies: only values up to `day` come out, and always as
    copies, so a strategy that edits what it gets cannot change the store for later days or others."""

    __slots__ = ("_store", "day")

    def __init__(self, store: FeatureStore, day: pd.Timestamp):
        self._store, self.day = store, pd.Timestamp(day)

    def at(self) -> pd.DataFrame:
        return self._store.at(self.day)

    def history(self, name: str, symbol: str, n: int) -> pd.Series:
        series = self._store.frames[name][symbol]
        return series.loc[: self.day].tail(n).copy()

    def index(self) -> pd.Series:
        return self._store.index_features.loc[self.day].copy()

    @property
    def names(self) -> list[str]:
        return list(self._store.names)


def build_features(panel: Panel, cfg: FeatureConfig | None = None, *, index_name: str | None = None) -> FeatureStore:
    cfg = cfg or FeatureConfig()
    index_name = index_name or (next(iter(panel.index)) if panel.index else None)
    index_close = panel.index[index_name]["close"] if index_name else None
    frames = _per_symbol(panel.fields, index_close, cfg)

    gaps = columns_with_gaps(panel.close)
    for symbol in gaps:
        rows = panel.close[symbol].notna().to_numpy()
        sub = {k: v.loc[rows, [symbol]] for k, v in panel.fields.items()}
        sub_index = index_close.loc[rows] if index_close is not None else None
        recomputed = _per_symbol(sub, sub_index, cfg)
        for name, frame in recomputed.items():
            frames[name][symbol] = frame[symbol].reindex(panel.dates)

    # No bar that day means no feature value that day (exponential averages would otherwise carry
    # the last value forward, and only on trailing gaps, which breaks point-in-time equality).
    has_bar = panel.close.notna()
    frames = {name: frame.where(has_bar) for name, frame in frames.items()}

    frames["mom_rank"] = ind.cs_rank(frames["mom_12_1"])
    frames["ret_21_rank"] = ind.cs_rank(frames["ret_21"])

    return FeatureStore(frames, index_feature_frame(index_close, panel.dates, cfg), gaps)


def index_feature_frame(index_close: pd.Series | None, dates: pd.DatetimeIndex, cfg: FeatureConfig) -> pd.DataFrame:
    """Regime inputs from the market index, computed on the index's own sessions (a missing index
    bar is skipped, not allowed to blank a moving average for the next 200 days)."""
    out = pd.DataFrame(index=dates)
    if index_close is None:
        return out
    close = index_close.dropna()
    out["index_close"] = index_close
    columns = {
        f"index_sma_{cfg.regime_sma}": close.rolling(cfg.regime_sma, min_periods=cfg.regime_sma).mean(),
        f"index_rv_{cfg.regime_vol}": ind.realized_vol(close, cfg.regime_vol, cfg.annualization_days),
        "index_ret_21": ind.rolling_return(close, 21),
        # From the 52-week high, not the all-time high, so the value does not depend on how much
        # history happens to be loaded (a live 600-session window must agree with the backtest).
        "index_dd_from_high": close / close.rolling(cfg.high_window, min_periods=cfg.high_window).max() - 1.0,
    }
    for name, series in columns.items():
        out[name] = series.reindex(dates).where(index_close.notna())
    return out


def lookahead_differences(panel: Panel, days: list[pd.Timestamp], cfg: FeatureConfig | None = None,
                          window: int | None = None, full: FeatureStore | None = None) -> list[str]:
    """Compare features on each day computed from the full panel vs from data ending that day.

    With `window` set, the truncated panel keeps only that many recent sessions (the live-trading
    view) and differences up to 1e-6 relative are tolerated (exponential averages forget slowly).
    Both the per-stock features and the index (regime) features are compared.
    Returns human-readable descriptions of any mismatch.
    """
    full = full or build_features(panel, cfg)
    problems = []
    for day in days:
        cut = panel.window(day, window) if window else panel.slice_to(day)
        part = build_features(cut, cfg)
        pairs = [(full.at(day), part.at(day))]
        if len(full.index_features.columns):
            pairs.append((full.index_features.loc[[day]], part.index_features.loc[[day]]))
        for a, b in pairs:
            x, y = a.to_numpy(dtype="float64"), b[a.columns].to_numpy(dtype="float64")
            both_nan = np.isnan(x) & np.isnan(y)
            if window:
                diff = np.abs(x - y) / np.maximum(np.abs(x), 1.0)
                bad = ~((diff <= 1e-6) | both_nan)
            else:
                bad = ~((x == y) | both_nan)
            if bad.any():
                cols = sorted({a.columns[j] for j in np.where(bad)[1]})
                problems.append(f"{day:%Y-%m-%d}: {int(bad.sum())} values differ in {cols[:6]}")
    return problems
