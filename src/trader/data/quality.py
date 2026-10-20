"""Data-quality checks for canonical daily bar frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trader.data.providers.base import PRICE_COLUMNS

Severity = Literal["info", "warn", "error"]
# Price ratios produced by common splits/bonuses (1:1 bonus = 0.5, 1:5 split = 0.2, 4:1 bonus with a
# 1:2 split = 0.1, 3:2 bonus = 0.4, ...) and their reverses. A day-over-day jump at one of these
# ratios suggests history that was not adjusted for a corporate action.
SPLIT_LIKE_RATIOS = (0.5, 1 / 3, 0.25, 0.2, 0.1, 0.4, 2 / 3, 1 / 6, 1 / 11, 0.6, 4 / 7,
                     1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class Finding:
    name: str
    check: str
    severity: Severity
    detail: str


def _sample(dates: pd.DatetimeIndex, n: int = 5) -> str:
    shown = ", ".join(d.strftime("%Y-%m-%d") for d in dates[:n])
    return shown + (f" (+{len(dates) - n} more)" if len(dates) > n else "")


def run_checks(
    name: str,
    bars: pd.DataFrame,
    *,
    reference_dates: pd.DatetimeIndex | None = None,
    expect_volume: bool = True,
    extreme_log_return: float = 0.25,
    recent_window: int = 60,
) -> list[Finding]:
    """Check one instrument's bars; `reference_dates` (e.g. Nifty 50 sessions) enables gap checks."""
    findings: list[Finding] = []

    def add(check: str, severity: Severity, detail: str) -> None:
        findings.append(Finding(name, check, severity, detail))

    if bars.empty:
        add("empty", "error", "no bars")
        return findings

    dates = pd.DatetimeIndex(bars["date"])
    if dates.has_duplicates:
        add("duplicate_dates", "error", f"{int(dates.duplicated().sum())} duplicated dates")
    if not dates.is_monotonic_increasing:
        add("unsorted", "error", "dates are not ascending")

    prices = bars[PRICE_COLUMNS]
    nan_rows = prices.isna().any(axis=1).to_numpy()
    if nan_rows.any():
        add("nan_prices", "error", f"{int(nan_rows.sum())} rows with missing prices: {_sample(dates[nan_rows])}")
    non_positive = (prices <= 0).any(axis=1).to_numpy()
    if non_positive.any():
        add("non_positive_prices", "error", f"{int(non_positive.sum())} rows: {_sample(dates[non_positive])}")

    eps = 1e-6
    body_high = bars[["open", "close"]].max(axis=1)
    body_low = bars[["open", "close"]].min(axis=1)
    bad = ((bars["high"] + eps < body_high) | (bars["low"] - eps > body_low) | (bars["low"] - eps > bars["high"])).to_numpy()
    if bad.any():
        add("ohlc_inconsistent", "error", f"{int(bad.sum())} rows where low/high do not bound open/close: {_sample(dates[bad])}")

    negative = (bars["volume"] < 0).to_numpy()
    if negative.any():
        add("negative_volume", "error", f"{int(negative.sum())} days with negative volume (vendor overflow?): {_sample(dates[negative])}")
    if expect_volume:
        zero = (bars["volume"] == 0).to_numpy()
        if zero.any():
            add("zero_volume", "warn", f"{int(zero.sum())} days with zero volume: {_sample(dates[zero])}")

    close = bars["close"].astype("float64")
    ratio = (close / close.shift(1)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(ratio)
    extreme = np.nan_to_num(np.abs(log_ret)) > extreme_log_return
    if extreme.any():
        moves = ", ".join(f"{d:%Y-%m-%d} ({r - 1:+.0%})" for d, r in zip(dates[extreme][:5], ratio[extreme][:5]))
        add("extreme_move", "warn", f"{int(extreme.sum())} days with |log return| > {extreme_log_return}: {moves}")
    # An unadjusted split opens at the new price level; a genuine crash opens near the previous
    # close and falls during the day. Require both the open and the close gap to match a ratio.
    open_ratio = (bars["open"].astype("float64") / close.shift(1)).to_numpy()
    split_like = np.zeros(len(bars), dtype=bool)
    for r in SPLIT_LIKE_RATIOS:
        split_like |= (np.nan_to_num(np.abs(ratio - r) / r, nan=1.0) < 0.05) & \
                      (np.nan_to_num(np.abs(open_ratio - r) / r, nan=1.0) < 0.05)
    split_like &= np.nan_to_num(np.abs(log_ret)) > 0.4
    if split_like.any():
        add("split_like_jump", "error", f"possible unadjusted split/bonus on {_sample(dates[split_like])}")

    gaps = dates.to_series().diff().dt.days.to_numpy()
    long = np.nan_to_num(gaps) > 30
    if long.any():
        spans = ", ".join(f"{d:%Y-%m-%d} ({int(g)} days)" for d, g in zip(dates[long][:5], gaps[long][:5]))
        add("long_gap", "warn", f"{int(long.sum())} gaps over 30 calendar days without bars (suspensions?), ending {spans}")

    if reference_dates is not None and len(reference_dates):
        ref = pd.DatetimeIndex(reference_dates).sort_values()
        span = ref[(ref >= dates.min()) & (ref <= dates.max())]
        missing = span.difference(dates)
        if len(missing):
            recent_missing = missing.intersection(ref[-recent_window:])
            severity: Severity = "error" if len(recent_missing) > 3 else "warn"
            add("missing_sessions", severity,
                f"{len(missing)} reference sessions missing ({len(recent_missing)} in the last {recent_window}): {_sample(missing)}")
        extra = dates.difference(ref)
        extra = extra[(extra >= ref.min()) & (extra <= ref.max())]
        if len(extra):
            add("extra_sessions", "warn", f"{len(extra)} dates absent from the reference: {_sample(extra)}")
        if dates.max() < ref.max():
            add("stale", "warn", f"last bar {dates.max():%Y-%m-%d} is before the reference's last session {ref.max():%Y-%m-%d}")

    return findings


def session_flags(activity: pd.DataFrame, *, min_coverage: float = 0.5, short_ratio: float = 0.25,
                  min_stocks: int = 50, baseline: int = 20) -> pd.DataFrame:
    """Classify each reference session from the whole market's bars, using only data up to that day.

    `activity` holds each stock's traded value (rows = the calendar index's sessions, NaN = no bar).
    * sparse: under `min_coverage` of the stocks that traded on the previous session have a bar, i.e.
      a vendor gap shared by the whole market (2015-2018 Muhurat and Budget-Saturday sessions).
    * short: the typical stock traded under `short_ratio` of its median over the previous `baseline`
      sessions: one-hour Muhurat sessions, NSE's 2024 disaster-recovery drills, the 2017-07-10 outage.
      Normal sessions are above 0.5 (2015-2026). Needs `min_stocks` stocks with a baseline.
    """
    present = activity.notna()
    before = present.shift(1, fill_value=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = (present & before).sum(axis=1) / before.sum(axis=1).replace(0, np.nan)
    logs = np.log1p(activity.clip(lower=0))
    relative = logs - logs.shift(1).rolling(baseline, min_periods=baseline // 2).median()
    judged = relative.notna().sum(axis=1)
    level = np.exp(relative.median(axis=1)).where(judged >= min_stocks)
    return pd.DataFrame({"stocks": present.sum(axis=1), "coverage": coverage, "activity": level,
                         "sparse": (coverage < min_coverage).to_numpy(), "short": (level < short_ratio).to_numpy()},
                        index=activity.index)


def trim_fragments(bars: pd.DataFrame, max_gap_days: int = 180) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Drop non-positive prices and any fragment of history that ends in a long gap.

    Vendors sometimes carry scattered or zero-price bars years before a stock actually listed
    (MAZDOCK: zero prices in 2017-2018, listed 2020; DELHIVERY: odd trades in 2016, listed 2022).
    Everything before the last gap longer than `max_gap_days` calendar days is dropped.
    Returns the trimmed frame and the first kept date if anything was cut, else None.
    """
    if bars.empty:
        return bars, None
    positive = (bars[PRICE_COLUMNS].to_numpy() > 0).all(axis=1)
    if positive.all():
        kept, cut = bars, None
    else:
        kept = bars.loc[positive]
        cut = pd.Timestamp(kept["date"].iloc[0]) if len(kept) else None
    if len(kept) > 1:
        days = kept["date"].to_numpy(dtype="datetime64[D]").astype("int64")
        long = np.flatnonzero(np.diff(days) > max_gap_days)
        if len(long):
            kept = kept.iloc[long[-1] + 1:]
            cut = pd.Timestamp(kept["date"].iloc[0])
    if kept is bars:
        return bars, None
    return kept.reset_index(drop=True), cut


def repair_volume(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Undo 32-bit overflow in vendor volumes (IDEA 2024-08-30 is stored as -81,259,413,
    i.e. 4,213,707,883 wrapped around 2**32). Returns the frame and the number of rows repaired."""
    negative = bars["volume"].to_numpy() < 0
    if not negative.any():
        return bars, 0
    out = bars.copy()
    out.loc[negative, "volume"] = out.loc[negative, "volume"] + 2**32
    return out, int(negative.sum())
