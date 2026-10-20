"""Panel: aligned daily price fields for many symbols, the unit the backtester works on.

Each field is a wide DataFrame (rows = sessions, columns = symbols, sorted). A missing bar is NaN:
the stock was not listed yet, was suspended, or the vendor lacks that day. Index bars (Nifty 50
etc.) are aligned to the same dates. Built from the data feed, the dates are the market's sessions
(whichever stocks are chosen), minus sessions the whole market lacks in the vendor data and short
sessions such as one-hour Muhurat trading, so the engine never trades on them. Both are judged from
each day's own data, so a backtest and the live view agree.

Prices are adjusted for corporate actions; `unadj_close` is the price the stock actually traded at
(Bar.traded_scale converts). Ticks, costs, price bands and share counts need traded prices.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from trader.core.types import Bar
from trader.data.feed import PANEL_FIELDS, panel_contract_check


class PanelError(ValueError):
    pass


@dataclass(eq=False)
class Panel:
    fields: dict[str, pd.DataFrame]
    index: dict[str, pd.DataFrame] = field(default_factory=dict)  # name -> open/high/low/close/volume by date

    def __post_init__(self) -> None:
        self.fields, self.index = dict(self.fields), dict(self.index)  # never add keys to the caller's dict
        self._prev_close: np.ndarray | None = None
        missing = [f for f in ("open", "high", "low", "close", "volume") if f not in self.fields]
        if missing:
            raise PanelError(f"panel is missing fields {missing}")
        close = self.fields["close"]
        if "unadj_close" not in self.fields:
            self.fields["unadj_close"] = close.copy()
        if "traded_value" not in self.fields:
            self.fields["traded_value"] = close * self.fields["volume"]

    # Accessors -------------------------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def symbols(self) -> list[str]:
        return list(self.fields["close"].columns)

    def __getattr__(self, name: str) -> pd.DataFrame:
        fields = self.__dict__.get("fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def index_close(self, name: str) -> pd.Series:
        return self.index[name]["close"]

    def bar(self, day: date | pd.Timestamp, symbol: str) -> Bar | None:
        ts = pd.Timestamp(day)
        try:
            row = self.dates.get_loc(ts)
            col = self.fields["close"].columns.get_loc(symbol)
        except KeyError:
            return None
        values = {f: self.fields[f].iat[row, col] for f in ("open", "high", "low", "close", "volume", "unadj_close")}
        if np.isnan(values["close"]):
            return None
        if self._prev_close is None:  # last close before each day, across gaps; built once
            self._prev_close = self.fields["close"].ffill().shift(1).to_numpy()
        prev = self._prev_close[row, col]
        return Bar(ts.date(), symbol, values["open"], values["high"], values["low"], values["close"], values["volume"],
                   None if np.isnan(prev) else float(prev), values["unadj_close"])

    # Slicing ---------------------------------------------------------------------------
    def _take(self, rows: slice | np.ndarray) -> Panel:
        # Copies, not views: modifying a slice must never change the panel it came from.
        return Panel({k: v.iloc[rows].copy() for k, v in self.fields.items()},
                     {k: v.loc[self.dates[rows]].copy() if len(v) else v.copy() for k, v in self.index.items()})

    def slice_to(self, day: date | pd.Timestamp) -> Panel:
        """Only sessions on or before `day`: what was knowable at that day's close."""
        end = int(self.dates.searchsorted(pd.Timestamp(day), side="right"))
        return self._take(slice(0, end))

    def window(self, day: date | pd.Timestamp, n_rows: int) -> Panel:
        """The last `n_rows` sessions up to and including `day` (the live-trading view)."""
        end = int(self.dates.searchsorted(pd.Timestamp(day), side="right"))
        return self._take(slice(max(0, end - n_rows), end))

    def between(self, start: date | pd.Timestamp, end: date | pd.Timestamp) -> Panel:
        lo = int(self.dates.searchsorted(pd.Timestamp(start), side="left"))
        hi = int(self.dates.searchsorted(pd.Timestamp(end), side="right"))
        return self._take(slice(lo, hi))

    def validate(self) -> None:
        problems = panel_contract_check(self.fields)
        for name, frame in self.index.items():
            if not frame.index.equals(self.dates):
                problems.append(f"index {name} is not aligned with the panel dates")
        if problems:
            raise PanelError("; ".join(problems))

    # Construction ----------------------------------------------------------------------
    @classmethod
    def from_feed(cls, feed, *, start: date | None = None, end: date | None = None, symbols: Iterable[str] | None = None,
                  adjustment: str = "economic", index_names: Iterable[str] | None = None,
                  drop_special: bool = True) -> Panel:
        raw = feed.load_panel(sorted(symbols) if symbols is not None else None, start, end, adjustment)
        calendar = feed.profile.calendar_index
        sessions = (pd.DatetimeIndex(feed.load_index(calendar, start, end)["date"])
                    if feed.index_store.exists(calendar) else pd.DatetimeIndex([]))
        dates = sessions.union(raw["close"].index)  # every market session, not only the chosen stocks' dates
        if drop_special and len(dates):
            flags = feed.session_flags()
            special = flags.index[(flags["sparse"] | flags["short"]).to_numpy()]
            dates = dates.difference(special)
        raw = {k: v.reindex(dates) for k, v in raw.items()}
        names = list(index_names) if index_names is not None else sorted({feed.profile.calendar_index, feed.profile.regime_index})
        index = {}
        for name in names:
            frame = feed.load_index(name).set_index("date")
            index[name] = frame.reindex(dates)
        return cls({k: raw[k] for k in PANEL_FIELDS if k in raw}, index)

    @classmethod
    def from_frames(cls, fields: Mapping[str, pd.DataFrame], index: Mapping[str, pd.DataFrame] | None = None) -> Panel:
        return cls(dict(fields), dict(index or {}))
