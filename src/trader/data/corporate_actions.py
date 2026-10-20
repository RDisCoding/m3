"""Corporate actions from NSE and the price adjustments they imply.

Upstox history is adjusted for splits and bonuses only. This module fetches NSE's corporate-action
list, parses each subject line and computes one factor per event on the *stored* series. Every
factor multiplies all prices strictly before the ex-date.

  split / bonus / consolidation  Checked against the stored series. Applied only if the vendor has
                                 not adjusted it, which shows up as an ex-date gap equal to the ratio.
  demerger                       factor = ex-date open / previous close, applied whenever that gap
                                 exceeds 0.5% in either direction. NSE runs a special pre-open session
                                 on the ex-date to discover the post-demerger price, so continuity at
                                 the open is the economically correct series. Computing it on the
                                 stored series fixes all three vendor behaviours seen in the data
                                 (verified against NSE bhavcopy): no adjustment (ITC 2025), partial
                                 adjustment (RELIANCE 2023) and over-adjustment (NMDC 2022, ABB 2019).
  arrangement                    "Scheme of Arrangement": often a demerger under another name. Same
                                 method, but only for gaps beyond 3%, since many schemes move no value.
  history break                  a gap beyond 10x on such a date means the stored history before it
                                 belongs to a different security; those bars are dropped.
  rights                         factor = theoretical ex-rights price / cum-rights close, applied only
                                 if the ex-date gap shows the vendor has not already adjusted it
                                 (Upstox adjusted 8 of the 12 large rights issues in the data).
  dividend                       factor = (close - dividend) / close. Used for total-return series only.

Factors are recomputed from the stored data whenever it changes, so a later vendor re-adjustment
never double-counts.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from itertools import combinations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NSE_BOOTSTRAP_URL = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
NSE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_BOOTSTRAP_URL,
}
EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}
SPLIT_KINDS = ("split", "bonus", "consolidation")
ECONOMIC_KINDS = (*SPLIT_KINDS, "demerger", "arrangement", "rights")
ADJUSTMENT_MODES = ("none", "economic", "total_return")
# A split-type event this small cannot be told apart from a normal overnight move, so it is
# assumed to be in the vendor data (|log 0.86| = 0.15, i.e. ratios between about 1:6 bonus and 7:6).
MIN_VERIFIABLE_LOG_RATIO = 0.15
DEMERGER_MIN_MOVE = 0.005  # demerger: correct ex-date gaps beyond 0.5% in either direction
ARRANGEMENT_MIN_MOVE = 0.03  # scheme of arrangement: only gaps beyond 3% are treated as value transfers
HISTORY_BREAK_RATIO = 10.0  # a gap this large means the earlier bars are a different security
# No single corporate action moves the vendor's basis by more than this; beyond it the stored bars
# before the event are not the stock's prices at all (PRIVISCL before 2020-08-21: a flat 0.15).
VENDOR_IMPLAUSIBLE_RATIO = 20.0
# Where NSE's file and the corrected vendor series disagree by more than this from one day to the
# next, the vendor changed its basis for a reason the event feed does not explain.
ANCHOR_JUMP = 0.03
RIGHTS_AMBIGUOUS = 0.98  # rights factors closer to 1 than this cannot be verified against market noise


# NSE client ---------------------------------------------------------------------------------
class NSEClient:
    """Minimal NSE website API client: cookie bootstrap, pacing, and re-bootstrap on blocks."""

    def __init__(self, *, client: httpx.Client | None = None, pace_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep, max_attempts: int = 3):
        self._client = client or httpx.Client(headers=NSE_HEADERS, timeout=httpx.Timeout(30.0, connect=5.0),
                                              follow_redirects=True)
        self._pace_s, self._sleep, self._max_attempts = pace_s, sleep, max_attempts
        self._bootstrapped = False
        self._last_call = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NSEClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _bootstrap(self) -> None:
        self._client.get(NSE_BOOTSTRAP_URL)
        self._bootstrapped = True

    def get_json(self, url: str, params: dict) -> object:
        last_error = "no attempt made"
        for attempt in range(self._max_attempts):
            if not self._bootstrapped:
                self._bootstrap()
            wait = self._pace_s - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.json()
                last_error = f"HTTP {resp.status_code}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            self._bootstrapped = False  # cookies expired or request blocked: start a fresh session
            self._sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"NSE request failed after {self._max_attempts} attempts: {last_error}")


def fetch_corporate_actions(client: NSEClient, start: date, end: date, chunk_days: int = 92) -> list[dict]:
    """All equity-segment corporate actions with ex-dates in [start, end], fetched in date chunks."""
    rows: list[dict] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        data = client.get_json(NSE_ACTIONS_URL, {"index": "equities", "from_date": f"{cursor:%d-%m-%Y}",
                                                 "to_date": f"{chunk_end:%d-%m-%Y}"})
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected NSE response for {cursor}..{chunk_end}: {str(data)[:200]}")
        rows.extend(data)
        cursor = chunk_end + timedelta(days=1)
    return rows


def _nse_date(text: object) -> str | None:
    if not isinstance(text, str) or not text.strip() or text.strip() == "-":
        return None
    try:
        return datetime.strptime(text.strip(), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def store_actions(conn: sqlite3.Connection, rows: Iterable[dict], fetched_at: str) -> int:
    records = []
    for row in rows:
        ex_date = _nse_date(row.get("exDate"))
        series = (row.get("series") or "").strip()
        if not ex_date or series not in EQUITY_SERIES or not row.get("symbol") or not row.get("subject"):
            continue
        try:
            face_value = float(row.get("faceVal")) if row.get("faceVal") not in (None, "", "-") else None
        except (TypeError, ValueError):
            face_value = None
        records.append((row["symbol"].strip(), (row.get("isin") or "").strip() or None, series, ex_date,
                        _nse_date(row.get("recDate")), " ".join(str(row["subject"]).split()), face_value,
                        (row.get("comp") or "").strip() or None, fetched_at))
    conn.executemany(
        "INSERT INTO corporate_actions (symbol, isin, series, ex_date, record_date, subject, face_value, company, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(symbol, ex_date, subject) DO UPDATE SET "
        "isin=excluded.isin, series=excluded.series, record_date=excluded.record_date, face_value=excluded.face_value, "
        "company=excluded.company, fetched_at=excluded.fetched_at",
        records,
    )
    return len(records)


def load_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT symbol, isin, series, ex_date, record_date, subject, face_value FROM corporate_actions", conn)


# Subject parsing ----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParsedAction:
    kind: str  # dividend | bonus | split | consolidation | rights | demerger
    ratio_new: float | None = None  # bonus/rights: new shares per `ratio_old` held; split: new face value
    ratio_old: float | None = None  # split/consolidation: old face value
    cash: float | None = None  # dividend per share (rupees)
    issue_price: float | None = None  # rights issue price per share (rupees), using the face value given
    premium: float | None = None  # rights premium over face value, when the issue is quoted as a premium

    def split_factor(self) -> float | None:
        """Price factor for split-type events (prices before the ex-date are multiplied by it)."""
        if self.ratio_new is None or self.ratio_old is None or self.ratio_new <= 0 or self.ratio_old <= 0:
            return None
        if self.kind == "bonus":
            return self.ratio_old / (self.ratio_new + self.ratio_old)
        if self.kind in ("split", "consolidation"):
            return self.ratio_new / self.ratio_old  # new face value / old face value
        return None


_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_CURRENCY = r"\b(?:rs|re|inr)(?![a-z])\.?\s*-?\s*"  # "Rs 10", "Rs.10", "Rs10", "Re 1", "Rs - 6"
_RUPEES = _CURRENCY + _NUM
_FROM_TO = r"(?:(?:from|frm)\s*)?" + _RUPEES + r".*?\bto\s*" + _RUPEES
# An amount right after a dividend word: "Dividend - Rs 19", "Special Dividend -Rs 3", "Int Div - Rs 4",
# "Div-7.5". Percentages of face value ("Dividend Of 150%") are not rupee amounts.
_DIVIDEND_AMOUNT = re.compile(r"(?:dividend|divdend|\bdiv)\b[^0-9%]{0,18}?(?:" + _CURRENCY + r")?"
                              r"([0-9]+(?:\.[0-9]+)?)(?![0-9.]*\s*%)")


def parse_subject(subject: str, face_value: float | None = None) -> list[ParsedAction]:
    """Parse an NSE corporate-action subject into zero or more price-relevant actions.

    Examples: "Bonus 1:2", "Demerger", "Rights 1:15 @ Premium Rs 1247",
    "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", "Fv Splt Frm Rs 10 To Re 1",
    "Interim Dividend - Rs 10 Per Share Special Dividend - Rs 66 Per Share",
    "Interim Div - Rs 6/- Per Share + Face Value Split (Sub-Division) - From Rs 10/- ... To Re 1/- ...".
    """
    s = " ".join(subject.lower().replace("/-", " ").split())
    out: list[ParsedAction] = []
    if "demerger" in s:
        out.append(ParsedAction("demerger"))
    elif re.search(r"s?c?heme of ar+angement", s):
        out.append(ParsedAction("arrangement"))
    if m := re.search(r"bonus\s*-?\s*" + _NUM + r"\s*:\s*" + _NUM, s):
        out.append(ParsedAction("bonus", ratio_new=float(m[1]), ratio_old=float(m[2])))
    # Ratios are read from the clause after the keyword, so a dividend's "Rs 6" earlier in the
    # subject is never mistaken for the old face value.
    if (m := re.search(r"(?:split|splt|sub.?division)(.*)", s)) and (r := re.search(_FROM_TO, m[1])):
        out.append(ParsedAction("split", ratio_old=float(r[1]), ratio_new=float(r[2])))
    if (m := re.search(r"consolidation(.*)", s)) and (r := re.search(_FROM_TO, m[1])):
        out.append(ParsedAction("consolidation", ratio_old=float(r[1]), ratio_new=float(r[2])))
    if "rights" in s:
        # One or more tranches, e.g. "Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share /
        # 2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share". Several tranches combine into one
        # equivalent issue: q rights shares per share held at the q-weighted issue price.
        tranches = re.findall(_NUM + r"\s*:\s*" + _NUM + r"[^@:/]*?@\s*(prem(?:ium)?\.?\s*(?:of\s*)?)?" + _RUPEES,
                              s.split("rights", 1)[1])
        if tranches:
            weights = [float(r) / float(n) for r, n, _, _ in tranches]
            q = sum(weights)
            all_premium = all(prem for _, _, prem, _ in tranches)
            fv = face_value or 0.0
            issue = sum(w * (float(px) + fv if prem else float(px)) for w, (_, _, prem, px) in zip(weights, tranches)) / q
            premium = sum(w * float(px) for w, (_, _, _, px) in zip(weights, tranches)) / q if all_premium else None
            if len(tranches) == 1:
                r, n, prem, px = tranches[0]
                out.append(ParsedAction("rights", ratio_new=float(r), ratio_old=float(n),
                                        issue_price=float(px) + fv if prem else float(px), premium=float(px) if prem else None))
            else:
                out.append(ParsedAction("rights", ratio_new=q, ratio_old=1.0, issue_price=issue, premium=premium))
    if re.search(r"dividend|divdend|\bdiv\b", s):
        # "Rs 8.50 Per Share (Including Special Dividend Of Rs 2 Per Share)" is 8.50, not 10.50.
        text = s.split("including", 1)[0]
        amounts = [float(x) for x in _DIVIDEND_AMOUNT.findall(text) if 0 < float(x) < 5_000]
        if amounts:
            out.append(ParsedAction("dividend", cash=sum(amounts)))
    return out


def unparsed_price_events(actions: pd.DataFrame) -> pd.DataFrame:
    """Subjects that may change the price but were not parsed (for manual review)."""
    pattern = r"ar+angement|capital reduction|reduction of capital|spin|amalgamation|rights|split|bonus|consolidation"
    mask = actions["subject"].str.contains(pattern, case=False, regex=True)
    parsed = actions["subject"].map(lambda s: bool(parse_subject(s)))
    return actions[mask & ~parsed]


# Factor computation -------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Factor:
    symbol: str
    ex_date: date
    kind: str
    factor: float
    applied: bool
    method: str
    detail: str
    subject: str = ""
    vendor_factor: float | None = None  # what the vendor applied at this date, measured against NSE's files
    restore_amount: float = 0.0  # rupees (stored basis) the vendor subtracted from earlier prices; added back
    evidence: str = "gaps"  # "official" when measured against NSE bhavcopy, "gaps" when inferred from price gaps


@dataclass(frozen=True, slots=True)
class _Measure:
    vendor: float  # vendor's adjustment at the event: (stored/official before) / (stored/official on the ex-date)
    prev_close: float  # official close of the session before the ex-date (rupees as traded)
    ex_open: float  # official open on the ex-date (NSE's discovery price for demergers)
    ex_close: float  # official close on the ex-date
    basis: float  # stored/official on the ex-date: converts as-traded rupees into the stored price basis


@dataclass
class _Event:
    ex_date: date
    kind: str
    subjects: list[str] = field(default_factory=list)
    parsed: list[ParsedAction] = field(default_factory=list)


def actions_for(actions: pd.DataFrame, symbol: str, isin: str | None = None) -> pd.DataFrame:
    """Rows for one stock: matched by NSE symbol, or by ISIN to catch symbol renames."""
    mask = actions["symbol"] == symbol
    if isin:
        mask |= actions["isin"] == isin
    return actions[mask].drop_duplicates(subset=["ex_date", "subject"])


def _group_events(rows: pd.DataFrame) -> list[_Event]:
    events: dict[tuple[date, str], _Event] = {}
    for row in rows.to_dict("records"):
        ex = date.fromisoformat(row["ex_date"])
        fv = row.get("face_value")
        for action in parse_subject(row["subject"], None if fv is None or pd.isna(fv) else float(fv)):
            event = events.setdefault((ex, action.kind), _Event(ex, action.kind))
            # The feed sometimes lists one dividend twice in different words (POLYMED 2017-09-20).
            if action.kind == "dividend" and any(p.cash == action.cash for p in event.parsed):
                continue
            event.subjects.append(row["subject"])
            event.parsed.append(action)
    for (ex, kind) in list(events):
        if kind == "arrangement" and (ex, "demerger") in events:
            del events[(ex, kind)]
    return sorted(events.values(), key=lambda e: (e.ex_date, e.kind))


def compute_factors(symbol: str, bars: pd.DataFrame, rows: pd.DataFrame, official: pd.DataFrame | None = None) -> list[Factor]:
    """Factors for one stock's events that fall inside its stored history.

    `official` holds the stock's NSE bhavcopy rows indexed by date (columns open, close). Where it
    covers the session before and the session of an event, the vendor's own adjustment is measured
    exactly and the factor is derived from it (evidence "official"). Otherwise the ex-date price gap
    in the stored series is used (evidence "gaps").
    """
    if bars.empty or rows.empty:
        return []
    off = official if official is not None and len(official) else None
    dates = bars["date"].to_numpy()
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    first, last = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()
    events = [e for e in _group_events(rows) if first < e.ex_date <= last]

    def around(ex: date) -> tuple[float, float]:
        i = int(np.searchsorted(dates, np.datetime64(ex), side="left"))
        return float(closes[i - 1]), float(opens[i])

    def ex_close(ex: date) -> float:
        return float(closes[int(np.searchsorted(dates, np.datetime64(ex), side="left"))])

    def measure(ex: date) -> _Measure | None:
        if off is None:
            return None
        i = int(np.searchsorted(dates, np.datetime64(ex), side="left"))
        before, on = pd.Timestamp(dates[i - 1]), pd.Timestamp(dates[i])
        if before not in off.index or on not in off.index:
            return None
        ob, oe = off.loc[before], off.loc[on]
        values = (float(ob["close"]), float(oe["open"]), float(oe["close"]), float(closes[i - 1]), float(closes[i]))
        if not all(np.isfinite(v) and v > 0 for v in values):
            return None
        off_prev, off_open, off_close, stored_prev, stored_ex = values
        return _Measure((stored_prev / off_prev) / (stored_ex / off_close), off_prev, off_open, off_close, stored_ex / off_close)

    current_fv = next((float(v) for v in rows["face_value"] if v is not None and not pd.isna(v)), None) \
        if "face_value" in rows else None
    split_events = [(e.ex_date, e.parsed[0]) for e in events if e.kind in ("split", "consolidation")]

    def face_value_at(ex: date) -> float | None:
        """Face value on `ex`: the feed reports today's, so undo later splits and consolidations."""
        if current_fv is None:
            return None
        fv = current_fv
        for d, action in split_events:
            if d > ex and action.ratio_old and action.ratio_new:
                fv *= action.ratio_old / action.ratio_new
        return fv

    def issue_price(p: ParsedAction, ex: date) -> float | None:
        if p.premium is not None and face_value_at(ex) is not None:
            return p.premium + face_value_at(ex)
        return p.issue_price

    def stale_open(open_gap: float, close_gap: float) -> bool:
        """Before NSE ran a discovery session on demerger ex-dates, the open could be a stale print
        (ADANIENT 2015: open 573, traded 106-110 all day). Value only leaves the stock, so only a
        large fall by the close that the open missed counts."""
        return close_gap < 0.90 and close_gap < open_gap and abs(close_gap - 1) > 2 * abs(open_gap - 1)

    factors: list[Factor] = []
    upstream: list[tuple[date, float]] = []  # split-type events already in the vendor data
    vendor_splits_at: dict[date, float] = {}  # product of split-type ratios the vendor applied on each date
    # Split-type events on the same date (e.g. "Bonus 4:1" plus a face-value split) act together,
    # so the ex-date gap is compared with their combined ratio.
    by_date: dict[date, list[tuple[_Event, float | None]]] = {}
    for e in (ev for ev in events if ev.kind in SPLIT_KINDS):
        by_date.setdefault(e.ex_date, []).append((e, e.parsed[0].split_factor()))
    for ex_date, group in sorted(by_date.items()):
        known = [(e, f) for e, f in group if f is not None]
        for e, f in group:
            if f is None:
                factors.append(Factor(symbol, ex_date, e.kind, 1.0, False, "unparsed", "ratio missing", " | ".join(e.subjects)))
        if not known:
            continue
        combined = float(np.prod([f for _, f in known]))
        prev_close, ex_open = around(ex_date)
        gap = ex_open / prev_close
        label = f"combined ratio {combined:.4f}" if len(known) > 1 else f"ratio {combined:.4f}"
        m = measure(ex_date)
        if m is not None:
            # What is still missing = expected ratio / what the vendor applied; match it to a subset.
            missing = combined / m.vendor
            subsets = [c for r in range(len(known) + 1) for c in combinations(range(len(known)), r)]
            distance = {c: abs(math.log(missing / float(np.prod([known[i][1] for i in c])))) for c in subsets}
            best = min(subsets, key=lambda c: distance[c])
            if distance[best] < 0.02:
                missed, method = best, ("vendor_unadjusted" if best else "verified_upstream")
            else:
                missed, method = (), "unexplained"
            detail = f"vendor applied {m.vendor:.4f} vs {label} (NSE bhavcopy)"
            upstream.append((ex_date, m.vendor))
            vendor_splits_at[ex_date] = float(np.prod([f for i, (_, f) in enumerate(known) if i not in missed]))
            for i, (e, f) in enumerate(known):
                applied = i in missed
                factors.append(Factor(symbol, ex_date, e.kind, f, applied,
                                      method if applied or method == "unexplained" or not missed else "verified_upstream",
                                      detail, " | ".join(e.subjects), vendor_factor=m.vendor, evidence="official"))
            continue
        if abs(math.log(combined)) < MIN_VERIFIABLE_LOG_RATIO:
            missed: tuple[int, ...] = ()
            method, detail = "assumed_upstream", f"{label} too small to verify; ex-date gap {gap:.4f}"
        else:
            # The vendor may have adjusted all, none or only some of the day's events (MINDACORP
            # 2015: the split but not the bonus). The subset whose ratio best explains the gap is
            # what is still missing from the data.
            subsets = [c for r in range(len(known) + 1) for c in combinations(range(len(known)), r)]
            distance = {c: abs(math.log(gap / float(np.prod([known[i][1] for i in c])))) for c in subsets}
            best = min(subsets, key=lambda c: distance[c])
            if best and distance[best] < 0.08 and distance[()] > MIN_VERIFIABLE_LOG_RATIO:
                missed, method = best, "vendor_unadjusted"
                detail = f"ex-date gap {gap:.4f} matches the unadjusted {'+'.join(known[i][0].kind for i in best)}"
            else:
                missed, method = (), "verified_upstream"
                detail = f"no gap at ex-date (gap {gap:.4f}, {label})"
        vendor_splits_at[ex_date] = float(np.prod([f for i, (_, f) in enumerate(known) if i not in missed]))
        for i, (e, f) in enumerate(known):
            applied = i in missed
            if not applied:
                upstream.append((ex_date, f))
            factors.append(Factor(symbol, ex_date, e.kind, f, applied, method if applied or not missed else "verified_upstream",
                                  detail, " | ".join(e.subjects)))

    def to_stored_basis(ex: date) -> float:
        """Multiplier that converts a per-share rupee amount on `ex` into the stored price basis.

        Includes split-type events on the same date: a dividend going ex with a bonus is paid on the
        pre-bonus shares (INFY 2015-06-15: Rs 29.50 with a 1:1 bonus)."""
        return float(np.prod([f for d, f in upstream if d >= ex])) if upstream else 1.0

    for e in (ev for ev in events if ev.kind not in SPLIT_KINDS):
        subject = " | ".join(e.subjects)
        prev_close, ex_open = around(e.ex_date)
        m = measure(e.ex_date)
        same_day_splits = vendor_splits_at.get(e.ex_date, 1.0)
        if m is not None and same_day_splits != 1.0:
            # Judge this event on what the vendor did beyond the same day's split or bonus.
            m = _Measure(m.vendor / same_day_splits, m.prev_close * same_day_splits, m.ex_open, m.ex_close, m.basis)
        if e.kind in ("demerger", "arrangement") and m is not None:
            if not 1 / VENDOR_IMPLAUSIBLE_RATIO < m.vendor < VENDOR_IMPLAUSIBLE_RATIO:
                factors.append(Factor(symbol, e.ex_date, "history_break", 1.0, True, "truncate",
                                      f"vendor prices before the ex-date are not the stock's traded prices "
                                      f"(vendor/official {m.vendor:.6f}); earlier bars dropped", subject,
                                      vendor_factor=m.vendor, evidence="official"))
                continue
            threshold = DEMERGER_MIN_MOVE if e.kind == "demerger" else ARRANGEMENT_MIN_MOVE
            true_factor, basis_note = m.ex_open / m.prev_close, "official ex-date open"
            close_gap = m.ex_close / m.prev_close
            if stale_open(true_factor, close_gap):
                true_factor, basis_note = close_gap, "official ex-date close (open looks stale)"
                factors.append(Factor(symbol, e.ex_date, "stale_open", 1.0, True, "repair",
                                      f"official open {m.ex_open:.2f} vs close {m.ex_close:.2f} after previous close "
                                      f"{m.prev_close:.2f}: the open is a stale print, so the day's open and high are "
                                      f"set to its close", subject, evidence="official"))
            wanted = true_factor if abs(true_factor - 1) > threshold else 1.0
            correction = wanted / m.vendor
            detail = f"true factor {true_factor:.4f} from {basis_note}; vendor applied {m.vendor:.4f}"
            if abs(math.log(correction)) > 0.0005:
                factors.append(Factor(symbol, e.ex_date, e.kind, correction, True, "official_correction", detail, subject,
                                      vendor_factor=m.vendor, evidence="official"))
            else:
                factors.append(Factor(symbol, e.ex_date, e.kind, 1.0, False, "verified_upstream" if wanted != 1.0 else "no_gap",
                                      detail, subject, vendor_factor=m.vendor, evidence="official"))
            continue
        if e.kind in ("demerger", "arrangement"):
            gap = ex_open / prev_close
            threshold = DEMERGER_MIN_MOVE if e.kind == "demerger" else ARRANGEMENT_MIN_MOVE
            detail = f"ex-date open {ex_open:.2f} / previous close {prev_close:.2f}"
            close_gap = ex_close(e.ex_date) / prev_close
            if stale_open(gap, close_gap):
                gap = close_gap
                detail = f"ex-date close {ex_close(e.ex_date):.2f} / previous close {prev_close:.2f} (open {ex_open:.2f} looks stale)"
                factors.append(Factor(symbol, e.ex_date, "stale_open", 1.0, True, "repair",
                                      f"{detail}: the day's open and high are set to its close", subject))
            if gap > HISTORY_BREAK_RATIO or gap < 1 / HISTORY_BREAK_RATIO:
                factors.append(Factor(symbol, e.ex_date, "history_break", 1.0, True, "truncate",
                                      f"{detail}: earlier bars belong to a different security and are dropped", subject))
            elif abs(gap - 1) > threshold:
                factors.append(Factor(symbol, e.ex_date, e.kind, gap, True, "ex_open_gap", detail, subject))
            else:
                factors.append(Factor(symbol, e.ex_date, e.kind, 1.0, False, "no_gap",
                                      f"ex-date gap {gap:.4f} within {threshold:.1%}; nothing to correct", subject))
        elif e.kind == "rights":
            p = e.parsed[0]
            if p.issue_price is None or p.ratio_new is None or p.ratio_old is None:
                factors.append(Factor(symbol, e.ex_date, "rights", 1.0, False, "unparsed", "rights terms missing", subject))
                continue
            if m is not None:
                cum, issue_raw = m.prev_close, issue_price(p, e.ex_date)
                if issue_raw >= cum:
                    factors.append(Factor(symbol, e.ex_date, "rights", 1.0, False, "no_discount",
                                          f"issue price {issue_raw:.2f} >= official cum-rights close {cum:.2f}", subject,
                                          vendor_factor=m.vendor, evidence="official"))
                    continue
                factor = (p.ratio_old * cum + p.ratio_new * issue_raw) / (p.ratio_old + p.ratio_new) / cum
                detail = (f"{p.ratio_new:g}:{p.ratio_old:g} at {issue_raw:.2f} vs official close {cum:.2f}: "
                          f"factor {factor:.4f}; vendor applied {m.vendor:.4f}")
                if abs(math.log(factor)) < 0.002:
                    method, applied = "ambiguous", False
                elif abs(math.log(m.vendor / factor)) < abs(math.log(m.vendor)):
                    method, applied = "verified_upstream", False
                else:
                    method, applied = "terp", True
                factors.append(Factor(symbol, e.ex_date, "rights", factor, applied, method, detail, subject,
                                      vendor_factor=m.vendor, evidence="official"))
                continue
            issue = issue_price(p, e.ex_date) * to_stored_basis(e.ex_date)
            if issue >= prev_close:
                factors.append(Factor(symbol, e.ex_date, "rights", 1.0, False, "no_discount",
                                      f"issue price {issue:.2f} >= cum-rights close {prev_close:.2f}", subject))
                continue
            terp = (p.ratio_old * prev_close + p.ratio_new * issue) / (p.ratio_old + p.ratio_new)
            factor, gap = terp / prev_close, ex_open / prev_close
            detail = f"{p.ratio_new:g}:{p.ratio_old:g} at {issue:.2f} vs close {prev_close:.2f}; ex-date gap {gap:.4f}"
            if factor >= RIGHTS_AMBIGUOUS:
                factors.append(Factor(symbol, e.ex_date, "rights", factor, False, "ambiguous",
                                      detail + " (too small to verify; left as the vendor has it)", subject))
            elif abs(gap - factor) < abs(gap - 1):
                factors.append(Factor(symbol, e.ex_date, "rights", factor, True, "terp", detail, subject))
            else:
                factors.append(Factor(symbol, e.ex_date, "rights", factor, False, "verified_upstream",
                                      detail + " (no gap: already adjusted by the vendor)", subject))
        elif e.kind == "dividend" and m is not None:
            cash_raw = sum(p.cash or 0.0 for p in e.parsed)
            dividend_yield = cash_raw / m.prev_close
            if not 0 < dividend_yield < 0.5:
                factors.append(Factor(symbol, e.ex_date, "dividend", 1.0, False, "implausible",
                                      f"dividend {cash_raw:.2f} vs official close {m.prev_close:.2f}", subject, evidence="official"))
                continue
            factor = 1 - dividend_yield
            # Some dividends (e.g. HEROMOTOCO's Rs 100 in 2024) the vendor subtracts from all earlier prices.
            removed = dividend_yield >= 0.002 and abs(math.log(m.vendor / factor)) < abs(math.log(m.vendor))
            detail = (f"dividend {cash_raw:.2f} = {dividend_yield:.2%} of official close {m.prev_close:.2f}; "
                      f"vendor applied {m.vendor:.4f}" + ("; vendor removed it from history" if removed else ""))
            factors.append(Factor(symbol, e.ex_date, "dividend", factor, True, "vendor_removed" if removed else "cash",
                                  detail, subject, vendor_factor=m.vendor,
                                  restore_amount=cash_raw * m.basis * same_day_splits if removed else 0.0, evidence="official"))
        elif e.kind == "dividend":
            cash = sum(p.cash or 0.0 for p in e.parsed) * to_stored_basis(e.ex_date)
            if 0 < cash < 0.5 * prev_close:
                factors.append(Factor(symbol, e.ex_date, "dividend", (prev_close - cash) / prev_close, True, "cash",
                                      f"dividend {cash:.4f} (stored basis) on close {prev_close:.2f}", subject))
            else:
                factors.append(Factor(symbol, e.ex_date, "dividend", 1.0, False, "implausible",
                                      f"dividend {cash:.4f} vs close {prev_close:.2f}", subject))
    unique: dict[tuple[date, str], Factor] = {}
    for f in factors:
        unique.setdefault((f.ex_date, f.kind), f)
    return sorted(unique.values(), key=lambda f: (f.ex_date, f.kind))


def store_factors(conn: sqlite3.Connection, symbol: str, factors: Iterable[Factor], computed_at: str) -> None:
    conn.execute("DELETE FROM adjustment_factors WHERE symbol = ?", (symbol,))
    conn.executemany(
        "INSERT INTO adjustment_factors (symbol, ex_date, kind, factor, applied, method, detail, subject, computed_at, "
        "vendor_factor, restore_amount, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(f.symbol, f.ex_date.isoformat(), f.kind, f.factor, int(f.applied), f.method, f.detail, f.subject, computed_at,
          f.vendor_factor, f.restore_amount, f.evidence) for f in factors],
    )


def load_factors(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT symbol, ex_date, kind, factor, applied, method, detail, subject, vendor_factor, restore_amount, evidence "
        "FROM adjustment_factors", conn)


# Applying factors ---------------------------------------------------------------------------
def _cumulative(dates: np.ndarray, events: list[tuple[np.datetime64, float]]) -> np.ndarray:
    out = np.ones(len(dates))
    for ex, f in events:
        out[dates < ex] *= f
    return out


def _additive(dates: np.ndarray, events: list[tuple[np.datetime64, float]]) -> np.ndarray:
    out = np.zeros(len(dates))
    for ex, amount in events:
        out[dates < ex] += amount
    return out


class OfficialBars:
    """One stock's NSE official bars as arrays, for fast alignment with its stored bars."""

    __slots__ = ("dates", "values")

    def __init__(self, dates: np.ndarray, values: dict[str, np.ndarray]):
        order = np.argsort(dates, kind="stable")
        self.dates = np.asarray(dates, dtype="datetime64[ns]")[order]
        self.values = {k: np.asarray(v, dtype="float64")[order] for k, v in values.items()}

    @classmethod
    def from_frame(cls, frame: pd.DataFrame | pd.Series) -> OfficialBars:
        if isinstance(frame, pd.Series):
            frame = frame.to_frame("close")
        return cls(pd.DatetimeIndex(frame.index).to_numpy(dtype="datetime64[ns]"),
                   {k: frame[k].to_numpy(dtype="float64") for k in frame.columns})

    def __len__(self) -> int:
        return len(self.dates)

    def align(self, dates: np.ndarray) -> dict[str, np.ndarray]:
        """Each field on `dates`; NaN where NSE has no bar that day."""
        pos = np.searchsorted(self.dates, dates)
        safe = np.minimum(pos, len(self.dates) - 1)
        hit = (pos < len(self.dates)) & (self.dates[safe] == dates)
        return {k: np.where(hit, v[safe], np.nan) for k, v in self.values.items()}


def _segment_median(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Median of `values` (ignoring NaN) over each run between consecutive `starts`, broadcast back.
    One sort per stock instead of a median call per run (equal to np.median, even counts averaged)."""
    segment = np.cumsum(starts) - 1
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    v, g = values[finite], segment[finite]
    order = np.lexsort((v, g))
    v, g = v[order], g[order]
    counts = np.bincount(g, minlength=int(segment[-1]) + 1)
    first = np.r_[0, np.cumsum(counts)[:-1]]
    has = counts > 0
    median = np.full(len(counts), np.nan)
    median[has] = (v[(first + (counts - 1) // 2)[has]] + v[(first + counts // 2)[has]]) / 2
    return median[segment]


def _last_per_segment(values: np.ndarray, valid: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """For each run between consecutive `starts`, the value at its last `valid` row, broadcast back."""
    segment = np.cumsum(starts) - 1
    out = np.full(len(values), np.nan)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return out
    seg = segment[rows]
    last = rows[np.r_[seg[1:] != seg[:-1], True]]
    table = np.full(int(segment[-1]) + 1, np.nan)
    table[segment[last]] = values[last]
    return table[segment]


def _level_changes(log_ratio: np.ndarray, known: np.ndarray, window: int = 5) -> np.ndarray:
    """Rows where the vendor/official level moves by more than ANCHOR_JUMP and stays there: the
    median of the next `window` days differs from the previous `window` days'. One bad vendor day or
    rounding noise (CUPID's 1/100-scale prices swing about 4% from day to day) is not a change."""
    lr = log_ratio[known]
    rows = []
    for i in np.flatnonzero(np.abs(np.diff(lr)) > ANCHOR_JUMP) + 1:
        if abs(np.median(lr[i:i + window]) - np.median(lr[max(0, i - window):i])) > ANCHOR_JUMP:
            rows.append(known[i])
    return np.asarray(rows, dtype=np.int64)


def _anchor_to_official(px: dict[str, np.ndarray], volume: np.ndarray, dates: np.ndarray,
                        off: dict[str, np.ndarray], event_dates: np.ndarray) -> np.ndarray:
    """Replace corrected vendor prices with NSE's official prices on the same adjustment basis.

    The vendor rounds old split-adjusted prices (TATAINVEST 2016: 494.45 stored as 49.0 after a 1:10
    split; CUPID at 1/100 scale), reports the last trade instead of the closing price on some special
    sessions (2025-02-01) and has a few wrong days (EMAMILTD 2023-04-17..19). Between two corporate
    actions the ratio corrected/official must be constant, so official OHLC times that ratio is the
    exact bar. The ratio used for a stretch is the one next to the stretch's end, where the factor at
    the next event was measured, so the series stays continuous across it: the median from the last
    lasting level change in the stretch (a vendor re-basing the event feed does not explain, COHANCE
    2020-03-24) to the stretch's end. Official prices never jump there, so neither does the result.
    Days without an official bar keep the vendor's values. Updates `px` and returns the new volume.
    """
    o, h, l, c = off["open"], off["high"], off["low"], off["close"]
    with np.errstate(invalid="ignore"):
        have = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0) & (o > 0) & (l > 0)
    if not have.any():
        return volume
    events = np.asarray(event_dates, dtype="datetime64[ns]")
    stretch = np.zeros(len(dates), dtype=bool)
    stretch[0] = True
    pos = np.searchsorted(dates, events)
    stretch[pos[(pos > 0) & (pos < len(dates))]] = True  # first bar on or after each ex-date

    def basis(numerator: np.ndarray, denominator: np.ndarray, valid: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, numerator / denominator, np.nan)
            log_ratio = np.log(ratio)
        parts = stretch.copy()
        parts[_level_changes(log_ratio, np.flatnonzero(valid))] = True
        return _last_per_segment(_segment_median(ratio, parts), valid, stretch)

    scale = basis(px["close"], c, have)
    use = have & np.isfinite(scale)
    for column, values in (("open", o), ("high", h), ("low", l), ("close", c)):
        px[column] = np.where(use, values * scale, px[column])
    vol = off.get("volume")
    if vol is None:
        return volume
    with np.errstate(invalid="ignore"):
        has_volume = use & (vol > 0) & (volume > 0)
    vscale = basis(volume.astype("float64"), vol, has_volume)
    ok = use & (vol > 0) & np.isfinite(vscale)
    return np.where(ok, np.round(vol * vscale), volume).astype("int64")


def adjust_bars(bars: pd.DataFrame, factors: pd.DataFrame, mode: str = "economic",
                as_traded: OfficialBars | pd.DataFrame | pd.Series | None = None, *, anchor: bool = True) -> pd.DataFrame:
    """Apply one stock's factors to its stored bars.

    Returns the canonical columns plus `unadj_close`, the price as it actually traded that day, for
    price filters and share counts: NSE's official close when `as_traded` (indexed by date) has it,
    otherwise the stored close with vendor split/bonus adjustments undone.
    mode: none = stored prices; economic = removed dividends restored, then splits the vendor missed,
          demergers and rights; total_return = economic plus dividends.
    With `anchor` and official open/high/low/close in `as_traded`, adjusted prices come from NSE's
    official bars on the corrected basis (see _anchor_to_official); stale ex-date opens are repaired.
    Works on arrays and builds one frame at the end (this runs for every stock on every panel load).
    """
    if mode not in ADJUSTMENT_MODES:
        raise ValueError(f"mode must be one of {ADJUSTMENT_MODES}")
    dates = bars["date"].to_numpy(dtype="datetime64[ns]")
    px = {c: bars[c].to_numpy(dtype="float64", copy=True) for c in ("open", "high", "low", "close")}
    volume = bars["volume"].to_numpy()
    extra = {c: bars[c].to_numpy() for c in bars.columns if c not in ("date", "open", "high", "low", "close", "volume")}
    # Money actually traded that day: vendor prices and volumes are both split-adjusted, so their
    # product is unaffected by splits; later price corrections must not change it.
    traded_value = px["close"] * volume
    official = OfficialBars.from_frame(as_traded) if isinstance(as_traded, (pd.DataFrame, pd.Series)) else as_traded
    off = official.align(dates) if official is not None and len(official) else None
    official_close = off["close"] if off is not None else np.full(len(dates), np.nan)
    if off is not None and "turnover" in off:
        # NSE's turnover is exact; the vendor's volume is not always split-adjusted.
        traded_value = np.where(np.isfinite(off["turnover"]), off["turnover"], traded_value)
    anchor_to = off if anchor and off is not None and mode != "none" and {"open", "high", "low"} <= set(off) else None

    if factors.empty:
        kind = method = np.array([], dtype=object)
        applied = np.array([], dtype=bool)
        ex = np.array([], dtype="datetime64[ns]")
        value = restore = np.array([], dtype="float64")
    else:
        kind = factors["kind"].to_numpy()
        applied = factors["applied"].to_numpy().astype(bool)
        method = factors["method"].to_numpy()
        ex = pd.to_datetime(factors["ex_date"]).to_numpy(dtype="datetime64[ns]")
        value = factors["factor"].to_numpy(dtype="float64")
        restore = (factors["restore_amount"].fillna(0.0).to_numpy(dtype="float64") if "restore_amount" in factors
                   else np.zeros(len(factors)))
    is_split = np.isin(kind, SPLIT_KINDS)

    def events(mask: np.ndarray) -> list[tuple[np.datetime64, float]]:
        return list(zip(ex[mask], value[mask]))

    restored = _additive(dates, list(zip(ex[restore > 0], restore[restore > 0])))
    vendor = is_split & ~applied & np.isin(method, ["verified_upstream", "assumed_upstream"])
    computed = (px["close"] + restored) / _cumulative(dates, events(vendor))
    unadj_close = np.where(np.isfinite(official_close), official_close, computed)

    def frame(keep: np.ndarray | None = None) -> pd.DataFrame:
        columns = {"date": dates, **px, "volume": volume, **extra, "traded_value": traded_value, "unadj_close": unadj_close}
        if keep is not None:
            columns = {k: v[keep] for k, v in columns.items()}
        return pd.DataFrame(columns)

    if mode == "none":
        return frame()
    if restored.any():
        for column in px:
            px[column] = px[column] + restored
    keep = None
    breaks = (kind == "history_break") & applied
    if breaks.any():
        keep = dates >= ex[breaks].max()
    kinds = ECONOMIC_KINDS if mode == "economic" else (*ECONOMIC_KINDS, "dividend")
    use = applied & np.isin(kind, kinds)
    price = _cumulative(dates, events(use))
    for column in px:
        px[column] = px[column] * price
    volume = np.round(volume / _cumulative(dates, events(use & is_split))).astype("int64")
    if anchor_to is not None:
        volume = _anchor_to_official(px, volume, dates, anchor_to, np.unique(ex))
    stale = (kind == "stale_open") & applied
    if stale.any():
        on = np.isin(dates, ex[stale])
        px["open"] = np.where(on, px["close"], px["open"])
        px["high"] = np.where(on, np.maximum(px["close"], px["low"]), px["high"])
    return frame(keep).reset_index(drop=True) if keep is not None else frame()


def official_for(by_stock: pd.DataFrame | None, stock: str) -> pd.DataFrame | None:
    """One stock's official rows indexed by date, from the per-stock bhavcopy table."""
    if by_stock is None or by_stock.empty:
        return None
    rows = by_stock[by_stock["stock"] == stock]
    return rows.set_index("date")[["open", "high", "low", "close", "prev_close", "volume"]] if len(rows) else None


def recompute_all(conn: sqlite3.Connection, store, isin_by_symbol: dict[str, str | None], computed_at: str,
                  by_stock: pd.DataFrame | None = None) -> list[Factor]:
    """Recompute and store factors for every stored stock, using NSE's files as evidence where available."""
    actions = load_actions(conn)
    grouped = dict(tuple(by_stock.groupby("stock"))) if by_stock is not None and len(by_stock) else {}
    everything: list[Factor] = []
    for stem in store.stems():
        bars = store.read_bars(stem)
        off = grouped.get(stem)
        official = off.set_index("date")[["open", "close"]] if off is not None else None
        factors = compute_factors(stem, bars, actions_for(actions, stem, isin_by_symbol.get(stem)), official)
        store_factors(conn, stem, factors, computed_at)
        everything.extend(factors)
    conn.commit()
    return everything
