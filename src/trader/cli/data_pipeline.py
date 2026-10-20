"""Data pipeline commands: universe, calendar, backfill, refresh, corporate actions, status, panel check."""

import json
import time
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from trader.cli.common import (
    IST,
    console,
    data_app,
    fmt_date,
    load_reference,
    make_provider,
    parse_date,
    split_values,
    today_ist,
)
from trader.data.bars import BarSync, SyncResult
from trader.data.calendar import (
    TradingCalendar,
    load_holidays,
    parse_holidays,
    store_holidays,
)
from trader.data.corporate_actions import (
    NSEClient,
    fetch_corporate_actions,
    load_actions,
    recompute_all,
    store_actions,
    unparsed_price_events,
)
from trader.data.feed import LocalDataFeed, panel_contract_check
from trader.data.instruments import SymbolMap
from trader.data.providers.base import Instrument
from trader.data.quality import Finding, run_checks
from trader.data.universe import (
    download_constituents,
    load_constituents,
    map_constituents,
    members,
    update_membership,
)
from trader.market import get_profile
from trader.settings import get_settings
from trader.storage.db import connect, utc_now
from trader.storage.parquet import BarStore

INDICES = ("Nifty 50", "Nifty 500")
DEFAULT_INDEX = "nifty500"
DEFAULT_START = date(2015, 1, 1)


# Helpers ------------------------------------------------------------------------------
def _stores() -> tuple[BarStore, BarStore]:
    settings = get_settings()
    return BarStore(settings.bars_dir / "daily"), BarStore(settings.bars_dir / "index")


def _start_run(conn, job: str, as_of: date | None) -> str:
    run_id = f"{job}-{datetime.now(IST):%Y%m%dT%H%M%S}"
    conn.execute("INSERT OR REPLACE INTO data_runs (run_id, job, as_of, started_at, status) VALUES (?, ?, ?, ?, 'running')",
                 (run_id, job, as_of.isoformat() if as_of else None, utc_now()))
    conn.commit()
    return run_id


def _finish_run(conn, run_id: str, status: str, summary: dict) -> None:
    conn.execute("UPDATE data_runs SET finished_at = ?, status = ?, summary_json = ? WHERE run_id = ?",
                 (utc_now(), status, json.dumps(summary, default=str), run_id))
    conn.commit()


def _update_universe(conn, index: str, symbols: SymbolMap):
    settings = get_settings()
    path = download_constituents(index, settings.raw_dir / "universe", today=today_ist())
    mapping = map_constituents(load_constituents(path), symbols)
    settings.ref_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(settings.ref_dir / f"{index}_mapping.csv", index=False)
    diff = update_membership(conn, index, mapping, today_ist())
    conn.commit()
    return diff, mapping


def _instruments(conn, index: str, symbols: SymbolMap, only: list[str] | None = None,
                 include_stored: bool = False) -> tuple[list[Instrument], list[str]]:
    """Index instruments plus current members of `index`, as Instrument objects from today's master.

    With `include_stored`, stocks already stored but no longer members are included while they still
    trade, so backtests keep their history current and open positions can still be valued.
    """
    current = members(conn, index)
    if current.empty:
        _update_universe(conn, index, symbols)
        current = members(conn, index)
    insts: list[Instrument] = [i for i in (symbols.index(name) for name in INDICES) if i is not None]
    missing: list[str] = []
    wanted = {s.upper() for s in only} if only else None
    for row in current.to_dict("records"):
        if wanted is not None and row["symbol"] not in wanted:
            continue
        inst = symbols.by_key(row["instrument_key"]) or symbols.equity(row["symbol"])
        (insts.append(inst) if inst else missing.append(row["symbol"]))
    if include_stored and wanted is None:
        known = {i.symbol for i in insts}
        for name, key in conn.execute("SELECT name, instrument_key FROM bar_meta WHERE kind = 'EQ'").fetchall():
            if name in known:
                continue
            inst = symbols.by_key(key) if key else None
            inst = inst or symbols.equity(name)
            if inst is not None and inst.symbol not in known:
                insts.append(inst)
                known.add(inst.symbol)
            elif inst is None:
                conn.execute("UPDATE bar_meta SET status = 'delisted' WHERE name = ? AND kind = 'EQ'", (name,))
        conn.commit()
    if wanted is not None:
        insts = [i for i in insts if i.kind == "INDEX" or i.symbol in wanted]
    return insts, missing


def _refresh_holidays(conn) -> int:
    with make_provider() as provider:
        holidays = parse_holidays(provider.market_holidays())
    count = store_holidays(conn, holidays, utc_now())
    conn.commit()
    return count


def _calendar(conn, index_store: BarStore) -> TradingCalendar:
    index = get_profile().calendar_index
    observed = [d.date() for d in index_store.read_bars(index)["date"]] if index_store.exists(index) else []
    return TradingCalendar(observed, load_holidays(conn))


def _progress() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    console=console, transient=False)


def _sync_summary(results: list[SyncResult], title: str) -> dict:
    status = Counter(r.status for r in results)
    summary = {
        "instruments": len(results), "ok": status.get("ok", 0), "skipped": status.get("skipped", 0),
        "errors": status.get("error", 0), "rows_added": sum(r.added for r in results),
        "rows_updated": sum(r.updated for r in results), "repulled": [r.name for r in results if r.repulled],
    }
    table = Table(title=title)
    for key in ("instruments", "ok", "skipped", "errors", "rows_added", "rows_updated"):
        table.add_column(key.replace("_", " "), justify="right")
    table.add_row(*(f"{summary[k]:,}" for k in ("instruments", "ok", "skipped", "errors", "rows_added", "rows_updated")))
    console.print(table)
    if summary["repulled"]:
        console.print(f"Full history re-pulled after a vendor re-adjustment: {', '.join(summary['repulled'])}")
    errors = [r for r in results if r.status == "error"]
    if errors:
        err = Table("Name", "Error", title="Errors")
        for r in errors[:30]:
            err.add_row(escape(r.name), escape(r.detail))
        console.print(err)
    return summary


def _run_quality(conn, run_id: str, daily: BarStore, index_store: BarStore) -> tuple[list[Finding], int]:
    """Check the corrected series, i.e. what strategies will read (known corporate-action gaps fixed,
    pre-listing fragments dropped)."""
    index = get_profile().calendar_index
    reference = pd.DatetimeIndex(index_store.read_bars(index)["date"]) if index_store.exists(index) else None
    feed = LocalDataFeed()
    bars = {stem: feed.load_bars(stem, adjustment="economic") for stem in daily.stems()}
    flags = feed.session_flags()
    special = flags.index[flags["sparse"].to_numpy()] if reference is not None else None  # missing for everyone
    ref_for_gaps = reference.difference(special) if reference is not None else None
    findings: list[Finding] = []
    for stem, frame in bars.items():
        findings.extend(run_checks(stem, frame, reference_dates=ref_for_gaps))
    for name in index_store.stems():
        findings.extend(run_checks(name, index_store.read_bars(name), expect_volume=False))
    now = utc_now()
    conn.execute("DELETE FROM dq_findings WHERE run_id = ?", (run_id,))
    conn.executemany("INSERT INTO dq_findings (run_id, name, check_name, severity, detail, checked_at) VALUES (?, ?, ?, ?, ?, ?)",
                     [(run_id, f.name, f.check, f.severity, f.detail, now) for f in findings])
    conn.commit()
    return findings, 0 if special is None else len(special)


def _print_quality(findings: list[Finding], n_special: int, n_files: int) -> None:
    errors = [f for f in findings if f.severity == "error"]
    by_check = Counter((f.check, f.severity) for f in findings)
    console.print(f"Quality: {n_files} files checked, {len(errors)} errors, "
                  f"{sum(f.severity == 'warn' for f in findings)} warnings; "
                  f"{n_special} market-wide special sessions excluded from gap checks")
    if by_check:
        console.print("  by check: " + ", ".join(f"{c} ({s}) x{n}" for (c, s), n in sorted(by_check.items())))
    if errors:
        table = Table("Name", "Check", "Detail", title="Errors")
        for f in errors[:25]:
            table.add_row(escape(f.name), f.check, escape(f.detail))
        console.print(table)


def _recompute_factors(conn, daily: BarStore, symbols: SymbolMap):
    from trader.data.bhavcopy import load_by_stock

    isin_by_symbol = {stem: (symbols.equity(stem).isin if symbols.equity(stem) else None) for stem in daily.stems()}
    return recompute_all(conn, daily, isin_by_symbol, utc_now(), load_by_stock(get_settings().official_dir))


def _rebuild_official_by_stock(conn) -> int:
    """Re-extract each stored stock's official history from the downloaded NSE files."""
    from trader.data.bhavcopy import build_by_stock, save_by_stock

    settings = get_settings()
    stocks = dict(conn.execute("SELECT name, isin FROM bar_meta WHERE kind = 'EQ'").fetchall())
    frame = build_by_stock(settings.official_dir, stocks)
    save_by_stock(frame, settings.official_dir)
    return int(frame["stock"].nunique()) if len(frame) else 0


def _sync_official(conn, cal: TradingCalendar, end: date) -> dict[str, int] | None:
    """Download any missing NSE daily files up to `end` (published around 18:00 IST on the day)."""
    from trader.data.bhavcopy import BhavcopyArchive

    settings = get_settings()
    if not settings.official_dir.exists():
        return None  # the archive is created by `trader data official-prices`
    done = {r[0] for r in conn.execute("SELECT date FROM official_days")}
    pending = [d for d in cal.sessions(DEFAULT_START, end) if d.isoformat() not in done]
    return BhavcopyArchive(settings.official_dir, conn).sync(pending[-30:]) if pending else {"ok": 0}


# Commands -----------------------------------------------------------------------------
@data_app.command()
def universe(index: Annotated[str, typer.Option(help="Constituent list: nifty500 or nifty50.")] = DEFAULT_INDEX) -> None:
    """Download today's constituent list and record it in the point-in-time membership table."""
    ref, _, _ = load_reference()
    conn = connect(get_settings().db_path)
    try:
        diff, mapping = _update_universe(conn, index, SymbolMap(ref))
        count = len(members(conn, index))
    finally:
        conn.close()
    kind = "initial load (start dates unknown: survivorship-biased before today)" if diff.initial_load else "update"
    console.print(f"{index} membership {kind} as of {diff.as_of}: {count} members; "
                  f"{len(diff.added)} added, {len(diff.removed)} removed, {diff.unchanged} unchanged; "
                  f"{int((mapping['status'] == 'placeholder').sum())} placeholder rows skipped")
    if diff.added:
        console.print("Added: " + ", ".join(diff.added))
    if diff.removed:
        console.print("Removed: " + ", ".join(diff.removed))


@data_app.command()
def calendar(
    refresh_holidays: Annotated[bool, typer.Option("--holidays/--no-holidays", help="Fetch this year's holiday list from Upstox.")] = True,
) -> None:
    """Show the trading calendar: today's status, recent and upcoming sessions, holidays, observed history."""
    conn = connect(get_settings().db_path)
    _, index_store = _stores()
    try:
        if refresh_holidays:
            console.print(f"Stored {_refresh_holidays(conn)} NSE holiday/special-session entries from Upstox")
        cal = _calendar(conn, index_store)
    finally:
        conn.close()
    now = datetime.now(IST)
    today = now.date()
    status = "a trading session" if cal.is_session(today) else (
        f"a holiday ({cal.holidays[today].description})" if today in cal.holidays else "a weekend day")
    console.print(f"Today {today} is {status}. Latest completed session: {cal.latest_completed_session(now, get_profile().session_close)}. "
                  f"Next session: {cal.next_session(today)}.")
    upcoming = [h for h in cal.holidays.values() if h.date >= today]
    if upcoming:
        table = Table("Date", "Day", "NSE", "Description", title=f"Upcoming NSE holidays and special sessions ({today.year})")
        for h in upcoming:
            table.add_row(h.date.isoformat(), h.date.strftime("%a"), "special session" if h.nse_special else "closed", escape(h.description))
        console.print(table)
    if cal.first_observed:
        table = Table("Year", "Sessions", "Weekday holidays", "Weekend sessions", title="Observed sessions (Nifty 50 bars)")
        for year in range(cal.first_observed.year, cal.last_observed.year + 1):
            start, end = max(date(year, 1, 1), cal.first_observed), min(date(year, 12, 31), cal.last_observed)
            days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
            sessions = [d for d in days if cal.is_session(d)]
            weekday_off = sum(1 for d in days if d.weekday() < 5 and not cal.is_session(d))
            weekend_on = sum(1 for d in sessions if d.weekday() >= 5)
            table.add_row(str(year), str(len(sessions)), str(weekday_off), str(weekend_on))
        console.print(table)


@data_app.command()
def backfill(
    index: Annotated[str, typer.Option(help="Universe to backfill (plus the Nifty 50 and Nifty 500 indices).")] = DEFAULT_INDEX,
    start: Annotated[str, typer.Option(help="First date, YYYY-MM-DD.")] = DEFAULT_START.isoformat(),
    end: Annotated[Optional[str], typer.Option(help="Last date (default: today in IST).")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-fetch instruments already backfilled from this start date.")] = False,
    only: Annotated[Optional[str], typer.Option(help="Comma-separated subset of symbols.")] = None,
) -> None:
    """Fetch full daily history for the universe and the indices. Resumable: completed instruments are skipped."""
    start_d, end_d = parse_date(start, DEFAULT_START), parse_date(end, today_ist())
    ref, _, _ = load_reference()
    symbols = SymbolMap(ref)
    daily, index_store = _stores()
    conn = connect(get_settings().db_path)
    run_id = _start_run(conn, "backfill", end_d)
    try:
        insts, missing = _instruments(conn, index, symbols, split_values([only]) if only else None)
        if missing:
            console.print(f"[yellow]Not in today's instrument master (skipped): {', '.join(missing)}[/]")
        started = time.perf_counter()
        with make_provider() as provider, _progress() as bar:
            task = bar.add_task("Backfilling", total=len(insts))
            results = BarSync(provider, daily, index_store, conn).backfill(
                insts, start_d, end_d, force=force, progress=lambda i: bar.update(task, advance=1, description=f"Backfilling {i.symbol:<12}"))
            requests = provider.request_count
        summary = _sync_summary(results, f"Backfill {start_d} -> {end_d}")
        summary.update(requests=requests, seconds=round(time.perf_counter() - started, 1), missing=missing)
        console.print(f"{requests:,} Upstox requests in {summary['seconds']:.0f}s")
        _finish_run(conn, run_id, "ok" if not summary["errors"] else "partial", summary)
    except BaseException as exc:
        conn.rollback()  # never keep a half-finished step
        _finish_run(conn, run_id, "failed", {"error": str(exc)})
        raise
    finally:
        conn.close()


@data_app.command()
def refresh(
    index: Annotated[str, typer.Option(help="Universe to refresh (plus the indices).")] = DEFAULT_INDEX,
    as_of: Annotated[Optional[str], typer.Option(help="Session to refresh through (default: latest completed session).")] = None,
    overlap: Annotated[int, typer.Option(help="Stored sessions re-fetched per instrument to catch revisions.")] = 10,
) -> None:
    """Daily update: recent history for every instrument, today's bars after the close, factors, quality checks."""
    ref, _, _ = load_reference()
    symbols = SymbolMap(ref)
    daily, index_store = _stores()
    conn = connect(get_settings().db_path)
    try:
        _refresh_holidays(conn)
        cal = _calendar(conn, index_store)
        now = datetime.now(IST)
        as_of_d = parse_date(as_of, cal.latest_completed_session(now, get_profile().session_close))
        run_id = _start_run(conn, "refresh", as_of_d)
        try:
            try:  # membership changes are recorded on the day NSE's list shows them
                diff, _ = _update_universe(conn, index, symbols)
                if diff.added or diff.removed:
                    console.print(f"{index}: added {diff.added or '-'}, removed {diff.removed or '-'}")
            except Exception as exc:
                console.print(f"[yellow]Constituent list not refreshed: {escape(str(exc))}[/]")
            insts, missing = _instruments(conn, index, symbols, include_stored=True)
            console.print(f"Refreshing {len(insts)} instruments through {as_of_d}"
                          + (" (today: live quotes supply today's bars)" if as_of_d == now.date() else ""))
            started = time.perf_counter()
            with make_provider() as provider, _progress() as bar:
                task = bar.add_task("Refreshing", total=len(insts))
                sync = BarSync(provider, daily, index_store, conn)
                for old, new in sync.reconcile_renames(insts):
                    console.print(f"Renamed {old} -> {new}")
                results = sync.refresh(
                    insts, as_of_d, today=now.date(), new_start=DEFAULT_START, overlap=overlap,
                    progress=lambda i: bar.update(task, advance=1, description=f"Refreshing {i.symbol:<12}"))
                requests = provider.request_count
            summary = _sync_summary(results, f"Refresh through {as_of_d}")
            summary.update(requests=requests, seconds=round(time.perf_counter() - started, 1), missing=missing)
            refreshed = {i.symbol for i in insts}
            stale = [r["name"] for r in conn.execute("SELECT name FROM bar_meta WHERE last_date < ? AND status = 'ok'",
                                                     (as_of_d.isoformat(),)) if r["name"] in refreshed]
            summary["stale"] = stale
            if stale:
                console.print(f"[yellow]{len(stale)} instruments have no bar for {as_of_d}: {', '.join(stale[:20])}[/]")
            official = _sync_official(conn, cal, as_of_d)
            if official is not None:
                summary["official_files"] = official.get("ok", 0)
                summary["official_stocks"] = _rebuild_official_by_stock(conn)
            try:  # pick up newly announced corporate actions (recent and upcoming ex-dates)
                with NSEClient() as nse:
                    recent = fetch_corporate_actions(nse, as_of_d - timedelta(days=45), now.date() + timedelta(days=120))
                summary["corporate_actions_fetched"] = store_actions(conn, recent, utc_now())
                conn.commit()
            except Exception as exc:  # NSE is independent of Upstox; stale actions are better than no refresh
                console.print(f"[yellow]NSE corporate actions not refreshed: {escape(str(exc))}[/]")
                summary["corporate_actions_fetched"] = None
            factors = _recompute_factors(conn, daily, symbols)
            summary["factors_applied"] = sum(f.applied for f in factors)
            findings, n_special = _run_quality(conn, run_id, daily, index_store)
            summary["dq_errors"] = sum(f.severity == "error" for f in findings)
            _print_quality(findings, n_special, len(daily.stems()) + len(index_store.stems()))
            console.print(f"{requests:,} Upstox requests in {summary['seconds']:.0f}s; "
                          f"{summary['corporate_actions_fetched'] or 0} recent NSE corporate actions checked; "
                          f"{summary['factors_applied']} corporate-action factors applied")
            _finish_run(conn, run_id, "ok" if not summary["errors"] and not stale else "partial", summary)
        except BaseException as exc:
            conn.rollback()  # never keep a half-finished step (e.g. factors deleted but not re-inserted)
            _finish_run(conn, run_id, "failed", {"error": str(exc)})
            raise
    finally:
        conn.close()


@data_app.command("corporate-actions")
def corporate_actions(
    fetch: Annotated[bool, typer.Option("--fetch/--no-fetch", help="Download NSE corporate actions first.")] = True,
    start: Annotated[str, typer.Option(help="First ex-date to download.")] = DEFAULT_START.isoformat(),
    show: Annotated[int, typer.Option(help="Applied events to list.")] = 25,
) -> None:
    """Download NSE corporate actions and compute the price factors for every stored stock."""
    ref, _, _ = load_reference()
    symbols = SymbolMap(ref)
    daily, _ = _stores()
    conn = connect(get_settings().db_path)
    try:
        if fetch:
            end = today_ist() + timedelta(days=120)  # include announced, upcoming ex-dates
            with NSEClient() as nse:
                rows = fetch_corporate_actions(nse, parse_date(start, DEFAULT_START), end)
            stored = store_actions(conn, rows, utc_now())
            conn.commit()
            console.print(f"Stored {stored:,} equity corporate actions from NSE ({len(rows):,} rows downloaded)")
        factors = _recompute_factors(conn, daily, symbols)
        actions = load_actions(conn)
    finally:
        conn.close()
    table = Table("Kind", "Events", "Applied by us", "Already in vendor data", "Not applicable", title="Corporate-action factors (stored stocks)")
    for kind in ("split", "bonus", "consolidation", "demerger", "arrangement", "history_break", "rights", "dividend"):
        rows_k = [f for f in factors if f.kind == kind]
        upstream = sum(1 for f in rows_k if f.method in ("verified_upstream", "assumed_upstream", "no_gap", "ambiguous"))
        applied = sum(f.applied for f in rows_k)
        table.add_row(kind, str(len(rows_k)), str(applied), str(upstream), str(len(rows_k) - applied - upstream))
    console.print(table)
    unadjusted = [f for f in factors if f.method == "vendor_unadjusted"]
    if unadjusted:
        console.print(f"[yellow]Split/bonus events missing from the vendor data (corrected by us): "
                      f"{', '.join(f'{f.symbol} {f.ex_date}' for f in unadjusted)}[/]")
    breaks = [f for f in factors if f.kind == "history_break"]
    if breaks:
        console.print("[yellow]History breaks (earlier bars dropped): "
                      + ", ".join(f"{f.symbol} before {f.ex_date}" for f in breaks) + "[/]")
    big = sorted((f for f in factors if f.applied and f.kind in ("demerger", "arrangement", "rights", "split", "bonus", "consolidation")),
                 key=lambda f: abs(1 - f.factor), reverse=True)
    if big:
        listing = Table("Symbol", "Ex-date", "Kind", "Factor", "Method", "Detail", title=f"Largest applied non-dividend adjustments (top {show})")
        for f in big[:show]:
            listing.add_row(escape(f.symbol), f.ex_date.isoformat(), f.kind, f"{f.factor:.4f}", f.method, escape(f.detail))
        console.print(listing)
    stored_symbols = set(daily.stems())
    review = unparsed_price_events(actions[actions["symbol"].isin(stored_symbols)])
    if not review.empty:
        listing = Table("Symbol", "Ex-date", "Subject", title="Possibly price-relevant subjects not parsed (review)")
        for row in review.sort_values("ex_date").to_dict("records")[-show:]:
            listing.add_row(escape(row["symbol"]), row["ex_date"], escape(row["subject"]))
        console.print(listing)


@data_app.command()
def status() -> None:
    """Coverage and health of the stored data."""
    settings = get_settings()
    conn = connect(settings.db_path)
    try:
        meta = pd.read_sql_query("SELECT * FROM bar_meta", conn)
        runs = pd.read_sql_query("SELECT run_id, status, started_at, finished_at, summary_json FROM data_runs ORDER BY started_at DESC LIMIT 5", conn)
        n_members = conn.execute("SELECT COUNT(*) FROM universe_membership WHERE end_date IS NULL").fetchone()[0]
        factor_counts = conn.execute("SELECT kind, SUM(applied), COUNT(*) FROM adjustment_factors GROUP BY kind").fetchall()
        n_actions = conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0]
        _, index_store = _stores()
        cal = _calendar(conn, index_store)
    finally:
        conn.close()
    latest = cal.latest_completed_session(datetime.now(IST), get_profile().session_close) if cal.first_observed else None
    console.print(f"Latest completed session: {latest}; current index members: {n_members}; "
                  f"corporate actions stored: {n_actions:,}")
    if not meta.empty:
        table = Table("Kind", "Files", "OK", "Errors", f"Current ({latest})", "Provisional last bar", "Bars", "Earliest", "Latest")
        for kind, g in meta.groupby("kind"):
            table.add_row(kind, str(len(g)), str(int((g["status"] == "ok").sum())), str(int((g["status"] == "error").sum())),
                          str(int((g["last_date"] == (latest.isoformat() if latest else "")).sum())), str(int(g["provisional"].sum())),
                          f"{int(g['rows'].sum()):,}", str(g["first_date"].min()), str(g["last_date"].max()))
        console.print(table)
    if factor_counts:
        console.print("Adjustment factors: " + ", ".join(f"{k} {int(a)}/{n} applied" for k, a, n in factor_counts))
    if not runs.empty:
        table = Table("Run", "Status", "Started (UTC)", "Summary")
        for row in runs.to_dict("records"):
            summary = json.loads(row["summary_json"] or "{}")
            short = ", ".join(f"{k}={v}" for k, v in summary.items() if k in ("ok", "skipped", "errors", "rows_added", "requests", "dq_errors"))
            table.add_row(row["run_id"], row["status"] or "-", row["started_at"], short)
        console.print(table)


@data_app.command("panel-check")
def panel_check(
    adjustment: Annotated[str, typer.Option(help="none, economic or total_return.")] = "economic",
    start: Annotated[Optional[str], typer.Option(help="First date (default: all history).")] = None,
) -> None:
    """Load the full adjusted panel the way the backtester will, time it and check its structure."""
    feed = LocalDataFeed()
    started = time.perf_counter()
    panel = feed.load_panel(start=parse_date(start, DEFAULT_START) if start else None, adjustment=adjustment)
    seconds = time.perf_counter() - started
    close = panel["close"]
    problems = panel_contract_check(panel)
    console.print(f"Panel ({adjustment}): {close.shape[1]} symbols x {close.shape[0]:,} dates, "
                  f"{fmt_date(close.index.min())} -> {fmt_date(close.index.max())}, "
                  f"{int(close.notna().sum().sum()):,} bars, loaded in {seconds:.2f}s")
    console.print("Contract check: " + ("[green]OK[/]" if not problems else "[red]" + escape("; ".join(problems)) + "[/]"))
    if problems:
        raise typer.Exit(1)


@data_app.command("official-prices")
def official_prices(
    start: Annotated[str, typer.Option(help="First session to download.")] = DEFAULT_START.isoformat(),
    end: Annotated[Optional[str], typer.Option(help="Last session (default: latest completed session).")] = None,
    retry_missing: Annotated[bool, typer.Option("--retry-missing", help="Retry days NSE had no file for.")] = False,
) -> None:
    """Download NSE's official daily price files (bhavcopy) for every session; resumable."""
    from trader.data.bhavcopy import BhavcopyArchive

    settings = get_settings()
    conn = connect(settings.db_path)
    _, index_store = _stores()
    try:
        cal = _calendar(conn, index_store)
        end_d = parse_date(end, cal.latest_completed_session(datetime.now(IST), get_profile().session_close))
        days = cal.sessions(parse_date(start, DEFAULT_START), end_d)
        archive = BhavcopyArchive(settings.official_dir, conn)
        started = time.perf_counter()
        with _progress() as bar:
            task = bar.add_task("Official prices", total=len(days))
            counts = archive.sync(days, retry_missing=retry_missing,
                                  progress=lambda d, s: bar.update(task, advance=1, description=f"Official prices {d}"))
        console.print(f"{len(days):,} sessions: {counts['ok']} downloaded, {counts['skipped']} already stored, "
                      f"{counts['missing']} not published, {counts['error']} errors ({time.perf_counter() - started:.0f}s)")
        console.print(f"Official history extracted for {_rebuild_official_by_stock(conn)} stored stocks")
    finally:
        conn.close()
