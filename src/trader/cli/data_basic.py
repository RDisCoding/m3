"""Data commands: auth check, instrument master, mapping, fetch, show, verify, smoke."""

import time
from datetime import date
from typing import Annotated, Optional

import pandas as pd
import pyarrow.parquet as pq
import typer
from rich.markup import escape
from rich.table import Table

from trader.cli.common import (
    DEFAULT_START,
    DEFAULT_TEST_SET,
    Fetched,
    bar_meta_dict,
    console,
    data_app,
    expiry_text,
    fmt_date,
    load_reference,
    make_provider,
    parse_date,
    split_values,
    store_for,
    summarize_findings,
    today_ist,
)
from trader.data.instruments import EQUITY_TYPES, SymbolMap
from trader.data.providers.base import Instrument, ProviderError
from trader.data.quality import Finding, run_checks
from trader.data.universe import (
    download_constituents,
    load_constituents,
    map_constituents,
    mapping_coverage,
)
from trader.settings import get_settings, token_source
from trader.storage.parquet import BAR_SCHEMA, BarStore


# Commands -----------------------------------------------------------------------------
@data_app.command("auth-check")
def auth_check() -> None:
    """Check that the Upstox token is configured and accepted (one read-only quote call)."""
    token = get_settings().token_value()
    if not token:
        console.print("[red]FAIL[/] UPSTOX_ACCESS_TOKEN not found in the environment, trader/.env or the workspace .env")
        raise typer.Exit(1)
    console.print(f"Token: found in {escape(str(token_source()))}, {len(token)} characters (value not shown), {expiry_text(token)}")
    with make_provider() as provider:
        health = provider.healthcheck()
    if not health.ok:
        console.print(f"[red]FAIL[/] {escape(health.detail)}")
        raise typer.Exit(1)
    console.print(f"[green]PASS[/] {escape(health.detail)} ({health.latency_ms:.0f} ms)")


@data_app.command()
def instruments(
    refresh: Annotated[bool, typer.Option("--refresh", help="Download even if today's copy exists.")] = False,
) -> None:
    """Download (at most once per day) and summarize the Upstox NSE instrument master."""
    ref, as_of, downloaded = load_reference(refresh)
    equity_segment = ref[ref["segment"] == "NSE_EQ"]
    shares = equity_segment[equity_segment["instrument_type"].isin(EQUITY_TYPES)]
    etfs = shares[(shares["instrument_type"] == "EQ") & shares["isin"].str.startswith("INF").fillna(False)]
    console.print(f"Instrument master as of {as_of} ({'downloaded now' if downloaded else 'cached copy from today'})")
    table = Table("Group", "Rows")
    table.add_row("NSE_EQ segment (all series, incl. bonds)", f"{len(equity_segment):,}")
    for series in EQUITY_TYPES:
        table.add_row(f"  share series {series}", f"{int((shares['instrument_type'] == series).sum()):,}")
    table.add_row("  of which ETFs (EQ series, INF ISIN)", f"{len(etfs):,}")
    table.add_row("NSE_INDEX", f"{int((ref['segment'] == 'NSE_INDEX').sum()):,}")
    console.print(table)
    ticks = shares[shares["instrument_type"] == "EQ"]["tick_size"].value_counts().sort_index()
    console.print("Tick sizes for EQ series (rupees): " + ", ".join(f"{t:.2f} x{n}" for t, n in ticks.items()))


@data_app.command("map")
def map_symbols(
    names: Annotated[Optional[list[str]], typer.Argument(help="Symbols, ISINs, instrument keys or index names.")] = None,
    index: Annotated[Optional[str], typer.Option(help="Also map a constituent list: nifty50 or nifty500.")] = None,
    show: Annotated[int, typer.Option(help="Matched constituent rows to print.")] = 5,
) -> None:
    """Show symbol -> ISIN -> Upstox instrument_key mappings."""
    ref, _, _ = load_reference()
    symbols = SymbolMap(ref)
    failed = False
    wanted = split_values(names)
    if wanted:
        table = Table("Input", "Symbol", "ISIN", "Instrument key", "Series", "Tick", "Name")
        for name in wanted:
            try:
                inst = symbols.resolve(name)
            except KeyError as exc:
                table.add_row(escape(name), "[red]not found[/]", "", "", "", "", escape(str(exc)))
                failed = True
                continue
            tick = f"{inst.tick_size:.2f}" if inst.tick_size is not None else "-"
            table.add_row(escape(name), escape(inst.symbol), inst.isin or "-", escape(inst.instrument_key),
                          inst.instrument_type or "-", tick, escape(inst.name))
        console.print(table)
    if index:
        settings = get_settings()
        path = download_constituents(index, settings.raw_dir / "universe", today=today_ist())
        mapping = map_constituents(load_constituents(path), symbols)
        out = settings.ref_dir / f"{index}_mapping.csv"
        mapping.to_csv(out, index=False)
        mapped, eligible = mapping_coverage(mapping)
        console.print(f"{index}: {mapped}/{eligible} tradable constituents mapped ({mapped / eligible:.1%}); "
                      f"statuses {mapping['status'].value_counts().to_dict()}; saved {escape(str(out))}")
        table = Table("Symbol", "ISIN", "Instrument key", "Upstox symbol", "Series", "Industry", "Status")
        problems = mapping[mapping["status"] != "ok"]
        for row in pd.concat([problems, mapping[mapping["status"] == "ok"].head(show)]).to_dict("records"):
            table.add_row(*(escape(str(row[c])) if row[c] is not None else "-" for c in
                            ("symbol", "isin", "instrument_key", "upstox_symbol", "upstox_type", "industry", "status")))
        console.print(table)
        failed = failed or mapped / eligible < 0.98
    if not wanted and not index:
        console.print("Nothing to map: pass symbols and/or --index nifty500")
    if failed:
        raise typer.Exit(1)


@data_app.command()
def fetch(
    names: Annotated[list[str], typer.Argument(help="Symbols, ISINs, instrument keys or index names (quote names with spaces).")],
    start: Annotated[str, typer.Option(help="First date, YYYY-MM-DD.")] = DEFAULT_START,
    end: Annotated[Optional[str], typer.Option(help="Last date, YYYY-MM-DD (default: today in IST).")] = None,
    store: Annotated[bool, typer.Option("--store/--no-store", help="Upsert into Parquet storage.")] = True,
) -> None:
    """Fetch daily candles from Upstox and (by default) store them as Parquet."""
    start_d, end_d = parse_date(start, date(2015, 1, 1)), parse_date(end, today_ist())
    ref, _, _ = load_reference()
    symbols = SymbolMap(ref)
    results: list[Fetched] = []
    failed = False
    with make_provider() as provider:
        started = time.perf_counter()
        for name in split_values(names):
            try:
                inst = symbols.resolve(name)
                bars = provider.fetch_daily(inst, start_d, end_d)
            except (KeyError, ProviderError) as exc:
                console.print(f"[red]{escape(name)}[/]: {escape(str(exc))}")
                failed = True
                continue
            upsert = store_for(inst).upsert(inst.symbol, bars, source="upstox", meta=bar_meta_dict(inst)) if store and not bars.empty else None
            results.append(Fetched(inst, bars, upsert))
        elapsed = time.perf_counter() - started
        requests = provider.request_count
    table = Table("Name", "Instrument key", "Bars", "First", "Last", "Last close", "Added", "Updated", "File written")
    for r in results:
        last_close = f"{r.bars['close'].iloc[-1]:,.2f}" if not r.bars.empty else "-"
        u = r.upsert
        table.add_row(escape(r.inst.symbol), escape(r.inst.instrument_key), f"{len(r.bars):,}",
                      fmt_date(r.bars["date"].min() if len(r.bars) else None),
                      fmt_date(r.bars["date"].max() if len(r.bars) else None), last_close,
                      str(u.added) if u else "-", str(u.updated) if u else "-", ("yes" if u.written else "no change") if u else "-")
    console.print(table)
    console.print(f"{requests} Upstox requests in {elapsed:.1f}s")
    if failed:
        raise typer.Exit(1)


@data_app.command()
def show(
    name: Annotated[str, typer.Argument(help="Stored symbol or index name.")],
    tail: Annotated[int, typer.Option(help="Rows to print from the end.")] = 5,
) -> None:
    """Print a stored Parquet file's metadata and last rows."""
    settings = get_settings()
    for kind in ("daily", "index"):
        store = BarStore(settings.bars_dir / kind)
        if not store.exists(name):
            continue
        df, meta = store.read(name), store.read_meta(name)
        console.print(f"{escape(str(store.path(name)))}: {len(df):,} rows {fmt_date(df['date'].min())} -> "
                      f"{fmt_date(df['date'].max())}; meta {escape(str(meta))}")
        table = Table(*df.columns)
        for row in df.tail(tail).itertuples(index=False):
            table.add_row(fmt_date(row.date), *(f"{v:,.2f}" for v in (row.open, row.high, row.low, row.close)),
                          f"{row.volume:,}", row.source, pd.Timestamp(row.ingested_at).strftime("%Y-%m-%d %H:%M:%S UTC"))
        console.print(table)
        return
    console.print(f"[red]No stored bars for {escape(name)}[/]")
    raise typer.Exit(1)


@data_app.command()
def verify(
    names: Annotated[Optional[list[str]], typer.Argument(help="Stored symbols to check (default: all).")] = None,
    reference: Annotated[str, typer.Option(help="Stored index whose sessions define the calendar.")] = "Nifty 50",
    raw: Annotated[bool, typer.Option("--raw", help="Check the vendor's stored files instead of the corrected series.")] = False,
) -> None:
    """Run data-quality checks; exits non-zero if any error is found.

    By default checks the corrected series strategies read (corporate actions applied, NSE official
    prices, pre-listing fragments dropped); --raw checks the vendor's files as stored. Sessions the
    whole market lacks in the vendor data are not counted as per-stock gaps."""
    from trader.data.feed import LocalDataFeed

    settings = get_settings()
    daily, index_store = BarStore(settings.bars_dir / "daily"), BarStore(settings.bars_dir / "index")
    feed = LocalDataFeed(settings)
    ref_dates = pd.DatetimeIndex(index_store.read_bars(reference)["date"]) if index_store.exists(reference) else None
    if ref_dates is None:
        console.print(f"[yellow]Reference index {escape(reference)} is not stored; gap checks skipped[/]")
    else:
        flags = feed.session_flags()
        ref_dates = ref_dates.difference(flags.index[flags["sparse"].to_numpy()])
    targets = split_values(names) or daily.stems()
    findings: list[Finding] = []
    for name in targets:
        if not daily.exists(name):
            findings.append(Finding(name, "not_stored", "error", "no Parquet file"))
            continue
        bars = daily.read_bars(name) if raw else feed.load_bars(name)
        findings.extend(run_checks(name, bars, reference_dates=ref_dates))
    if index_store.exists(reference):
        findings.extend(run_checks(reference, index_store.read_bars(reference), expect_volume=False))
    table = Table("Name", "Severity", "Check", "Detail")
    for f in findings:
        colour = {"error": "red", "warn": "yellow"}.get(f.severity, "white")
        table.add_row(escape(f.name), f"[{colour}]{f.severity}[/]", f.check, escape(f.detail))
    if findings:
        console.print(table)
    errors = sum(f.severity == "error" for f in findings)
    console.print(f"Checked {len(targets)} symbols ({'stored vendor files' if raw else 'corrected series'}): "
                  f"{errors} errors, {sum(f.severity == 'warn' for f in findings)} warnings")
    if errors:
        raise typer.Exit(1)


@data_app.command()
def smoke(
    symbols: Annotated[str, typer.Option(help="Comma-separated test symbols.")] = DEFAULT_TEST_SET,
    start: Annotated[str, typer.Option(help="First date to fetch, YYYY-MM-DD.")] = DEFAULT_START,
    index: Annotated[str, typer.Option(help="Index fetched alongside; its sessions are the gap-check calendar.")] = "Nifty 50",
    constituents: Annotated[str, typer.Option(help="Constituent list for the mapping-coverage check.")] = "nifty500",
) -> None:
    """End-to-end check: auth, instrument master, mapping, daily candles, Parquet storage."""
    settings = get_settings()
    steps: list[tuple[str, str, str]] = []

    def record(step: str, ok: Optional[bool], detail: str) -> None:
        steps.append((step, "PASS" if ok else ("WARN" if ok is None else "FAIL"), detail))

    def finish(rows: Optional[list[tuple[str, ...]]] = None) -> None:
        if rows:
            detail = Table("Name", "Instrument key", "Bars", "First", "Last", "Last close", "Quality")
            for row in rows:
                detail.add_row(*(escape(c) for c in row))
            console.print(detail)
        summary = Table("Step", "Result", "Detail")
        for step, status, text in steps:
            colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[status]
            summary.add_row(step, f"[{colour}]{status}[/]", escape(text))
        console.print(summary)
        if any(status == "FAIL" for _, status, _ in steps):
            raise typer.Exit(1)

    token = settings.token_value()
    if not token:
        record("1 Upstox authentication", False, "UPSTOX_ACCESS_TOKEN not configured")
        finish()
        return

    with make_provider() as provider:
        # 1. Authentication -------------------------------------------------------------
        health = provider.healthcheck()
        latency = f" ({health.latency_ms:.0f} ms)" if health.latency_ms else ""
        record("1 Upstox authentication", health.ok, f"{health.detail}{latency}; token {expiry_text(token)}")
        if not health.ok:
            finish()
            return

        # 2. Instrument master ----------------------------------------------------------
        ref, as_of, downloaded = load_reference()
        symbol_map = SymbolMap(ref)
        n_indices = int((ref["segment"] == "NSE_INDEX").sum())
        index_inst = symbol_map.index(index)
        record("2 NSE instrument master",
               len(symbol_map.equity_symbols) >= 2000 and index_inst is not None and symbol_map.index("Nifty 500") is not None,
               f"as of {as_of} ({'downloaded' if downloaded else 'cached today'}): "
               f"{len(symbol_map.equity_symbols):,} share-series symbols, {n_indices} indices")

        # 3. Mapping ----------------------------------------------------------------------
        wanted = split_values([symbols])
        resolved: list[Instrument] = []
        unresolved: list[str] = []
        for name in wanted:
            try:
                resolved.append(symbol_map.resolve(name))
            except KeyError as exc:
                unresolved.append(str(exc))
        coverage_ok: Optional[bool]
        try:
            path = download_constituents(constituents, settings.raw_dir / "universe", today=today_ist())
            mapping = map_constituents(load_constituents(path), symbol_map)
            settings.ref_dir.mkdir(parents=True, exist_ok=True)
            mapping.to_csv(settings.ref_dir / f"{constituents}_mapping.csv", index=False)
            mapped, eligible = mapping_coverage(mapping)
            coverage_ok = mapped / eligible >= 0.98
            coverage = (f"; {constituents}: {mapped}/{eligible} tradable constituents mapped ({mapped / eligible:.1%}), "
                        f"statuses {mapping['status'].value_counts().to_dict()}")
        except Exception as exc:  # the constituent source is independent of Upstox
            coverage_ok, coverage = None, f"; constituent list unavailable: {exc}"
        base_ok = not unresolved and index_inst is not None
        record("3 Symbol/ISIN -> instrument_key",
               (base_ok and coverage_ok) if coverage_ok is not None else (None if base_ok else False),
               f"{len(resolved)}/{len(wanted)} test symbols resolved{coverage}"
               + (f"; unresolved: {unresolved}" if unresolved else ""))

        # 4. Daily candles --------------------------------------------------------------
        targets = resolved + ([index_inst] if index_inst is not None else [])
        start_d, end_d = parse_date(start, date(2015, 1, 1)), today_ist()
        fetched: list[Fetched] = []
        errors: list[str] = []
        before, started = provider.request_count, time.perf_counter()
        for inst in targets:
            try:
                fetched.append(Fetched(inst, provider.fetch_daily(inst, start_d, end_d), None))
            except ProviderError as exc:
                errors.append(f"{inst.symbol}: {exc}")
        empty = [f.inst.symbol for f in fetched if f.bars.empty]
        record("4 Daily OHLCV candles", not errors and not empty and len(fetched) == len(targets),
               f"{len(fetched)}/{len(targets)} instruments, {sum(len(f.bars) for f in fetched):,} bars "
               f"{start_d} -> {end_d}, {provider.request_count - before} requests in {time.perf_counter() - started:.1f}s"
               + (f"; errors: {errors}" if errors else "") + (f"; empty: {empty}" if empty else ""))

        # 4b. Today's candle (informational: the historical endpoint stops at the previous session)
        if resolved:
            try:
                today_bar = provider.fetch_today(resolved[0])
                if today_bar.empty:
                    record("  today's candle (intraday)", None, f"{resolved[0].symbol}: none returned (no session today, or not yet available)")
                else:
                    row = today_bar.iloc[-1]
                    last_hist = max((f.bars["date"].max() for f in fetched if not f.bars.empty), default=None)
                    record("  today's candle (intraday)", True,
                           f"{resolved[0].symbol} {row['date']:%Y-%m-%d} close {row['close']:,.2f} (provisional; not stored); "
                           f"historical data ends {fmt_date(last_hist)}")
            except ProviderError as exc:
                record("  today's candle (intraday)", None, f"intraday endpoint failed: {exc}")

    # 5. Parquet storage ----------------------------------------------------------------
    ref_dates = next((pd.DatetimeIndex(f.bars["date"]) for f in fetched if f.inst.kind == "INDEX"), None)
    problems: list[str] = []
    all_findings: list[Finding] = []
    rows: list[tuple[str, ...]] = []
    for f in fetched:
        if f.bars.empty:
            continue
        store = store_for(f.inst)
        store.upsert(f.inst.symbol, f.bars, source="upstox", meta=bar_meta_dict(f.inst))
        stored = store.read_bars(f.inst.symbol)
        on_fetched_dates = stored[stored["date"].isin(f.bars["date"])].reset_index(drop=True)
        if not on_fetched_dates.equals(f.bars.reset_index(drop=True)):
            problems.append(f"{f.inst.symbol}: stored values differ from fetched values")
        if not pq.read_schema(store.path(f.inst.symbol)).remove_metadata().equals(BAR_SCHEMA):
            problems.append(f"{f.inst.symbol}: file schema differs from BAR_SCHEMA")
        if store.upsert(f.inst.symbol, f.bars, source="upstox", meta=bar_meta_dict(f.inst)).written:
            problems.append(f"{f.inst.symbol}: re-upserting identical bars rewrote the file")
        findings = run_checks(f.inst.symbol, stored, reference_dates=ref_dates if f.inst.kind == "EQ" else None,
                              expect_volume=f.inst.kind == "EQ")
        all_findings.extend(findings)
        rows.append((f.inst.symbol, f.inst.instrument_key, f"{len(stored):,}", fmt_date(stored["date"].min()),
                     fmt_date(stored["date"].max()), f"{stored['close'].iloc[-1]:,.2f}", summarize_findings(findings)))
    n_errors = sum(f.severity == "error" for f in all_findings)
    n_warn = sum(f.severity == "warn" for f in all_findings)
    record("5 Parquet storage", bool(rows) and not problems and n_errors == 0,
           f"{len(rows)} files under {settings.bars_dir}; schema matches, stored = fetched, re-upsert is a no-op"
           if not problems else "; ".join(problems))
    record("  data quality", n_errors == 0 if not n_warn else (None if n_errors == 0 else False),
           f"{n_errors} errors, {n_warn} warnings" + (" (details: uv run trader data verify)" if n_warn or n_errors else ""))
    finish(rows)
