"""Commands for the market profile and the feature store."""

import time
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from rich.markup import escape
from rich.table import Table

from trader.cli.common import app, console, parse_date
from trader.core.panel import Panel
from trader.data.feed import LocalDataFeed
from trader.features.features import (
    FeatureConfig,
    build_features,
    lookahead_differences,
)
from trader.market import get_profile

market_app = typer.Typer(no_args_is_help=True, help="The active market profile.")
features_app = typer.Typer(no_args_is_help=True, help="Indicators computed for strategies.")
app.add_typer(market_app, name="market")
app.add_typer(features_app, name="features")


@market_app.command("show")
def market_show() -> None:
    """Print the active market profile: the only place market-specific rules live."""
    p = get_profile()
    table = Table("Setting", "Value", title=f"Market profile {p.code}")
    rows = [
        ("Name", p.name), ("Currency", p.currency), ("Time zone", p.timezone),
        ("Session", f"{p.session_open:%H:%M}-{p.session_close:%H:%M}"),
        ("Calendar index", p.calendar_index), ("Regime index", p.regime_index), ("Benchmarks", ", ".join(p.benchmarks)),
        ("Risk-free rate", f"{p.risk_free_rate:.2%}"), ("Trading days per year", str(p.trading_days_per_year)),
        ("Settlement", f"T+{p.settlement_days}"),
        ("Share quantities", "fractional" if p.shares.fractional else f"whole shares, minimum {p.shares.min_quantity:g}"),
        ("Tick sizes", ", ".join(f"<{u:g}: {t:g}" if u != float("inf") else f"above: {t:g}" for u, t in p.ticks.tiers)),
        ("Price bands", type(p.price_bands).__name__), ("Cost model", p.cost_model.name),
        ("Short selling", "allowed" if p.shorting_allowed else "not used"),
    ]
    for key, value in rows:
        table.add_row(key, escape(str(value)))
    console.print(table)


@features_app.command("check")
def features_check(
    days: Annotated[int, typer.Option(help="Random sessions to test for look-ahead.")] = 12,
    window: Annotated[int, typer.Option(help="Live-trading window (sessions) compared with full history.")] = 600,
    start: Annotated[Optional[str], typer.Option(help="First date of the panel (default: all history).")] = None,
    seed: Annotated[int, typer.Option(help="Seed for picking the test dates.")] = 7,
) -> None:
    """Build every feature on the real panel, time it, and prove no feature uses future data."""
    profile = get_profile()
    feed = LocalDataFeed(profile=profile)
    started = time.perf_counter()
    panel = Panel.from_feed(feed, start=parse_date(start, None) if start else None)
    load_s = time.perf_counter() - started
    cfg = FeatureConfig.for_profile(profile)
    started = time.perf_counter()
    store = build_features(panel, cfg, index_name=profile.regime_index)
    build_s = time.perf_counter() - started
    console.print(f"Panel: {len(panel.symbols)} symbols x {len(panel.dates):,} sessions (loaded in {load_s:.1f}s); "
                  f"{len(store.names)} features built in {build_s:.1f}s; "
                  f"{len(store.gap_symbols)} symbols with gaps recomputed on their own bars")

    eligible = panel.dates[cfg.warmup_bars:]
    rng = np.random.default_rng(seed)
    picks = sorted(pd.DatetimeIndex(rng.choice(eligible, size=min(days, len(eligible)), replace=False)))
    started = time.perf_counter()
    exact = lookahead_differences(panel, picks, cfg, full=store)
    live = lookahead_differences(panel, [d for d in picks if panel.dates.get_loc(d) >= window], cfg, window=window, full=store)
    console.print(f"Look-ahead test on {len(picks)} random sessions ({time.perf_counter() - started:.0f}s): "
                  + ("[green]features identical with data ending that day[/]" if not exact else "[red]" + escape("; ".join(exact)) + "[/]"))
    console.print(f"Live-window test ({window} sessions): "
                  + ("[green]within 1e-6 of full history[/]" if not live else "[red]" + escape("; ".join(live)) + "[/]"))

    last = store.at(panel.dates[-1])
    table = Table("Feature", "Coverage on last day", "Median", title=f"Features on {panel.dates[-1]:%Y-%m-%d}")
    for name in store.names:
        col = last[name]
        table.add_row(name, f"{col.notna().mean():.0%}", f"{col.median():,.3f}" if col.notna().any() else "-")
    console.print(table)
    if exact or live:
        raise typer.Exit(1)
