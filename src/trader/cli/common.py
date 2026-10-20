"""Shared CLI objects and helpers."""

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from trader.data.bars import file_meta
from trader.data.instruments import ensure_reference
from trader.data.providers.base import Instrument
from trader.data.providers.upstox import UpstoxProvider, token_expiry
from trader.data.quality import Finding
from trader.settings import get_settings
from trader.storage.parquet import BarStore, UpsertResult

if hasattr(sys.stdout, "reconfigure"):
    # UTF-8 for pipes and log files; real consoles already use UTF-8 via the Windows console API.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console(width=None if sys.stdout.isatty() else 160)
app = typer.Typer(no_args_is_help=True, add_completion=False, help="NSE paper-trading research system.")
data_app = typer.Typer(no_args_is_help=True, help="Market data: Upstox auth, instruments, candles, storage.")
app.add_typer(data_app, name="data")

IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_TEST_SET = "RELIANCE,TCS,INFY,HDFCBANK,ITC,BAJFINANCE"
DEFAULT_START = "2015-01-01"


# Helpers ------------------------------------------------------------------------------
def today_ist() -> date:
    return datetime.now(IST).date()


def parse_date(text: Optional[str], default: date) -> date:
    if not text:
        return default
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {text!r}") from exc


def split_values(values: Optional[list[str]]) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def make_provider() -> UpstoxProvider:
    settings = get_settings()
    return UpstoxProvider(
        settings.token_value(),
        base_url=settings.upstox_base_url,
        timeout_s=settings.http_timeout_s,
        connect_timeout_s=settings.http_connect_timeout_s,
    )


def load_reference(refresh: bool = False) -> tuple[pd.DataFrame, date, bool]:
    settings = get_settings()
    return ensure_reference(settings.raw_dir, settings.ref_dir, refresh=refresh, today=today_ist())


def store_for(inst: Instrument) -> BarStore:
    return BarStore(get_settings().bars_dir / ("index" if inst.kind == "INDEX" else "daily"))


def bar_meta_dict(inst: Instrument) -> dict:
    return file_meta(inst)


def fmt_date(value: object) -> str:
    return "-" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def expiry_text(token: str) -> str:
    expiry = token_expiry(token)
    if expiry is None:
        return "expiry unknown (token is not a JWT)"
    days_left = (expiry - datetime.now(timezone.utc)).days
    return f"expires {expiry.astimezone(IST):%Y-%m-%d %H:%M} IST ({days_left} days left)"


def summarize_findings(findings: list[Finding]) -> str:
    if not findings:
        return "clean"
    return "; ".join(f"{f.severity}: {f.check}" for f in findings)


@dataclass
class Fetched:
    inst: Instrument
    bars: pd.DataFrame
    upsert: Optional[UpsertResult]
