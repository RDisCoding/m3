"""Command-line interface. Run `uv run trader --help`."""

from typing import Annotated

import typer

from trader.cli.common import app
from trader.logging_setup import setup_logging
from trader.settings import get_settings


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False) -> None:
    settings = get_settings()
    token = settings.token_value()
    setup_logging("DEBUG" if verbose else settings.log_level, settings.logs_dir, secrets=[token] if token else [])


# Importing the command modules registers their commands on the shared Typer apps.
from trader.cli import data_basic, data_pipeline, research  # noqa: E402,F401

__all__ = ["app"]
