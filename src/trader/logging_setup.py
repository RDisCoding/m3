"""Logging: rich console output plus a rotating file, with secret redaction on every handler."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

REDACTED = "***REDACTED***"


class RedactSecretsFilter(logging.Filter):
    """Replaces any configured secret appearing in a log message.

    Attached to handlers (not loggers) so it also covers records from child loggers.
    """

    def __init__(self, secrets: Iterable[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s and len(s) >= 8]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, REDACTED)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


_configured = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None, secrets: Iterable[str] = ()) -> None:
    global _configured
    if _configured:
        return
    redact = RedactSecretsFilter(secrets)

    console = RichHandler(show_path=False, rich_tracebacks=False, markup=False)
    console.setLevel(level)
    console.addFilter(redact)
    handlers: list[logging.Handler] = [console]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "trader.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        file_handler.addFilter(redact)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in handlers:
        root.addHandler(handler)
    # httpx logs every request line at INFO; keep it quiet (headers are never logged either way).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True
