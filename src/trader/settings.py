"""Runtime settings loaded from environment variables and .env files.

Secrets (the Upstox analytics token) are read from, in increasing priority:
  1. <workspace>/.env      e.g. D:\\eta\\.env (where the token currently lives)
  2. <project>/.env        e.g. D:\\eta\\trader\\.env
  3. process environment   UPSTOX_ACCESS_TOKEN=...
Both .env files are outside version control; the project's .gitignore excludes .env.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/trader/settings.py -> parents[2] is the project root (D:\eta\trader).
PROJECT_ROOT = Path(os.environ.get("TRADER_HOME") or Path(__file__).resolve().parents[2])
WORKSPACE_ENV = PROJECT_ROOT.parent / ".env"
PROJECT_ENV = PROJECT_ROOT / ".env"
TOKEN_ENV_NAME = "UPSTOX_ACCESS_TOKEN"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRADER_",
        env_file=(WORKSPACE_ENV, PROJECT_ENV),  # later files win
        env_file_encoding="utf-8-sig",  # tolerate a BOM written by Windows editors
        extra="ignore",
    )

    market: str = "IN_NSE"  # market profile code (see trader.market)
    data_dir: Path = PROJECT_ROOT / "data"
    tz: str = "Asia/Kolkata"
    log_level: str = "INFO"
    http_timeout_s: float = 30.0
    http_connect_timeout_s: float = 3.0

    upstox_base_url: str = "https://api.upstox.com"
    upstox_access_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(TOKEN_ENV_NAME, f"TRADER_{TOKEN_ENV_NAME}"),
    )

    # Derived locations -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def ref_dir(self) -> Path:
        return self.data_dir / "ref"

    @property
    def bars_dir(self) -> Path:
        return self.data_dir / "bars"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def official_dir(self) -> Path:
        return self.data_dir / "official" / "bhavcopy"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "trader.sqlite"

    def token_value(self) -> str | None:
        """The raw token for building the HTTP client. Never log or print this."""
        if self.upstox_access_token is None:
            return None
        value = self.upstox_access_token.get_secret_value().strip()
        return value or None


def token_source() -> str | None:
    """Where the token was found, without reading it into any log or output."""
    if os.environ.get(TOKEN_ENV_NAME) or os.environ.get(f"TRADER_{TOKEN_ENV_NAME}"):
        return "process environment"
    for path in (PROJECT_ENV, WORKSPACE_ENV):  # same priority order as Settings
        if path.is_file():
            values = dotenv_values(path, encoding="utf-8-sig")
            if values.get(TOKEN_ENV_NAME) or values.get(f"TRADER_{TOKEN_ENV_NAME}"):
                return str(path)
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
