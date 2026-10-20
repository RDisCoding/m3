import logging

import pytest

from trader.logging_setup import REDACTED, RedactSecretsFilter
from trader.settings import Settings

FAKE = "eyJfake.payload-not-a-real-token.signature"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TRADER_UPSTOX_ACCESS_TOKEN", raising=False)


def test_token_is_read_from_env_file_and_never_shown(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"UPSTOX_ACCESS_TOKEN={FAKE}\n", encoding="utf-8")
    settings = Settings(_env_file=env)
    assert settings.token_value() == FAKE
    assert FAKE not in repr(settings) and FAKE not in str(settings.model_dump())


def test_env_file_with_bom_is_accepted(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfUPSTOX_ACCESS_TOKEN=" + FAKE.encode() + b"\r\n")
    assert Settings(_env_file=env).token_value() == FAKE


def test_process_environment_overrides_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("UPSTOX_ACCESS_TOKEN=from-file-value\n", encoding="utf-8")
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "from-env-value")
    assert Settings(_env_file=env).token_value() == "from-env-value"


def test_later_env_file_wins(tmp_path):
    workspace, project = tmp_path / "workspace.env", tmp_path / "project.env"
    workspace.write_text("UPSTOX_ACCESS_TOKEN=workspace-value\n", encoding="utf-8")
    project.write_text("UPSTOX_ACCESS_TOKEN=project-value\n", encoding="utf-8")
    assert Settings(_env_file=(workspace, project)).token_value() == "project-value"


def test_blank_token_counts_as_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("UPSTOX_ACCESS_TOKEN=\n", encoding="utf-8")
    assert Settings(_env_file=env).token_value() is None


def test_log_filter_redacts_the_token():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "calling with %s", (FAKE,), None)
    RedactSecretsFilter([FAKE]).filter(record)
    assert FAKE not in record.getMessage() and REDACTED in record.getMessage()
