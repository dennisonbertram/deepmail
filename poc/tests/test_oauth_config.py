"""Tests for OAuthConfig.from_env() and to_client_config().

We do NOT exercise the live OAuth flow — that needs a real GCP client and a
browser. We test the config shape only: env-var reading, defaults, .env file
discovery, and the dict format the InstalledAppFlow consumes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.oauth import (  # noqa: E402
    DEFAULT_SCOPES,
    GMAIL_READONLY_SCOPE,
    OAuthConfig,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip Google env vars before every test; restore via monkeypatch."""
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    # Also pin cwd to a tmp so a stray .env in the repo root can't leak in.
    yield


def test_from_env_reads_client_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "my-client-id.apps.googleusercontent.com")
    cfg = OAuthConfig.from_env()
    assert cfg.client_id == "my-client-id.apps.googleusercontent.com"
    assert cfg.client_secret is None
    assert cfg.scopes == DEFAULT_SCOPES


def test_from_env_reads_client_secret(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id-1")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "shhh-not-really-secret")
    cfg = OAuthConfig.from_env()
    assert cfg.client_id == "id-1"
    assert cfg.client_secret == "shhh-not-really-secret"


def test_from_env_raises_when_client_id_missing(monkeypatch, tmp_path):
    # Run from a directory with no .env so dotenv discovery can't find one.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        OAuthConfig.from_env()


def test_from_env_empty_string_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "   ")  # whitespace only
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        OAuthConfig.from_env()


def test_from_env_loads_dotenv(monkeypatch, tmp_path):
    """A .env file in CWD should be picked up if env vars aren't already set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_CLIENT_ID=dotenv-client-id\n"
        "GOOGLE_CLIENT_SECRET=dotenv-secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = OAuthConfig.from_env()
    assert cfg.client_id == "dotenv-client-id"
    assert cfg.client_secret == "dotenv-secret"


def test_from_env_custom_scopes(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id-1")
    cfg = OAuthConfig.from_env(scopes=["https://www.googleapis.com/auth/userinfo.email"])
    assert cfg.scopes == ["https://www.googleapis.com/auth/userinfo.email"]


def test_to_client_config_shape():
    cfg = OAuthConfig(
        client_id="abc.apps.googleusercontent.com",
        client_secret="some-secret",
        scopes=[GMAIL_READONLY_SCOPE],
    )
    out = cfg.to_client_config()
    assert "installed" in out
    inst = out["installed"]
    assert inst["client_id"] == "abc.apps.googleusercontent.com"
    assert inst["client_secret"] == "some-secret"
    assert inst["auth_uri"] == "https://accounts.google.com/o/oauth2/auth"
    assert inst["token_uri"] == "https://oauth2.googleapis.com/token"
    assert inst["auth_provider_x509_cert_url"] == "https://www.googleapis.com/oauth2/v1/certs"
    assert inst["redirect_uris"] == ["http://localhost"]


def test_to_client_config_secret_none_becomes_empty_string():
    """Desktop apps under PKCE may omit the secret; the config still must
    include the key (empty string) because Credentials.from_authorized_user_info
    requires it to be present."""
    cfg = OAuthConfig(client_id="abc", client_secret=None, scopes=DEFAULT_SCOPES)
    out = cfg.to_client_config()
    assert out["installed"]["client_secret"] == ""


def test_to_client_config_is_consumable_by_installedappflow():
    """Smoke check: the dict shape we produce is what InstalledAppFlow expects.
    We don't run the flow — just instantiate from_client_config to verify it
    doesn't reject our shape."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    cfg = OAuthConfig(client_id="abc", client_secret="def", scopes=DEFAULT_SCOPES)
    # If our shape were wrong, this would raise ValueError("Client secrets must
    # be for a web or installed app.")
    flow = InstalledAppFlow.from_client_config(cfg.to_client_config(), cfg.scopes)
    assert flow.client_type == "installed"
