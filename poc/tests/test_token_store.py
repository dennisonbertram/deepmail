"""Tests for TokenStore: round-trip, file permissions, atomic write."""

from __future__ import annotations

import datetime as _dt
import json
import os
import stat
import sys
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from google.oauth2.credentials import Credentials  # noqa: E402

from pi_email.token_store import TokenStore  # noqa: E402


def _make_creds(token: str = "access-tok", refresh: str = "refresh-tok") -> Credentials:
    """Build a Credentials object with all the fields we serialize."""
    expiry = _dt.datetime(2026, 5, 18, 15, 30, 0)
    return Credentials(
        token=token,
        refresh_token=refresh,
        token_uri="https://oauth2.googleapis.com/token",
        client_id="abc.apps.googleusercontent.com",
        client_secret="some-secret",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        expiry=expiry,
    )


# ---------- Basic round-trip ----------


def test_load_returns_none_when_file_missing(tmp_path):
    store = TokenStore(path=tmp_path / "nonexistent.json")
    assert store.load() is None


def test_save_and_load_round_trip(tmp_path):
    store = TokenStore(path=tmp_path / "tokens.json")
    creds = _make_creds()
    store.save(creds)

    loaded = store.load()
    assert loaded is not None
    assert loaded.token == "access-tok"
    assert loaded.refresh_token == "refresh-tok"
    assert loaded.client_id == "abc.apps.googleusercontent.com"
    assert loaded.client_secret == "some-secret"
    assert loaded.token_uri == "https://oauth2.googleapis.com/token"
    assert "https://www.googleapis.com/auth/gmail.readonly" in (loaded.scopes or [])


def test_save_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "tokens.json"
    store = TokenStore(path=nested)
    store.save(_make_creds())
    assert nested.exists()


def test_save_with_empty_client_secret_still_round_trips(tmp_path):
    """Desktop-app PKCE may have an empty client_secret. The key MUST still
    be present in the serialized JSON or from_authorized_user_info will raise."""
    store = TokenStore(path=tmp_path / "tokens.json")
    creds = Credentials(
        token="t",
        refresh_token="r",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="id-1",
        client_secret=None,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )
    store.save(creds)
    # Inspect JSON directly to confirm the key is present.
    with open(store.path) as f:
        data = json.load(f)
    assert "client_secret" in data
    assert data["client_secret"] == ""
    # And confirm load works.
    loaded = store.load()
    assert loaded is not None
    assert loaded.client_id == "id-1"


# ---------- Permissions ----------


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_save_sets_0o600_permissions(tmp_path):
    store = TokenStore(path=tmp_path / "tokens.json")
    store.save(_make_creds())
    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


# ---------- Atomic write ----------


def test_atomic_write_does_not_corrupt_on_fsync_failure(tmp_path, monkeypatch):
    """If fsync() raises mid-write, the destination file must be left
    untouched (atomic-rename guarantees this — we only mutate dst when the
    tmp is fully written and fsynced)."""
    store = TokenStore(path=tmp_path / "tokens.json")
    # Pre-populate with a known-good token so we can verify it survives a
    # failed second save.
    store.save(_make_creds(token="old-token", refresh="old-refresh"))
    before = store.path.read_text(encoding="utf-8")

    # Mock os.fsync to raise — this happens AFTER the temp file is written
    # but BEFORE the rename. Atomicity means the destination is unchanged.
    real_fsync = os.fsync

    def boom(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="simulated"):
        store.save(_make_creds(token="new-token", refresh="new-refresh"))

    # Restore for cleanup.
    monkeypatch.setattr(os, "fsync", real_fsync)

    # Destination must still be the old contents — atomic rename never happened.
    after = store.path.read_text(encoding="utf-8")
    assert before == after

    # No leftover .tmp files in the dir.
    tmps = list(tmp_path.glob(".tokens.*.tmp"))
    assert tmps == [], f"stale tmp file(s): {tmps}"


def test_clear_removes_file(tmp_path):
    store = TokenStore(path=tmp_path / "tokens.json")
    store.save(_make_creds())
    assert store.path.exists()
    store.clear()
    assert not store.path.exists()
    # No-op when called again.
    store.clear()


# ---------- Default path uses platformdirs ----------


def test_default_path_uses_platformdirs(tmp_path, monkeypatch):
    """Without an explicit path, TokenStore derives the location via platformdirs.
    We just confirm the path includes the app name; the exact location is
    OS-dependent."""
    store = TokenStore(app_name="pi-email-test-app")
    assert "pi-email-test-app" in str(store.path)
    assert store.path.name == "tokens.json"
