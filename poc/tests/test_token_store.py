"""Tests for TokenStore: round-trip, file permissions, atomic write, multi-account."""

from __future__ import annotations

import datetime as _dt
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

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


# ---------- Basic round-trip (legacy single-file mode via path=) ----------


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
    # Patch _maybe_migrate to avoid side effects in default mode.
    with patch.object(TokenStore, "_maybe_migrate"):
        store = TokenStore(app_name="pi-email-test-app")
    assert "pi-email-test-app" in str(store.path)
    assert store.path.name == "tokens.json"


# ==========================================================================
# Multi-account tests
# ==========================================================================


class TestMultiAccountSaveLoad:

    def test_save_and_load_two_accounts(self, tmp_path):
        """Save two accounts, load_all returns both."""
        store = TokenStore(path=tmp_path / "tokens.json")
        # Use the accounts dir manually since we're in legacy-mode path.
        # Instead, create a store that simulates multi-account.
        accounts_dir = tmp_path / "accounts"
        store._accounts_dir = accounts_dir
        store._legacy_mode = False

        creds1 = _make_creds(token="tok-1", refresh="ref-1")
        creds2 = _make_creds(token="tok-2", refresh="ref-2")

        store.save(creds1, email="work@example.com")
        store.save(creds2, email="personal@gmail.com")

        all_accounts = store.load_all()
        assert len(all_accounts) == 2
        assert "work@example.com" in all_accounts
        assert "personal@gmail.com" in all_accounts
        assert all_accounts["work@example.com"].token == "tok-1"
        assert all_accounts["personal@gmail.com"].token == "tok-2"

    def test_load_specific_account(self, tmp_path):
        """load(email) returns only that account's creds."""
        store = TokenStore(path=tmp_path / "tokens.json")
        accounts_dir = tmp_path / "accounts"
        store._accounts_dir = accounts_dir
        store._legacy_mode = False

        creds1 = _make_creds(token="tok-1")
        creds2 = _make_creds(token="tok-2")

        store.save(creds1, email="a@example.com")
        store.save(creds2, email="b@example.com")

        loaded = store.load(email="b@example.com")
        assert loaded is not None
        assert loaded.token == "tok-2"

    def test_load_no_email_returns_first(self, tmp_path):
        """load() with no email in multi-account mode returns the first account."""
        store = TokenStore(path=tmp_path / "tokens.json")
        accounts_dir = tmp_path / "accounts"
        store._accounts_dir = accounts_dir
        store._legacy_mode = False

        store.save(_make_creds(token="tok-1"), email="a@example.com")
        store.save(_make_creds(token="tok-2"), email="b@example.com")

        loaded = store.load()
        assert loaded is not None
        # Should get the first (sorted) account.
        assert loaded.token == "tok-1"


class TestMultiAccountListAccounts:

    def test_list_accounts(self, tmp_path):
        """list_accounts returns email list."""
        store = TokenStore(path=tmp_path / "tokens.json")
        accounts_dir = tmp_path / "accounts"
        store._accounts_dir = accounts_dir
        store._legacy_mode = False

        store.save(_make_creds(), email="z@example.com")
        store.save(_make_creds(), email="a@example.com")

        result = store.list_accounts()
        assert result == ["a@example.com", "z@example.com"]

    def test_list_accounts_empty(self, tmp_path):
        """list_accounts returns empty list when no accounts exist."""
        store = TokenStore(path=tmp_path / "tokens.json")
        store._accounts_dir = tmp_path / "accounts"
        assert store.list_accounts() == []


class TestMultiAccountRemove:

    def test_remove_account(self, tmp_path):
        """Remove one account, other remains."""
        store = TokenStore(path=tmp_path / "tokens.json")
        accounts_dir = tmp_path / "accounts"
        store._accounts_dir = accounts_dir
        store._legacy_mode = False

        store.save(_make_creds(token="tok-1"), email="keep@example.com")
        store.save(_make_creds(token="tok-2"), email="remove@example.com")

        assert store.remove("remove@example.com") is True
        assert store.list_accounts() == ["keep@example.com"]

        # Can still load the remaining account.
        loaded = store.load(email="keep@example.com")
        assert loaded is not None
        assert loaded.token == "tok-1"

    def test_remove_nonexistent(self, tmp_path):
        """Remove non-existent account returns False."""
        store = TokenStore(path=tmp_path / "tokens.json")
        store._accounts_dir = tmp_path / "accounts"
        assert store.remove("nobody@example.com") is False


class TestMultiAccountMigration:

    def test_migration_from_legacy_tokens_json(self, tmp_path):
        """Old tokens.json is migrated to accounts/<email>.json on construction."""
        # Write a legacy tokens.json.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        legacy_path = data_dir / "tokens.json"
        creds = _make_creds(token="legacy-tok")
        payload = TokenStore._serialize(creds)
        legacy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Mock user_data_dir to return our tmp_path and mock _resolve_email.
        with patch("pi_email.token_store.user_data_dir", return_value=str(data_dir)), \
             patch.object(TokenStore, "_resolve_email", return_value="migrated@example.com"):
            store = TokenStore()

        # Legacy file should be gone.
        assert not legacy_path.exists()

        # New account file should exist.
        account_file = data_dir / "accounts" / "migrated@example.com.json"
        assert account_file.exists()

        # Can load the migrated account.
        loaded = store.load(email="migrated@example.com")
        assert loaded is not None
        assert loaded.token == "legacy-tok"

    def test_migration_unknown_email(self, tmp_path):
        """If email can't be resolved, migrate under unknown@migrated."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        legacy_path = data_dir / "tokens.json"
        creds = _make_creds(token="unknown-tok")
        payload = TokenStore._serialize(creds)
        legacy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with patch("pi_email.token_store.user_data_dir", return_value=str(data_dir)), \
             patch.object(TokenStore, "_resolve_email", return_value=None):
            store = TokenStore()

        assert not legacy_path.exists()
        account_file = data_dir / "accounts" / "unknown@migrated.json"
        assert account_file.exists()

    def test_no_migration_when_accounts_exist(self, tmp_path):
        """If accounts/ already has files, just delete the legacy file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        legacy_path = data_dir / "tokens.json"
        legacy_path.write_text('{"token": "old"}', encoding="utf-8")

        accounts_dir = data_dir / "accounts"
        accounts_dir.mkdir()
        (accounts_dir / "existing@example.com.json").write_text(
            json.dumps(TokenStore._serialize(_make_creds())),
            encoding="utf-8",
        )

        with patch("pi_email.token_store.user_data_dir", return_value=str(data_dir)):
            store = TokenStore()

        # Legacy file removed, existing account untouched.
        assert not legacy_path.exists()
        assert (accounts_dir / "existing@example.com.json").exists()
        assert store.list_accounts() == ["existing@example.com"]
