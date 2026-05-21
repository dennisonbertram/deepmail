"""Per-user OAuth token persistence (multi-account).

Tokens live OUTSIDE the repo (refresh tokens are long-lived credentials —
checking them in would be a security incident). We use platformdirs to derive
the right per-OS user-data path:

  - macOS:   ~/Library/Application Support/pi-email-deep-context-library/
  - Linux:   ~/.local/share/pi-email-deep-context-library/   (XDG_DATA_HOME)
  - Windows: %APPDATA%/pi-email-deep-context-library/

Multi-account layout:
  accounts/<email>.json   — one file per authenticated Google account

Migration: if the legacy ``tokens.json`` exists, it is automatically migrated
to the ``accounts/`` directory on first use.  The email is discovered by
calling Gmail's ``getProfile`` endpoint; if that fails the file is stored
under ``unknown@migrated.json`` and renamed the next time the account is
identified.

The token files are written atomically (tmp -> fsync -> rename) with mode
0o600 so a half-finished refresh or a hostile coresident process can't read
the refresh token from disk.

Storage shape is exactly what google.oauth2.credentials.Credentials round-trips
via `from_authorized_user_info` / `to_json` — see oauth.py for the schema.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from pathlib import Path

from google.oauth2.credentials import Credentials
from platformdirs import user_data_dir

log = logging.getLogger(__name__)

DEFAULT_APP_NAME = "pi-email-deep-context-library"
_TOKEN_FILENAME = "tokens.json"          # legacy single-account filename
_ACCOUNTS_DIR = "accounts"
_FILE_MODE = 0o600
_DIR_MODE = 0o700


class TokenStore:
    """JSON-backed per-user token persistence with multi-account support.

    Each authenticated Google account is stored as a separate JSON file under
    ``<data_dir>/accounts/<email>.json``.  The legacy single-file layout
    (``<data_dir>/tokens.json``) is auto-migrated on first access.

    The ``path`` constructor parameter is retained for test isolation: when
    set, the store operates in **legacy single-account mode** (no accounts/
    directory, no migration).  Production code should omit it.
    """

    def __init__(
        self,
        app_name: str = DEFAULT_APP_NAME,
        path: Path | None = None,
    ):
        self.app_name = app_name
        self._legacy_mode = path is not None

        if path is not None:
            # Test-isolation / legacy mode: single file, no accounts dir.
            self.path = Path(path)
            self._data_dir = self.path.parent
            self._accounts_dir = self._data_dir / _ACCOUNTS_DIR
        else:
            self._data_dir = Path(user_data_dir(app_name))
            self.path = self._data_dir / _TOKEN_FILENAME   # legacy path
            self._accounts_dir = self._data_dir / _ACCOUNTS_DIR

        # Auto-migrate on construction (no-op if already migrated).
        if not self._legacy_mode:
            self._maybe_migrate()

    # ================================================================
    # Multi-account public API
    # ================================================================

    def save(self, creds: Credentials, email: str | None = None) -> None:
        """Save credentials for a specific account.

        If *email* is ``None`` **and** we're in legacy mode (test path set),
        fall back to writing to ``self.path`` for backward compat.  In
        production (no explicit path) email is required.
        """
        if email is None and self._legacy_mode:
            self._atomic_write(self.path, creds)
            return

        if email is None:
            raise ValueError("email is required for multi-account save")

        self._accounts_dir.mkdir(parents=True, exist_ok=True)
        self._chmod_dir(self._accounts_dir)
        dest = self._accounts_dir / f"{email}.json"
        self._atomic_write(dest, creds)

    def load(self, email: str | None = None) -> Credentials | None:
        """Load credentials for a specific account.

        If *email* is ``None``:
          - Legacy mode (path set): read from ``self.path``.
          - Multi-account mode: return the **first** account found (for
            backward compat with callers that aren't account-aware yet).
        """
        if email is not None:
            return self._read_creds(self._accounts_dir / f"{email}.json")

        if self._legacy_mode:
            return self._read_creds(self.path)

        # Multi-account fallback: return first account.
        accounts = self.load_all()
        if not accounts:
            return None
        return next(iter(accounts.values()))

    def load_all(self) -> dict[str, Credentials]:
        """Load all authenticated accounts.  Returns ``{email: creds}``."""
        accounts: dict[str, Credentials] = {}
        if not self._accounts_dir.exists():
            return accounts
        for f in sorted(self._accounts_dir.glob("*.json")):
            creds = self._read_creds(f)
            if creds is not None:
                accounts[f.stem] = creds
        return accounts

    def list_accounts(self) -> list[str]:
        """Return email addresses of all authenticated accounts."""
        if not self._accounts_dir.exists():
            return []
        return sorted(f.stem for f in self._accounts_dir.glob("*.json"))

    def remove(self, email: str) -> bool:
        """Remove an account's credentials.  Returns True if deleted."""
        path = self._accounts_dir / f"{email}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Remove the legacy token file.  No-op if it doesn't exist.

        For multi-account, use ``remove(email)`` instead.
        """
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # ================================================================
    # Migration from legacy single-file layout
    # ================================================================

    def _maybe_migrate(self) -> None:
        """If the legacy ``tokens.json`` exists, migrate to accounts/."""
        if not self.path.exists():
            return
        if self._accounts_dir.exists() and any(self._accounts_dir.glob("*.json")):
            # Already migrated but old file lingers — just remove it.
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return

        creds = self._read_creds(self.path)
        if creds is None:
            return

        # Determine the email address.
        email = self._resolve_email(creds)
        if not email:
            email = "unknown@migrated"

        self._accounts_dir.mkdir(parents=True, exist_ok=True)
        self._chmod_dir(self._accounts_dir)
        dest = self._accounts_dir / f"{email}.json"
        self._atomic_write(dest, creds)

        # Remove old file.
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        log.info("Migrated legacy tokens.json → accounts/%s.json", email)

    @staticmethod
    def _resolve_email(creds: Credentials) -> str | None:
        """Best-effort email resolution via Gmail getProfile."""
        try:
            from googleapiclient.discovery import build
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            profile = service.users().getProfile(userId="me").execute()
            return profile.get("emailAddress")
        except Exception:
            return None

    # ================================================================
    # Internals
    # ================================================================

    @staticmethod
    def _read_creds(path: Path) -> Credentials | None:
        """Read a single credentials JSON file.  Returns None if missing."""
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return Credentials.from_authorized_user_info(info)

    def _atomic_write(self, dest: Path, creds: Credentials) -> None:
        """Atomic write to *dest* with 0o600 perms."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._chmod_dir(dest.parent)

        payload = self._serialize(creds)

        fd, tmp_path = tempfile.mkstemp(
            prefix=".tokens.", suffix=".tmp", dir=str(dest.parent)
        )
        try:
            os.fchmod(fd, _FILE_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        try:
            os.chmod(dest, _FILE_MODE)
        except (OSError, NotImplementedError):  # pragma: no cover
            pass

    @staticmethod
    def _chmod_dir(d: Path) -> None:
        try:
            os.chmod(d, _DIR_MODE)
        except (OSError, NotImplementedError):  # pragma: no cover
            pass

    @staticmethod
    def _serialize(creds: Credentials) -> dict:
        """Produce the dict shape `from_authorized_user_info` consumes.

        We don't just delegate to `creds.to_json()` because that drops keys
        whose values are None — and `client_secret` being None would cause
        `from_authorized_user_info` to raise on load. We force the key in.
        """
        payload: dict = {
            "client_id": creds.client_id or "",
            # Empty string is a load-bearing default — see docstring.
            "client_secret": creds.client_secret or "",
            "refresh_token": creds.refresh_token,
            "token": creds.token,
            "token_uri": creds.token_uri,
            "scopes": list(creds.scopes) if creds.scopes else [],
        }
        if creds.expiry is not None:
            # The format from_authorized_user_info expects: ISO-8601 with a Z.
            payload["expiry"] = creds.expiry.isoformat() + "Z"
        # Drop keys whose value is None so the JSON is tidy — but keep
        # client_secret even if empty (above).
        return {k: v for k, v in payload.items() if v is not None}
