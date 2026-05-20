"""Per-user OAuth token persistence.

Tokens live OUTSIDE the repo (refresh tokens are long-lived credentials —
checking them in would be a security incident). We use platformdirs to derive
the right per-OS user-data path:

  - macOS:   ~/Library/Application Support/pi-email-deep-context-library/
  - Linux:   ~/.local/share/pi-email-deep-context-library/   (XDG_DATA_HOME)
  - Windows: %APPDATA%/pi-email-deep-context-library/

The token file is written atomically (tmp -> fsync -> rename) with mode 0o600
so a half-finished refresh or a hostile coresident process can't read the
refresh token from disk.

Storage shape is exactly what google.oauth2.credentials.Credentials round-trips
via `from_authorized_user_info` / `to_json` — see oauth.py for the schema.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from google.oauth2.credentials import Credentials
from platformdirs import user_data_dir

DEFAULT_APP_NAME = "pi-email-deep-context-library"
_TOKEN_FILENAME = "tokens.json"
_FILE_MODE = 0o600
_DIR_MODE = 0o700


class TokenStore:
    """JSON-backed per-user token persistence with 0o600 file permissions."""

    def __init__(
        self,
        app_name: str = DEFAULT_APP_NAME,
        path: Path | None = None,
    ):
        self.app_name = app_name
        if path is not None:
            self.path = Path(path)
        else:
            # platformdirs handles macOS / Linux / Windows correctly without
            # us hardcoding ~/.config or ~/Library/Application Support.
            self.path = Path(user_data_dir(app_name)) / _TOKEN_FILENAME

    # ----------------------------------------------------------------

    def load(self) -> Credentials | None:
        """Return Credentials if a token file exists and is parseable, else None.

        Returns None (not a raise) for the missing-file case because the
        caller's flow is "load -> if None, run acquire_credentials". A corrupt
        file IS surfaced (raises JSONDecodeError) — that signals tampering or
        a partial write we should not silently overwrite.
        """
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        # Credentials.from_authorized_user_info requires client_id, client_secret,
        # refresh_token keys (even if client_secret is ""). save() preserves them.
        return Credentials.from_authorized_user_info(info)

    def save(self, creds: Credentials) -> None:
        """Atomic write to self.path with 0o600 perms.

        Writes JSON of the shape `Credentials.from_authorized_user_info` consumes.
        Always includes `client_secret` (possibly empty) so the round-trip works.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Best-effort dir perms — we own this dir; don't fail if chmod is a noop
        # (e.g. on Windows where POSIX modes don't apply meaningfully).
        try:
            os.chmod(self.path.parent, _DIR_MODE)
        except (OSError, NotImplementedError):  # pragma: no cover
            pass

        payload = self._serialize(creds)

        # Atomic write: write to a sibling tmp file, fsync, rename.
        # Using mkstemp(dir=...) so the tmp file lands on the same filesystem
        # as the destination — that's what makes the rename atomic.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tokens.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            # Restrict perms BEFORE writing the secret, not after — closes the
            # race where another process could read the temp file's content
            # between create and chmod.
            os.fchmod(fd, _FILE_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            # On POSIX, rename is atomic on the same filesystem and replaces
            # any existing destination.
            os.replace(tmp_path, self.path)
        except BaseException:
            # Best-effort cleanup; never mask the original exception.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        # Belt-and-suspenders: re-chmod the final path in case the FS dropped
        # the mode through the rename (some exotic FSes do; macOS HFS+/APFS
        # preserves it but we don't want to depend on that).
        try:
            os.chmod(self.path, _FILE_MODE)
        except (OSError, NotImplementedError):  # pragma: no cover
            pass

    def clear(self) -> None:
        """Remove the token file. No-op if it doesn't exist."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # ----------------------------------------------------------------

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
