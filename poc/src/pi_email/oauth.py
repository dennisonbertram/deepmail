"""OAuth configuration + installed-app credential acquisition for Gmail.

We mirror the credential setup used by the parent pi-google project: a Desktop-app
GCP OAuth client, PKCE S256, loopback redirect. The same GCP client app is shared
across projects via env vars; tokens live separately per-project (see token_store).

Required env vars (loaded from .env via python-dotenv if available):
  GOOGLE_CLIENT_ID       — required. Desktop-app client ID.
  GOOGLE_CLIENT_SECRET   — optional. Desktop-app "secret" is not truly secret
                           under PKCE, but Google's installed-app schema still
                           accepts it; google-auth-oauthlib will round-trip it
                           via Credentials.

This module does NOT touch token storage — that's token_store.py — and it does
NOT implement Gmail API calls. It's the auth seam only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Lazy / optional: python-dotenv is already a dependency, but we only need
# load_dotenv if the env vars aren't set yet.
try:
    from dotenv import find_dotenv as _find_dotenv
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - dotenv is in deps; defensive only
    _load_dotenv = None  # type: ignore[assignment]
    _find_dotenv = None  # type: ignore[assignment]


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
# Pass 12A: contacts.readonly. The People API is the missing input that
# unlocks non-spouse family identification — the user's own address book
# labels relatives directly via the Family group, the `relations` field, and
# biography notes. Adding the scope here means new auth flows request it by
# default; existing tokens minted before Pass 12A are missing it (see
# `contacts.credentials_have_contacts_scope`) and the user must re-run
# `deep-email auth --refresh-auth` to upgrade.
CONTACTS_READONLY_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"
# Pass 14B: calendar.readonly. Family members routinely appear in calendar
# event titles ("Vitus Birthday", "Mom's anniversary", "Family Dinner") and
# as attendees of recurring family events — but the email + contacts
# pipeline never sees those signals. Pass 14B's GoogleCalendar client
# (`calendar_evidence.py`) reads them. Tokens minted before Pass 14B are
# missing this scope (see `calendar_evidence.credentials_have_calendar_scope`);
# the user must re-run `deep-email auth --refresh-auth` to upgrade.
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
DEFAULT_SCOPES: list[str] = [
    GMAIL_READONLY_SCOPE,
    CONTACTS_READONLY_SCOPE,
    CALENDAR_READONLY_SCOPE,
]


# Google's published endpoints — hardcoded so we don't depend on a
# discovery document just to start the flow.
_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_TOKEN_URI = "https://oauth2.googleapis.com/token"
_CERT_URI = "https://www.googleapis.com/oauth2/v1/certs"


@dataclass
class OAuthConfig:
    """Materialized installed-app OAuth config.

    `client_secret` is optional: Google's Desktop client type technically
    issues one, but under PKCE it's not load-bearing. We accept None and
    serialize as empty string when handing to google-auth-oauthlib.
    """

    client_id: str
    client_secret: str | None
    scopes: list[str] = field(default_factory=lambda: list(DEFAULT_SCOPES))

    @classmethod
    def from_env(cls, scopes: list[str] | None = None) -> "OAuthConfig":
        """Read GOOGLE_CLIENT_ID (required) and GOOGLE_CLIENT_SECRET (optional).

        Loads .env via python-dotenv if installed and present. .env is searched
        from the current working directory upward — the standard dotenv default.
        Raises RuntimeError with a remediation hint if GOOGLE_CLIENT_ID is unset.
        """
        if (
            _load_dotenv is not None
            and _find_dotenv is not None
            and not os.environ.get("GOOGLE_CLIENT_ID")
        ):
            # find_dotenv(usecwd=True) walks up from CWD looking for .env
            # (the default walks up from this module's __file__ — wrong for a
            # POC where the user runs from the project root). load_dotenv
            # itself does not override already-set env vars by default.
            path = _find_dotenv(usecwd=True)
            if path:
                _load_dotenv(path)

        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        if not client_id:
            raise RuntimeError(
                "GOOGLE_CLIENT_ID not set.\n\n"
                "You need a Google Cloud OAuth client ID to use Deep Email.\n"
                "Set it in your environment:\n"
                "  export GOOGLE_CLIENT_ID='your-id.apps.googleusercontent.com'\n\n"
                "Or run 'deep-email setup' for interactive setup.\n"
                "See: https://github.com/user/deep-email#google-cloud-setup"
            )

        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip() or None
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            scopes=list(scopes) if scopes is not None else list(DEFAULT_SCOPES),
        )

    def to_client_config(self) -> dict:
        """Return the dict shape InstalledAppFlow.from_client_config expects.

        google-auth-oauthlib accepts either a "web" or "installed" top-level
        key and treats their inner structure identically; "installed" is the
        correct one for Desktop / loopback flow.
        """
        return {
            "installed": {
                "client_id": self.client_id,
                # Always include the key (even empty) — Credentials.from_authorized_user_info
                # requires client_secret to be present in the JSON.
                "client_secret": self.client_secret or "",
                "auth_uri": _AUTH_URI,
                "token_uri": _TOKEN_URI,
                "auth_provider_x509_cert_url": _CERT_URI,
                # Loopback. google-auth-oauthlib's run_local_server replaces this
                # with the actual ephemeral port at redirect time.
                "redirect_uris": ["http://localhost"],
            }
        }


def acquire_credentials(
    config: OAuthConfig,
    open_browser: bool = True,
    port: int = 0,
) -> Credentials:
    """Run the installed-app flow with PKCE on a loopback ephemeral port.

    PKCE is on by default in google-auth-oauthlib (`autogenerate_code_verifier=True`).
    We explicitly set `code_verifier=None` so the library generates one and uses
    S256 — never roll your own.

    `prompt="consent"` and `access_type="offline"` are passed through
    `run_local_server`'s **kwargs to the authorization URL; they're what get
    us a refresh_token back instead of just an access token.
    """
    flow = InstalledAppFlow.from_client_config(
        config.to_client_config(),
        scopes=config.scopes,
        # Library will autogenerate a PKCE verifier + S256 challenge.
        code_verifier=None,
        autogenerate_code_verifier=True,
    )
    creds = flow.run_local_server(
        port=port,
        open_browser=open_browser,
        # Forwarded to the authorization URL.
        prompt="consent",
        access_type="offline",
    )
    return creds


def refresh_if_needed(creds: Credentials) -> Credentials:
    """Refresh `creds` in place if the access token is expired.

    Raises if the access token is expired and no refresh_token is available —
    callers should catch this and trigger re-consent via `acquire_credentials`.
    """
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif creds.expired and not creds.refresh_token:
        raise RuntimeError(
            "Access token expired and no refresh_token available; "
            "re-run acquire_credentials() to re-consent."
        )
    return creds
