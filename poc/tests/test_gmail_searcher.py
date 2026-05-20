"""Unit tests for `pi_email.gmail_searcher.GmailSearcher`.

We mock `googleapiclient.discovery.build` so no real network is touched. The
tests exercise the four behaviors that matter for the production wiring:

  - Single-page listing + per-message batch fetch produces the expected
    SearchBatch with the correct quota_units_used.
  - Multi-page listing follows `nextPageToken` correctly.
  - `max_results_per_query` truncates exhaustively (sets `truncated=True`).
  - A transient batch error is retried; the second-pass success is captured.
  - A token-refresh failure (invalid_grant) raises a clear GmailAuthError
    that mentions `pi-email auth`.
  - Message conversion: text/plain payload + RFC 2822 date + quote stripping.

The MagicMock-with-chained-calls pattern is the standard googleapiclient
test approach — `service.users().messages().list(...).execute()` etc. are
all callable mocks that return whatever we configure.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

import httplib2  # noqa: E402
from googleapiclient.errors import HttpError  # noqa: E402

from pi_email.gmail_searcher import (  # noqa: E402
    GmailAuthError,
    GmailSearcher,
    _message_from_payload,
)


# ---------------- helpers ----------------


def _b64url(text: str) -> str:
    """Encode `text` as Gmail-style URL-safe base64 (no padding)."""
    raw = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def _make_payload(
    msg_id: str,
    *,
    body: str = "hello world",
    headers: dict[str, str] | None = None,
    thread_id: str | None = None,
) -> dict:
    """Build a fake `messages.get(format='full')` response."""
    if headers is None:
        headers = {
            "From": "alice@example.com",
            "To": "me@example.com",
            "Subject": "test subject",
            "Date": "Mon, 18 May 2026 09:00:00 +0000",
        }
    return {
        "id": msg_id,
        "threadId": thread_id or msg_id,
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": k, "value": v} for k, v in headers.items()],
            "body": {"data": _b64url(body)},
        },
    }


def _http_error(status: int, reason: str = "rateLimitExceeded") -> HttpError:
    """Construct an HttpError carrying a status + Google-shaped error body."""
    resp = httplib2.Response({"status": status})
    body = {"error": {"errors": [{"reason": reason}]}}
    return HttpError(resp, json.dumps(body).encode("utf-8"))


class _FakeBatch:
    """Stand-in for googleapiclient.http.BatchHttpRequest in tests.

    Records every (request_id) that was added; on `execute()` it calls the
    configured callback once per request with either a payload OR an
    HttpError exception based on `responses_by_id`.
    """

    def __init__(self, callback, responses_by_id: dict[str, object]):
        self._cb = callback
        self._added: list[tuple[object, str]] = []
        self._responses = responses_by_id

    def add(self, request, request_id):
        self._added.append((request, request_id))

    def execute(self):
        for _req, rid in self._added:
            resp = self._responses.get(rid)
            if isinstance(resp, HttpError):
                self._cb(rid, None, resp)
            else:
                self._cb(rid, resp, None)


def _make_service(
    *,
    list_pages: list[dict] | None = None,
    list_raises: list[Exception | None] | None = None,
    get_responses: dict[str, object] | None = None,
    profile: dict | None = None,
    list_call_log: list[dict] | None = None,
    batch_call_log: list[list[str]] | None = None,
    batch_responses_per_call: list[dict[str, object]] | None = None,
) -> MagicMock:
    """Build a MagicMock that quacks like the discovery-built gmail service.

    `list_pages` is a queue of dicts the messages().list().execute() should
    return (one per call). `list_raises[i]` (if non-None) is raised instead
    of returning page[i]. `get_responses` is consulted by the fake batch.

    `batch_responses_per_call`, if provided, overrides `get_responses` and
    lets per-call mappings drive each batch.execute() round individually —
    used for the retry test where attempt 1 throws and attempt 2 succeeds.
    """
    list_pages = list_pages or []
    list_raises = list_raises or []
    get_responses = get_responses or {}
    batch_responses_per_call = batch_responses_per_call or []

    service = MagicMock(name="gmail_service")

    # --- list path -------------------------------------------------
    list_idx = {"i": 0}

    def list_execute_factory(args, kwargs):
        # snapshot the call args
        if list_call_log is not None:
            list_call_log.append({**kwargs})
        i = list_idx["i"]
        list_idx["i"] += 1
        if i < len(list_raises) and list_raises[i] is not None:
            raise list_raises[i]
        if i >= len(list_pages):
            return {"messages": []}
        return list_pages[i]

    def list_call(*args, **kwargs):
        m = MagicMock()
        m.execute.side_effect = lambda: list_execute_factory(args, kwargs)
        return m

    service.users.return_value.messages.return_value.list.side_effect = list_call

    # --- get path (for the simple single-message .fetch tests) -----
    def get_call(*args, **kwargs):
        m = MagicMock()
        mid = kwargs.get("id") or args[0]
        resp = get_responses.get(mid)
        if isinstance(resp, HttpError):
            m.execute.side_effect = resp
        else:
            m.execute.return_value = resp
        return m

    service.users.return_value.messages.return_value.get.side_effect = get_call

    # --- batch path ------------------------------------------------
    batch_idx = {"i": 0}

    def new_batch(callback):
        i = batch_idx["i"]
        batch_idx["i"] += 1
        if batch_responses_per_call:
            responses = batch_responses_per_call[
                min(i, len(batch_responses_per_call) - 1)
            ]
        else:
            responses = get_responses
        b = _FakeBatch(callback, responses)
        if batch_call_log is not None:
            batch_call_log.append([rid for _, rid in b._added])
            # record on execute so we capture the final added set
            original_execute = b.execute

            def wrapped_execute():
                batch_call_log[-1] = [rid for _, rid in b._added]
                return original_execute()

            b.execute = wrapped_execute  # type: ignore[assignment]
        return b

    service.new_batch_http_request.side_effect = new_batch

    # --- getProfile path (whoami test) -----------------------------
    if profile is not None:
        service.users.return_value.getProfile.return_value.execute.return_value = profile

    return service


# ---------------- Tests ----------------


def test_single_page_list_three_ids_quota_65():
    """3 hits: 1 list call (5 units) + 3 gets (20 units each) = 65 total."""
    pages = [
        {
            "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
            # no nextPageToken
        }
    ]
    gets = {
        "m1": _make_payload("m1", body="body 1"),
        "m2": _make_payload("m2", body="body 2"),
        "m3": _make_payload("m3", body="body 3"),
    }
    service = _make_service(list_pages=pages, get_responses=gets)
    creds = MagicMock()

    with patch("pi_email.gmail_searcher.build", return_value=service):
        s = GmailSearcher(creds, max_results_per_query=500)
        batch = s.search_and_fetch("from:alice")

    assert len(batch.hits) == 3
    assert batch.quota_units_used == 5 + 20 * 3 == 65
    assert s.quota_used == 65
    assert batch.truncated is False
    assert batch.error is None
    # Ordering matters for stable diffs downstream.
    assert [m.message_id for m in batch.hits] == ["m1", "m2", "m3"]


def test_multi_page_list_follows_pagination():
    """Two list pages -> 2 list calls (10 quota) + N gets (20 each)."""
    pages = [
        {"messages": [{"id": "a1"}, {"id": "a2"}], "nextPageToken": "tok-2"},
        {"messages": [{"id": "a3"}]},  # final page
    ]
    gets = {mid: _make_payload(mid) for mid in ["a1", "a2", "a3"]}
    list_calls: list[dict] = []
    service = _make_service(
        list_pages=pages, get_responses=gets, list_call_log=list_calls
    )
    creds = MagicMock()

    with patch("pi_email.gmail_searcher.build", return_value=service):
        s = GmailSearcher(creds, max_results_per_query=500)
        batch = s.search_and_fetch("family")

    assert len(batch.hits) == 3
    # 2 list calls => 10 units, 3 gets => 60 units, total = 70.
    assert batch.quota_units_used == 10 + 60 == 70
    assert batch.truncated is False
    # The second list call must carry the pageToken from the first response.
    assert list_calls[0].get("pageToken") in (None,)
    assert list_calls[1].get("pageToken") == "tok-2"


def test_max_results_truncates():
    """50 IDs available + max_results_per_query=10 -> truncated=True; only 10 fetched."""
    pages = [
        {
            "messages": [{"id": f"id{i}"} for i in range(50)],
            "nextPageToken": "more",
        },
    ]
    gets = {f"id{i}": _make_payload(f"id{i}") for i in range(50)}
    service = _make_service(list_pages=pages, get_responses=gets)
    creds = MagicMock()

    with patch("pi_email.gmail_searcher.build", return_value=service):
        s = GmailSearcher(creds, max_results_per_query=10)
        batch = s.search_and_fetch("subject:test")

    assert len(batch.hits) == 10
    assert batch.truncated is True
    # 1 list call (capped before second page) + 10 gets = 5 + 200 = 205.
    assert batch.quota_units_used == 5 + 10 * 20 == 205


def test_batch_429_retries_then_succeeds(monkeypatch):
    """Two IDs: round 1 errors on m2 with 429, round 2 succeeds.

    The successful m1 in round 1 should be kept; m2's eventual success
    appears in round 2.
    """
    # Disable backoff sleep so the test is fast.
    monkeypatch.setattr("pi_email.gmail_searcher.time.sleep", lambda _s: None)

    pages = [{"messages": [{"id": "m1"}, {"id": "m2"}]}]
    rate_err = _http_error(429, "rateLimitExceeded")
    round1 = {
        "m1": _make_payload("m1", body="ok 1"),
        "m2": rate_err,
    }
    round2 = {
        "m2": _make_payload("m2", body="ok 2"),
    }
    service = _make_service(
        list_pages=pages,
        batch_responses_per_call=[round1, round2],
    )
    creds = MagicMock()

    with patch("pi_email.gmail_searcher.build", return_value=service):
        s = GmailSearcher(creds, max_results_per_query=500)
        batch = s.search_and_fetch("q")

    assert len(batch.hits) == 2
    assert {m.message_id for m in batch.hits} == {"m1", "m2"}
    # batch.retry_count is "rounds beyond first"; we did 2 rounds total -> 1.
    assert batch.retry_count == 1
    # Quota: 1 list (5) + 2 gets in round 1 (40, even though m2 errored —
    # google bills per sub-request) + 1 get in round 2 (20) = 65.
    assert batch.quota_units_used == 5 + 40 + 20 == 65


def test_invalid_grant_raises_clear_error(monkeypatch):
    """A 401 followed by an invalid_grant refresh raises GmailAuthError
    mentioning `pi-email auth`."""
    pages = [{"messages": [{"id": "m1"}]}]
    auth_err = _http_error(401, "authError")
    # First the list call returns 401; the refresh path will raise.
    service = _make_service(
        list_pages=pages,
        list_raises=[auth_err],
    )
    creds = MagicMock()

    def fake_refresh(c):
        raise RuntimeError("invalid_grant: token revoked")

    monkeypatch.setattr("pi_email.gmail_searcher.refresh_if_needed", fake_refresh)
    monkeypatch.setattr("pi_email.gmail_searcher.time.sleep", lambda _s: None)

    with patch("pi_email.gmail_searcher.build", return_value=service):
        s = GmailSearcher(creds)
        with pytest.raises(GmailAuthError) as exc:
            s.search_and_fetch("q")

    assert "pi-email auth" in str(exc.value)


def test_message_conversion_strips_quoted_text():
    """Verify that the payload converter pulls headers, body, decodes base64,
    converts RFC 2822 -> ISO 8601, AND strips the reply chain."""
    body = (
        "Sure, see you Saturday.\n"
        "\n"
        "On Mon, May 18 2026, alice@example.com wrote:\n"
        "> Hey, are we still on for Saturday?\n"
        "> Let me know.\n"
    )
    payload = _make_payload(
        "msg-abc",
        body=body,
        headers={
            "From": "Bob Smith <bob@example.com>",
            "To": "Alice <alice@example.com>",
            "Subject": "Re: Saturday plan",
            "Date": "Mon, 18 May 2026 09:00:00 +0000",
            "Message-ID": "<abc@mail.example.com>",
        },
    )
    msg = _message_from_payload(payload)

    assert msg.message_id == "msg-abc"
    assert msg.from_addr == "Bob Smith <bob@example.com>"
    assert msg.subject == "Re: Saturday plan"
    # Body was decoded from base64 and contains the original reply attribution.
    assert "On Mon, May 18 2026, alice@example.com wrote:" in msg.body
    # body_clean should have the reply attribution + quoted lines stripped.
    assert msg.body_clean is not None
    assert "On Mon, May 18 2026, alice@example.com wrote:" not in msg.body_clean
    assert "Sure, see you Saturday." in msg.body_clean
    # Date converted to ISO 8601 (parsedate yields offset-aware datetime).
    assert msg.date.startswith("2026-05-18")


# ---------------------------------------------------------------------------
# Pass 17A — calendar-notification persons attached to Message
# ---------------------------------------------------------------------------


def test_calendar_notification_message_carries_persons() -> None:
    """Pass 17A: a calendar-notification email's `Message.calendar_persons`
    must be populated from the subject + body. Non-calendar messages must
    have an empty `calendar_persons` list (default factory).
    """
    body = (
        "You have a new event.\n\n"
        "Guests\n"
        "  Jana Bertram <jana@example.com>\n"
        "  alice@gmail.com\n"
    )
    payload = _make_payload(
        "cal-1",
        body=body,
        headers={
            "From": "Google Calendar <calendar-notification@google.com>",
            "To": "me@example.com",
            "Subject": "Accepted: Vitus Birthday in school",
            "Date": "Mon, 18 May 2026 09:00:00 +0000",
            "List-Unsubscribe": "<mailto:unsubscribe@google.com>",
        },
    )
    msg = _message_from_payload(payload)
    # Calendar mailers are NOT bulk (Pass-17A exemption).
    assert msg.is_bulk is False
    # Persons populated. Should include the title-derived "Vitus" AND
    # the attendees.
    names = {p.name for p in msg.calendar_persons}
    emails = {p.email for p in msg.calendar_persons if p.email}
    assert "Vitus" in names
    assert "jana@example.com" in emails
    assert "alice@gmail.com" in emails


def test_non_calendar_message_has_empty_calendar_persons() -> None:
    """Regular (non-calendar) mail has an empty calendar_persons list."""
    payload = _make_payload(
        "regular-1",
        body="just a normal email",
        headers={
            "From": "alice@example.com",
            "To": "me@example.com",
            "Subject": "lunch tomorrow?",
            "Date": "Mon, 18 May 2026 09:00:00 +0000",
        },
    )
    msg = _message_from_payload(payload)
    assert msg.calendar_persons == []
