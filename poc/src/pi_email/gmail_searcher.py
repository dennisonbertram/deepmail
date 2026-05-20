"""Real Gmail-backed Searcher.

Implements the `Searcher` protocol against `users.messages.list` +
`users.messages.get` via the Google API client. The design mirrors
research/01-gmail-api-mechanics.md:

  - Paginate `messages.list` by following `nextPageToken` until exhausted
    or `max_results_per_query` is reached. Each `list` call = 5 quota units.
  - Batch-fetch IDs in groups of 50 with `BatchHttpRequest` (50 is Google's
    recommended ceiling for stability — much larger batches start failing
    intermittently). Each `messages.get(format="full")` = 20 quota units.
  - On a retryable error (`gmail_errors.classify`), back off and retry the
    failing call up to 3 attempts. For batched fetches we retry the *batch
    subset* that failed; surviving IDs from the first attempt are kept.
  - On 401 (REFRESH_TOKEN_AND_RETRY) we refresh credentials in-place and
    re-issue. On `FATAL_INVALID_GRANT` we raise a clear error pointing the
    user at `pi-email auth`. On `FATAL_QUOTA_EXHAUSTED` we raise
    `GmailQuotaExhausted` so the caller can surface today's spend.

The conversion from Gmail's payload tree to our `Message` dataclass:
  - Prefer `text/plain` parts; fall back to `text/html` with a regex strip
    if no plaintext exists. We do NOT pull in `html2text` — adding a
    dependency just for the HTML fallback isn't worth the surface area for
    a POC, and the regex strip is good enough for the rare html-only mail.
  - Headers are normalized into the dataclass (`from_addr`, `to_addr`,
    `subject`, `date`). The original raw header map isn't stored on Message
    today; we keep the conversion side small so the rest of the loop is
    unchanged.
  - The plaintext body has reply quotes stripped via
    `strip_quotes.strip_quotes_and_signatures` BEFORE being stored as
    `body_clean` (the loop does the same for fixture messages — we just
    do it eagerly here so the canary `query` command shows clean text).
"""

from __future__ import annotations

import base64
import binascii
import email.utils
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .calendar_email_parser import (
    is_calendar_notification_sender,
    looks_like_calendar_invite_response,
    parse_calendar_email,
)
from .corpus import Message
from .filters import is_bulk_message
from .gmail_errors import ErrorAction, RetryDecision, classify
from .html_strip import clean_html_and_css
from .oauth import refresh_if_needed
from .searcher import SearchBatch
from .strip_quotes import strip_quotes_and_signatures


log = logging.getLogger(__name__)


# Per research/01-gmail-api-mechanics.md (May 2026 quota schedule):
LIST_QUOTA_COST = 5
GET_QUOTA_COST = 20

# Google recommends <= 50 sub-requests per batch for stability.
DEFAULT_BATCH_SIZE = 50

# Cap on retry attempts per single HTTP call / per batch retry round.
MAX_ATTEMPTS = 3

# Per-page size we ask the list endpoint for. The endpoint silently caps at
# 500 today, so we just ask for that.
LIST_PAGE_SIZE = 500


class GmailQuotaExhausted(RuntimeError):
    """Raised when Gmail returns a daily-cap error.

    The exception text includes `quota_used` (the cumulative count tracked by
    this `GmailSearcher` instance) so the operator can diagnose how close to
    the cap their workload is running.
    """

    def __init__(self, quota_used: int, detail: str = ""):
        self.quota_used = quota_used
        msg = (
            f"Gmail daily quota exhausted (this instance used {quota_used} units"
            f" before the cap was reported)."
        )
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg)


class GmailAuthError(RuntimeError):
    """Raised when the refresh token is no longer usable.

    Points the operator at `pi-email auth` to re-consent.
    """


# ---------------- HTTP-error parsing ----------------


def _http_error_status_and_body(err: HttpError) -> tuple[int, dict | None]:
    """Pull (status, parsed-json-body) out of an HttpError.

    HttpError's `resp.status` is reliable; the body is bytes that may or may
    not be JSON. We try JSON first and fall back to None so `classify` only
    sees structured data.
    """
    status = int(getattr(err.resp, "status", 0) or 0)
    body: dict | None = None
    content = getattr(err, "content", None)
    if content:
        try:
            if isinstance(content, (bytes, bytearray)):
                text = content.decode("utf-8", errors="replace")
            else:
                text = str(content)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                body = parsed
        except (ValueError, TypeError, UnicodeDecodeError):
            body = None
    return status, body


def _retry_after(err: HttpError) -> str | None:
    """Best-effort Retry-After header pull from the HttpError response."""
    resp = getattr(err, "resp", None)
    if resp is None:
        return None
    # httplib2 Response is dict-like (case-insensitive header access).
    try:
        return resp.get("retry-after") or resp.get("Retry-After")
    except Exception:
        return None


# ---------------- Body / header conversion ----------------


# NOTE: HTML / CSS stripping lives in `html_strip.py` (`clean_html_and_css`).
# We call it from `_message_from_payload` against whichever body part we
# selected (text/plain preferred, text/html fallback) so font-stack tokens
# like "Arial", "Helvetica Neue", "Segoe UI" never reach the entity
# extractor — they were landing as `[[people/...]]` entries in real
# profiles before this cleanup step was added.


def _decode_b64url(data: str | None) -> str:
    """Decode Gmail's url-safe base64 body. Returns "" on any failure."""
    if not data:
        return ""
    # Gmail uses URL-safe alphabet without padding; pad to a multiple of 4.
    try:
        s = data.replace("-", "+").replace("_", "/")
        pad = (-len(s)) % 4
        s += "=" * pad
        raw = base64.b64decode(s)
        return raw.decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        return ""


def _walk_parts(payload: dict | None) -> list[dict]:
    """DFS over a Gmail payload tree, returning every leaf-or-multipart part."""
    out: list[dict] = []
    if not payload:
        return out
    out.append(payload)
    for child in payload.get("parts", []) or []:
        out.extend(_walk_parts(child))
    return out


def _extract_body(payload: dict | None) -> str:
    """Walk the part tree and return the best raw body text we can get.

    Strategy: collect every text/plain part's decoded text first; if any
    exist, concatenate them. Otherwise fall back to text/html parts (the
    HTML/CSS cleanup happens later in `_message_from_payload` via
    `clean_html_and_css`, so we hand the raw HTML back unchanged here).
    We deliberately don't try to handle attachments — the body is what
    we materialize into the corpus.
    """
    parts = _walk_parts(payload)
    plain_chunks: list[str] = []
    html_chunks: list[str] = []
    for part in parts:
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body") or {}
        data = body.get("data")
        if not data:
            continue
        decoded = _decode_b64url(data)
        if not decoded:
            continue
        if mime == "text/plain":
            plain_chunks.append(decoded)
        elif mime == "text/html":
            html_chunks.append(decoded)

    if plain_chunks:
        return "\n\n".join(plain_chunks).strip()
    if html_chunks:
        return "\n\n".join(html_chunks).strip()
    return ""


def _headers_to_map(payload: dict | None) -> dict[str, str]:
    """Flatten payload.headers[] into a case-insensitive dict (keep last)."""
    out: dict[str, str] = {}
    if not payload:
        return out
    for h in payload.get("headers", []) or []:
        name = (h.get("name") or "").strip()
        value = h.get("value")
        if not name or value is None:
            continue
        # Gmail returns canonical header names; lowercase for lookup.
        out[name.lower()] = str(value)
    return out


# Headers we preserve onto Message.headers. The first set (From/To/...) is
# what the existing code already pulled; the second set is what the bulk
# filter inspects. We don't try to enumerate every RFC 5322 header — just
# the ones we use today plus a couple of bulk-mail tell-tales.
_PRESERVED_HEADER_NAMES = (
    "from",
    "to",
    "cc",
    "bcc",
    "subject",
    "date",
    "message-id",
    "references",
    "in-reply-to",
    "list-unsubscribe",
    "list-id",
    "list-help",
    "precedence",
    "auto-submitted",
)


def _select_preserved_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter a lowercase header map down to the ones we want on Message.

    Returns canonical-case keys (`List-Unsubscribe`, not `list-unsubscribe`)
    so downstream consumers see header names that resemble the wire format.
    `is_bulk_message` does case-insensitive lookups anyway, so this is just
    cosmetics — but it means a debugger-printed Message looks right.
    """
    canonical = {
        "from": "From",
        "to": "To",
        "cc": "Cc",
        "bcc": "Bcc",
        "subject": "Subject",
        "date": "Date",
        "message-id": "Message-ID",
        "references": "References",
        "in-reply-to": "In-Reply-To",
        "list-unsubscribe": "List-Unsubscribe",
        "list-id": "List-Id",
        "list-help": "List-Help",
        "precedence": "Precedence",
        "auto-submitted": "Auto-Submitted",
    }
    out: dict[str, str] = {}
    for name in _PRESERVED_HEADER_NAMES:
        if name in headers:
            out[canonical[name]] = headers[name]
    return out


def _rfc2822_to_iso(date_str: str) -> str:
    """Convert an RFC 2822 date to ISO 8601. Returns input on failure."""
    if not date_str:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        if dt is None:
            return date_str
        return dt.isoformat()
    except (TypeError, ValueError):
        return date_str


def _message_from_payload(raw: dict) -> Message:
    """Convert a Gmail messages.get(format='full') response into our Message.

    The Message dataclass we use across the loop (corpus.Message) stores
    string headers in the flat fields (`from_addr`, `to_addr`, ...). The
    full preserved-header map (`Message.headers`) is also populated so the
    bulk-mail filter can inspect `List-Unsubscribe`, `Precedence`, etc.
    Headers we don't care about (DKIM, Received chains, X-* random junk)
    are dropped to keep the in-memory footprint small.
    """
    msg_id = str(raw.get("id") or "")
    thread_id = str(raw.get("threadId") or msg_id)
    payload = raw.get("payload") or {}
    headers = _headers_to_map(payload)
    preserved = _select_preserved_headers(headers)

    body_raw = _extract_body(payload)
    # Strip HTML tags / CSS declarations / signature font-stacks BEFORE
    # quote stripping. We apply this to BOTH text/plain and text/html paths
    # because plaintext signatures sometimes copy CSS declarations
    # literally (the original symptom: "Arial" landing as a person entity).
    body_cleaned = clean_html_and_css(body_raw)
    body_clean = strip_quotes_and_signatures(body_cleaned)

    from_addr = headers.get("from", "")
    subject = headers.get("subject", "")
    # Compute is_bulk against the preserved header map so the case-insensitive
    # lookup in `is_bulk_message` finds `List-Unsubscribe` and friends. (It
    # would also work with the all-lowercase `headers` map, but using the
    # canonical map keeps test fixtures readable.)
    is_bulk = is_bulk_message(preserved, from_addr)

    # Pass-19: calendar acceptance / invite-response emails that come
    # from the user's own custom domain (e.g. `dennison@dennisonbertram.com`)
    # or the user's spouse's gmail (`jana.uhlarikova@gmail.com`) are NOT
    # caught by `is_calendar_notification_sender(from_addr)` — they carry
    # the human's own address. But they're still authentic calendar
    # responses whose subject/body shape gives them away ("Accepted: Vitus
    # Birthday in school @ ..." + a calendar.google.com URL in the body).
    # Detect the shape via `looks_like_calendar_invite_response` so:
    #   (a) the bulk-filter doesn't drop them (the user's own-domain
    #       acceptance carries `Auto-Submitted: auto-replied` which the
    #       header check otherwise flags as bulk), and
    #   (b) the calendar parser is invoked to mine kid names from the
    #       title + attendees from the body.
    is_calendar_shaped = looks_like_calendar_invite_response(subject, body_raw)
    if is_calendar_shaped:
        is_bulk = False

    # Pass-17A: when the sender is a Google Calendar notification mailer
    # OR the subject+body look like a calendar invite response (Pass-19),
    # mine the subject + body for event title + attendees. The bulk filter
    # exemptions above ensure the message survives into the corpus and
    # reaches downstream entity extraction; the parsed persons ride along
    # on `Message.calendar_persons` for the loop to consume after
    # extraction.
    #
    # IMPORTANT: feed the RAW body, not body_clean. `clean_html_and_css`
    # treats `<jana@example.com>` (a legit angle-bracketed email in a
    # plaintext attendee list) as an HTML tag and strips it — losing the
    # email entirely. The parser is internally defensive about footer /
    # URL noise so re-using the raw body is safe.
    calendar_persons: list = []
    if is_calendar_notification_sender(from_addr) or is_calendar_shaped:
        calendar_persons = parse_calendar_email(subject, body_raw)

    # `Message-ID` header (canonical RFC ID) vs. Gmail's opaque `id` — keep
    # Gmail's `id` as the primary message_id because the rest of the loop
    # uses it as the dedupe key and provenance citation in materializer.py.
    msg = Message(
        message_id=msg_id,
        thread_id=thread_id,
        from_addr=from_addr,
        to_addr=headers.get("to", ""),
        subject=subject,
        date=_rfc2822_to_iso(headers.get("date", "")),
        body=body_raw,
        source_path=Path(f"gmail:{msg_id}"),
        body_clean=body_clean,
        headers=preserved,
        is_bulk=is_bulk,
        calendar_persons=calendar_persons,
    )
    return msg


# ---------------- Retry wrapper ----------------


@dataclass
class _RetryContext:
    """Mutable context threaded through retries — lets us update credentials
    in place when the access token needs to be refreshed."""

    creds: Credentials
    service: Any  # googleapiclient.discovery.Resource
    quota_delta: int = 0


def _execute_with_retry(
    request_builder,
    ctx: _RetryContext,
    *,
    quota_cost: int,
    description: str,
):
    """Run `request_builder()` -> execute, with classify-driven retries.

    `request_builder` is a zero-arg callable that returns a fresh request
    object — it's a callable (not a request) so we can rebuild the request
    against a refreshed service if the token was rotated mid-flight.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            req = request_builder()
            result = req.execute()
            ctx.quota_delta += quota_cost
            return result
        except HttpError as err:
            status, body = _http_error_status_and_body(err)
            retry_after = _retry_after(err)
            decision = classify(
                response_status=status,
                response_body=body,
                retry_after=retry_after,
                attempt=attempt,
                max_attempts=MAX_ATTEMPTS,
            )
            _act_on_decision(decision, ctx, description=description)
            # Keep looping for retryable cases. If _act_on_decision returned
            # without raising it means we should retry.
        except Exception:
            # Non-HTTPError exceptions are programmer errors / network blips —
            # don't paper over them.
            raise


def _act_on_decision(
    decision: RetryDecision,
    ctx: _RetryContext,
    *,
    description: str,
) -> None:
    """Mutate ctx (refresh creds) or sleep, or raise — based on the decision."""
    action = decision.action
    if action == ErrorAction.SUCCESS:
        return
    if action == ErrorAction.REFRESH_TOKEN_AND_RETRY:
        try:
            refresh_if_needed(ctx.creds)
        except Exception as exc:  # google-auth wraps invalid_grant here
            raise GmailAuthError(
                "Token refresh failed; re-run `pi-email auth` to re-consent. "
                f"({exc})"
            ) from exc
        # Rebuild the service with the refreshed creds so subsequent requests
        # pick up the new token.
        ctx.service = build(
            "gmail", "v1", credentials=ctx.creds, cache_discovery=False
        )
        return
    if action in (ErrorAction.BACKOFF_AND_RETRY, ErrorAction.BACKOFF_LONG_AND_RETRY):
        time.sleep(decision.delay_seconds)
        return
    if action == ErrorAction.FATAL_QUOTA_EXHAUSTED:
        raise GmailQuotaExhausted(ctx.quota_delta, detail=description)
    if action == ErrorAction.FATAL_INVALID_GRANT:
        raise GmailAuthError(
            "Refresh token revoked; re-run `pi-email auth` to re-consent."
        )
    # FATAL_CLIENT_ERROR or anything else.
    raise RuntimeError(
        f"Gmail API call failed ({description}); classifier action={action.name}, "
        f"attempt cap reached."
    )


# ---------------- The Searcher ----------------


class GmailSearcher:
    """Production Searcher: real `users.messages.list` + `users.messages.get`.

    Quota tracking is cumulative across the instance's lifetime — `_quota_used`
    sums every list (5 units) and every get (20 units) call. `SearchBatch.
    quota_units_used` reports the per-call delta so the loop can log per
    iteration; `quota_used` is the running total.
    """

    def __init__(
        self,
        credentials: Credentials,
        *,
        max_results_per_query: int = 500,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self._creds = credentials
        self._service = build(
            "gmail", "v1", credentials=credentials, cache_discovery=False
        )
        self._max = int(max_results_per_query)
        self._batch_size = int(batch_size)
        self._quota_used = 0

    # ---- public properties ----

    @property
    def quota_used(self) -> int:
        """Cumulative quota units used by this searcher since construction."""
        return self._quota_used

    # ---- Searcher protocol ----

    def search(self, query: str) -> list[str]:
        """List message IDs matching `query`. Paginates until exhausted.

        Use `search_and_fetch` in the loop — `search`/`fetch` are kept for
        compatibility with the original `Searcher` protocol.
        """
        ids, _truncated = self._list_ids(query)
        return ids

    def fetch(self, msg_id: str) -> Message:
        """Single-message fetch with retry. 20 quota units."""
        ctx = _RetryContext(creds=self._creds, service=self._service)
        raw = _execute_with_retry(
            lambda: ctx.service.users().messages().get(
                userId="me", id=msg_id, format="full"
            ),
            ctx,
            quota_cost=GET_QUOTA_COST,
            description=f"messages.get(id={msg_id})",
        )
        self._service = ctx.service  # in case retries rebuilt it
        self._quota_used += ctx.quota_delta
        return _message_from_payload(raw)

    def search_and_fetch(self, query: str) -> SearchBatch:
        """Primary path: list IDs (paginated) then batch-fetch full messages."""
        per_call_quota_start = self._quota_used

        ids, truncated = self._list_ids(query)
        hits, errors, batch_retries = self._batch_fetch(ids)

        per_call_quota = self._quota_used - per_call_quota_start
        error_str = None
        if errors:
            # Compact, non-secret error summary — never include token text or
            # full payloads in case the caller logs this.
            error_str = (
                f"{len(errors)} of {len(ids)} message(s) failed to fetch: "
                + "; ".join(sorted({e for e in errors})[:3])
            )

        return SearchBatch(
            query=query,
            hits=hits,
            quota_units_used=per_call_quota,
            retry_count=batch_retries,
            truncated=truncated,
            error=error_str,
        )

    # ---- internals ----

    def _list_ids(self, query: str) -> tuple[list[str], bool]:
        """Paginate users.messages.list. Returns (ids, truncated).

        `truncated` is True iff we stopped before exhausting `nextPageToken`
        because `max_results_per_query` was reached.
        """
        ids: list[str] = []
        page_token: str | None = None
        truncated = False
        ctx = _RetryContext(creds=self._creds, service=self._service)

        while True:
            page_size = min(LIST_PAGE_SIZE, max(1, self._max - len(ids)))
            page = _execute_with_retry(
                lambda pt=page_token, ps=page_size: ctx.service.users()
                .messages()
                .list(userId="me", q=query, maxResults=ps, pageToken=pt),
                ctx,
                quota_cost=LIST_QUOTA_COST,
                description=f"messages.list(q={query!r}, page_token={page_token!r})",
            )
            for m in page.get("messages", []) or []:
                mid = m.get("id")
                if not mid:
                    continue
                ids.append(str(mid))
                if len(ids) >= self._max:
                    break
            page_token = page.get("nextPageToken")
            if not page_token:
                break
            if len(ids) >= self._max:
                truncated = True
                break

        self._service = ctx.service
        self._quota_used += ctx.quota_delta
        return ids, truncated

    def _batch_fetch(
        self, ids: list[str]
    ) -> tuple[list[Message], list[str], int]:
        """Batch-fetch every id in chunks of `batch_size`.

        Returns (messages, error-summaries, total-retry-count).
        On a transient batch error we retry the SUBSET that failed up to two
        times after the first attempt (so 3 attempts total). Messages that
        succeed in earlier rounds are preserved.
        """
        out: list[Message] = []
        errors: list[str] = []
        retry_count = 0

        for start in range(0, len(ids), self._batch_size):
            chunk = ids[start : start + self._batch_size]
            chunk_hits, chunk_retries, chunk_errors = self._fetch_one_batch(chunk)
            out.extend(chunk_hits)
            errors.extend(chunk_errors)
            retry_count += chunk_retries

        return out, errors, retry_count

    def _fetch_one_batch(
        self, chunk: list[str]
    ) -> tuple[list[Message], int, list[str]]:
        """Fetch one batch (<=batch_size IDs) with up to MAX_ATTEMPTS rounds.

        Returns (messages, retry-rounds-used, error-summaries).
        """
        remaining = list(chunk)
        collected: list[Message] = []
        errors: list[str] = []
        attempt = 0

        ctx = _RetryContext(creds=self._creds, service=self._service)

        while remaining:
            attempt += 1
            round_results: dict[str, dict] = {}
            round_errors: dict[str, HttpError] = {}

            def make_callback(results=round_results, errors_map=round_errors):
                def _cb(request_id, response, exception):
                    if exception is not None:
                        errors_map[request_id] = exception
                    elif response is not None:
                        results[request_id] = response
                return _cb

            batch = ctx.service.new_batch_http_request(callback=make_callback())
            for mid in remaining:
                batch.add(
                    ctx.service.users().messages().get(
                        userId="me", id=mid, format="full"
                    ),
                    request_id=mid,
                )
            # Executing the batch is itself a single HTTP round-trip whose
            # internal sub-requests each consume `messages.get` quota (20 ea).
            try:
                batch.execute()
            except HttpError as err:
                # Whole-batch transport-level error — apply the classifier.
                status, body = _http_error_status_and_body(err)
                decision = classify(
                    response_status=status,
                    response_body=body,
                    retry_after=_retry_after(err),
                    attempt=attempt,
                    max_attempts=MAX_ATTEMPTS,
                )
                _act_on_decision(decision, ctx, description="batch.execute")
                # Retry the entire chunk.
                continue

            # Per-message accounting: every sub-request that came back
            # (success OR error) consumed 20 units. That matches Google's
            # documented behavior — quota is debited per sub-request, not
            # per outer batch.
            ctx.quota_delta += GET_QUOTA_COST * (len(round_results) + len(round_errors))

            # Convert successes.
            for mid, raw in round_results.items():
                try:
                    collected.append(_message_from_payload(raw))
                except Exception as conv_err:  # pragma: no cover - defensive
                    errors.append(f"convert {mid}: {conv_err!r}")

            # Decide what to do about per-message errors.
            if not round_errors:
                remaining = []
                break

            # If we're out of attempts, give up on the remaining IDs.
            if attempt >= MAX_ATTEMPTS:
                for mid, err in round_errors.items():
                    errors.append(f"{mid}: HTTP {getattr(err.resp, 'status', '?')}")
                remaining = []
                break

            # Classify each error; the strictest decision wins.
            should_retry: list[str] = []
            max_delay = 0.0
            for mid, err in round_errors.items():
                status, body = _http_error_status_and_body(err)
                decision = classify(
                    response_status=status,
                    response_body=body,
                    retry_after=_retry_after(err),
                    attempt=attempt,
                    max_attempts=MAX_ATTEMPTS,
                )
                if decision.action == ErrorAction.SUCCESS:
                    continue
                if decision.action == ErrorAction.REFRESH_TOKEN_AND_RETRY:
                    _act_on_decision(decision, ctx, description=f"batch get id={mid}")
                    should_retry.append(mid)
                elif decision.action in (
                    ErrorAction.BACKOFF_AND_RETRY,
                    ErrorAction.BACKOFF_LONG_AND_RETRY,
                ):
                    should_retry.append(mid)
                    max_delay = max(max_delay, decision.delay_seconds)
                elif decision.action == ErrorAction.FATAL_QUOTA_EXHAUSTED:
                    # Fatal — raise so the caller learns immediately.
                    self._service = ctx.service
                    self._quota_used += ctx.quota_delta
                    raise GmailQuotaExhausted(
                        self._quota_used, detail=f"batch get id={mid}"
                    )
                elif decision.action == ErrorAction.FATAL_INVALID_GRANT:
                    self._service = ctx.service
                    self._quota_used += ctx.quota_delta
                    raise GmailAuthError(
                        "Refresh token revoked; re-run `pi-email auth`."
                    )
                else:
                    # Fatal per-message error (e.g. 404, 400). Drop it.
                    errors.append(
                        f"{mid}: HTTP {status} ({decision.action.name})"
                    )

            if max_delay > 0:
                time.sleep(max_delay)
            remaining = should_retry
            if not remaining:
                break

        self._service = ctx.service
        self._quota_used += ctx.quota_delta
        # retry-rounds-used = attempts beyond the first.
        rounds_used = max(0, attempt - 1)
        return collected, rounds_used, errors
