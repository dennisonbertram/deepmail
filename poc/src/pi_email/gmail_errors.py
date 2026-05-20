"""Gmail API error classifier with retry decisions.

Plain exponential backoff misses three distinct Gmail behaviours:

  - 401 means "refresh the access token", not "wait and retry the same call".
  - 403 has THREE meaningfully different `errors[].reason` values:
      * rateLimitExceeded       — project-level burst; short backoff.
      * userRateLimitExceeded   — per-user burst; longer backoff (4x).
      * quotaExceeded /
        dailyLimitExceeded      — per-day cap; no retry, fail loudly.
  - 401 with `error: "invalid_grant"` in the body means the refresh token
    itself is revoked; the user needs to re-consent. Retrying buys nothing.

See research/01-gmail-api-mechanics.md §"Quotas" and Google's
[handle-errors guide](https://developers.google.com/workspace/gmail/api/guides/handle-errors).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

# Backoff parameters. Public constants so tests / callers can tune.
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 60.0
LONG_BACKOFF_MULTIPLIER = 4.0
JITTER_FRAC = 0.20


class ErrorAction(Enum):
    """What the caller should do in response to a Gmail API response."""

    SUCCESS = auto()
    REFRESH_TOKEN_AND_RETRY = auto()      # 401: access token expired
    BACKOFF_AND_RETRY = auto()            # 429, 403 rateLimitExceeded, 5xx
    BACKOFF_LONG_AND_RETRY = auto()       # 403 userRateLimitExceeded
    FATAL_QUOTA_EXHAUSTED = auto()        # 403 quotaExceeded / dailyLimitExceeded
    FATAL_CLIENT_ERROR = auto()           # 4xx with no retry path
    FATAL_INVALID_GRANT = auto()          # 401 invalid_grant: refresh revoked


@dataclass
class RetryDecision:
    """Result of classifying one API response."""

    action: ErrorAction
    delay_seconds: float
    attempt_remaining: int


# Reasons we treat as "burst limiter" — short backoff.
_BURST_REASONS = frozenset({
    "rateLimitExceeded",
    "backendError",
    # 5xx-shaped errors sometimes surface as a reason; the status check
    # also catches them but a belt-and-suspenders set helps.
})

# Reasons we treat as "per-user limiter" — longer backoff.
_USER_BURST_REASONS = frozenset({
    "userRateLimitExceeded",
})

# Reasons we treat as fatal-day-cap.
_QUOTA_FATAL_REASONS = frozenset({
    "quotaExceeded",
    "dailyLimitExceeded",
    "dailyLimitExceededUnreg",
})


def _exp_backoff(attempt: int, multiplier: float = 1.0) -> float:
    """Compute jittered exponential backoff in seconds for the given attempt (1-indexed).

    Attempts: 1 -> base, 2 -> base*2, 3 -> base*4, ..., capped at MAX_DELAY_SECONDS,
    then multiplied (for the long-backoff case) and jittered by ±JITTER_FRAC.
    """
    raw = BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    raw *= multiplier
    raw = min(raw, MAX_DELAY_SECONDS)
    # ±20% multiplicative jitter — avoids thundering-herd on synchronized clients.
    jitter = 1.0 + random.uniform(-JITTER_FRAC, JITTER_FRAC)
    return max(0.0, raw * jitter)


def _parse_retry_after(retry_after: str | None) -> float | None:
    """Return seconds, or None if the header is missing/unparseable.

    Per RFC 7231 Retry-After may be an integer (seconds) or an HTTP-date. We
    only support seconds — that's what Gmail emits in practice, and an
    HTTP-date parse adds dependency surface for a header we rarely see.
    """
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def _reasons(body: dict | None) -> list[str]:
    """Pull all `error.errors[].reason` strings from a Google JSON error body."""
    if not body:
        return []
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return []
    out: list[str] = []
    for e in err.get("errors", []) or []:
        if isinstance(e, dict):
            r = e.get("reason")
            if isinstance(r, str):
                out.append(r)
    # OAuth-style "error" + "error_description" sometimes appears at the body
    # root (for token endpoint errors) — surface that too.
    return out


def _oauth_error_code(body: dict | None) -> str | None:
    """Return the OAuth-style `error` field for invalid_grant detection."""
    if not isinstance(body, dict):
        return None
    val = body.get("error")
    if isinstance(val, str):
        return val
    # Some clients nest the OAuth error inside .error.message
    if isinstance(val, dict):
        msg = val.get("message")
        if isinstance(msg, str) and "invalid_grant" in msg.lower():
            return "invalid_grant"
    return None


def classify(
    response_status: int,
    response_body: dict | None,
    retry_after: str | None,
    attempt: int,
    max_attempts: int = 3,
) -> RetryDecision:
    """Inspect status + error.reason + Retry-After and decide what to do.

    `attempt` is the 1-indexed count of THIS attempt (i.e. 1 on the first try).
    `max_attempts` is the inclusive cap — when attempt >= max_attempts we
    return FATAL_CLIENT_ERROR even for retryable conditions.
    """
    remaining = max(0, max_attempts - attempt)

    # Success path. 200/201/204/etc all collapse to SUCCESS.
    if 200 <= response_status < 300:
        return RetryDecision(ErrorAction.SUCCESS, 0.0, remaining)

    reasons = _reasons(response_body)
    oauth_err = _oauth_error_code(response_body)

    # 401 path — refresh OR invalid_grant.
    if response_status == 401:
        # invalid_grant comes back from the *token* endpoint as the OAuth-style
        # `error` field, not from Gmail's `error.errors[].reason`. We check both
        # for safety.
        if oauth_err == "invalid_grant" or "invalid_grant" in reasons:
            return RetryDecision(ErrorAction.FATAL_INVALID_GRANT, 0.0, 0)
        if remaining <= 0:
            return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
        return RetryDecision(ErrorAction.REFRESH_TOKEN_AND_RETRY, 0.0, remaining)

    # 403 path — three sub-cases differ by reason.
    if response_status == 403:
        if any(r in _QUOTA_FATAL_REASONS for r in reasons):
            return RetryDecision(ErrorAction.FATAL_QUOTA_EXHAUSTED, 0.0, 0)
        if any(r in _USER_BURST_REASONS for r in reasons):
            if remaining <= 0:
                return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
            delay = _parse_retry_after(retry_after)
            if delay is None:
                delay = _exp_backoff(attempt, multiplier=LONG_BACKOFF_MULTIPLIER)
            return RetryDecision(ErrorAction.BACKOFF_LONG_AND_RETRY, delay, remaining)
        if any(r in _BURST_REASONS for r in reasons):
            if remaining <= 0:
                return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
            delay = _parse_retry_after(retry_after) or _exp_backoff(attempt)
            return RetryDecision(ErrorAction.BACKOFF_AND_RETRY, delay, remaining)
        # 403 with no recognized reason — treat as fatal. Permission denied
        # ("forbidden") is the most likely cause; retrying buys nothing.
        return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)

    # 429 path — always a rate limiter; respect Retry-After if present.
    if response_status == 429:
        if remaining <= 0:
            return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
        delay = _parse_retry_after(retry_after) or _exp_backoff(attempt)
        return RetryDecision(ErrorAction.BACKOFF_AND_RETRY, delay, remaining)

    # 5xx path — transient server error; exponential backoff.
    if 500 <= response_status < 600:
        if remaining <= 0:
            return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
        delay = _parse_retry_after(retry_after) or _exp_backoff(attempt)
        return RetryDecision(ErrorAction.BACKOFF_AND_RETRY, delay, remaining)

    # Everything else (400, 404, 405, 410, ...) is a fatal client error.
    return RetryDecision(ErrorAction.FATAL_CLIENT_ERROR, 0.0, 0)
