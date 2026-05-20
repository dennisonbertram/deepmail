"""Tests for the Gmail error classifier.

Every branch of `classify` has a test. Where backoff is exponential we don't
pin the exact value (it's jittered), just the bounds.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.gmail_errors import (  # noqa: E402
    BASE_DELAY_SECONDS,
    ErrorAction,
    JITTER_FRAC,
    LONG_BACKOFF_MULTIPLIER,
    MAX_DELAY_SECONDS,
    classify,
)


@pytest.fixture(autouse=True)
def _stable_jitter(monkeypatch):
    """Pin random.uniform so backoff math is testable. Disables jitter (returns 0)
    so we can assert exact delay values."""
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)


def _err(reason: str) -> dict:
    """Build a Gmail-style error body for a given reason."""
    return {
        "error": {
            "code": 0,  # value doesn't matter; classify uses HTTP status
            "errors": [{"reason": reason, "message": f"test-{reason}"}],
        }
    }


# ---------- Success ----------


def test_success_200():
    d = classify(200, None, None, attempt=1)
    assert d.action == ErrorAction.SUCCESS
    assert d.delay_seconds == 0.0


def test_success_204():
    d = classify(204, None, None, attempt=1)
    assert d.action == ErrorAction.SUCCESS


# ---------- 401 ----------


def test_401_triggers_refresh():
    d = classify(401, None, None, attempt=1)
    assert d.action == ErrorAction.REFRESH_TOKEN_AND_RETRY
    assert d.delay_seconds == 0.0
    assert d.attempt_remaining == 2  # 3 - 1


def test_401_invalid_grant_is_fatal():
    body = {"error": "invalid_grant", "error_description": "Token revoked"}
    d = classify(401, body, None, attempt=1)
    assert d.action == ErrorAction.FATAL_INVALID_GRANT
    assert d.delay_seconds == 0.0


def test_401_at_max_attempts_is_fatal():
    d = classify(401, None, None, attempt=3, max_attempts=3)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


# ---------- 403 ----------


def test_403_rate_limit_exceeded_backoff():
    d = classify(403, _err("rateLimitExceeded"), None, attempt=1)
    assert d.action == ErrorAction.BACKOFF_AND_RETRY
    assert d.delay_seconds == BASE_DELAY_SECONDS  # attempt=1 -> base, no jitter


def test_403_user_rate_limit_exceeded_long_backoff():
    d = classify(403, _err("userRateLimitExceeded"), None, attempt=1)
    assert d.action == ErrorAction.BACKOFF_LONG_AND_RETRY
    # Long backoff: base * LONG_MULT
    assert d.delay_seconds == BASE_DELAY_SECONDS * LONG_BACKOFF_MULTIPLIER


def test_403_quota_exceeded_fatal():
    d = classify(403, _err("quotaExceeded"), None, attempt=1)
    assert d.action == ErrorAction.FATAL_QUOTA_EXHAUSTED
    assert d.delay_seconds == 0.0
    assert d.attempt_remaining == 0


def test_403_daily_limit_exceeded_fatal():
    d = classify(403, _err("dailyLimitExceeded"), None, attempt=1)
    assert d.action == ErrorAction.FATAL_QUOTA_EXHAUSTED


def test_403_unknown_reason_is_fatal():
    """Forbidden with no recognized retryable reason — most likely permission denied."""
    d = classify(403, _err("forbidden"), None, attempt=1)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


def test_403_rate_limit_exceeded_exponential():
    """attempt=3 -> base*4 (no jitter)."""
    d = classify(403, _err("rateLimitExceeded"), None, attempt=3, max_attempts=5)
    assert d.delay_seconds == BASE_DELAY_SECONDS * 4


def test_403_burst_at_max_attempts_is_fatal():
    d = classify(403, _err("rateLimitExceeded"), None, attempt=3, max_attempts=3)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


# ---------- 429 ----------


def test_429_with_retry_after_uses_header():
    d = classify(429, None, "30", attempt=1)
    assert d.action == ErrorAction.BACKOFF_AND_RETRY
    assert d.delay_seconds == 30.0


def test_429_without_retry_after_exponential():
    d = classify(429, None, None, attempt=1)
    assert d.action == ErrorAction.BACKOFF_AND_RETRY
    assert d.delay_seconds == BASE_DELAY_SECONDS


def test_429_malformed_retry_after_falls_back_to_exponential():
    d = classify(429, None, "soon", attempt=1)
    assert d.action == ErrorAction.BACKOFF_AND_RETRY
    assert d.delay_seconds == BASE_DELAY_SECONDS


def test_429_at_max_attempts_is_fatal():
    d = classify(429, None, "5", attempt=3, max_attempts=3)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


# ---------- 5xx ----------


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_retries_with_exponential(status):
    d = classify(status, None, None, attempt=1)
    assert d.action == ErrorAction.BACKOFF_AND_RETRY
    assert d.delay_seconds == BASE_DELAY_SECONDS


def test_5xx_at_max_attempts_is_fatal():
    d = classify(503, None, None, attempt=3, max_attempts=3)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


def test_5xx_respects_retry_after():
    d = classify(503, None, "7", attempt=1)
    assert d.delay_seconds == 7.0


# ---------- Other 4xx ----------


def test_400_fatal():
    d = classify(400, {"error": {"errors": [{"reason": "invalid"}]}}, None, attempt=1)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


def test_404_fatal():
    d = classify(404, None, None, attempt=1)
    assert d.action == ErrorAction.FATAL_CLIENT_ERROR


# ---------- Backoff math ----------


def test_backoff_caps_at_max(monkeypatch):
    """High attempt counts should cap at MAX_DELAY_SECONDS."""
    d = classify(503, None, None, attempt=20, max_attempts=999)
    assert d.delay_seconds == MAX_DELAY_SECONDS


def test_long_backoff_uses_4x_multiplier_then_caps():
    # attempt=1, long-mult = 4. base*4 = 2.0 (well under cap).
    d = classify(403, _err("userRateLimitExceeded"), None, attempt=1)
    assert d.delay_seconds == BASE_DELAY_SECONDS * LONG_BACKOFF_MULTIPLIER
    # attempt=20, long-mult = 4. Should cap.
    d = classify(403, _err("userRateLimitExceeded"), None, attempt=20, max_attempts=999)
    assert d.delay_seconds == MAX_DELAY_SECONDS


def test_jitter_applied(monkeypatch):
    """With jitter enabled, the delay should be within ±JITTER_FRAC of nominal."""
    # Restore real jitter for this test.
    monkeypatch.setattr(random, "uniform", lambda a, b: -JITTER_FRAC)
    d = classify(503, None, None, attempt=1)
    # nominal = BASE; jitter * nominal = BASE*(1-0.2)
    assert d.delay_seconds == pytest.approx(BASE_DELAY_SECONDS * (1.0 - JITTER_FRAC))


# ---------- Attempt accounting ----------


def test_attempt_remaining_decreases():
    d1 = classify(503, None, None, attempt=1, max_attempts=5)
    d2 = classify(503, None, None, attempt=2, max_attempts=5)
    assert d1.attempt_remaining == 4
    assert d2.attempt_remaining == 3
