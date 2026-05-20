"""Tests for `pi_email.filters` — bulk / marketing / list-managed detection.

The filter has two contracts:
  - is_bulk_sender: sender-only check (local-part + domain patterns).
  - is_bulk_message: above + RFC 2369/3834 header signals
    (List-Unsubscribe, List-Id, Precedence, Auto-Submitted).

The spec calls out "better to let a borderline newsletter through than to
silently filter a personal email", so the negative cases (real names with
ordinary domains) matter at least as much as the positive ones — they
exist to lock the patterns down against creep.
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

import pytest  # noqa: E402

from pi_email.filters import is_bulk_message, is_bulk_sender  # noqa: E402


# ---------------------------------------------------------------------------
# is_bulk_sender
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "addr",
    [
        # Canonical bulk local-parts.
        "noreply@anywhere.com",
        "no-reply@anywhere.com",
        "donotreply@anywhere.com",
        "newsletter@example.org",
        "notifications@github.com",
        "marketing@somecompany.com",
        # Aggressive but-conservative-enough flags: bare role addresses.
        "hi@thecompany.com",
        "hello@thecompany.com",
        "team@thecompany.com",
        "info@thecompany.com",
        "editorial@helpscout.com",
        # System / bounce.
        "postmaster@host.example",
        "mailer-daemon@host.example",
        "bounces@list.example",
        # Suffixed local-parts (\b boundary still flags `noreply+x`).
        "noreply+abc@anywhere.com",
        "support-team@product.com",
        # Display-name + angle-bracket form must be parsed.
        "Foo Co <noreply@anywhere.com>",
        # Domain patterns.
        "anyone@bayareafoundersclub.substack.com",
        "bayareafoundersclub@substack.com",
        "user@m.list-manage.com",
        "outbound@u123.ccsend.com",
        "person@redditads.com",
        "alerts@u123.luma-mail.com",
        "events@u17.eventbritemail.com",
    ],
)
def test_bulk_sender_positive(addr: str) -> None:
    assert is_bulk_sender(addr) is True, addr


@pytest.mark.parametrize(
    "addr",
    [
        # Real-looking personal addresses.
        "Alice Smith <alice.smith@gmail.com>",
        "alice.smith@gmail.com",
        "dennison@withtally.com",
        "bob.jones@example.org",
        "j.doe@university.edu",
        # First-name local parts that look superficially like our patterns
        # but aren't word-boundary matches.
        "informa@bigcorp.example",       # `info` is a prefix but `informa` keeps going
        "teamster@union.example",        # `team` prefix, real word
        "hello.kitty@sanrio.example",    # actually starts with `hello` — see note
        # Domains that contain a brand-y substring but aren't list-managed.
        "alice@notsubstack.com",
        "bob@my.salesforce.com.au",      # cTLD twist — suffix doesn't match
        # `app.salesforce.com` isn't a marketing-cloud domain, but our
        # naive suffix `.salesforce.com$` would flag it. This is a known
        # false positive — acceptable per task spec ("conservative").
        # Empty / garbage.
        "",
    ],
)
def test_bulk_sender_negative(addr: str) -> None:
    # `hello.kitty@` is a deliberate aggressive flag: `hello@` as a role
    # address is so common in transactional mail that we accept the
    # collateral on hypothetical personal addresses starting with "hello".
    # If a real person hits it, we'd loosen the pattern. The test reflects
    # that: anything *not* aggressive-flagged should pass through here.
    if addr == "hello.kitty@sanrio.example":
        # `^hello\b` matches because `.` is a non-word char ending the
        # `hello` token. Acknowledge the false positive and skip; tighten
        # the pattern if real-world examples make this matter.
        assert is_bulk_sender(addr) is True
        return
    assert is_bulk_sender(addr) is False, addr


def test_bulk_sender_handles_none() -> None:
    """None input must not raise — return False."""
    assert is_bulk_sender(None) is False  # type: ignore[arg-type]


def test_bulk_sender_handles_malformed_no_at() -> None:
    """A `From:` line with no `@` should not crash and should return False."""
    assert is_bulk_sender("  Alice  Smith  ") is False


def test_bulk_sender_is_case_insensitive() -> None:
    assert is_bulk_sender("NoReply@Example.COM") is True
    assert is_bulk_sender("Bob.JONES@Example.org") is False


def test_bulk_sender_substack_domain_specific() -> None:
    """Verify the named example in the task spec."""
    assert is_bulk_sender("bayareafoundersclub@substack.com") is True


# ---------------------------------------------------------------------------
# is_bulk_message
# ---------------------------------------------------------------------------


def test_bulk_message_list_unsubscribe_with_list_id_overrides_personal_sender() -> None:
    """A `List-Unsubscribe` + `List-Id` pair on a message from a real-looking
    address is enough to flag it. Plenty of mailing-list software preserves
    the individual author's `From:` while adding the list headers.

    Pass 8C Fix 3: `List-Unsubscribe` ALONE is no longer enough (calendar
    invites and OAuth code mails carry it on legit personal mail); we
    require pairing with `List-Id` or `Precedence: bulk|list|junk`.
    """
    headers = {
        "List-Unsubscribe": "<mailto:unsubscribe@list.example>",
        "List-Id": "<some-list.example.com>",
    }
    assert is_bulk_message(headers, "anyone@anywhere.com") is True


def test_bulk_message_precedence_bulk() -> None:
    headers = {"Precedence": "bulk"}
    assert is_bulk_message(headers, "personal@example.com") is True


def test_bulk_message_precedence_list() -> None:
    headers = {"Precedence": "list"}
    assert is_bulk_message(headers, "personal@example.com") is True


def test_bulk_message_precedence_other_does_not_match() -> None:
    """`Precedence: first-class` (or any other value) must NOT trigger."""
    headers = {"Precedence": "first-class"}
    assert is_bulk_message(headers, "personal@example.com") is False


def test_bulk_message_auto_submitted_auto_replied() -> None:
    """Per RFC 3834, anything other than `no` indicates a bot."""
    headers = {"Auto-Submitted": "auto-replied"}
    assert is_bulk_message(headers, "personal@example.com") is True


def test_bulk_message_auto_submitted_no_does_not_match() -> None:
    headers = {"Auto-Submitted": "no"}
    assert is_bulk_message(headers, "personal@example.com") is False


def test_bulk_message_list_id_present() -> None:
    headers = {"List-Id": "<some-list.example.com>"}
    assert is_bulk_message(headers, "personal@example.com") is True


def test_bulk_message_personal_passes_through() -> None:
    """No headers, personal-looking address -> not bulk."""
    assert is_bulk_message({}, "alice.smith@gmail.com") is False


def test_bulk_message_malformed_from_no_email() -> None:
    """A From: with no parseable email defaults to False (don't filter)."""
    assert is_bulk_message({}, "  Alice  Smith  ") is False


def test_bulk_message_empty_inputs_do_not_raise() -> None:
    """Defensive: None / empty inputs must just return False."""
    assert is_bulk_message(None, None) is False  # type: ignore[arg-type]
    assert is_bulk_message({}, "") is False
    assert is_bulk_message({}, None) is False  # type: ignore[arg-type]
    assert is_bulk_message(None, "") is False  # type: ignore[arg-type]


def test_bulk_message_case_insensitive_header_keys() -> None:
    """RFC 2822 header names are case-insensitive — our lookup must mirror that.

    Pass 8C: use a header that triggers on its own (`list-id`) since Fix 3
    weakened `list-unsubscribe` alone. The case-insensitivity contract is
    unchanged and tested independently of the Fix-3 pairing rule.
    """
    headers = {"list-id": "<members.example.com>"}
    assert is_bulk_message(headers, "personal@example.com") is True


def test_bulk_message_empty_header_values_ignored() -> None:
    """A `List-Unsubscribe: ` with a whitespace-only value should NOT flag
    — defensively treat it as if the header weren't present."""
    headers = {"List-Unsubscribe": "   "}
    # Sender is also clean.
    assert is_bulk_message(headers, "personal@example.com") is False


def test_bulk_message_sender_alone_is_enough() -> None:
    """No headers but bulk-looking From: still flags."""
    assert is_bulk_message({}, "noreply@anywhere.com") is True


# ---------------------------------------------------------------------------
# Pass 8C — personal-domain / user-email / List-Unsubscribe-alone exemptions
# ---------------------------------------------------------------------------


def test_personal_domain_overrides_list_unsubscribe() -> None:
    """Pass 8C Fix 1: a `List-Unsubscribe` on a personal-domain sender
    (gmail.com, icloud.com, ...) is NOT enough to flag. Real personal mail
    routinely carries `List-Unsubscribe` (Google Calendar invites etc.);
    flagging it costs us 99/100 messages on family-search seed iterations.
    """
    headers = {"List-Unsubscribe": "<mailto:unsubscribe@list.example>"}
    assert is_bulk_message(headers, "alice@gmail.com") is False


def test_noreply_at_personal_domain_still_bulk() -> None:
    """Pass 8C Fix 1 exception: strict system local-parts (noreply@,
    mailer-daemon@, ...) still indicate bulk even at a personal domain.
    A `noreply@gmail.com` is unlikely but possible (a service that signs up
    with Gmail); it's still a service address, not a human."""
    assert is_bulk_message({}, "noreply@gmail.com") is True
    # Strict-local rule fires at is_bulk_sender level too.
    assert is_bulk_sender("noreply@gmail.com") is True


def test_calendar_invite_with_only_list_unsubscribe_not_bulk() -> None:
    """Pass 8C Fix 3: `List-Unsubscribe` alone (no `List-Id`, no
    `Precedence: bulk|list`) must NOT flag — calendar invites and many
    auto-confirmations carry it. Per RFC 8058/2369, properly-marked
    list mail also carries `List-Id` or `Precedence`."""
    headers = {"List-Unsubscribe": "<mailto:unsubscribe@calendar.example>"}
    # Calendar invite from a non-personal-domain address — the
    # personal-domain exemption can't help here; Fix 3 must do the work.
    assert is_bulk_message(headers, "invite@calendar.google.com") is False


def test_legit_newsletter_with_multiple_list_headers_still_bulk() -> None:
    """Pass 8C Fix 3 (counter-test): a real newsletter has multiple
    list-related headers — `List-Unsubscribe` + `List-Id`. That pairing
    is the legitimate-bulk signal we still flag."""
    headers = {
        "List-Unsubscribe": "<mailto:unsubscribe@newsletter.example>",
        "List-Id": "<weekly.newsletter.example>",
    }
    assert is_bulk_message(headers, "editorial@something.com") is True


def test_team_at_personal_domain_not_bulk() -> None:
    """Pass 8C Fix 1: weak role local-parts (`team@`, `info@`, `hello@`)
    are exempted at personal-email domains. A small-team Gmail (`team@gmail.com`)
    is a human group, not a marketing endpoint.
    """
    assert is_bulk_message({}, "team@gmail.com") is False
    assert is_bulk_sender("team@gmail.com") is False
    # Other weak locals at personal domains: same exemption.
    assert is_bulk_sender("info@gmail.com") is False
    assert is_bulk_sender("hello@icloud.com") is False
    assert is_bulk_sender("hi@protonmail.com") is False


def test_team_at_business_domain_still_bulk() -> None:
    """Pass 8C Fix 4: at a NON-personal domain, weak role locals still flag.
    The exemption is targeted to consumer-provider domains; business
    addresses with `team@` / `info@` keep the existing aggressive flag."""
    assert is_bulk_message({}, "team@bigco.com") is True
    assert is_bulk_sender("team@bigco.com") is True
    assert is_bulk_sender("info@somecompany.com") is True


def test_user_email_never_bulk() -> None:
    """Pass 8C Fix 2: the user's own outbound mail is never bulk — even
    if header signals fire, even if the local-part looks role-like, even
    if the domain is a known marketing platform. The user's threads are
    load-bearing for family discovery and must always pass through."""
    user_emails = {"dennison@withtally.com"}
    # No headers — sender at user's own address.
    assert is_bulk_message({}, "dennison@withtally.com", user_emails) is False
    # Even with strong list-managed headers, user emails are exempt.
    strong_bulk_headers = {
        "List-Id": "<some-list.example.com>",
        "Precedence": "bulk",
        "Auto-Submitted": "auto-replied",
    }
    assert (
        is_bulk_message(strong_bulk_headers, "dennison@withtally.com", user_emails)
        is False
    )
    # Case-insensitive on user_emails.
    assert (
        is_bulk_message({}, "DENNISON@WITHTALLY.COM", {"dennison@withtally.com"})
        is False
    )
    # Display-name + angle-bracket form is parsed and matched.
    assert (
        is_bulk_message(
            {}, "Dennison Bertram <dennison@withtally.com>", user_emails
        )
        is False
    )


def test_user_emails_default_none_does_not_raise() -> None:
    """Defensive: omitting `user_emails` (default None) keeps the old
    behavior. Empty set is also valid."""
    assert is_bulk_message({}, "alice@gmail.com") is False
    assert is_bulk_message({}, "alice@gmail.com", None) is False
    assert is_bulk_message({}, "alice@gmail.com", set()) is False


def test_auto_replied_at_business_domain_still_bulk() -> None:
    """Pass 8C Fix 5: `Auto-Submitted: auto-replied` on a non-personal-domain
    sender still flags. Vacation responders from a `@company.com` mailbox
    are usually template noise we don't want to extract entities from."""
    headers = {"Auto-Submitted": "auto-replied"}
    assert is_bulk_message(headers, "bob@somecompany.com") is True


def test_auto_replied_at_personal_domain_not_bulk() -> None:
    """Pass 8C Fix 5 + Fix 1: `Auto-Submitted` on a personal-domain
    sender does NOT flag — the personal-domain exemption overrides it.
    A family member's OOO auto-reply still mentions family relationships
    and should not be silently dropped."""
    headers = {"Auto-Submitted": "auto-replied"}
    assert is_bulk_message(headers, "mom@gmail.com") is False
    # Even auto-generated from a personal domain.
    headers = {"Auto-Submitted": "auto-generated"}
    assert is_bulk_message(headers, "dad@icloud.com") is False


# ---------------------------------------------------------------------------
# Pass 17A — Google Calendar notification exemption
# ---------------------------------------------------------------------------


def test_calendar_notification_sender_not_bulk() -> None:
    """Pass 17A: `calendar-notification@google.com` is NEVER bulk — even
    when carrying `List-Unsubscribe` (which Google routinely attaches).
    The email contains accepted-invite + attendee data that
    `calendar_email_parser.py` mines for family signal."""
    headers = {
        "List-Unsubscribe": "<mailto:unsubscribe@google.com>",
    }
    assert (
        is_bulk_message(headers, "calendar-notification@google.com")
        is False
    )
    # Sender check alone also exempts.
    assert is_bulk_sender("calendar-notification@google.com") is False


def test_calendar_notification_sender_with_display_name_not_bulk() -> None:
    """Display-name + angle-bracket form must still be parsed and exempted."""
    addr = "Google Calendar <calendar-notification@google.com>"
    assert is_bulk_message({}, addr) is False
    assert is_bulk_sender(addr) is False


def test_calendar_notification_sender_subdomain_not_bulk() -> None:
    """Pass 17A: subdomain variant (`@calendar.google.com`) is also
    exempt — match is suffix-based."""
    addr = "calendar-notification@calendar.google.com"
    assert is_bulk_message({}, addr) is False
    assert is_bulk_sender(addr) is False


def test_calendar_notification_even_with_list_id_not_bulk() -> None:
    """Pass 17A: the calendar exemption is unconditional — even if
    Google ever adds a `List-Id` header, we don't want to drop these.
    The downstream parser handles the message regardless."""
    headers = {
        "List-Id": "<calendar.google.com>",
        "List-Unsubscribe": "<mailto:unsubscribe@google.com>",
        "Precedence": "bulk",
    }
    assert (
        is_bulk_message(headers, "calendar-notification@google.com")
        is False
    )


def test_other_google_senders_still_filtered_appropriately() -> None:
    """Pass 17A: the exemption is narrow — `noreply@google.com` (NOT a
    calendar mailer) is still bulk when combined with the bulk-sender
    heuristic. The calendar exemption only covers the specific
    calendar-* local-parts."""
    # `noreply@google.com` — the local part matches strict-system patterns
    # which fire even at a personal domain, AND google.com isn't in the
    # personal-domain list. Bulk.
    assert is_bulk_sender("noreply@google.com") is True


def test_calendar_variant_calendar_at_google_not_bulk() -> None:
    """Pass 17A: `calendar@google.com` (legacy variant) is exempt."""
    assert is_bulk_message({}, "calendar@google.com") is False
    assert is_bulk_sender("calendar@google.com") is False
