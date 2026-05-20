"""Tests for `pi_email.calendar_email_parser` — Google Calendar email mining.

Pass 17A extracts family signal from calendar-notification emails already
in the corpus. We test:

  * Sender detection (true positives + true negatives).
  * Subject prefix stripping ("Accepted: ..." -> "...").
  * Title-name extraction:
      - kinship-word adjacency (e.g. "Vitus Birthday in school").
      - possessive form (e.g. "Mom's anniversary lunch" / "Vitus's school").
  * Attendee parsing in the two common body formats:
      - Name + angle-bracket email.
      - Bare-email lines.
  * Family-signal scoring tiers.
  * End-to-end `parse_calendar_email` over a realistic mocked body.
"""

from __future__ import annotations

import sys
from pathlib import Path


POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.calendar_email_parser import (  # noqa: E402
    CalendarEmailPerson,
    extract_event_title,
    is_calendar_notification_sender,
    looks_like_calendar_invite_response,
    parse_attendees_from_body,
    parse_calendar_email,
)


# ---------------------------------------------------------------------------
# Subject prefix stripping
# ---------------------------------------------------------------------------


def test_strip_accepted_prefix() -> None:
    assert extract_event_title("Accepted: Vitus Birthday") == "Vitus Birthday"


def test_strip_invitation_prefix() -> None:
    assert extract_event_title("Invitation: Family Dinner") == "Family Dinner"


def test_strip_updated_invitation_prefix() -> None:
    # "Updated invitation:" is longer than "Invitation:" — must be matched first.
    out = extract_event_title("Updated invitation: Anniversary")
    assert out == "Anniversary"


def test_strip_tentative_declined_canceled_prefixes() -> None:
    assert extract_event_title("Tentative: Anniversary lunch") == "Anniversary lunch"
    assert extract_event_title("Declined: Wedding") == "Wedding"
    assert extract_event_title("Canceled event: Birthday Party") == "Birthday Party"
    assert extract_event_title("Cancelled event: Family Dinner") == "Family Dinner"


def test_strip_reminder_prefix() -> None:
    assert extract_event_title("Reminder: Dad's birthday") == "Dad's birthday"


def test_strip_prefix_case_insensitive() -> None:
    """User-forwarded mails sometimes have uppercased prefixes."""
    assert extract_event_title("ACCEPTED: Vitus Birthday") == "Vitus Birthday"


def test_no_prefix_returns_input_trimmed() -> None:
    assert extract_event_title("Regular subject") == "Regular subject"
    assert extract_event_title("  spaces  ") == "spaces"


def test_empty_subject_returns_empty() -> None:
    assert extract_event_title("") == ""
    assert extract_event_title(None) == ""


# Pass-19 — Google Calendar appends `@ <Weekday> <Month> <DD>, <YYYY> ...`
# to the subject. We strip it so the title parser doesn't pick up date
# tokens as bogus person names.


def test_strip_date_suffix_from_title() -> None:
    raw = (
        "Accepted: Vitus Birthday in school @ Thu May 28, 2026 "
        "9:30am - 10:30am (EDT) (dennison@withtally.com)"
    )
    assert extract_event_title(raw) == "Vitus Birthday in school"


def test_strip_date_suffix_with_elio() -> None:
    raw = (
        "Accepted: Elio Birthday Party @ Sun Mar 8, 2026 "
        "10am - 2pm (EDT) (dennison@withtally.com)"
    )
    assert extract_event_title(raw) == "Elio Birthday Party"


def test_at_symbol_in_event_title_not_stripped() -> None:
    # The tight " @ <Weekday> <Month> <DD>, <YYYY>" pattern means a real
    # event title that happens to contain ` @ <Place> ` is NOT truncated
    # (no canonical date suffix follows). Verifies we didn't over-strip.
    assert extract_event_title("Invitation: Coffee @ Joe's") == "Coffee @ Joe's"


# ---------------------------------------------------------------------------
# Sender detection
# ---------------------------------------------------------------------------


def test_is_calendar_notification_sender_primary() -> None:
    assert is_calendar_notification_sender("calendar-notification@google.com") is True


def test_is_calendar_notification_sender_with_display_name() -> None:
    assert (
        is_calendar_notification_sender(
            "Google Calendar <calendar-notification@google.com>"
        )
        is True
    )


def test_is_calendar_notification_sender_calendar_at_google() -> None:
    assert is_calendar_notification_sender("calendar@google.com") is True


def test_is_calendar_notification_sender_subdomain() -> None:
    """Subdomain variant — calendar-notification@calendar.google.com."""
    assert (
        is_calendar_notification_sender(
            "calendar-notification@calendar.google.com"
        )
        is True
    )


def test_is_calendar_notification_sender_calendar_noreply() -> None:
    assert (
        is_calendar_notification_sender("calendar-noreply@google.com") is True
    )


def test_is_calendar_notification_sender_negatives() -> None:
    # Personal address — not a calendar mailer.
    assert is_calendar_notification_sender("alice@gmail.com") is False
    # Generic noreply at google — NOT a calendar mailer (local-part doesn't match).
    assert is_calendar_notification_sender("noreply@google.com") is False
    # Looks-like-calendar but at a non-google domain.
    assert is_calendar_notification_sender("calendar@example.com") is False
    # Empty / None / malformed.
    assert is_calendar_notification_sender("") is False
    assert is_calendar_notification_sender(None) is False
    assert is_calendar_notification_sender("not an email") is False


# ---------------------------------------------------------------------------
# Pass-19: calendar-shape detection (sender-agnostic).
#
# Real Gmail acceptance emails come From: the user's own address or their
# spouse's gmail, NOT From: calendar-notification@google.com. The shape
# detector identifies them by subject prefix + characteristic body
# markers (calendar.google.com URL, "has accepted this invitation" line).
# ---------------------------------------------------------------------------


_REAL_GMAIL_ACCEPTANCE_BODY = """\
Jana Bertram has accepted this invitation.

Vitus Birthday in school
Thursday May 28, 2026 — 9:30am - 10:30am
Eastern Time - New York

Organizer
dennison@withtally.com

Guests
dennison@withtally.com - organizer
Jana Bertram

View all guest info
https://calendar.google.com/calendar/event?action=VIEW&eid=...
Invitation from Google Calendar: https://calendar.google.com/calendar/
"""


def test_calendar_shape_detected_from_personal_gmail_sender() -> None:
    """Acceptance email From: spouse's gmail should still be detected."""
    subj = (
        "Accepted: Vitus Birthday in school @ Thu May 28, 2026 "
        "9:30am - 10:30am (EDT) (dennison@withtally.com)"
    )
    assert looks_like_calendar_invite_response(
        subj, _REAL_GMAIL_ACCEPTANCE_BODY
    ) is True


def test_calendar_shape_detected_from_custom_domain_sender() -> None:
    """Acceptance email From: user's own custom domain should be detected."""
    subj = (
        "Accepted: Elio Birthday Party @ Sun Mar 8, 2026 "
        "10am - 2pm (EDT) (dennison@withtally.com)"
    )
    assert looks_like_calendar_invite_response(
        subj, _REAL_GMAIL_ACCEPTANCE_BODY
    ) is True


def test_calendar_shape_requires_prefix() -> None:
    """Body looks calendarish but subject has no calendar prefix — reject."""
    assert (
        looks_like_calendar_invite_response(
            "Lunch at Joe's", _REAL_GMAIL_ACCEPTANCE_BODY
        )
        is False
    )


def test_calendar_shape_requires_body_marker() -> None:
    """Subject has prefix but body has no calendar markers — reject."""
    assert (
        looks_like_calendar_invite_response(
            "Accepted: Lunch", "ok sounds good"
        )
        is False
    )


def test_calendar_shape_empty_inputs() -> None:
    assert looks_like_calendar_invite_response(None, "body") is False
    assert looks_like_calendar_invite_response("subj", None) is False
    assert looks_like_calendar_invite_response("", "") is False


def test_parse_calendar_email_real_gmail_acceptance_extracts_kid() -> None:
    """End-to-end: a real-shape acceptance email yields the kid's name at
    family_signal_strength >= 0.80, and the title-stripper kept date
    tokens out of the candidate list."""
    subj = (
        "Accepted: Vitus Birthday in school @ Thu May 28, 2026 "
        "9:30am - 10:30am (EDT) (dennison@withtally.com)"
    )
    persons = parse_calendar_email(subj, _REAL_GMAIL_ACCEPTANCE_BODY)
    names = {p.name for p in persons}
    # Vitus is the kid name we MUST surface.
    assert "Vitus" in names
    # And we must NOT surface the date tokens that polluted Pass-18.
    assert "Thu" not in names
    assert "May" not in names
    # Vitus must be tagged as a kinship event (strength >= 0.80).
    vitus = next(p for p in persons if p.name == "Vitus")
    assert vitus.family_signal_strength >= 0.80
    assert vitus.family_signal_source == "kinship_event"


# ---------------------------------------------------------------------------
# Attendee parsing
# ---------------------------------------------------------------------------


def test_attendee_parsing_format_a_name_angle_email() -> None:
    """Google's 'Name <email>' attendee format."""
    body = """
    Guests
      Jana Bertram <jana@example.com>
      Bob Smith <bob.smith@example.com>
    """
    result = parse_attendees_from_body(body)
    assert ("Jana Bertram", "jana@example.com") in result
    assert ("Bob Smith", "bob.smith@example.com") in result


def test_attendee_parsing_format_b_bare_emails() -> None:
    """Older / simpler attendee format with bare email per line."""
    body = """
    Attendees:
      alice@gmail.com (organizer)
      vitus.dad@gmail.com
      jana@example.com
    """
    result = parse_attendees_from_body(body)
    emails = [e for _, e in result]
    assert "alice@gmail.com" in emails
    assert "vitus.dad@gmail.com" in emails
    assert "jana@example.com" in emails


def test_attendee_parsing_mixed_format() -> None:
    """Both formats can coexist in a single body."""
    body = """
    Guests
      Jana Bertram <jana@example.com>
      alice@gmail.com
    """
    result = parse_attendees_from_body(body)
    # Both should appear, the bracketed one keeps its display name.
    assert ("Jana Bertram", "jana@example.com") in result
    emails = [e for _, e in result]
    assert "alice@gmail.com" in emails


def test_attendee_parsing_drops_calendar_self_mailer() -> None:
    """The calendar mailer itself must never appear as an attendee."""
    body = "Sent from calendar-notification@google.com\nalice@gmail.com"
    result = parse_attendees_from_body(body)
    emails = [e for _, e in result]
    assert "calendar-notification@google.com" not in emails
    assert "alice@gmail.com" in emails


def test_attendee_parsing_drops_resource_calendars() -> None:
    """Room / equipment calendars are not people."""
    body = "alice@gmail.com\nconfroom@resource.calendar.google.com"
    result = parse_attendees_from_body(body)
    emails = [e for _, e in result]
    assert "alice@gmail.com" in emails
    assert all(not e.endswith(".calendar.google.com") for e in emails)


def test_attendee_parsing_empty_inputs() -> None:
    assert parse_attendees_from_body("") == []
    assert parse_attendees_from_body(None) == []


def test_attendee_parsing_dedupe_same_email() -> None:
    """A duplicate email across two attendee mentions collapses to one entry,
    preserving the better (longer non-fallback) display name."""
    body = """
    Guests
      jana@example.com
      Jana Bertram <jana@example.com>
    """
    result = parse_attendees_from_body(body)
    emails = [e for _, e in result]
    assert emails.count("jana@example.com") == 1
    name = next(n for n, e in result if e == "jana@example.com")
    # Either the longer "Jana Bertram" or the local-part fallback survives;
    # the merge should prefer the longer display name.
    assert name == "Jana Bertram"


# ---------------------------------------------------------------------------
# Title-name extraction (via parse_calendar_email end-to-end)
# ---------------------------------------------------------------------------


def test_kinship_word_in_title_extracts_name() -> None:
    """'Accepted: Vitus Birthday in school' -> a CalendarEmailPerson 'Vitus'."""
    persons = parse_calendar_email("Accepted: Vitus Birthday in school", "")
    by_name = {p.name: p for p in persons}
    assert "Vitus" in by_name
    p = by_name["Vitus"]
    # Score >= 0.80 because the title contains a kinship word ("Birthday").
    assert p.family_signal_strength >= 0.80
    assert p.family_signal_source == "kinship_event"
    assert p.event_title == "Vitus Birthday in school"
    assert p.source == "event_title"


def test_possessive_in_title() -> None:
    """'Mom's anniversary lunch' is detected as a kinship/relation event;
    'Mom' itself is a relation-only word so we DON'T emit a CalendarEmailPerson
    for it, but a possessive-form NAME ("Vitus's school") does emit."""
    persons = parse_calendar_email("Accepted: Vitus's school pickup", "")
    by_name = {p.name: p for p in persons}
    assert "Vitus" in by_name
    p = by_name["Vitus"]
    assert p.family_signal_source == "possessive_in_title"
    assert p.family_signal_strength >= 0.90

    # And the relation-only case — "Mom's anniversary" should NOT emit a
    # "Mom" person record (Mom is a relation word, not a name).
    persons = parse_calendar_email("Accepted: Mom's anniversary lunch", "")
    assert "Mom" not in {p.name for p in persons}


def test_personal_domain_attendee_signal() -> None:
    """An attendee at a personal email domain (gmail.com) on a non-family
    event title yields the personal_attendee signal at 0.65."""
    body = "Guests\n  Jana Bertram <jana@gmail.com>\n"
    persons = parse_calendar_email("Accepted: Lunch meeting", body)
    by_email = {p.email: p for p in persons}
    p = by_email["jana@gmail.com"]
    assert p.family_signal_source == "personal_attendee"
    assert p.family_signal_strength == 0.65


def test_attendee_at_family_event_gets_family_event_attendee_signal() -> None:
    """Attendees on a title with a kinship word are tagged
    family_event_attendee even if they sit at a business domain."""
    body = "Guests\n  Coworker <person@bigco.com>\n"
    persons = parse_calendar_email("Accepted: Family Dinner", body)
    by_email = {p.email: p for p in persons}
    p = by_email["person@bigco.com"]
    assert p.family_signal_source == "family_event_attendee"
    assert p.family_signal_strength == 0.65


def test_title_person_and_attendee_merge() -> None:
    """A title-only name AND an attendee with the same display name collapse
    into one CalendarEmailPerson tagged source='both' with the stronger
    signal."""
    body = "Guests\n  Vitus <vitus@example.com>\n"
    persons = parse_calendar_email("Accepted: Vitus Birthday in school", body)
    by_name_lc = {p.name.lower(): p for p in persons}
    p = by_name_lc["vitus"]
    assert p.source == "both"
    assert p.email == "vitus@example.com"
    # Take the stronger (kinship_event 0.85) over personal_attendee (0.65).
    assert p.family_signal_strength >= 0.80
    assert p.family_signal_source == "kinship_event"


def test_parse_calendar_email_empty_inputs() -> None:
    assert parse_calendar_email("", "") == []
    assert parse_calendar_email(None, None) == []


def test_parse_calendar_email_returns_sorted_by_strength() -> None:
    """Output is sorted descending by family_signal_strength then name."""
    body = "Guests\n  Bob <bob@bigco.com>\n  Alice <alice@gmail.com>\n"
    persons = parse_calendar_email("Accepted: Lunch meeting", body)
    # alice (personal domain, 0.65) > bob (weak, 0.50).
    assert persons[0].email == "alice@gmail.com"
    assert persons[1].email == "bob@bigco.com"


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


def test_calendar_email_person_default_field_factory() -> None:
    """matched_kinship_words defaults to a fresh empty list (not shared)."""
    a = CalendarEmailPerson(
        name="A",
        email=None,
        source="event_title",
        event_title="title",
        family_signal_strength=0.5,
        family_signal_source="weak",
    )
    b = CalendarEmailPerson(
        name="B",
        email=None,
        source="event_title",
        event_title="title",
        family_signal_strength=0.5,
        family_signal_source="weak",
    )
    a.matched_kinship_words.append("birthday")
    assert b.matched_kinship_words == []
