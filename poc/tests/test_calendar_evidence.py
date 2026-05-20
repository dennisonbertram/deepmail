"""Tests for `pi_email.calendar_evidence`.

We patch `googleapiclient.discovery.build` so no real network call is made.
The MagicMock pattern mirrors test_gmail_searcher.py: chained `.events().list(...)
.execute()` returns whatever we configure.

Test surface:
  1. Title parsing — kinship-word + birthday detector.
  2. Title parsing — possessive form detector.
  3. Attendee detector — personal-domain attendee surfaces.
  4. Recurring family-event high-score path.
  5. Paginated `list_recent_events` correctly concatenates multi-page
     responses.
  6. `credentials_have_calendar_scope` happy + unhappy paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.calendar_evidence import (  # noqa: E402
    CALENDAR_READONLY_SCOPE,
    CalendarPerson,
    GoogleCalendar,
    calendar_person_to_contact_evidence,
    credentials_have_calendar_scope,
    score_family_signal,
)


# ---------------- helpers ----------------


def _make_service(pages: list[dict]) -> MagicMock:
    """Build a MagicMock that quacks like a discovery-built calendar service.

    `pages` is a list of dicts the events().list().execute() should return,
    one per pagination call. Each dict may contain `items` and
    `nextPageToken` keys.
    """
    service = MagicMock(name="calendar_service")
    idx = {"i": 0}

    def list_execute_factory(*args, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        if i >= len(pages):
            return {"items": []}
        return pages[i]

    def events_list(**kwargs):
        m = MagicMock()
        m.execute.side_effect = lambda: list_execute_factory(**kwargs)
        return m

    service.events.return_value.list.side_effect = events_list
    return service


# ---------------- 1. kinship-word birthday detector ----------------


def test_extract_kinship_word_title_birthday():
    """"Vitus Birthday" -> CalendarPerson(name='Vitus', source contains
    'birthday_event')."""
    service = _make_service(
        [{"items": [{"summary": "Vitus Birthday"}]}]
    )
    creds = MagicMock()
    creds.scopes = [CALENDAR_READONLY_SCOPE]
    with patch("pi_email.calendar_evidence.build", return_value=service):
        cal = GoogleCalendar(creds)
        events = cal.list_recent_events()
        persons = cal.extract_family_signals(events)

    by_name = {p.name.lower(): p for p in persons}
    assert "vitus" in by_name, f"expected 'Vitus' in {[p.name for p in persons]}"
    v = by_name["vitus"]
    assert v.name == "Vitus"
    assert "birthday_event" in v.family_signal_source, v.family_signal_source
    assert v.family_signal_strength >= 0.75


# ---------------- 2. possessive form detector ----------------


def test_extract_possessive_title():
    """"Mom's anniversary" — the possessive relation-only word 'Mom' should
    NOT produce a person (relation-only filter), but a possessive form
    with a proper name like "Jana's birthday" should produce 'Jana' with
    a possessive source tag."""
    service = _make_service(
        [
            {
                "items": [
                    {"summary": "Mom's anniversary"},
                    {"summary": "Jana's birthday"},
                ]
            }
        ]
    )
    creds = MagicMock()
    creds.scopes = [CALENDAR_READONLY_SCOPE]
    with patch("pi_email.calendar_evidence.build", return_value=service):
        cal = GoogleCalendar(creds)
        events = cal.list_recent_events()
        persons = cal.extract_family_signals(events)

    names_lower = {p.name.lower() for p in persons}
    # 'Mom' is relation-only and must not be surfaced as a person.
    assert "mom" not in names_lower, f"got: {names_lower}"
    # 'Jana' should be surfaced with a possessive_in_title source tag.
    by_name = {p.name.lower(): p for p in persons}
    assert "jana" in by_name, f"got: {names_lower}"
    j = by_name["jana"]
    assert "possessive_in_title" in j.family_signal_source, j.family_signal_source
    # Possessive is graded at >= 0.9.
    assert j.family_signal_strength >= 0.85, j.family_signal_strength


# ---------------- 3. attendee personal-domain ----------------


def test_extract_attendee_personal_domain():
    """An attendee at a personal-email domain who appears in 5+ events
    surfaces with a `frequent_attendee_personal` source tag."""
    # Build 5 events where the same gmail-domain attendee shows up.
    items = []
    for i in range(5):
        items.append(
            {
                "summary": f"Coffee #{i}",
                "attendees": [
                    {"email": "auntcarol@gmail.com", "displayName": "Aunt Carol"},
                    {"email": "self@example.com", "self": True},
                ],
            }
        )
    service = _make_service([{"items": items}])
    creds = MagicMock()
    creds.scopes = [CALENDAR_READONLY_SCOPE]
    with patch("pi_email.calendar_evidence.build", return_value=service):
        cal = GoogleCalendar(creds)
        events = cal.list_recent_events()
        persons = cal.extract_family_signals(events)

    by_email = {p.email: p for p in persons if p.email}
    assert "auntcarol@gmail.com" in by_email, [p.email for p in persons]
    carol = by_email["auntcarol@gmail.com"]
    assert carol.is_attendee
    assert carol.attendee_event_count == 5
    assert "frequent_attendee_personal" in carol.family_signal_source, (
        carol.family_signal_source
    )
    # `self` attendee must be excluded.
    assert "self@example.com" not in by_email


# ---------------- 4. recurring family event high score ----------------


def test_recurring_family_event_high_score():
    """Same person mentioned in 3+ kinship-word events -> score >= 0.9 and
    `recurring_family_event` in the source tag."""
    service = _make_service(
        [
            {
                "items": [
                    {"summary": "Vitus Birthday"},
                    {"summary": "Family Dinner with Vitus"},
                    {"summary": "Vitus school anniversary"},
                ]
            }
        ]
    )
    creds = MagicMock()
    creds.scopes = [CALENDAR_READONLY_SCOPE]
    with patch("pi_email.calendar_evidence.build", return_value=service):
        cal = GoogleCalendar(creds)
        events = cal.list_recent_events()
        persons = cal.extract_family_signals(events)

    by_name = {p.name.lower(): p for p in persons}
    assert "vitus" in by_name, [p.name for p in persons]
    v = by_name["vitus"]
    assert v.family_signal_strength >= 0.9, v.family_signal_strength
    assert "recurring_family_event" in v.family_signal_source, (
        v.family_signal_source
    )
    # And the matched titles list should hold all three.
    assert len(v.appears_in_titles) == 3


# ---------------- 5. pagination ----------------


def test_list_recent_events_paginates():
    """A two-page response is concatenated by list_recent_events."""
    pages = [
        {
            "items": [
                {"summary": "Vitus Birthday"},
            ],
            "nextPageToken": "tok1",
        },
        {
            "items": [
                {"summary": "Elio Birthday"},
            ]
            # no nextPageToken => last page
        },
    ]
    service = _make_service(pages)
    creds = MagicMock()
    creds.scopes = [CALENDAR_READONLY_SCOPE]
    with patch("pi_email.calendar_evidence.build", return_value=service):
        cal = GoogleCalendar(creds)
        events = cal.list_recent_events()
    titles = [e.get("summary") for e in events]
    assert titles == ["Vitus Birthday", "Elio Birthday"]


# ---------------- 6. credentials_have_calendar_scope ----------------


def test_credentials_have_calendar_scope_happy():
    creds = MagicMock()
    creds.scopes = [
        "https://www.googleapis.com/auth/gmail.readonly",
        CALENDAR_READONLY_SCOPE,
    ]
    assert credentials_have_calendar_scope(creds) is True


def test_credentials_have_calendar_scope_missing():
    creds = MagicMock()
    creds.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    assert credentials_have_calendar_scope(creds) is False


def test_credentials_have_calendar_scope_none():
    """None credentials -> False (no crash)."""
    assert credentials_have_calendar_scope(None) is False


# ---------------- bonus: ContactEvidence bridge ----------------


def test_calendar_person_to_contact_evidence_round_trip():
    """The bridge to family_judge.ContactEvidence preserves the calendar
    person's name, score, and source tag prefix `calendar:`."""
    person = CalendarPerson(
        name="Vitus",
        email=None,
        is_attendee=False,
        appears_in_titles=["Vitus Birthday", "Vitus school pickup"],
        attendee_event_count=0,
        family_signal_strength=0.95,
        family_signal_source="recurring_family_event+birthday_event",
        matched_kinship_words=["birthday"],
    )
    ce = calendar_person_to_contact_evidence(person)
    assert ce.contact_name == "Vitus"
    assert ce.family_signal_strength == 0.95
    assert ce.family_signal_source.startswith("calendar:")
    assert "Vitus Birthday" in (ce.biography or "")


# ---------------- bonus: score_family_signal direct ----------------


def test_score_family_signal_weak_default():
    """A person with no recognized signal still gets a `weak` tag, not 0."""
    p = CalendarPerson(name="random", appears_in_titles=["random event"])
    strength, source = score_family_signal(p)
    assert strength == 0.40
    assert "weak" in source
