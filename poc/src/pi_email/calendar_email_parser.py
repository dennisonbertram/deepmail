"""Calendar-notification email parsing (Pass 17A).

Why this module exists:
  The user's actual family members (kids: Vitus, Elio; partner: Jana; etc.)
  routinely appear in Google Calendar event titles ("Accepted: Vitus
  Birthday in school") and attendee lists, yet they rarely show up in the
  *body* of normal email — the entity extractor never sees them. Pass 14B
  added a Calendar API client (`calendar_evidence.py`) but that path needs
  the `calendar.readonly` OAuth scope; until the user re-consents we can't
  hit the Calendar API directly.

  HOWEVER — when the user accepts a calendar invite, Google sends a
  confirmation email FROM `calendar-notification@google.com` whose
  subject contains the event title and whose body contains the attendee
  list. Those emails are already in Gmail, already in the corpus, and
  available with no new scopes. This module mines them.

The pipeline:

  1. `is_calendar_notification_sender(from_addr)` — first-line filter
     for Google's calendar mailers (and a couple of legacy variants we
     might encounter).
  2. `parse_calendar_email(subject, body)` — extract event title,
     attendee list, and kinship-shaped names from the title. Returns
     a list of `CalendarEmailPerson` records.
  3. Each `CalendarEmailPerson` carries a `family_signal_strength` so
     downstream (loop.py -> materializer's candidate gather) can
     prioritize them.

We deliberately reuse the kinship-word vocabulary from
`calendar_evidence.py` so signals from the Calendar API path (when
available) and this email-derived path produce the same tags. The
title-parser borrows the same possessive + adjacency heuristics —
calendar event titles look the same whether they reach us via the
API or via a confirmation email subject.

This is a NO-RE-AUTH FALLBACK to Pass 14B's full Calendar API ingest.
When the user grants `calendar.readonly` and we run the full path,
this module still fires but contributes overlapping signal — duplicate
persons get merged by canonicalization downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .calendar_evidence import (
    _KINSHIP_WORDS,
    _RELATION_ONLY_WORDS,
    _extract_names_from_title,
    _title_has_kinship_word,
)


# ---------------------------------------------------------------------------
# Sender detection
# ---------------------------------------------------------------------------

# Local-part + domain pairs from which Google Calendar sends event
# confirmations / invites. The primary one is
# `calendar-notification@google.com`; older / less common variants are
# tolerated. Domain matching is suffix-based so a `@calendar.google.com`
# subdomain variant (if Google ever uses one) is covered.
#
# We keep the set narrow — overly broad coverage would re-introduce the
# false positives the bulk filter was added to suppress. If real-world
# examples of other Calendar mailers surface, add them here.
CALENDAR_NOTIFICATION_SENDERS = frozenset({
    "calendar-notification@google.com",
    "calendar-noreply@google.com",
    "calendar@google.com",
})

# Local-parts that, paired with `google.com` or a Calendar subdomain,
# indicate Google Calendar mail. Kept separate from
# CALENDAR_NOTIFICATION_SENDERS so we can match
# `calendar-notification@calendar.google.com` (subdomain variant) without
# enumerating every possible host.
CALENDAR_NOTIFICATION_LOCAL_PARTS = frozenset({
    "calendar-notification",
    "calendar-noreply",
    "calendar",
})

# Domains (suffix-matched) that host Google Calendar mailers.
CALENDAR_NOTIFICATION_DOMAINS = frozenset({
    "google.com",
    "calendar.google.com",
})


# Match `local@domain` out of either bare or `Name <addr>` From: forms.
# Duplicated from filters.py — keeping this module's import surface small.
_ANGLE_ADDR_RE = re.compile(r"<([^<>@\s]+@[^<>@\s]+)>")
_BARE_ADDR_RE = re.compile(r"([^\s<>,;]+@[^\s<>,;]+)")


def _extract_address(from_addr: str | None) -> str:
    """Pull the bare `local@domain` out of a header value, lowercased.

    Returns "" on parse failure / None input. Duplicated from
    `filters._extract_address` — keeping this module independent.
    """
    if not from_addr:
        return ""
    s = from_addr.strip()
    if not s:
        return ""
    m = _ANGLE_ADDR_RE.search(s)
    if m:
        return m.group(1).strip().lower()
    m = _BARE_ADDR_RE.search(s)
    if m:
        return m.group(1).strip().lower()
    return ""


def is_calendar_notification_sender(from_addr: str | None) -> bool:
    """Return True if the From: address is a Google Calendar mailer.

    Accepts either a bare `addr@domain` or the `Name <addr>` header
    form. Match logic:

      1. Exact match against `CALENDAR_NOTIFICATION_SENDERS`.
      2. Local-part in `CALENDAR_NOTIFICATION_LOCAL_PARTS` AND
         domain (suffix-matched) in `CALENDAR_NOTIFICATION_DOMAINS`.

    Tolerates None / empty / malformed input — returns False.
    """
    addr = _extract_address(from_addr)
    if not addr or "@" not in addr:
        return False
    if addr in CALENDAR_NOTIFICATION_SENDERS:
        return True
    local, _, domain = addr.rpartition("@")
    local = local.lower()
    domain = domain.lower()
    if local not in CALENDAR_NOTIFICATION_LOCAL_PARTS:
        return False
    # Domain suffix match — accepts `google.com` AND `calendar.google.com`.
    for d in CALENDAR_NOTIFICATION_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return True
    return False


# ---------------------------------------------------------------------------
# Calendar-shape detection (Pass 19)
# ---------------------------------------------------------------------------

# Pass-19 finding: most "Accepted: <event>" mails the user actually
# receives are NOT From: calendar-notification@google.com. When an
# attendee RSVPs through Google Calendar, the resulting confirmation
# is sent FROM the attendee's own address (e.g. the user's spouse at
# `jana.uhlarikova@gmail.com`) carrying `Auto-Submitted: auto-replied`
# and a body whose text starts "Jana Bertram has accepted this
# invitation." with a `calendar.google.com/calendar/event?...` URL
# in the trailing footer. So `is_calendar_notification_sender` returns
# False for every one of those even though they carry the kids'
# birthday-event titles ("Accepted: Vitus Birthday in school @ ...").
#
# This module's calendar-shape detector closes the gap: subject must
# start with a known calendar prefix AND body must reference a Google
# Calendar URL or one of the characteristic "has accepted/declined/
# tentatively accepted this invitation" sentinel phrases. Both signals
# combined are tight enough that we're confident the email is a real
# calendar response (a casual personal email is extremely unlikely to
# carry both a subject like "Accepted: Birthday" AND a google calendar
# URL in its body).
_CALENDAR_BODY_MARKERS: tuple[str, ...] = (
    "calendar.google.com/calendar",
    "invitation from google calendar",
    "has accepted this invitation",
    "has tentatively accepted this invitation",
    "has declined this invitation",
    "you are receiving this email because you are subscribed to calendar",
)


def looks_like_calendar_invite_response(
    subject: str | None,
    body: str | None,
) -> bool:
    """True iff (subject, body) look like a Google Calendar invite response.

    Detection logic:

      1. Subject must start (case-insensitive) with one of the known
         Google Calendar prefixes (`Accepted:`, `Declined:`, `Invitation:`,
         `Updated invitation:`, `Reminder:`, ...).
      2. Body must contain at least one of `_CALENDAR_BODY_MARKERS`:
         - a `calendar.google.com/calendar` URL (universal in the
           footer of any GCal-sent body)
         - the `invitation from google calendar` footer line
         - the `has (accepted|tentatively accepted|declined) this invitation`
           opener that GCal uses on RSVP confirmations
         - the `you are receiving this email because you are subscribed to
           calendar` unsubscribe footer (line-wrapped variants tolerated by
           substring match)

    Both must hold. This is intentionally tighter than either signal
    alone — a personal email whose subject begins "Reminder: ..." won't
    be misidentified unless its body also has a calendar.google.com URL,
    which is unusual outside actual GCal traffic.

    Tolerates None / empty input — returns False.
    """
    if not subject or not body:
        return False
    s = subject.strip().lower()
    if not any(s.startswith(p.lower()) for p in CALENDAR_SUBJECT_PREFIXES):
        return False
    bl = body.lower()
    for marker in _CALENDAR_BODY_MARKERS:
        if marker in bl:
            return True
    return False


# ---------------------------------------------------------------------------
# Subject prefix stripping
# ---------------------------------------------------------------------------

# Google Calendar prefixes the user's RSVP / event-action onto the
# subject. Order matters when stripping: the longer / more-specific
# variants must be tried before short prefixes (e.g. "Updated invitation"
# before "Invitation"). We strip case-insensitively and accept either
# colon or space separator.
CALENDAR_SUBJECT_PREFIXES: tuple[str, ...] = (
    "Updated invitation:",
    "Updated invitation",
    "Canceled event:",
    "Cancelled event:",
    "Canceled:",
    "Cancelled:",
    "Accepted:",
    "Tentative:",
    "Declined:",
    "Invitation:",
    "Reminder:",
    # Date-prefixed forms that Google sometimes uses on RSVP confirmations.
    # Conservative: we don't try to enumerate every locale-specific
    # variant — only the canonical English forms.
)

_PREFIX_RES = [
    re.compile(rf"^\s*{re.escape(p)}\s*", re.IGNORECASE)
    for p in CALENDAR_SUBJECT_PREFIXES
]

# Match the date-and-time tail Google Calendar appends to subjects when
# the user accepts/declines/forwards an invite, e.g.
#   "Vitus Birthday in school @ Thu May 28, 2026 9:30am - 10:30am (EDT) (dennison@withtally.com)"
# We strip from the first ` @ <Weekday> ` token onward — `Weekday` is a 3-9
# letter capitalized token, followed by another capitalized word
# (the month) and a numeric day, comma, year. The pattern is intentionally
# tight so a real event title with " @ " in it (e.g. "Coffee @ Joe's") is
# not over-eagerly truncated; we require the canonical Calendar suffix
# shape (Day Month Date, Year) immediately after.
_DATE_SUFFIX_RE = re.compile(
    r"\s+@\s+[A-Z][a-z]{2,8}\s+[A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4}.*$"
)


def extract_event_title(subject: str | None) -> str:
    """Strip Google Calendar's subject prefix to recover the raw event title.

    Examples (from real fixtures):
      "Accepted: Vitus Birthday in school" -> "Vitus Birthday in school"
      "Invitation: Family Dinner @ Sat"    -> "Family Dinner @ Sat"
      "Updated invitation: Anniversary"    -> "Anniversary"
      "Just a regular subject"             -> "Just a regular subject"
      ""                                   -> ""

    If no known prefix matches, returns the trimmed input unchanged.

    Pass-19: Real Gmail calendar acceptance subjects include a canonical
    date-and-time tail:
      "Accepted: Vitus Birthday in school @ Thu May 28, 2026 9:30am - 10:30am (EDT) (dennison@withtally.com)"
    The trailing date tokens (Thu, May, 28, 2026, ...) leaked into
    `_extract_names_from_title` as bogus person candidates ("Thu" became a
    kinship-event "person"). We now strip the canonical suffix here so the
    title-parser sees only the real event title text.
    """
    if not subject:
        return ""
    s = subject.strip()
    for rx in _PREFIX_RES:
        m = rx.match(s)
        if m:
            s = s[m.end():].strip()
            break
    # Strip the canonical " @ <Day> <Month> <DD>, <YYYY> ..." suffix.
    # Tight pattern — only the well-formed Google Calendar tail is removed.
    s = _DATE_SUFFIX_RE.sub("", s).strip()
    return s


# ---------------------------------------------------------------------------
# Attendee list parsing
# ---------------------------------------------------------------------------

# Lines that look like "Name <email>" or "email" inside the body. The
# Google Calendar body format has changed over the years; we accept
# several styles. Examples (from observed real messages):
#
#   Guests
#     - alice@gmail.com - yes
#     - bob.smith@example.com
#     - Jana Bertram <jana@example.com>
#
#   Attendees:
#     alice@example.com (organizer)
#     bob.smith@example.com
#
# We don't try to recover the RSVP status — we just want the (name, email)
# pairs. The display-name regex is intentionally lenient: a single
# capitalized token ("Jana") qualifies if the email follows in angle
# brackets.

# Name-then-email form: "Jana Bertram <jana@example.com>" or
# "Jana <jana@example.com>". The name capture is up to the `<`.
_NAME_ANGLE_EMAIL_RE = re.compile(
    r"([A-Za-z][\w .,'\-]*?)\s*<([^<>@\s]+@[^<>@\s]+)>"
)

# Bare-email form anywhere on a line, captured with a generous boundary.
_BARE_EMAIL_LINE_RE = re.compile(
    r"\b([\w.+\-]+@[\w.\-]+\.[A-Za-z]{2,})\b"
)


def parse_attendees_from_body(body: str | None) -> list[tuple[str, str]]:
    """Extract `(display_name, email)` pairs from a calendar notification body.

    Returns a list (preserving first-occurrence order, deduplicated by
    lowercased email). When no display name is found for an address we
    fall back to the local-part — better than dropping the attendee
    entirely.

    Filters:
      * Empty body / None -> []
      * `calendar-notification@google.com` / `calendar.google.com` self-
        references are dropped — they're the mailer, not a person.
      * Resource-calendar addresses ending in `.calendar.google.com`
        (rooms, equipment) are dropped.

    Skips lines starting with `Unsubscribe`, `Manage`, or anything that
    looks like Google's footer — those frequently embed a URL with
    `@` in a query string and would otherwise read as an attendee.
    """
    if not body:
        return []

    # Reject obvious footer / unsubscribe sections that frequently embed
    # `?email=...` URLs. We split on lines and drop everything after the
    # first footer marker. Conservative — if neither marker appears, we
    # keep the whole body.
    text = body
    for marker in (
        "Unsubscribe ",
        "You are receiving this",
        "Forwarding this invitation",
        "google.com/calendar/event?",
        "https://www.google.com/calendar",
    ):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]

    seen: dict[str, tuple[str, str]] = {}
    order: list[str] = []

    # First pass — name-then-angle-email (preserves the display name).
    for m in _NAME_ANGLE_EMAIL_RE.finditer(text):
        raw_name = m.group(1).strip(" -\t,;:")
        email = m.group(2).strip().lower()
        if not _is_attendee_email_acceptable(email):
            continue
        # Reject obvious junk leading text: a name made entirely of
        # punctuation or single character is dropped in favour of the
        # local-part.
        clean_name = _clean_display_name(raw_name)
        if not clean_name:
            clean_name = email.split("@", 1)[0]
        if email in seen:
            # Prefer the longest non-fallback name we encounter.
            prev_name, _ = seen[email]
            if len(clean_name) > len(prev_name):
                seen[email] = (clean_name, email)
            continue
        seen[email] = (clean_name, email)
        order.append(email)

    # Second pass — bare-email lines (no display name). Only adds new
    # emails not already captured with a display name.
    for m in _BARE_EMAIL_LINE_RE.finditer(text):
        email = m.group(1).strip().lower()
        if not _is_attendee_email_acceptable(email):
            continue
        if email in seen:
            continue
        seen[email] = (email.split("@", 1)[0], email)
        order.append(email)

    return [seen[e] for e in order]


def _is_attendee_email_acceptable(email: str) -> bool:
    """Filter out self-mailer / resource / footer-junk emails."""
    if not email or "@" not in email:
        return False
    # The Calendar mailer's own address must not show up as an attendee.
    if email in CALENDAR_NOTIFICATION_SENDERS:
        return False
    local, _, domain = email.rpartition("@")
    if not local or not domain:
        return False
    # Resource calendars (rooms / equipment) are not people.
    if domain.endswith(".calendar.google.com"):
        return False
    # `noreply@google.com` style is not a person — drop.
    if local in {"noreply", "no-reply", "donotreply", "support"}:
        return False
    return True


# Filter for display-name cleanup: drop obvious non-name garbage.
_NAME_CHAR_RE = re.compile(r"[A-Za-z]")


def _clean_display_name(raw: str) -> str:
    """Sanitize a display-name fragment pulled before `<email>`.

    Rules:
      * Strip surrounding quotes / whitespace / list bullets.
      * If the cleaned name has no letters, drop it (return "").
      * If the cleaned name is a single word and that word is one of
        the kinship-relation-only stopwords ("mom", "dad", ...), keep
        it as-is — we DO want to surface "Mom <mom@gmail.com>" as a
        person record, the judge will figure out the rest.
    """
    s = raw.strip().strip("\"'`").strip(" -\t,;:")
    if not s:
        return ""
    if not _NAME_CHAR_RE.search(s):
        return ""
    # Collapse internal whitespace.
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Public dataclass + main parser
# ---------------------------------------------------------------------------


@dataclass
class CalendarEmailPerson:
    """A person derived from a Google Calendar notification email.

    `source` records WHICH parser produced this record:
      * "event_title"       — extracted from the subject's event-title text.
      * "attendee"          — pulled from the body's attendee/guest list.
      * "both"              — same name surfaced in both (rare but possible).

    `event_title` is the cleaned event title (post-`extract_event_title`).
    Carrying it forward lets the materializer cite the originating event
    in the candidate biography ("Calendar: 'Vitus Birthday in school'").

    `family_signal_strength` ∈ [0.0, 1.0] mirrors `calendar_evidence`'s
    scoring scale; `family_signal_source` is the tag describing which
    detector fired.
    """

    name: str
    email: str | None
    source: str
    event_title: str
    family_signal_strength: float
    family_signal_source: str
    matched_kinship_words: list[str] = field(default_factory=list)


# Family-signal scoring constants. Mirrors `calendar_evidence.score_family_signal`
# but expressed against the smaller per-email view (no recurring-event
# aggregation — we'd need cross-message correlation for that, which the
# materializer can do later if we feed it the raw records).
_SIGNAL_POSSESSIVE = (0.90, "possessive_in_title")
_SIGNAL_KINSHIP_EVENT = (0.85, "kinship_event")
_SIGNAL_PERSONAL_ATTENDEE = (0.65, "personal_attendee")
_SIGNAL_FAMILY_EVENT_ATTENDEE = (0.65, "family_event_attendee")
_SIGNAL_WEAK = (0.50, "weak")


# Personal-domain set — duplicated from filters.py to keep this module
# importable without circular pulls. Drift risk acknowledged; if the
# filters list grows we should refactor both to a shared constants
# module (out of scope for Pass 17A).
_PERSONAL_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com",
    "icloud.com", "me.com", "mac.com",
    "yahoo.com", "yahoo.co.uk", "yahoo.ca", "yahoo.fr", "yahoo.de",
    "hotmail.com", "outlook.com", "live.com", "msn.com",
    "aol.com", "protonmail.com", "proton.me",
    "fastmail.com", "fastmail.fm",
    "tutanota.com", "tuta.io",
    "zoho.com",
    "duck.com",
})


def _email_at_personal_domain(email: str | None) -> bool:
    """True iff `email`'s domain is in the personal-providers set."""
    if not email or "@" not in email:
        return False
    domain = email.rpartition("@")[2].lower()
    return domain in _PERSONAL_EMAIL_DOMAINS


def parse_calendar_email(
    subject: str | None,
    body: str | None,
) -> list[CalendarEmailPerson]:
    """Extract persons from a single Google Calendar notification email.

    Steps:
      1. Strip the subject prefix (Accepted: / Invitation: / ...) to
         recover the raw event title.
      2. Run the event title through the kinship-word + possessive
         detectors borrowed from `calendar_evidence` to surface
         title-only person candidates (kids' first names, "Vitus
         Birthday in school" -> "Vitus").
      3. Parse the body for an attendee list.
      4. Merge by lowercased name: an attendee whose display-name
         matches a title-extracted person collapses into one record
         tagged source="both" with the stronger of the two signals.
      5. Score: possessive > kinship_event > attendee at personal
         domain > attendee at family-themed event > weak.

    Returns the merged + scored list. Empty list if neither the subject
    nor the body yields a person.

    Tolerates None inputs by treating them as empty strings.
    """
    event_title = extract_event_title(subject)

    # --- Title-derived candidates ---
    by_name: dict[str, CalendarEmailPerson] = {}
    title_has_kinship = (
        _title_has_kinship_word(event_title) if event_title else False
    )

    for name, kw, detector in _extract_names_from_title(event_title):
        key = name.lower()
        if detector == "possessive_in_title":
            strength, source = _SIGNAL_POSSESSIVE
        else:
            strength, source = _SIGNAL_KINSHIP_EVENT
        by_name[key] = CalendarEmailPerson(
            name=name,
            email=None,
            source="event_title",
            event_title=event_title,
            family_signal_strength=strength,
            family_signal_source=source,
            matched_kinship_words=[kw],
        )

    # --- Attendees ---
    by_email: dict[str, CalendarEmailPerson] = {}
    for display, email in parse_attendees_from_body(body):
        # Skip attendees that are obviously the user (we can't know without
        # user_emails, but the caller will dedupe self-references via the
        # downstream materializer).
        if email in by_email:
            continue
        # Determine signal strength for this attendee. A title-themed
        # event upgrades the attendee tag to "family_event_attendee";
        # a personal-domain attendee at a non-themed event still
        # qualifies as "personal_attendee".
        if title_has_kinship:
            strength, source = _SIGNAL_FAMILY_EVENT_ATTENDEE
        elif _email_at_personal_domain(email):
            strength, source = _SIGNAL_PERSONAL_ATTENDEE
        else:
            strength, source = _SIGNAL_WEAK
        person = CalendarEmailPerson(
            name=display or email.split("@", 1)[0],
            email=email,
            source="attendee",
            event_title=event_title,
            family_signal_strength=strength,
            family_signal_source=source,
            matched_kinship_words=[],
        )
        by_email[email] = person

    # --- Merge title-only + attendees by name ---
    merged: list[CalendarEmailPerson] = []
    used_name_keys: set[str] = set()

    # Walk attendees first because they carry an email (the more
    # specific identifier). If an attendee's display name matches a
    # title-only candidate, fold them together as source="both".
    for email, attendee in by_email.items():
        name_key = attendee.name.lower()
        sibling = by_name.get(name_key)
        if sibling is not None:
            # Take the stronger signal; preserve both source tags.
            if sibling.family_signal_strength > attendee.family_signal_strength:
                attendee.family_signal_strength = sibling.family_signal_strength
                attendee.family_signal_source = sibling.family_signal_source
                attendee.matched_kinship_words = list(sibling.matched_kinship_words)
            attendee.source = "both"
            used_name_keys.add(name_key)
        merged.append(attendee)

    for name_key, person in by_name.items():
        if name_key in used_name_keys:
            continue
        merged.append(person)

    # Sort by descending signal strength then name for deterministic
    # output. The fixture demo test snapshots rely on stable ordering.
    merged.sort(
        key=lambda p: (-p.family_signal_strength, p.name.lower())
    )
    return merged


# Re-export the kinship-word constants under module-local names so
# callers + tests don't need to know they originated in
# calendar_evidence.py. This is a thin alias layer; mutating one
# mutates the other (frozenset, so neither is mutable anyway).
KINSHIP_WORDS = _KINSHIP_WORDS
RELATION_ONLY_WORDS = _RELATION_ONLY_WORDS
