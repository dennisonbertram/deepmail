"""Google Calendar API client + family-event detection (Pass 14B).

After 13 passes of email-only + contacts-only family extraction the system
finds the spouse via surname match but no other family members. The user's
actual kids (Vitus, Elio) routinely show up in *calendar event titles* —
"Vitus Birthday", "Elio swim class", "Family Dinner Jana" — yet they never
surface as judge candidates because email content rarely names them and the
user doesn't curate a "Family" contact group either.

This module is the missing input. It fetches recent events from the user's
primary calendar, parses kinship-shaped event titles for capitalized names,
extracts attendee lists from family-themed events, and emits
`CalendarPerson` records with a `family_signal_strength` ∈ [0.0, 1.0] +
tagged source.

Why these signals, in priority order:
  1. Recurring family-themed event (3+ events): "Vitus Birthday" + "Vitus
     school pickup" + "Vitus dentist" — extremely strong. The user is
     organizing their life around this person, name-by-name.
  2. Possessive in event title ("Vitus's birthday", "Mom's anniversary"):
     the apostrophe-s grammar is almost always familial in personal
     calendars.
  3. Single kinship-word event ("Vitus Birthday", "Family Dinner"):
     medium-strong. Could also be a coworker's birthday tracked
     personally, hence not 1.0.
  4. Frequent attendee at personal-domain email (5+ events): the user
     spends significant time with this person, often family.
  5. Attendee at a family-themed event: weaker tiebreaker.

The output is a list of `CalendarPerson` objects. A `to_contact_evidence()`
helper converts them into the `ContactEvidence` shape already plumbed
through the judge — Pass 14A will wire that into the materializer.

Public surface:
  * `CALENDAR_READONLY_SCOPE` — the OAuth scope constant.
  * `CalendarPerson` dataclass — normalized person record.
  * `GoogleCalendar` — the client.
    `list_recent_events`, `extract_family_signals`,
    `list_family_calendar_persons`.
  * `score_family_signal` — pure scorer (testable without API).
  * `credentials_have_calendar_scope` — auth helper.
  * `calendar_person_to_contact_evidence` — bridge to family_judge.ContactEvidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


log = logging.getLogger(__name__)


# OAuth scope. Read-only is sufficient — we never mutate calendars.
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"


# Page size for events.list. The API caps at 2500; we ask for the page-size
# default and paginate. 250 keeps response bodies small without inflating
# round-trips for typical personal calendars.
_EVENTS_PAGE_SIZE = 250

# Default lookback window for `list_recent_events`. A year balances recall
# (annual birthdays / anniversaries fire) against noise (deeply stale
# events from people no longer in your life).
_DEFAULT_DAYS_BACK = 365

# Hard cap on the number of events we'll page through. Defensive — a
# pathological calendar could blow past this; we'd rather log a warning
# than spin forever.
_DEFAULT_MAX_EVENTS = 1000


# Kinship vocabulary. Aligns with `contacts._BIOGRAPHY_KINSHIP_WORDS` so
# the calendar source produces signals consistent with the contacts
# source. We additionally include event-shape words ("birthday",
# "anniversary", "wedding") which name an *event* rather than a relation
# but in a personal calendar almost always imply family.
_KINSHIP_WORDS: frozenset[str] = frozenset({
    # Event-shape kinship words.
    "birthday", "anniversary", "wedding", "baby",
    # Family group words.
    "kids", "family",
    # Parental.
    "mom", "dad", "mother", "father", "mommy", "daddy", "mama", "papa",
    # Marital.
    "wife", "husband", "spouse", "partner",
    # Child.
    "son", "daughter", "child",
    # Sibling.
    "brother", "sister", "sibling",
    # Grandparent.
    "grandma", "grandpa", "grandmother", "grandfather",
    # Extended.
    "aunt", "uncle", "cousin", "niece", "nephew",
    # In-law shapes.
    "in-law", "in-laws",
})

# Words that look like a kinship word but actually describe a *relation*
# rather than a *named person* — when these appear in a title without an
# adjacent capitalized name, we still want to surface the event but we
# DON'T want to create a "Mom" CalendarPerson (the user has a real
# mother; capturing her as "Mom" would be useless to the judge). These
# words DO still upweight any attendees on the event though.
_RELATION_ONLY_WORDS: frozenset[str] = frozenset({
    "mom", "dad", "mother", "father", "mommy", "daddy", "mama", "papa",
    "wife", "husband", "spouse", "partner",
    "brother", "sister", "sibling",
    "grandma", "grandpa", "grandmother", "grandfather",
    "aunt", "uncle",
    "in-law", "in-laws",
    "family", "kids",
})

# Tokens that should NEVER be treated as a person name even if they're
# title-cased in the calendar event. Calendars are full of these.
_NAME_STOPWORDS: frozenset[str] = frozenset({
    "and", "or", "the", "a", "an",
    "dinner", "lunch", "breakfast", "brunch", "drinks", "coffee", "tea",
    "meeting", "call", "appointment", "checkup", "checkin", "checkout",
    "doctor", "dentist", "school", "pickup", "dropoff",
    "party", "celebration", "event", "trip", "vacation",
    "day", "night", "morning", "evening", "afternoon",
    "with", "at", "in", "on", "for", "to", "from", "of",
    "happy", "celebrating",
    # Common stand-alone date / day words that show up in calendar titles.
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}) | _KINSHIP_WORDS


# Free-mail domains that signal "personal" rather than "work". Mirrors the
# materializer's heuristic; kept duplicated here so this module has no
# dependency on materializer.
_PERSONAL_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com",
    "yahoo.com", "ymail.com",
    "hotmail.com", "outlook.com", "live.com", "msn.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com", "proton.me", "protonmail.com",
    "fastmail.com", "fastmail.fm",
    "duck.com",
})


# Possessive form. Matches "Vitus's birthday", "Mom's anniversary",
# "Jana's school pickup". Two captures:
#   group(1): the bare name token (case-preserved).
#   group(2): the following kinship word (we use this to confirm intent).
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z'\-]+)['’]s\s+(\w+)",
)

# Bare possessive without trailing kinship word — e.g., "Vitus's school"
# where "school" isn't in the kinship list but the apostrophe-s grammar
# itself is signal enough in a personal calendar context.
_BARE_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z'\-]+)['’]s\b",
)


# ---------------- Data ----------------


@dataclass
class CalendarPerson:
    """A person derived from Calendar events.

    A `CalendarPerson` may represent either:
      (a) An attendee (`is_attendee=True`, `email` populated) — extracted
          from `event.attendees[].email`.
      (b) A name pulled from event titles (`is_attendee=False`,
          `email=None`) — title-only candidates whose existence we
          inferred from kinship-word adjacency.

    `appears_in_titles` is the deduplicated list of event titles where
    this person was named or attended; the materializer uses it to
    render a one-line bio for the judge ("Calendar: 'Vitus Birthday',
    'Vitus school pickup', ...").

    `family_signal_strength` ∈ [0.0, 1.0] — see `score_family_signal`.
    `family_signal_source` is a "+"-joined tag listing every detector
    that fired (e.g., "recurring_family_event+possessive_in_title").
    """

    name: str
    email: str | None = None
    is_attendee: bool = False
    appears_in_titles: list[str] = field(default_factory=list)
    attendee_event_count: int = 0
    family_signal_strength: float = 0.0
    family_signal_source: str = ""
    matched_kinship_words: list[str] = field(default_factory=list)


# ---------------- Title parsing ----------------


def _tokenize_title(title: str) -> list[str]:
    """Split a title into whitespace tokens, dropping pure-punctuation
    tokens but preserving original case. Apostrophes inside tokens are
    preserved so possessive forms remain detectable downstream.
    """
    if not title:
        return []
    # Replace common non-alphanumeric separators with whitespace so
    # "Vitus-Birthday" tokenizes as two words. Hyphens INSIDE a word
    # ("in-law") are handled by the regex layer, not here.
    cleaned = re.sub(r"[\(\)\[\]\{\},;:!\?\.]", " ", title)
    return [t for t in cleaned.split() if t]


def _strip_possessive(token: str) -> str:
    """Strip a trailing `'s` / `’s` from `token`. Otherwise return the
    token unchanged. CRITICAL: we must NOT strip a plain trailing `s`
    — that would turn "Vitus" into "Vitu".
    """
    if not token:
        return token
    for suffix in ("'s", "’s"):
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[: -len(suffix)]
    return token


def _looks_like_name(token: str) -> bool:
    """Cheap heuristic: token is a plausible person-name fragment.

    Rules:
      * Starts with an uppercase letter (proper noun).
      * Contains at least one lowercase letter (rules out all-caps acronyms
        like "RSVP", "TBD", "ETA").
      * Lowercase form is not in the stopword list.
      * Length >= 2.

    The caller is responsible for stripping a trailing possessive `'s`
    (see `_strip_possessive`) before passing the token in.
    """
    if not token or len(token) < 2:
        return False
    if not token[0].isupper():
        return False
    if not any(c.islower() for c in token):
        return False
    if token.lower() in _NAME_STOPWORDS:
        return False
    return True


def _find_kinship_word_indices(tokens: list[str]) -> list[tuple[int, str]]:
    """Return [(index, lowercased_word), ...] for every kinship word in
    `tokens`. Case-insensitive."""
    out: list[tuple[int, str]] = []
    for i, tok in enumerate(tokens):
        lc = tok.lower().rstrip(".,!?")
        if lc in _KINSHIP_WORDS:
            out.append((i, lc))
    return out


def _extract_names_from_title(title: str) -> list[tuple[str, str, str]]:
    """Find candidate (name, kinship_word, detector_tag) triples in a
    single event title.

    Detector tags:
      * "possessive_in_title" — "Vitus's birthday" form.
      * "kinship_adjacent" — "<Kinship> Name" or "Name <Kinship>" form.

    The returned list is deduplicated by lowercase name. Multiple
    detectors firing for the same name are collapsed; the first detector
    wins for the tag.
    """
    found: dict[str, tuple[str, str, str]] = {}

    # 1. Possessive forms. "Vitus's birthday" / "Mom's anniversary".
    #    "Mom's anniversary" should NOT produce a "Mom" CalendarPerson
    #    (relation-only), but we DO let the event mark the next-word
    #    kinship signal for any attendees. Possessive with a real name
    #    ("Vitus's school") fires. The "possessive" sentinel is ALWAYS
    #    appended on top of any kinship-word match so the scorer can
    #    grant both the possessive_in_title and the kinship-word
    #    detectors.
    for m in _POSSESSIVE_RE.finditer(title):
        name = m.group(1)
        following = m.group(2).lower()
        if name.lower() in _RELATION_ONLY_WORDS:
            continue
        if not _looks_like_name(name):
            continue
        # If the following word is itself a kinship word, use it;
        # otherwise mark the possessive grammar alone.
        kinship = following if following in _KINSHIP_WORDS else "possessive"
        found.setdefault(
            name.lower(), (name, kinship, "possessive_in_title")
        )

    # Catch bare possessives like "Vitus's school pickup" where "school"
    # isn't in the kinship list.
    for m in _BARE_POSSESSIVE_RE.finditer(title):
        name = m.group(1)
        if name.lower() in _RELATION_ONLY_WORDS:
            continue
        if not _looks_like_name(name):
            continue
        found.setdefault(
            name.lower(), (name, "possessive", "possessive_in_title")
        )

    # 2. Kinship-adjacent forms. Tokenize and scan for name-shaped
    #    tokens. When a kinship word appears in the title, we scan the
    #    ENTIRE title (rather than a tight adjacency window) because
    #    family-themed titles like "Family Dinner with Vitus" place the
    #    name several tokens away from the kinship word ("Family ...
    #    Vitus" with two tokens between). For non-family-themed titles
    #    we don't extract anything (the kinship-word presence is the
    #    family signal). _NAME_STOPWORDS keeps everyday calendar nouns
    #    ("dinner", "lunch", "school") out.
    tokens = _tokenize_title(title)
    kinship_idxs = _find_kinship_word_indices(tokens)
    if kinship_idxs:
        # Pick the first kinship-word index as the canonical "anchor"
        # for tagging; if multiple kinship words appear we record only
        # the first to avoid noisy multi-tag output.
        anchor_kw = kinship_idxs[0][1]
        for j, tok in enumerate(tokens):
            if any(j == idx for idx, _ in kinship_idxs):
                continue  # don't re-treat the kinship word itself as a name
            bare = _strip_possessive(tok)
            if not _looks_like_name(bare):
                continue
            if bare.lower() in _RELATION_ONLY_WORDS:
                continue
            found.setdefault(
                bare.lower(), (bare, anchor_kw, "kinship_adjacent")
            )

    return list(found.values())


def _title_has_kinship_word(title: str) -> bool:
    """True if the title contains ANY kinship word (including relation-
    only words like 'family'). Used to classify whole events as
    family-themed for the attendee detector."""
    tokens = _tokenize_title(title)
    return bool(_find_kinship_word_indices(tokens))


# ---------------- Scoring ----------------


def score_family_signal(person: CalendarPerson) -> tuple[float, str]:
    """Compute (strength, source) for a CalendarPerson.

    Strength ∈ [0.0, 1.0]; source is a "+"-joined tag listing every
    detector that fired. Multiple detectors compound the source tag but
    the score is the MAX of the per-detector values — additivity would
    push two weak signals above the threshold of one strong signal,
    which isn't the semantics we want (mirrors `contacts.score_family_signal`).

    Priority table:
      3+ events with kinship-word titles:    0.95  "recurring_family_event"
      Possessive-form name in title:         0.90  "possessive_in_title"
      Single kinship-word event title:       0.80  "birthday_event"
      Frequent attendee, personal domain:    0.70  "frequent_attendee_personal"
      Attendee at family-themed event:       0.65  "family_event_attendee"
      Otherwise:                             0.40  "weak"
    """
    sources: list[str] = []
    score = 0.0

    titles_with_kinship = sum(
        1 for t in person.appears_in_titles if _title_has_kinship_word(t)
    )

    # 1. Recurring family event (same person across 3+ family-themed
    #    titles). "Vitus Birthday" + "Vitus school pickup" + "Vitus
    #    dentist" — assuming all three have kinship words on them.
    if titles_with_kinship >= 3:
        sources.append("recurring_family_event")
        score = max(score, 0.95)

    # 2. Possessive form ("possessive_in_title" in matched kinship
    #    words). The token "possessive" is the sentinel value stored by
    #    the bare-possessive regex; "possessive_in_title" is the
    #    detector tag. We accept either.
    if "possessive" in person.matched_kinship_words or any(
        kw.endswith("'s") for kw in person.matched_kinship_words
    ):
        sources.append("possessive_in_title")
        score = max(score, 0.90)

    # 3. Single kinship-word event. Birthday is the canonical case but
    #    any kinship word fires this rule.
    if titles_with_kinship >= 1 and "recurring_family_event" not in sources:
        # Tag the source by the specific kinship word when we have one
        # to render — "birthday_event" reads better than the generic
        # "kinship_event" in logs/profiles.
        if "birthday" in person.matched_kinship_words:
            sources.append("birthday_event")
        elif "anniversary" in person.matched_kinship_words:
            sources.append("anniversary_event")
        elif "wedding" in person.matched_kinship_words:
            sources.append("wedding_event")
        else:
            sources.append("kinship_event")
        score = max(score, 0.80)

    # 4. Frequent attendee at personal domain. >=5 events is the
    #    threshold; we trust the input `attendee_event_count` and check
    #    the email's domain.
    if person.is_attendee and person.email and person.attendee_event_count >= 5:
        domain = person.email.split("@", 1)[-1].lower() if "@" in person.email else ""
        if domain in _PERSONAL_EMAIL_DOMAINS:
            sources.append("frequent_attendee_personal")
            score = max(score, 0.70)

    # 5. Attendee at family-themed event. Weak but useful tiebreaker;
    #    only fires when at least one of their appearances was on a
    #    title that itself was family-themed.
    if person.is_attendee and titles_with_kinship >= 1 and score < 0.65:
        sources.append("family_event_attendee")
        score = max(score, 0.65)

    if score == 0.0:
        sources.append("weak")
        score = 0.40

    return score, "+".join(sources)


# ---------------- The client ----------------


class GoogleCalendar:
    """Wrapper around the Google Calendar API for family-signal extraction.

    One instance keeps the discovery service alive across multiple calls.
    Methods are synchronous and blocking — single-user scale doesn't need
    async or pagination parallelism.
    """

    def __init__(self, credentials: Credentials):
        self._creds = credentials
        self._service = build(
            "calendar", "v3", credentials=credentials, cache_discovery=False
        )

    # ---- events fetch ----

    def list_recent_events(
        self,
        days_back: int = _DEFAULT_DAYS_BACK,
        max_results: int = _DEFAULT_MAX_EVENTS,
    ) -> list[dict]:
        """Fetch events from primary calendar in the last N days.

        `singleEvents=true` expands recurring events into individual
        occurrences — without this we'd see the master event only and
        miss the per-occurrence titles+attendees that a real family
        calendar accumulates ("Vitus Birthday 2024", "Vitus Birthday
        2025").

        `orderBy="startTime"` is required when singleEvents=true. The
        order itself doesn't matter to us — we de-dup by attendee
        email + collapse titles — but the API rejects calls without
        it.

        Paginates via `nextPageToken`. Stops at `max_results` events
        regardless of remaining pages (defensive cap; a calendar with
        hundreds of small daily events could otherwise spin).
        """
        # Compute timeMin in RFC3339. We build it without external deps —
        # `datetime.now(timezone.utc).isoformat()` produces what the
        # Calendar API expects.
        from datetime import datetime, timedelta, timezone

        time_min = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).isoformat()

        events: list[dict] = []
        page_token: str | None = None
        while True:
            req = self._service.events().list(
                calendarId="primary",
                timeMin=time_min,
                singleEvents=True,
                orderBy="startTime",
                maxResults=_EVENTS_PAGE_SIZE,
                pageToken=page_token,
            )
            resp = req.execute()
            for ev in resp.get("items", []) or []:
                events.append(ev)
                if len(events) >= max_results:
                    log.info(
                        "calendar: hit max_results cap (%d events); "
                        "truncating",
                        max_results,
                    )
                    return events
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return events

    # ---- family-signal extraction ----

    def extract_family_signals(
        self, events: list[dict]
    ) -> list[CalendarPerson]:
        """Parse `events` for family-shaped signals.

        Steps per event:
          1. Title parsing — kinship-word adjacency + possessive
             forms produce title-only CalendarPerson records.
          2. Attendee extraction — each `attendees[].email` becomes a
             CalendarPerson; if the event's title is family-themed all
             attendees are upweighted.
          3. Multi-event correlation — duplicate names/emails across
             events collapse into a single record with merged title
             list + bumped attendee counts.

        Returns the deduplicated, scored list sorted by descending
        family_signal_strength.
        """
        # Two indices: one for title-name candidates (keyed by lowercased
        # name) and one for attendees (keyed by lowercased email).
        by_name: dict[str, CalendarPerson] = {}
        by_email: dict[str, CalendarPerson] = {}

        for ev in events or []:
            title = str(ev.get("summary") or "").strip()
            # The description may also mention people in narrative form;
            # we don't currently parse it but include the field in the
            # title pool for keyword presence (cheap upweight).
            if not title and not ev.get("attendees"):
                continue

            title_has_kinship = _title_has_kinship_word(title)

            # ---- Title-name candidates ----
            for name, kw, detector in _extract_names_from_title(title):
                key = name.lower()
                person = by_name.get(key)
                if person is None:
                    person = CalendarPerson(name=name)
                    by_name[key] = person
                if title and title not in person.appears_in_titles:
                    person.appears_in_titles.append(title)
                if kw and kw not in person.matched_kinship_words:
                    person.matched_kinship_words.append(kw)
                # When the possessive-form detector fired, also stamp
                # the "possessive" sentinel so the scorer can grant
                # `possessive_in_title` even when the next word was a
                # real kinship word like "birthday".
                if detector == "possessive_in_title":
                    if "possessive" not in person.matched_kinship_words:
                        person.matched_kinship_words.append("possessive")

            # ---- Attendees ----
            for attendee in ev.get("attendees", []) or []:
                email = str(attendee.get("email") or "").strip().lower()
                if not email:
                    continue
                # Skip the user themselves — `self: true` flag, or
                # `responseStatus` of the owner.
                if attendee.get("self"):
                    continue
                # Skip resource calendars (rooms / equipment).
                if attendee.get("resource"):
                    continue
                # Skip group / mailing-list calendars.
                if attendee.get("organizer") and not attendee.get(
                    "displayName"
                ) and email.endswith(".calendar.google.com"):
                    continue

                display = (
                    str(attendee.get("displayName") or "").strip()
                    or email.split("@", 1)[0]
                )
                person = by_email.get(email)
                if person is None:
                    person = CalendarPerson(
                        name=display,
                        email=email,
                        is_attendee=True,
                    )
                    by_email[email] = person
                # Prefer the display name once we have one.
                if attendee.get("displayName") and (
                    not person.name or "@" in person.name
                ):
                    person.name = str(attendee["displayName"]).strip()
                if title and title not in person.appears_in_titles:
                    person.appears_in_titles.append(title)
                person.attendee_event_count += 1
                if title_has_kinship:
                    # Tag with a generic "attendee_family_event" kinship
                    # marker so the scorer can use it.
                    if "attendee_family_event" not in person.matched_kinship_words:
                        person.matched_kinship_words.append(
                            "attendee_family_event"
                        )

        # Merge: if a title-name person and an attendee-by-email share a
        # name (case-insensitive), fold them together. The attendee
        # version is canonical because it has the email; bring over the
        # title-only fields.
        merged: list[CalendarPerson] = []
        used_name_keys: set[str] = set()
        for email_key, p in by_email.items():
            # Look for a title-only person with the same name.
            name_key = p.name.lower()
            sibling = by_name.get(name_key)
            if sibling is not None:
                for t in sibling.appears_in_titles:
                    if t not in p.appears_in_titles:
                        p.appears_in_titles.append(t)
                for kw in sibling.matched_kinship_words:
                    if kw not in p.matched_kinship_words:
                        p.matched_kinship_words.append(kw)
                used_name_keys.add(name_key)
            merged.append(p)
        for name_key, p in by_name.items():
            if name_key in used_name_keys:
                continue
            merged.append(p)

        # Score every person.
        for p in merged:
            strength, source = score_family_signal(p)
            p.family_signal_strength = strength
            p.family_signal_source = source

        # Sort by descending strength, then attendee_event_count, then
        # name — deterministic.
        merged.sort(
            key=lambda p: (
                -p.family_signal_strength,
                -p.attendee_event_count,
                p.name.lower(),
            )
        )
        return merged

    def list_family_calendar_persons(
        self,
        days_back: int = _DEFAULT_DAYS_BACK,
        max_results: int = _DEFAULT_MAX_EVENTS,
    ) -> list[CalendarPerson]:
        """Convenience: fetch + extract in one call. Returns only persons
        with a non-weak signal (strength >= 0.5) so the materializer
        doesn't get drowned in random attendees from a single meeting."""
        events = self.list_recent_events(
            days_back=days_back, max_results=max_results
        )
        all_persons = self.extract_family_signals(events)
        # Filter to strength >= 0.5: drops the catch-all "weak" tier
        # (0.40) but keeps the family-event-attendee tier (0.65). The
        # judge can decide whether a 0.65 is real family.
        return [p for p in all_persons if p.family_signal_strength >= 0.5]


# ---------------- Bridge to family_judge.ContactEvidence ----------------


def calendar_person_to_contact_evidence(person: CalendarPerson):
    """Convert a CalendarPerson into a `ContactEvidence` so the existing
    judge prompt machinery (designed for Pass 12A contacts) can score
    calendar-derived candidates with the same plumbing.

    Import is local to keep this module importable without the
    family_judge dependency (lets `pi-email calendar` debug-listing run
    without pulling in the entire judging stack).
    """
    from .family_judge import ContactEvidence

    relations: list[str] = []
    # Render matched kinship words as pseudo-relations the judge can read.
    for kw in person.matched_kinship_words:
        if kw == "attendee_family_event":
            continue
        if kw == "possessive":
            continue
        relations.append(f"calendar_{kw}: (event title)")

    # Render a compact "Calendar: 'event1', 'event2', ..." biography.
    titles_snippet = ", ".join(
        f"'{t}'" for t in person.appears_in_titles[:3]
    )
    bio = (
        f"Calendar: {titles_snippet}"
        if titles_snippet
        else "Calendar evidence (no title preview available)"
    )

    return ContactEvidence(
        contact_name=person.name,
        emails=tuple([person.email]) if person.email else tuple(),
        family_signal_strength=person.family_signal_strength,
        family_signal_source=f"calendar:{person.family_signal_source}",
        relations=tuple(relations),
        biography=bio,
        # Calendar evidence never indicates a true Google "Family" group
        # membership — that's a Contacts API-only signal.
        in_family_group=False,
    )


# ---------------- Helpers used by callers ----------------


def credentials_have_calendar_scope(creds: Credentials) -> bool:
    """True iff `creds.scopes` includes the calendar.readonly scope.

    Used by callers (cli auth, materializer) to detect tokens minted
    before Pass 14B. When False, callers should either prompt the user
    to re-authenticate (`pi-email auth --refresh-auth`) or skip the
    Calendar code path entirely.
    """
    if creds is None:
        return False
    scopes = getattr(creds, "scopes", None) or []
    return CALENDAR_READONLY_SCOPE in set(scopes)
