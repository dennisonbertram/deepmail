"""Materializer: writes profiles/<slug>.md with YAML frontmatter + candidate list + footnote provenance.

All extracted PERSON entities are candidates. The calling model (Claude Code,
Cursor, etc.) classifies them based on the original query. No family-specific
rules, no surname auto-accept, no family-graph expansion.

Rules 2/3/5/6 are retained as RANKING signals (metadata on each candidate),
not inclusion gates.
"""

from __future__ import annotations

import datetime
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from .corpus import Corpus, Message, MessageId
from .entities import Entity


# Tokens that, when they appear as the LAST token of a "person name" canonical,
# are almost always role/title or subject-line cruft that bled into the entity
# label. We strip them iteratively (so "Kelly Ebeling Founder/CEO Intro" peels
# back to "Kelly Ebeling"). Stored lowercased; comparison is case-insensitive.
#
# Conservative by design: tokens here can NEVER legitimately end a personal
# name. Adding a token like "Lee" or "Brown" would break real names, so the
# list is roles / titles / mail-subject prefixes / business-function dept
# words only.
#
# Round-5: mirrored against entities._ROLE_TAIL_TOKENS so the extractor and
# the materializer agree on what counts as "definitely a role tail." The
# extractor strips at NER time; this layer is defense-in-depth for any
# canonical that slipped past (older cached entities, materializer-only
# callers, etc).
_NAME_TAIL_STOPLIST: frozenset[str] = frozenset({
    # Roles / titles
    "founder", "ceo", "cto", "cfo", "coo", "president", "director",
    "manager", "engineer", "developer", "designer", "consultant",
    "analyst", "advisor", "investor", "lead", "head", "chief",
    "vp", "evp", "svp", "principal", "partner", "owner",
    "founder/ceo",
    # Subject-line / newsletter cruft
    "intro", "introduction", "re", "fwd", "subject", "update",
    "newsletter", "weekly", "monthly", "daily", "digest",
    # Business-function / department tokens — Round-5 addition.
    "marketing", "engineering", "sales", "operations", "product",
    "design", "research", "strategy", "legal", "finance", "hr",
    "it", "pr", "bd", "growth", "analytics", "partnerships",
    "community",
})


# Extract the display name from a "Display Name <addr@host>" header. Kept
# local to avoid importing from entities.py (parallel worker territory).
_FROM_DISPLAY_RE = re.compile(r"^\s*(.+?)\s*<[^<>]+@[^<>]+>\s*$")
_EMAIL_IN_FROM_RE = re.compile(r"<([^<>]+@[^<>]+)>")


def _normalize_canonical_name(name: str) -> str:
    """Clean a canonical entity name for use as a header/slug/wikilink.

    Three passes:
      1. Adjacent-token dedup: "Mitra Mitra Martin" -> "Mitra Martin".
         Triggered when canonicalize() picks the longest cluster member and
         the longest one was itself a duplicated form ("Mitra" alias merged
         into "Mitra Martin" produced "Mitra Mitra Martin" somewhere upstream).
      2. Trailing role/title strip: "Kelly Ebeling Founder" -> "Kelly Ebeling".
         Iterates so "Foo Bar Founder/CEO" peels both tokens.
      3. Length cap: after the above, names of 4+ tokens are truncated to the
         first 3. Real person names are rarely 4+ tokens; when we see one in
         practice it's because subject text leaked in ("Branson Bollinger
         Wagmi Intro" -> after stripping "Intro" we still have 3 tokens, but
         the cap kicks in on cases like "Branson Bollinger Wagmi Intro Call"
         that would otherwise survive). Trade-off documented: we accept the
         odd legitimate-but-mangled 4-token name in exchange for cutting the
         long tail of subject-line cruft.

    If the result is empty (every token was stripped), return the original
    name unchanged — better to keep noise than lose the entry entirely. Flag
    is the empty-tokens-after-strip branch; callers can rely on a non-empty
    return whenever `name` was non-empty.
    """
    if not name or not name.strip():
        return name

    tokens = name.split()

    # 1. Adjacent-token dedup (case-insensitive).
    deduped: list[str] = []
    for tok in tokens:
        if deduped and deduped[-1].lower() == tok.lower():
            continue
        deduped.append(tok)
    tokens = deduped

    # 2. Trailing role/title strip — iterate so multi-role suffixes peel.
    while tokens:
        tail = tokens[-1].lower().strip(".,;:!?")
        if tail in _NAME_TAIL_STOPLIST:
            tokens.pop()
            continue
        break

    # 3. Length cap. Real names are rarely 4+ tokens; if they are, the last
    #    1-2 tokens are usually noise. We accept the false-positive on rare
    #    legitimate 4+ token names (e.g. "Maria Carmen Sofia de la Cruz")
    #    in exchange for catching the long tail of subject-text leaks.
    if len(tokens) >= 4:
        tokens = tokens[:3]

    result = " ".join(tokens).strip()
    if not result:
        # Pathological case — everything stripped. Better to keep the noisy
        # name than silently nuke the entry. Flagged here.
        return name
    return result


def _tokenize_normalized(name: str) -> list[str]:
    """Lowercase whitespace tokens of the normalized form — used by self-filter
    for first+last fuzzy match."""
    return _normalize_canonical_name(name).lower().split()


def _derive_self_aliases(
    corpus: Corpus, user_self: dict | None
) -> set[tuple[str, ...]]:
    """Collect a set of normalized-token-tuples that should be considered "the
    user themselves" for member-list filtering.

    Sources:
      * `user_self["display_name"]` if provided.
      * The local-part of `user_self["email"]` as a single-token alias —
        e.g. dennison@withtally.com -> ("dennison",). This is the fallback
        path that always works even when the corpus has zero sent-mail
        coverage; the materializer's _matches_self treats a 1-token alias
        as a first-name match against any canonical, which captures the
        common case of "the user appears in their own family list as
        <First> <Last>".
      * Sender display names parsed from any message in the corpus whose
        from_addr contains `user_self["email"]`. This recovers the user's
        FULL name from their own sent mail when Gmail's getProfile path
        leaves display_name as None (see cli._run_loop_gmail).

    Each alias is stored as a token-tuple so the caller can do both
    full-match and first+last token comparisons.
    """
    aliases: set[tuple[str, ...]] = set()
    if user_self is None:
        return aliases

    display = user_self.get("display_name")
    if display:
        toks = _tokenize_normalized(display)
        if toks:
            aliases.add(tuple(toks))

    email = (user_self.get("email") or "").lower()
    if email:
        # Local-part alias — always fires, doesn't require the user's sent
        # mail to be in the corpus. "dennison@withtally.com" -> ("dennison",).
        # The `+suffix` part of an addressing-suffixed email is stripped so
        # `dennison+noreply@...` still yields ("dennison",).
        local_part = email.split("@", 1)[0]
        local_part = local_part.split("+", 1)[0]
        if local_part:
            aliases.add((local_part,))
        for msg in corpus.messages.values():
            from_addr = msg.from_addr or ""
            # Cheap substring match first so we don't regex every header.
            if email not in from_addr.lower():
                continue
            disp = _parse_sender_display(from_addr)
            if disp:
                toks = _tokenize_normalized(disp)
                if toks:
                    aliases.add(tuple(toks))
    return aliases




def _parse_sender_display(from_addr: str) -> str:
    """Pull the display name out of a `Display Name <addr@host>` header.

    Returns "" if the header is a bare email or unparseable. Local copy —
    we don't import from entities.py (parallel-worker territory)."""
    if not from_addr:
        return ""
    m = _FROM_DISPLAY_RE.match(from_addr)
    if not m:
        return ""
    raw = m.group(1).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        raw = raw[1:-1].strip()
    return raw


def _matches_self(
    canonical: str, self_aliases: set[tuple[str, ...]]
) -> tuple[bool, str]:
    """Decide whether `canonical` refers to the user themselves.

    Returns (matched, rule) — `rule` is a short label naming WHICH rule
    fired so the on_log callback can surface it in the self-filter line.
    When `matched` is False, `rule` is "".

    Match rules (first match wins; tried in order):
      * "exact"          — token-tuple equality after normalization.
      * "first+last"     — both sides 2+ tokens and (first, last) match.
        Catches the common case where the alias is "Dennison Bertram" and
        the candidate is "Dennison Foo Bertram" or similar.
      * "first-name"     — one side is a single token matching the other
        side's first token. Catches the email-local-part alias case
        (dennison -> "Dennison <Anything>") and the inverse (the user's
        name comes through canonicalize as a bare first-name).
    """
    cand = tuple(_tokenize_normalized(canonical))
    if not cand:
        return False, ""
    for alias in self_aliases:
        if not alias:
            continue
        if cand == alias:
            return True, "exact"
        if len(cand) >= 2 and len(alias) >= 2:
            if cand[0] == alias[0] and cand[-1] == alias[-1]:
                return True, "first+last"
        if len(cand) == 1 and cand[0] == alias[0]:
            return True, "first-name"
        if len(alias) == 1 and alias[0] == cand[0]:
            return True, "first-name"
    return False, ""


@dataclass
class ProfileMember:
    """One candidate in the materialized profile."""

    name: str
    excerpts: list[tuple[MessageId, str]] = field(default_factory=list)
    # Ranking signals (metadata, not gates).
    bidirectional: bool = False
    personal_domain: bool = False
    calendar_signal: bool = False


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "unknown"


def _extract_sentence_around(text: str, term: str, max_len: int = 220) -> str:
    """Pull the sentence containing `term`. Falls back to a window."""
    if not term:
        return ""
    pattern = re.compile(rf"[^.!?\n]*\b{re.escape(term)}\b[^.!?\n]*[.!?]?", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        idx = text.lower().find(term.lower())
        if idx < 0:
            return ""
        start = max(0, idx - 80)
        end = min(len(text), idx + len(term) + 80)
        return text[start:end].strip()
    snippet = m.group(0).strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rsplit(" ", 1)[0] + "..."
    return snippet


# Pass 10 — hard cap on the number of PERSON candidates handed to the
# LLM judge. With loose extraction (the live-Gmail default after Pass 10)
# the pre-cap pool can balloon into the hundreds; 50 keeps the judge cost
# bounded (~$0.15 per run at Claude Sonnet rates with prompt caching) while
# leaving plenty of headroom for the real-Gmail "32 SEED candidates" peak.
_GATHER_CANDIDATE_CAP = 50

# Regex used to pull email addresses out of "Display Name <addr@host>" or
# bare-address headers. We only need the domain for personal-domain checks
# (rule 3), so we capture the host directly.
_EMAIL_DOMAIN_RE = re.compile(r"[A-Za-z0-9._+\-]+@([A-Za-z0-9.\-]+)")


def _msg_has_personal_domain_address(msg: Message) -> bool:
    """True if msg's from_addr or to_addr contains an email at a personal
    domain (gmail.com, icloud.com, hotmail.com, ...). Domain list is
    imported lazily to avoid a top-level cycle with frontier.py."""
    from .frontier import PERSONAL_EMAIL_DOMAINS

    for header in (msg.from_addr or "", msg.to_addr or ""):
        for m in _EMAIL_DOMAIN_RE.finditer(header):
            domain = m.group(1).lower().strip(".")
            if domain in PERSONAL_EMAIL_DOMAINS:
                return True
    return False


# ---------------------------------------------------------------------------
# Pass 17B: Rule 5 — address-book frequency analysis
#
# Real family often appears in a user's inbox WITHOUT explicit kinship words —
# they just exchange emails about everyday things. Rule 5 widens the
# candidate pool to include personal-mail-domain addresses (gmail.com,
# icloud.com, ...) that the user exchanges messages with BIDIRECTIONALLY (at
# least one sent + one received) and with substantial volume (>=3 total
# messages). Family-domain over-inclusion is OK; the judge filters downstream.
# ---------------------------------------------------------------------------


# Regex used to extract a bare email from a 'Display Name <addr@host>' or
# bare-email header. Reused by Rule 5 helpers. Distinct from
# `_EMAIL_DOMAIN_RE` (which captures only the host) because the Rule 5
# accounting keys by the full address.
_EMAIL_FULL_RE = re.compile(r"([A-Za-z0-9._+\-]+@[A-Za-z0-9.\-]+)")


def _extract_email_from_header(header: str) -> str | None:
    """Pull the bare email address out of a 'Display <email>' or bare email
    header. Returns lowercased or None if no email is present."""
    if not header:
        return None
    m = _EMAIL_FULL_RE.search(header)
    return m.group(1).lower() if m else None


def _is_personal_domain(addr: str) -> bool:
    """True if `addr` (a full header or bare email) contains an email at a
    domain in `PERSONAL_EMAIL_DOMAINS`. Lazily imports the domain set to
    avoid a top-level cycle with frontier.py."""
    from .frontier import PERSONAL_EMAIL_DOMAINS

    email = _extract_email_from_header(addr)
    if email is None:
        return False
    if "@" not in email:
        return False
    domain = email.split("@", 1)[1].lower().strip(".")
    return domain in PERSONAL_EMAIL_DOMAINS


def _derive_canonical_name_for_email(corpus: Corpus, email_addr: str) -> str:
    """Find a display name for `email_addr` by walking corpus messages where
    the From header contains this email. Returns the first display name
    found (parsed out of `Display Name <email>`), or falls back to
    title-casing the email's local part.

    Local-part fallback splits on common separators (. _ - +) so
    `jana.smith@gmail.com` → "Jana Smith" and `jana@gmail.com` → "Jana".
    """
    target = email_addr.lower()
    for msg in corpus.messages.values():
        from_lc = (msg.from_addr or "").lower()
        if target in from_lc:
            display = _parse_sender_display(msg.from_addr or "")
            if display:
                return display
    # Fall back to title-cased local-part.
    local = target.split("@", 1)[0] if "@" in target else target
    parts = [p for p in re.split(r"[._\-+]", local) if p]
    if not parts:
        return target
    return " ".join(p.capitalize() for p in parts)


def _build_excerpts_for_email(
    corpus: Corpus,
    email_addr: str,
    max_excerpts: int = 8,
) -> tuple[list[dict], list[MessageId]]:
    """Build excerpt dicts AND candidate message-ids for messages where
    `email_addr` appears as sender or recipient.

    Returns (excerpt_dicts, message_ids). Most recent first (by date string,
    descending; ISO-ish dates sort correctly). Capped at `max_excerpts`.
    Each excerpt dict has keys msg_id / from_addr / subject / snippet
    (first 300 chars of body)."""
    target = email_addr.lower()
    matched: list[tuple[MessageId, Message]] = []
    for mid, msg in corpus.messages.items():
        from_lc = (msg.from_addr or "").lower()
        to_lc = (msg.to_addr or "").lower()
        if target in from_lc or target in to_lc:
            matched.append((mid, msg))
    matched.sort(key=lambda x: x[1].date or "", reverse=True)
    matched = matched[:max_excerpts]
    excerpts: list[dict] = []
    mids: list[MessageId] = []
    for mid, msg in matched:
        body = msg.body_clean or msg.body or ""
        excerpts.append(
            {
                "msg_id": mid,
                "from_addr": msg.from_addr or "",
                "subject": msg.subject or "",
                "snippet": body[:300].strip(),
            }
        )
        mids.append(mid)
    return excerpts, mids


# Minimum bidirectional total below which Rule 5 does not fire. Tuned for
# "substantial correspondence" — Run 16 saw real family members at
# 5-30 messages each over the time window, so 3 is a conservative floor
# that still admits even sparse family contacts.
_RULE_5_MIN_TOTAL_MESSAGES = 3


# Pass 18: Rule 6 minimum family-signal strength for a CalendarEmailPerson
# to be promoted to a candidate. 0.80 maps to the kinship-event /
# possessive-in-title tier in `calendar_email_parser` — strictly higher
# than the personal-attendee floor (0.65). This keeps Rule 6 narrow:
# only persons whose name appeared inside a calendar-themed event title
# get promoted, attendees-without-kinship-context are deliberately out.
_RULE_6_MIN_FAMILY_SIGNAL_STRENGTH = 0.80


def _gather_rule_5_frequent_personal_correspondents(
    corpus: Corpus,
    user_emails: set[str],
) -> list[tuple[str, list[dict], list[MessageId]]]:
    """Find personal-domain email addresses that have substantial
    bidirectional correspondence with the user.

    Returns: list of (canonical_name, excerpt_dicts, message_ids). Each
    survivor satisfies:
      * personal-mail-domain (from `PERSONAL_EMAIL_DOMAINS`)
      * at least 1 message FROM the user TO this address
      * at least 1 message FROM this address TO the user
      * total messages (sent + received) >= _RULE_5_MIN_TOTAL_MESSAGES (3)

    Excerpt dicts are produced by `_build_excerpts_for_email` (most recent
    first, capped at 8). Caller is responsible for synthesizing a
    candidate ProfileMember from these tuples."""
    user_emails_lc = {ue.lower() for ue in user_emails if ue}
    if not user_emails_lc:
        return []

    # email_addr -> {"sent": N, "received": M}
    direction_counts: dict[str, dict[str, int]] = {}

    for msg in corpus.messages.values():
        from_full = msg.from_addr or ""
        to_full = msg.to_addr or ""
        from_lc = from_full.lower()
        to_lc = to_full.lower()

        from_email = _extract_email_from_header(from_full)
        to_email = _extract_email_from_header(to_full)

        from_is_user = any(ue in from_lc for ue in user_emails_lc)
        to_is_user = any(ue in to_lc for ue in user_emails_lc)

        from_personal = _is_personal_domain(from_full)
        to_personal = _is_personal_domain(to_full)

        # User sent TO a personal-domain recipient (other than self).
        if (
            from_is_user
            and to_personal
            and to_email is not None
            and not to_is_user
        ):
            direction_counts.setdefault(
                to_email, {"sent": 0, "received": 0}
            )["sent"] += 1
        # User received FROM a personal-domain sender (other than self).
        if (
            to_is_user
            and from_personal
            and from_email is not None
            and not from_is_user
        ):
            direction_counts.setdefault(
                from_email, {"sent": 0, "received": 0}
            )["received"] += 1

    candidates: list[tuple[str, list[dict], list[MessageId]]] = []
    # Sort by email for deterministic output.
    for email_addr in sorted(direction_counts):
        counts = direction_counts[email_addr]
        if counts["sent"] < 1 or counts["received"] < 1:
            continue
        total = counts["sent"] + counts["received"]
        if total < _RULE_5_MIN_TOTAL_MESSAGES:
            continue
        canonical_name = _derive_canonical_name_for_email(corpus, email_addr)
        excerpts, mids = _build_excerpts_for_email(
            corpus, email_addr, max_excerpts=8
        )
        candidates.append((canonical_name, excerpts, mids))
    return candidates


# ---------------------------------------------------------------------------
# Pass 18: Rule 6 — promote persons extracted from calendar-notification
# emails (Pass 17A) into the candidate pool.
#
# Pass 17A added `_inject_calendar_person_entities` in loop.py: when Gmail
# delivers a Google Calendar notification (e.g. "Accepted: Vitus Birthday
# in school"), the parser produces `CalendarEmailPerson` records on the
# message AND emits synthetic Entity rows. But the existing Rule 1-5
# admission logic in `_gather_family_members` only fires on relation
# co-occurrence, body-name matches, personal-domain headers, or surname
# matches — calendar-derived persons (e.g. a child's first name in an
# event title) often satisfy NONE of those. Rule 6 closes that gap: any
# PERSON whose name matches a CalendarEmailPerson with high family-signal
# strength (>= 0.80, i.e. kinship-event or possessive-in-title tier) is
# admitted as a candidate.
# ---------------------------------------------------------------------------


def _gather_rule_6_calendar_signal(
    corpus: Corpus,
    entities: list[Entity],
) -> list[tuple[str, list[dict], list[MessageId]]]:
    """Find PERSON entities matching high-signal calendar-notification persons.

    For each PERSON entity in `entities`:
      1. Walk corpus messages with non-empty `calendar_persons`.
      2. If any of those messages carries a `CalendarEmailPerson` whose
         normalized name matches the entity AND whose
         `family_signal_strength >= _RULE_6_MIN_FAMILY_SIGNAL_STRENGTH`,
         promote the entity.
      3. Build excerpts from the calendar-notification messages — subject
         line plus first 200 chars of body, capped at 5.

    Returns: list of (canonical_name, excerpt_dicts, message_ids). The
    canonical_name is the entity's label (already normalized at extraction
    time). Excerpt dicts mirror Rule 5's shape so the build-phase
    precomputed-snippet path can consume them transparently.

    Name matching is case-insensitive on first-whitespace-token equality —
    a `CalendarEmailPerson` named "Vitus" matches an entity labelled
    "Vitus" or "Vitus Bertram". This is intentional: calendar event
    titles routinely surface only the first name ("Vitus Birthday"),
    while the body extractor may have surfaced the full name from
    elsewhere in the corpus, and we want them to merge.
    """
    # Pre-walk: which messages are calendar-notification messages?
    # Build a name -> list[(msg_id, CalendarEmailPerson)] index keyed by
    # lowercased first-token of the person's name.
    cal_index: dict[str, list[tuple[MessageId, object]]] = defaultdict(list)
    for mid, msg in corpus.messages.items():
        persons = getattr(msg, "calendar_persons", None) or []
        if not persons:
            continue
        for p in persons:
            name = getattr(p, "name", None) or ""
            if not name:
                continue
            first_token = name.strip().split()[0].lower() if name.strip() else ""
            if first_token:
                cal_index[first_token].append((mid, p))

    if not cal_index:
        return []

    out: list[tuple[str, list[dict], list[MessageId]]] = []
    seen_canonical: set[str] = set()

    for e in entities:
        if e.kind != "person":
            continue
        label = e.label or ""
        tokens = label.strip().split()
        if not tokens:
            continue
        first_token = tokens[0].lower()
        candidates = cal_index.get(first_token, [])
        if not candidates:
            continue

        # Find the matching CalendarEmailPerson with the strongest signal.
        # We require strength >= threshold — anything below is "not
        # a strong-enough calendar signal" and Rule 6 abstains.
        qualifying: list[tuple[MessageId, object]] = []
        for mid, p in candidates:
            strength = float(getattr(p, "family_signal_strength", 0.0) or 0.0)
            if strength >= _RULE_6_MIN_FAMILY_SIGNAL_STRENGTH:
                qualifying.append((mid, p))
        if not qualifying:
            continue

        # Deduplicate by canonical entity label.
        if label in seen_canonical:
            continue
        seen_canonical.add(label)

        # Build excerpts from the calendar-notification messages.
        # Cap at 5; subject + first 200 chars of body per excerpt.
        excerpt_dicts: list[dict] = []
        mids: list[MessageId] = []
        seen_mids: set[MessageId] = set()
        for mid, _p in qualifying[:5]:
            if mid in seen_mids:
                continue
            seen_mids.add(mid)
            msg = corpus.get(mid)
            if msg is None:
                continue
            body = (msg.body_clean or msg.body or "")[:200].strip()
            subject = msg.subject or ""
            snippet_parts = [s for s in (subject, body) if s]
            snippet = " — ".join(snippet_parts)
            excerpt_dicts.append(
                {
                    "msg_id": mid,
                    "from_addr": msg.from_addr or "",
                    "subject": subject,
                    "snippet": snippet,
                }
            )
            mids.append(mid)

        if excerpt_dicts:
            out.append((label, excerpt_dicts, mids))

    return out


def _gather_candidates(
    corpus: Corpus,
    entities: set[Entity],
    seen_by_message: dict[MessageId, set[Entity]],
    canonical_map: dict[str, str] | None = None,
    user_emails: set[str] | None = None,
    on_log: Callable[[str], None] | None = None,
    max_candidates: int = _GATHER_CANDIDATE_CAP,
) -> dict[str, ProfileMember]:
    """Gather ALL extracted PERSON entities as candidates.

    Every PERSON entity passing stoplists is a candidate. Rules 2/3/5/6 are
    retained as ranking signals (metadata on each candidate), not inclusion
    gates. The calling model classifies based on the original query.

    Returns: dict of canonical-name -> ProfileMember (with up to 8
    excerpts each). The optional `canonical_map` folds e.g. "Jane" into
    "Jane Smith" so they share one section. `on_log` receives a diagnostic
    line summarizing signal counts.

    Cap: at most `max_candidates` (default 50).
    """
    canonical_map = canonical_map or {}
    log = on_log or (lambda s: None)

    def _canon(label: str) -> str:
        return canonical_map.get(label, label)

    # Inverse index: which messages does each PERSON entity appear in?
    msgs_by_person: dict[Entity, set[MessageId]] = defaultdict(set)
    for mid, ents in seen_by_message.items():
        for e in ents:
            if e.kind == "person":
                msgs_by_person[e].add(mid)

    person_entities = sorted(
        {e for e in entities if e.kind == "person"},
        key=lambda e: e.label,
    )

    # Signal counters for diagnostic log.
    signal_counts = {
        "total": 0, "sender_recipient": 0, "personal_domain": 0,
        "bidirectional": 0, "calendar": 0,
    }

    # Each entry: (signal_score, entity, candidate_msg_ids, precomputed_excerpts)
    cands: list[
        tuple[int, Entity, set[str], list[MessageId], list[dict] | None]
    ] = []

    for e in person_entities:
        signals: set[str] = set()
        person_msgs = msgs_by_person.get(e, set())

        # Signal: sender/recipient + body match
        label_lc = e.label.lower()
        tokens = e.label.split()
        first_token_lc = tokens[0].lower() if tokens else ""
        addr_hit = False
        body_msgs: list[MessageId] = []
        for mid, msg in corpus.messages.items():
            from_lc = (msg.from_addr or "").lower()
            to_lc = (msg.to_addr or "").lower()
            if first_token_lc and (
                first_token_lc in from_lc or first_token_lc in to_lc
            ):
                if (
                    " " not in label_lc
                    or label_lc in from_lc
                    or label_lc in to_lc
                ):
                    addr_hit = True
            body = (msg.body_clean or msg.body or "").lower()
            if label_lc and label_lc in body:
                body_msgs.append(mid)
        if addr_hit and body_msgs:
            signals.add("sender_recipient")

        # Signal: personal-domain repeat
        personal_msgs: list[MessageId] = []
        for mid in person_msgs:
            msg = corpus.get(mid)
            if msg is None:
                continue
            if _msg_has_personal_domain_address(msg):
                personal_msgs.append(mid)
        if len(personal_msgs) >= 2:
            signals.add("personal_domain")

        for s in signals:
            signal_counts[s] += 1
        signal_counts["total"] += 1

        # Candidate message-IDs for excerpt synthesis.
        candidate_mids: list[MessageId] = []
        seen_mids: set[MessageId] = set()
        for mid in sorted(person_msgs):
            if mid not in seen_mids:
                candidate_mids.append(mid)
                seen_mids.add(mid)
        for mid in sorted(body_msgs):
            if mid not in seen_mids:
                candidate_mids.append(mid)
                seen_mids.add(mid)
        candidate_mids = candidate_mids[:8]

        # Score for ranking: more signals = higher priority.
        score = len(signals)
        cands.append((score, e, signals, candidate_mids, None))

    # --- Rule 5: bidirectional personal-domain correspondents
    rule_5_cands: list[tuple[str, list[dict], list[MessageId]]] = []
    if user_emails:
        rule_5_cands = _gather_rule_5_frequent_personal_correspondents(
            corpus, user_emails
        )

    canon_to_idx: dict[str, int] = {}
    for i, (_, e, _, _, _) in enumerate(cands):
        canon_to_idx[_canon(e.label)] = i

    for canonical_name, excerpts_dicts, mids in rule_5_cands:
        canon = _canon(canonical_name)
        if canon in canon_to_idx:
            idx = canon_to_idx[canon]
            score_i, e_i, signals_i, mids_i, ex_i = cands[idx]
            signals_i.add("bidirectional")
            cands[idx] = (score_i + 1, e_i, signals_i, mids_i, ex_i)
        else:
            synth_e = Entity(
                kind="person",
                key=canonical_name.lower(),
                label=canonical_name,
            )
            cands.append((1, synth_e, {"bidirectional"}, mids[:8], excerpts_dicts))
            canon_to_idx[canon] = len(cands) - 1
        signal_counts["bidirectional"] += 1

    # --- Rule 6: calendar-notification persons
    rule_6_cands = _gather_rule_6_calendar_signal(corpus, person_entities)
    for canonical_name, excerpts_dicts, mids in rule_6_cands:
        canon = _canon(canonical_name)
        if canon in canon_to_idx:
            idx = canon_to_idx[canon]
            score_i, e_i, signals_i, mids_i, ex_i = cands[idx]
            signals_i.add("calendar")
            merged_mids = list(mids_i)
            for m in mids:
                if m not in merged_mids:
                    merged_mids.append(m)
            merged_mids = merged_mids[:8]
            new_ex = ex_i if ex_i else excerpts_dicts
            cands[idx] = (score_i + 1, e_i, signals_i, merged_mids, new_ex)
        else:
            synth_e = Entity(
                kind="person",
                key=canonical_name.lower(),
                label=canonical_name,
            )
            cands.append((1, synth_e, {"calendar"}, mids[:8], excerpts_dicts))
            canon_to_idx[canon] = len(cands) - 1
        signal_counts["calendar"] += 1

    # Sort by descending score, then alphabetical label for determinism.
    cands.sort(key=lambda x: (-x[0], x[1].label))
    capped = cands[:max_candidates]

    log(
        f"[gather] {len(person_entities)} person entities -> "
        f"{len(capped)} candidates (capped at {max_candidates}); "
        f"sender_recipient={signal_counts['sender_recipient']}, "
        f"personal_domain={signal_counts['personal_domain']}, "
        f"bidirectional={signal_counts['bidirectional']}, "
        f"calendar={signal_counts['calendar']}"
    )

    # --- Build ProfileMembers
    members: dict[str, ProfileMember] = {}
    seen_member_msgs: set[tuple[str, MessageId]] = set()
    for _score, e, signals_set, candidate_mids, precomputed_excerpts in capped:
        canon = _canon(e.label)
        pm = members.setdefault(canon, ProfileMember(name=canon))
        # Populate ranking signals.
        if "bidirectional" in signals_set:
            pm.bidirectional = True
        if "personal_domain" in signals_set:
            pm.personal_domain = True
        if "calendar" in signals_set:
            pm.calendar_signal = True
        tokens = e.label.split()
        first_token = tokens[0] if tokens else e.label
        precomputed_by_mid: dict[MessageId, str] = {}
        if precomputed_excerpts:
            for ex in precomputed_excerpts:
                mid_val = ex.get("msg_id")
                snip_val = ex.get("snippet") or ""
                if mid_val and snip_val:
                    precomputed_by_mid[mid_val] = snip_val
        for mid in candidate_mids:
            if (canon, mid) in seen_member_msgs:
                continue
            seen_member_msgs.add((canon, mid))
            msg = corpus.get(mid)
            if msg is None:
                continue
            text = msg.body_clean or msg.body or ""
            snippet = _extract_sentence_around(text, first_token)
            if not snippet and e.label != first_token:
                snippet = _extract_sentence_around(text, e.label)
            if not snippet and (
                "bidirectional" in signals_set
                or "calendar" in signals_set
            ):
                snippet = precomputed_by_mid.get(mid, "")
                if not snippet:
                    snippet = text[:300].strip()
            if snippet:
                pm.excerpts.append((mid, snippet))

    # Filter: keep only members that have at least one excerpt.
    members = {k: v for k, v in members.items() if v.excerpts}
    return members


# ---------------------------------------------------------------------------
# Simplified profile output
# ---------------------------------------------------------------------------


def write_family_profile(
    out_path: Path,
    corpus: Corpus,
    entities: set[Entity],
    seen_by_message: dict[MessageId, set[Entity]],
    seed: str,
    stop_reason: str,
    queries_run: list[str],
    canonical_map: dict[str, str] | None = None,
    user_self: dict | None = None,
    on_log: Callable[[str], None] | None = None,
    skip_judge: bool = False,
    **kwargs,
) -> Path:
    """Write profiles/<slug>.md and return the absolute path.

    All extracted PERSON entities are candidates. The calling model
    classifies them based on the original query.

    `canonical_map` (optional) folds e.g. "Jane" into "Jane Smith".
    `user_self` (optional) excludes the user from their own results.
    `skip_judge` is accepted for backward compat but ignored (no judge).

    Extra kwargs are accepted and ignored for backward compatibility with
    callers that still pass `judge`, `family_contacts`, or `query_mode`.
    """
    user_emails_for_gather: set[str] = set()
    if user_self:
        primary = (user_self.get("email") or "").strip().lower()
        if primary:
            user_emails_for_gather.add(primary)
            if "+" in primary.split("@", 1)[0]:
                local, domain = primary.split("@", 1)
                clean_local = local.split("+", 1)[0]
                if clean_local:
                    user_emails_for_gather.add(f"{clean_local}@{domain}")

    members = _gather_candidates(
        corpus,
        entities,
        seen_by_message,
        canonical_map=canonical_map,
        user_emails=user_emails_for_gather or None,
        on_log=on_log,
    )

    # ---- Self-filter: drop "you" from your own results.
    self_aliases = _derive_self_aliases(corpus, user_self)
    if self_aliases:
        if on_log is not None:
            email = (user_self or {}).get("email") or "?"
            alias_strs = sorted(" ".join(a) for a in self_aliases)
            on_log(
                f"[self-filter] active for email={email}, "
                f"name_aliases={alias_strs}"
            )
        filtered: dict[str, ProfileMember] = {}
        for canon, pm in members.items():
            matched, rule = _matches_self(canon, self_aliases)
            if matched:
                if on_log is not None:
                    pretty = _normalize_canonical_name(canon)
                    email = (user_self or {}).get("email") or "?"
                    on_log(
                        f"[self-filter] excluding canonical={pretty} "
                        f"(matched={rule})"
                    )
                    on_log(f"excluded self: {pretty} ({email})")
                continue
            filtered[canon] = pm
        members = filtered

    # ---- Frontmatter
    slug = _slugify_query(seed)
    seen_slugs: set[str] = set()
    member_wikilinks: list[str] = []
    for m in members.values():
        s = _slugify(_normalize_canonical_name(m.name))
        if s in seen_slugs:
            continue
        seen_slugs.add(s)
        member_wikilinks.append(f"[[people/{s}]]")
    fingerprint = corpus.fingerprint()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    frontmatter = {
        "schema_version": 1,
        "kind": slug or "profile",
        "canonical_name": seed or "Query results",
        "query": seed,
        "profile_slug": slug or "profile",
        "members": member_wikilinks,
        "aliases": [slug] if slug else ["profile"],
        "last_derived": now,
        "source_fingerprint": fingerprint,
        "source_messages": len(corpus),
        "candidates": len(members),
        "confidence": "medium" if len(members) >= 3 else "low",
        "derivation": {
            "seed": seed,
            "stop_reason": stop_reason,
            "queries_run": queries_run,
        },
    }

    # ---- Body
    body_parts: list[str] = []
    body_parts.append(f"# Candidates\n")
    body_parts.append(
        f"All extracted persons from the corpus. Evaluate based on the "
        f"original query: \"{seed}\".\n"
    )

    # Build footnote numbering.
    msg_order: list[MessageId] = []
    msg_index: dict[MessageId, int] = {}

    def cite(mid: MessageId) -> str:
        if mid not in msg_index:
            msg_order.append(mid)
            msg_index[mid] = len(msg_order)
        return f"[^{msg_index[mid]}]"

    if not members:
        body_parts.append("(no candidates extracted from the corpus)\n")
    else:
        grouped: dict[str, list[ProfileMember]] = defaultdict(list)
        for name, pm in members.items():
            grouped[_normalize_canonical_name(name)].append(pm)
        for display_name in sorted(grouped):
            pms = grouped[display_name]
            excerpts: list[tuple[MessageId, str]] = []
            is_bidirectional = False
            is_personal_domain = False
            is_calendar = False
            for pm in pms:
                excerpts.extend(pm.excerpts)
                if pm.bidirectional:
                    is_bidirectional = True
                if pm.personal_domain:
                    is_personal_domain = True
                if pm.calendar_signal:
                    is_calendar = True
            body_parts.append(f"### {display_name}\n")
            body_parts.append(f"- Appears in {len(excerpts)} messages")
            if is_bidirectional:
                body_parts.append("- Bidirectional correspondence: yes")
            if is_personal_domain:
                body_parts.append("- Personal-domain contact: yes")
            if is_calendar:
                body_parts.append("- Calendar signal: yes")
            # Extract domain from first excerpt message.
            if excerpts:
                first_mid = excerpts[0][0]
                first_msg = corpus.get(first_mid)
                if first_msg and first_msg.from_addr:
                    domain_match = _EMAIL_DOMAIN_RE.search(first_msg.from_addr)
                    if domain_match:
                        body_parts.append(f"- Domain: {domain_match.group(1)}")
            body_parts.append("- Evidence:")
            for mid, snippet in excerpts[:3]:
                body_parts.append(f"  - \"{snippet}\" {cite(mid)}")
            body_parts.append("")

    body_parts.append("## Open questions\n")
    body_parts.append(
        "- Disambiguation between people sharing first names is not performed.\n"
    )

    # ---- Provenance
    body_parts.append("## Provenance\n")
    for mid in msg_order:
        msg = corpus.get(mid)
        subj = msg.subject if msg else "(unknown)"
        body_parts.append(f'[^{msg_index[mid]}]: gmail:{mid} - "{subj}"')

    # ---- Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    content = "---\n" + fm_yaml + "\n---\n\n" + "\n".join(body_parts) + "\n"
    out_path.write_text(content, encoding="utf-8")
    return out_path.resolve()


def _slugify_query(query: str) -> str:
    """Convert query to a filesystem-safe slug."""
    if not query or not query.strip():
        return "profile"
    slug = re.sub(r'[^a-z0-9]+', '-', query.lower()).strip('-')
    return slug[:50] or "profile"


