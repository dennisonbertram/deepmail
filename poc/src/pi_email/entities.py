"""Entity extraction over a corpus delta.

Extracts PERSON and EMAIL entities from email messages using spaCy NER. All
spaCy PERSON entities that pass the stoplists are emitted -- no kinship-word
gating, no relation-grounding, no adjacency binding. The calling model
(Claude Code, Cursor, etc.) does all classification via the SKILL.md.

Bulk messages -- flagged earlier by `filters.is_bulk_message(...)` -- are
skipped wholesale: newsletters and list-managed mail dump too many proper
nouns to be trustworthy entity sources at all.

Public API:
  * `Entity(kind, key, label[, confidence])` dataclass
  * `EntityExtraction(entities, by_message)` aggregate
  * `extract_from_corpus(corpus, message_ids=None) -> EntityExtraction`
  * `canonicalize(...)` (embedding-based clustering)

spaCy is loaded lazily on first call (~0.4s + model file ~12 MB). The download
happens on demand if the model isn't already installed.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .corpus import Corpus, Message, MessageId
from .strip_quotes import strip_quotes_and_signatures

if TYPE_CHECKING:  # avoid runtime import of sentence-transformers at module load
    from .embedder import Embedder
    from .embedding_store import EmbeddingStore


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation vocabulary (kept as a set for _in_stoplist filtering only)
# ---------------------------------------------------------------------------

# Relation words used ONLY by _in_stoplist to reject bare relation words
# ("Mom", "Dad") that spaCy mistags as PERSON. No longer used for gating
# or grounding.
RELATION_WORDS: frozenset[str] = frozenset({
    "mom", "mum", "mommy", "mother", "mothers", "ma",
    "dad", "daddy", "father", "fathers", "papa", "pa",
    "wife", "husband", "spouse", "partner", "fiancé", "fiancée", "fiance", "fiancee",
    "son", "sons", "daughter", "daughters",
    "kid", "kids", "child", "children", "baby",
    "brother", "brothers", "bro",
    "sister", "sisters", "sis", "sibling", "siblings",
    "grandma", "grandmother", "nana", "grandpa", "grandfather", "papaw",
    "grandson", "grandsons", "granddaughter", "granddaughters",
    "grandchild", "grandchildren", "grandkids", "grandparent", "grandparents",
    "aunt", "aunts", "auntie", "uncle", "uncles",
    "cousin", "cousins", "niece", "nieces", "nephew", "nephews",
    "family", "relative", "relatives", "in-law", "in-laws", "stepmom", "stepdad",
})



# ---------------------------------------------------------------------------
# Stoplists
# ---------------------------------------------------------------------------

# Tokens that, if they appear *as part of* a PERSON span (case-insensitive),
# mean the span is almost certainly a company / org / institution mislabeled
# by spaCy. The NER model is already pretty good; this is belt-and-suspenders
# against the residual noise in real mail.
_COMPANY_TOKENS: frozenset[str] = frozenset({
    "labs", "inc", "llc", "corp", "co", "ltd", "gmbh",
    "partners", "group", "holdings", "foundation", "institute",
    "capital", "ventures",
    # Pass 7A: broaden the company-suffix vocabulary. Each of these has
    # been observed in real-Gmail runs as a constituent token of a span
    # that spaCy mistagged PERSON (Coinbase Wealth, Apple Music, Edge
    # Network, etc.). None of them legitimately appear in personal names
    # in the inboxes we've examined.
    "studios", "studio", "media", "network", "networks", "solutions",
    "systems", "technologies", "technology", "tech", "ai",
    "bank", "fund", "trust", "industries", "enterprises",
    "international", "global", "worldwide",
})


# Tokens that, in the FIRST or LAST position of a multi-token PERSON span,
# strongly indicate the span is a place / venue / institution rather than a
# personal name. Examples we've seen real-Gmail-runs mislabel:
#   "Corto Cafe"     -> last token "Cafe" -> reject
#   "Clinton St"     -> last token "St"   -> reject
#   "Park Avenue Smith" -> first token "Park" -> reject (street name)
#
# Cohen is intentionally NOT in this list — it's a real surname ("David
# Cohen" must pass). The org-context regex below handles "Morrison Cohen as
# a law firm".
_ORG_INDICATOR_TOKENS: frozenset[str] = frozenset({
    "cafe", "bar", "restaurant", "hotel", "inn", "club", "association",
    "foundation", "institute", "center", "centre", "society", "council",
    "committee", "department", "office", "bureau", "agency", "studio",
    "gallery", "museum", "library", "school", "academy", "university",
    "college", "hospital", "clinic", "church", "temple", "synagogue",
    "mosque",
    # Place / venue suffixes commonly mistagged as PERSON
    "park", "plaza", "square", "court", "avenue", "street",
    "st", "rd", "blvd",
})


# spaCy entity labels that must NOT be accepted as PERSON candidates, even
# when munged through other code paths (e.g. the from-header extractor used
# to admit short ORG names). The body NER pass already gates on label
# == "PERSON"; this set is the authoritative "anything-but-PERSON" list used
# by the cross-label overlap check (Rule 4).
_NON_PERSON_NER_LABELS: frozenset[str] = frozenset({
    "ORG", "GPE", "LOC", "PRODUCT", "FAC", "EVENT", "NORP",
    "WORK_OF_ART", "LAW", "MONEY", "PERCENT", "DATE", "TIME",
    "ORDINAL", "CARDINAL", "QUANTITY",
})


# Common English nouns / adjectives / verbs / months / weekdays / generic
# role words / crypto asset names that spaCy NER sometimes mislabels as
# PERSON when they're title-cased in subjects or salutations.
#
# Lowercased on init. Applied AFTER spaCy POS gating because POS tagging is
# noisy on short title-cased snippets ("Bitcoin" tags as PROPN even though
# it's an asset name).
#
# Rule: for single-token candidates, reject if the lowercased name is in the
# set. For multi-token candidates, only reject if EVERY token is in the set
# (so "Email Johnson" survives because "Johnson" is not a stop word).
_COMMON_NOUN_STOPLIST: frozenset[str] = frozenset({
    # From the real-Gmail run that surfaced this need.
    "email", "schedule", "subscribe", "healthy", "remaining",
    # Salutations / sign-offs that get title-cased and look like names.
    "thanks", "thank", "regards", "best", "sincerely", "cheers",
    "hello", "hi", "hey", "dear", "greetings",
    # Email structural words.
    "subject", "body", "message", "reply", "forward",
    # Time words.
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "spring", "summer", "fall", "autumn", "winter",
    # Crypto / web3 / asset names that NER misreads as people.
    "bitcoin", "ethereum", "ether", "solana", "polygon", "arbitrum",
    "nft", "defi", "dao", "web3",
    # Generic role nouns.
    "founder", "ceo", "cto", "cfo", "coo", "president", "director",
    "manager", "engineer", "developer", "designer", "consultant",
    "analyst", "advisor", "investor",
    # Common false-positives from email signatures + marketing footers.
    "preview", "unsubscribe", "share", "view",
    # Tiny function words (capitalized at sentence start).
    "the", "and", "or", "but", "for", "with", "without",
})


# Sentence-level org context: when one of these phrases appears in the same
# sentence as a multi-token PERSON candidate, treat the candidate as an
# organization. This is the only mechanism that distinguishes
# "Morrison Cohen as a law firm" (reject) from "David Cohen is visiting"
# (accept) — both spans are syntactically identical [PROPN PROPN], so the
# disambiguator has to be context.
#
# The pattern is intentionally narrow ("as a/an/the law firm") so it doesn't
# fire on incidental mentions of "the company picnic" or "a great agency".
_ORG_CONTEXT_RE = re.compile(
    r"\bas\s+(?:a|an|the)\s+"
    r"(?:law\s+firm|law\s+office|firm|lawyer|attorney|counsel|"
    r"agency|company|corporation|consulting|consultancy|llp|llc)\b",
    re.IGNORECASE,
)


# Tokens that, if they appear *anywhere* in a from-header display name
# (case-insensitive), mean the sender is an organization / role mailbox /
# bulk-mail program — not a person. Real-Gmail runs surfaced this need
# because the body-NER bulk filter doesn't always fire (e.g. transactional
# notification mail from Google Cloud Platform lacks `List-Unsubscribe`)
# yet the sender display name is clearly an org. Each token here is one
# that has never legitimately appeared as part of a personal name in the
# inboxes we've examined, so a single token-match is sufficient.
#
# Failure modes this catches (real entries from the bad profile):
#   "Google Cloud Platform"  -> "Cloud" / "Platform"
#   "Sonoma Art School"      -> "School"
#   "Edge Esmeralda"         -> (no match here — needs different gate)
#   "Summer Workshops Team"  -> "Workshops" / "Team"
_SENDER_ORG_TOKENS: frozenset[str] = frozenset({
    # Platform / service / system mailboxes.
    "cloud", "platform", "service", "services", "notifications",
    "team", "support", "help", "customer", "customers",
    "account", "accounts", "security", "billing", "sales",
    "marketing", "newsletter", "updates", "community",
    # Institutional / venue.
    "foundation", "school", "academy", "university", "college",
    "institute", "project", "workshop", "workshops",
    "class", "classes", "lab", "labs", "studio",
    # Event / gathering.
    "conference", "event", "events", "summit", "meetup",
    "group", "network", "club", "festival",
})


# Honorifics + relation prefixes we strip from the front of a NER span. spaCy
# regularly returns "Aunt Carol" / "Dr. Smith" as PERSON; we keep "Carol" /
# "Smith". The list is intentionally narrow — anything not on it is preserved
# verbatim so we don't strip parts of legitimate compound names.
_LEADING_DROP: frozenset[str] = frozenset({
    "dr", "mr", "mrs", "ms", "prof",
    "aunt", "auntie", "uncle", "grandma", "grandmother", "grandpa", "grandfather",
    "nana", "papaw",
    "mom", "mum", "mommy", "mother", "ma",
    "dad", "daddy", "father", "papa", "pa",
})


# Trailing tokens we strip from a PERSON span — same idea as _LEADING_DROP but
# at the tail, and centered on role / business-function words rather than
# family / honorific. spaCy regularly returns spans like "Ryan Rigney
# Marketing" or "Kelly Ebeling Founder" as a single PERSON entity when the
# role / dept word follows the name in a salutation or signature; we peel it
# off so the canonical entity is just the name.
#
# Mirrors materializer's _NAME_TAIL_STOPLIST (defense-in-depth — that layer
# still runs at slug time) plus the business-function tokens that
# materializer never had to deal with because they were already in NER spans
# before reaching the materializer. Lowercased; compared case-insensitively
# with trailing punctuation stripped.
_ROLE_TAIL_TOKENS: frozenset[str] = frozenset({
    # Roles / titles (mirrors materializer._NAME_TAIL_STOPLIST)
    "founder", "ceo", "cto", "cfo", "coo", "president", "director",
    "manager", "engineer", "developer", "designer", "consultant",
    "analyst", "advisor", "investor", "lead", "head", "chief",
    "vp", "evp", "svp", "principal", "partner", "owner",
    "founder/ceo",
    # Subject-line / newsletter cruft
    "intro", "introduction", "re", "fwd", "subject", "update",
    "newsletter", "weekly", "monthly", "daily", "digest",
    # Business-function / department tokens — new in this round; never
    # legitimately end a personal name in the inboxes we've examined.
    "marketing", "engineering", "sales", "operations", "product",
    "design", "research", "strategy", "legal", "finance", "hr",
    "it", "pr", "bd", "growth", "analytics", "partnerships",
    "community",
})


# Multi-token PERSON spans that contain any of these tokens are almost
# always events, conferences, festivals, communities, or place names that
# spaCy occasionally mistags as PERSON when both tokens are title-cased
# (e.g. "Edge Esmeralda" — a famous pop-up city / community event).
#
# Single-token candidates whose lowercased form is in this set are also
# rejected — "Edge" alone leaks through as PERSON when it's a community
# name fragment.
#
# Trade-off: if the user has a real friend named "Edge" or "Esmeralda" they
# will be filtered out. Acceptable for v1 — the spec calls this out
# explicitly. The list is intentionally short and biased toward tokens
# that pattern-match events/communities rather than common English words
# (those live in _COMMON_NOUN_STOPLIST).
_EVENT_PLACE_TOKENS: frozenset[str] = frozenset({
    # Events / conferences / festivals
    "festival", "conference", "summit", "expo", "convention", "fair",
    "hackathon", "meetup", "demo", "demoday", "pitch", "launch",
    # Camps / retreats
    "camp", "retreat", "weekend", "getaway",
    # Communities / events with proper-noun-y names that surfaced as FPs.
    "esmeralda",  # known FP — Edge Esmeralda is a famous pop-up city
    "edge",       # paired with Esmeralda; also bare "Edge" surfaced as FP
    # Locations
    "city", "town", "village", "country", "state",
    # Sports / leagues
    "league", "tournament", "championship", "playoffs", "finals",
})


# Public figures, politicians, notable VCs, and crypto/web3 brand names
# that look syntactically like personal names. Newsletter and news-summary
# content drops these into family inboxes despite the bulk filter;
# spaCy correctly tags them PERSON so the only way to remove them is an
# explicit list.
#
# Lowercased. Membership rules in `_in_stoplist`:
#   * single-token candidate -> reject if cleaned.lower() in the set
#   * multi-token candidate  -> reject only if the FULL lowercased phrase
#     is in the set (so "Jane Trumpington" survives — the check is whole-
#     name, not substring).
#
# Trade-off: if the user has a real friend with one of these last names —
# e.g. a real "Trump" or "Sanders" — they will be filtered out. Acceptable
# for v1; the spec calls this out explicitly. Better to drop a real name
# than admit a stream of news-summary noise.
_PUBLIC_FIGURE_STOPLIST: frozenset[str] = frozenset({
    # US politicians (extremely common in news/newsletter context)
    "trump", "biden", "harris", "obama", "clinton", "bush",
    "pelosi", "schumer", "mcconnell", "sanders", "warren",
    "musk", "zuckerberg", "bezos", "gates",
    # Notable VCs / tech figures (newsletter content)
    "tim draper", "marc andreessen", "ben horowitz", "paul graham",
    "fred wilson", "naval ravikant", "balaji srinivasan",
    "vitalik buterin", "vitalik", "satoshi", "satoshi nakamoto",
    "sam altman", "altman", "ilya sutskever",
    # Crypto / web3 brand names that look like people
    "coinbase wealth", "coinbase prime", "binance",
    "ethereum foundation", "solana foundation",
    # Pass 7A: known two-word organization names that look syntactically
    # like personal names. Belt-and-suspenders for NER mistakes where
    # both tokens are common short generic words and the org-shape
    # token list above doesn't cover them.
    "double zero", "edge esmeralda", "binance smart",
    "apple music", "google cloud", "amazon web",
})


# ---------------------------------------------------------------------------
# Email regex (preserved verbatim from the prior implementation)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


# Embedded-image Content-ID addresses (RFC 2392) look like real emails to
# the regex above — they have the @ separator and dotted-host shape — but
# they are NOT contact addresses: they're internal MIME references for
# inline images / attachments. We saw e.g.
#   image001.jpg@01dce229.53871350
# materialize as a "contact" entity and seed an iterative search. The
# patterns below catch the four common CID shapes:
#   * local part `image\d+\.<ext>` (Outlook standard)
#   * local part `image\d+_<hash>` (some Office variants)
#   * local part `part\d+\.\d+` (older Apple Mail / Mozilla)
#   * domain `<hex>.<hex>` with both segments 8+ hex chars (Outlook MUID-y)
# Plus a broad belt-and-suspenders check: any local part containing a
# common image file extension as a substring (no real personal email has
# `.jpg` in the local part).
_CID_LOCAL_IMAGE_EXT_RE = re.compile(
    r"^image\d+\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff)$",
    re.IGNORECASE,
)
_CID_LOCAL_IMAGE_HASH_RE = re.compile(r"^image\d+_", re.IGNORECASE)
_CID_LOCAL_PART_RE = re.compile(r"^part\d+\.\d+", re.IGNORECASE)
_CID_LOCAL_IMG_EXT_SUBSTR_RE = re.compile(
    r"\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff)\b",
    re.IGNORECASE,
)
_CID_DOMAIN_HEXHEX_RE = re.compile(r"^[0-9a-f]{8,}\.[0-9a-f]{8,}$", re.IGNORECASE)


def _is_cid_email(addr: str) -> bool:
    """True if `addr` looks like an embedded-image Content-ID address rather
    than a real contact email.

    Conservative: a plain `image@somedomain.com` (possible "Image Department"
    mailbox) is NOT caught — we require either a digits-after-`image` local
    part, an image-extension substring in the local part, the `partN.M`
    shape, or the hex.hex CID-y domain shape.
    """
    if "@" not in addr:
        return False
    local, _, domain = addr.partition("@")
    if not local or not domain:
        return False
    if _CID_LOCAL_IMAGE_EXT_RE.match(local):
        return True
    if _CID_LOCAL_IMAGE_HASH_RE.match(local):
        return True
    if _CID_LOCAL_PART_RE.match(local):
        return True
    if _CID_LOCAL_IMG_EXT_SUBSTR_RE.search(local):
        return True
    if _CID_DOMAIN_HEXHEX_RE.match(domain):
        return True
    return False


# Pulls the display name out of a "Display Name <addr@host>" header. RFC 2822
# allows quoted strings and embedded angle brackets; for our purposes a simple
# match of "anything before the first <addr>" is good enough.
_SENDER_NAME_RE = re.compile(r"^\s*(.+?)\s*<[^<>]+@[^<>]+>\s*$")


# ---------------------------------------------------------------------------
# Lazy spaCy loader
# ---------------------------------------------------------------------------

_NLP_LOCK = threading.Lock()
_NLP = None  # type: ignore[var-annotated]
_NLP_LOG_DONE = False  # so the "downloading model" message prints at most once


def _load_spacy_model():
    """Load `en_core_web_sm`, downloading it once if not already installed.

    Import is deferred so module-import time stays cheap (sub-millisecond) and
    test files that never extract entities don't pay the spaCy startup cost.
    Concurrent first-callers are serialized through `_NLP_LOCK` so we never
    spawn two `spacy download` subprocesses.
    """
    global _NLP, _NLP_LOG_DONE
    with _NLP_LOCK:
        if _NLP is not None:
            return _NLP
        import spacy  # local import — module import must not pay this cost

        model = "en_core_web_sm"
        try:
            _NLP = spacy.load(model)
        except OSError:
            if not _NLP_LOG_DONE:
                _log.info(
                    "[entity-extractor] downloading spaCy en_core_web_sm "
                    "model (~12 MB, first-run only)"
                )
                # Also print to stderr so the demo's stdout-only logger
                # surfaces it even when logging is configured at WARNING.
                print(
                    "[entity-extractor] downloading spaCy en_core_web_sm "
                    "model (~12 MB, first-run only)",
                    file=sys.stderr,
                    flush=True,
                )
                _NLP_LOG_DONE = True
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model],
                check=True,
            )
            _NLP = spacy.load(model)
        return _NLP


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A canonicalized entity surfaced from a message.

    `kind` is one of:
      * "person"   -- a human, extracted via spaCy NER + stoplists
      * "email"    -- an RFC 822 address (from body, From, or To)

    The frontier dedupes by (kind, key); the proposer renders the human-readable
    `label`. `confidence` is metadata that downstream code may use to weight
    or filter; it does NOT participate in equality or hashing -- two Entities
    with the same (kind, key, label) and different confidence collapse to one.
    """

    kind: str
    key: str   # canonical form, lowercased
    label: str  # display form (title-case for names, lowercased for emails)
    confidence: str = field(default="medium", compare=False)

    def __str__(self) -> str:
        return f"{self.kind}:{self.label}"


@dataclass
class EntityExtraction:
    """Result of extracting entities from a corpus delta."""

    entities: set[Entity] = field(default_factory=set)
    # message_id -> entities surfaced from that message; used for provenance.
    by_message: dict[MessageId, set[Entity]] = field(default_factory=lambda: defaultdict(set))


# ---------------------------------------------------------------------------
# Name cleanup + stoplist helpers
# ---------------------------------------------------------------------------


def _canon_name(raw: str) -> tuple[str, str]:
    """Return (key, label) for a person name. Key is lowercased; label keeps
    the cleaned casing produced by `_clean_person_span`."""
    cleaned = re.sub(r"\s+", " ", raw.strip())
    return cleaned.lower(), cleaned


def _clean_person_span(raw: str) -> str:
    """Strip honorifics + leading relation words + trailing role tokens +
    trailing punctuation from a spaCy PERSON span. Returns "" if nothing
    usable remains.

    Examples (run on actual spaCy output):
      "Aunt Carol"                 -> "Carol"
      "Dr. Smith"                  -> "Smith"      (spaCy usually drops "Dr." but not always)
      "Grandma Helen"              -> "Helen"
      "Jane"                       -> "Jane"
      "Jane Smith Jr."             -> "Jane Smith Jr"
      "Ryan Rigney Marketing"      -> "Ryan Rigney"      (Round-5: role tail strip)
      "Kelly Ebeling Founder"      -> "Kelly Ebeling"
      "Sam Lee CEO Founder"        -> "Sam Lee"          (iterates tail strip)
      ""                           -> ""

    Role-tail stripping iterates so "X Y CEO Founder" peels both. It runs
    AFTER the leading-honorific strip so a single-token leftover like
    "Founder" is preserved (everything-stripped means we return ""
    upstream from where it's checked).
    """
    s = raw.strip().strip("\"'`")
    # Drop trailing punctuation but keep periods inside (e.g. "Jr.")
    while s and s[-1] in ".,;:!?":
        s = s[:-1]
    s = s.strip()
    if not s:
        return ""
    tokens = s.split()
    # Drop leading honorific / relation prefixes.
    while tokens:
        head = tokens[0].lower().rstrip(".")
        if head in _LEADING_DROP:
            tokens.pop(0)
            continue
        break
    # Drop trailing role / business-function tokens, but only when MORE THAN
    # ONE token remains — peeling a single-token "Founder" would erase the
    # span, and the downstream stoplist already kills bare-role single-token
    # candidates via _COMMON_NOUN_STOPLIST. The materializer's
    # _NAME_TAIL_STOPLIST applies the same defense-in-depth at slug time.
    while len(tokens) > 1:
        tail = tokens[-1].lower().rstrip(".,;:!?")
        if tail in _ROLE_TAIL_TOKENS:
            tokens.pop()
            continue
        break
    return " ".join(tokens).strip()


def _is_company_like(name: str) -> bool:
    """True if any token in `name` is a known company suffix (case-insensitive)."""
    for tok in name.split():
        clean = tok.strip(".,").lower()
        if clean in _COMPANY_TOKENS:
            return True
    return False


def _in_stoplist(name: str) -> bool:
    """Belt-and-suspenders post-NER drop.

    Drops a PERSON candidate if any of these apply:
      * empty / fewer than 2 letters after stripping
      * all-caps single token (acronym: "IBM", "NASA")
      * 4+ consecutive capital letters anywhere (mid-string acronym leakage)
      * contains digits ("Web3", "iPhone16")
      * 50%+ non-letter characters (URL fragments, junk)
      * 100% lowercase (NER noise on lowercase text)
      * matches a company-suffix token ("Labs", "Inc", "LLC", ...)
      * single-token name in `_COMMON_NOUN_STOPLIST` (case-insensitive)
      * multi-token name where every token is in `_COMMON_NOUN_STOPLIST`
      * multi-token name whose FIRST or LAST token is an org indicator
        ("Park", "Cafe", "St", ...) — kills "Clinton St" / "Corto Cafe" /
        "Park Avenue Smith"
      * single-token name in `_EVENT_PLACE_TOKENS` OR multi-token name
        containing ANY token in `_EVENT_PLACE_TOKENS` — kills "Edge",
        "Edge Esmeralda", "Esmeralda Festival", etc.
      * multi-token name containing ANY token in `_SENDER_ORG_TOKENS` —
        same any-position check that catches "Google Cloud Platform" in
        sender-name parsing, now applied to body NER candidates too.
      * single-token name in `_PUBLIC_FIGURE_STOPLIST` (politicians /
        notable VCs / brand-names that look like people) OR multi-token
        whose lowercased WHOLE phrase is in `_PUBLIC_FIGURE_STOPLIST`
        (so "Coinbase Wealth" matches but "Jane Trumpington" does not).
      * is itself a relation word ("Mom", "Dad" — when spaCy mistags a
        relation-word as PERSON)
    """
    if not name:
        return True
    stripped = name.strip()
    # Too short to be a real name. Single-letter "names" are noise.
    if len(stripped) < 2:
        return True
    tokens = stripped.split()
    # Single-token all-caps acronym.
    if len(tokens) == 1 and stripped.isupper():
        return True
    # Run of 4+ consecutive uppercase letters anywhere — embedded acronym
    # (e.g. "IBM Corp" might have been already span-stripped but a residual
    # "IBM" leaks through). 4+ is intentionally above legitimate first-name
    # caps like "Jane" (one capital letter at start) and avoids hitting
    # camelCase that the lowercase-check below already covers.
    if re.search(r"[A-Z]{4,}", stripped):
        return True
    # Digits anywhere.
    if any(ch.isdigit() for ch in stripped):
        return True
    # 50%+ non-letter characters (excluding whitespace). Catches URL
    # fragments, mostly-punctuation noise.
    letters = sum(1 for ch in stripped if ch.isalpha())
    non_ws = sum(1 for ch in stripped if not ch.isspace())
    if non_ws and letters / non_ws < 0.5:
        return True
    # All-lowercase (e.g. mis-cased models output "kiddo" as PERSON sometimes).
    if stripped == stripped.lower() and any(ch.isalpha() for ch in stripped):
        return True
    # Company stem.
    if _is_company_like(stripped):
        return True
    # Bare relation word — "Mom" alone wouldn't be a useful target.
    if stripped.lower() in RELATION_WORDS:
        return True
    # Common English noun / asset name / generic word — single-token
    # candidates die outright; multi-token candidates die only when EVERY
    # token is in the list (so real personal names with a common-word in
    # them survive).
    lower_tokens = [t.strip(".,;:!?").lower() for t in tokens]
    if len(tokens) == 1:
        if lower_tokens[0] in _COMMON_NOUN_STOPLIST:
            return True
    else:
        if all(t in _COMMON_NOUN_STOPLIST for t in lower_tokens):
            return True
    # Multi-token spans whose first or last token is an org indicator —
    # "Clinton St", "Corto Cafe", "Park Avenue Smith". Single-token names
    # like a bare "Park" survive (could be a surname).
    if len(tokens) >= 2:
        if lower_tokens[0] in _ORG_INDICATOR_TOKENS:
            return True
        if lower_tokens[-1] in _ORG_INDICATOR_TOKENS:
            return True
    # Event / place / community tokens — single-token in the list dies
    # ("Edge"); multi-token containing ANY listed token dies
    # ("Edge Esmeralda"). Real personal names will not contain
    # "festival" / "esmeralda" / "summit" etc., so the any-position
    # check is safe here.
    if len(tokens) == 1:
        if lower_tokens[0] in _EVENT_PLACE_TOKENS:
            return True
    else:
        if any(t in _EVENT_PLACE_TOKENS for t in lower_tokens):
            return True
    # Sender-org tokens applied at ANY position for multi-token candidates.
    # The same check that catches "Google Cloud Platform" in sender-name
    # parsing — extended to the body NER path so e.g. spans like
    # "Foo Workshops Group" or "Acme Marketing Team" don't slip through.
    # Single-token "marketing" / "cloud" / "team" are already covered
    # via _COMMON_NOUN_STOPLIST (or could be added there). We only apply
    # at multi-token because legitimate first names like "Cloud" or
    # "Sky" (rare but exist) shouldn't be blocked outright.
    if len(tokens) >= 2:
        if any(t in _SENDER_ORG_TOKENS for t in lower_tokens):
            return True
    # Public-figure / brand-name list. Single-token check is membership;
    # multi-token check is whole-phrase equality so "Jane Trumpington"
    # survives even though "trump" prefixes Trumpington.
    if len(tokens) == 1:
        if lower_tokens[0] in _PUBLIC_FIGURE_STOPLIST:
            return True
    else:
        full_phrase = " ".join(lower_tokens)
        if full_phrase in _PUBLIC_FIGURE_STOPLIST:
            return True
    return False


def _looks_like_person_name(s: str) -> bool:
    """Cheap heuristic for "this is structurally a personal name": 2-4 tokens,
    each starting with an uppercase letter, no digits.

    Used as a fast pre-filter before we run spaCy on a from-header display
    name; the spaCy check is then the authoritative arbiter.
    """
    if not s:
        return False
    tokens = s.split()
    if not (2 <= len(tokens) <= 4):
        return False
    for tok in tokens:
        if not tok:
            return False
        if not tok[0].isupper():
            return False
        if any(ch.isdigit() for ch in tok):
            return False
    return True


def _parse_sender_display_name(from_addr: str) -> str:
    """Extract the display name from a "Name <addr@host>" header. Returns ""
    if the header is a bare address (no display name) or unparseable."""
    if not from_addr:
        return ""
    m = _SENDER_NAME_RE.match(from_addr)
    if not m:
        return ""
    raw = m.group(1).strip()
    # Strip surrounding quotes if present.
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    return raw


def _sender_name_is_personal(nlp, name: str) -> bool:
    """Decide whether a from-header display name refers to a person.

    Strategy:
      * Must pass `_looks_like_person_name` (cheap structural filter).
      * If any token (case-insensitive) is in `_SENDER_ORG_TOKENS`, reject —
        these are tokens that *never* appear in real personal names but
        constantly show up in transactional / notification sender names that
        the bulk-mail filter doesn't catch (no List-Unsubscribe header on
        e.g. "Google Cloud Platform <cloudplatform-noreply@google.com>").
        This is additive to the existing single-token / structural / spaCy
        gates — it catches multi-token org names that spaCy occasionally
        labels PERSON because both tokens are title-cased.
      * spaCy must NOT tag a substantive portion of it as ORG / GPE / PRODUCT
        / FAC. We let it through even when spaCy returns no PERSON tag at all
        — short personal names like "Priya Kumar" sometimes only get one of
        the two tokens labeled.
    """
    if not _looks_like_person_name(name):
        return False
    # Token blocklist — rejects e.g. "Google Cloud Platform", "Sonoma Art
    # School", "Edge Esmeralda", "Foo Workshops". A single token-match is
    # enough; a real personal name will never contain these words.
    lower_tokens = {tok.strip(".,;:!?").lower() for tok in name.split()}
    if lower_tokens & _SENDER_ORG_TOKENS:
        return False
    doc = nlp(name)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE", "LOC", "PRODUCT", "FAC", "EVENT", "WORK_OF_ART"):
            # If the non-person span covers most of the name, treat as non-person.
            if len(ent.text) >= max(1, int(len(name) * 0.6)):
                return False
    return True


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_entities(
    message: Message,
    user_emails: set[str] | None = None,
) -> list[Entity]:
    """Extract PERSON and EMAIL entities from a message.

    All spaCy PERSON entities passing stoplists are emitted.
    No kinship-word gating. No relation-grounding. The calling model classifies.

    Behavior:
      * Bulk-flagged messages return `[]` -- no extraction.
      * Emails are extracted from body, From, and To unconditionally.
      * Sender-name (from "Name <addr>" headers) is extracted at confidence
        "medium" when it parses as a personal name (spaCy-gated).
      * ALL PERSON entities from the body NER pass are emitted when they
        pass the stoplist and NER quality filters.
        * Multi-token names ("Jane Smith") -> confidence "high".
        * Single-token names ("Jane") -> confidence "low".

    Returns a list (may contain duplicates of the same Entity if e.g. the
    sender name also appears in the body); the caller is expected to fold
    into a set if dedup matters.
    """
    if getattr(message, "is_bulk", False):
        return []

    body = message.body_clean if message.body_clean is not None else message.body
    out: list[Entity] = []

    # ---- Emails: body + From + To.
    # CID addresses ("image001.jpg@01dce229.53871350") match the regex but
    # are MIME-internal references, not contacts -- drop them via
    # `_is_cid_email` before they enter the entity stream.
    emails: set[str] = set()
    for source in (body or "", message.from_addr or "", message.to_addr or ""):
        for m in _EMAIL_RE.finditer(source):
            addr = m.group(0).lower()
            if _is_cid_email(addr):
                continue
            emails.add(addr)
    for addr in sorted(emails):
        out.append(Entity(kind="email", key=addr, label=addr, confidence="high"))

    # Lazy-load spaCy. We need it for sender-name validation AND body NER, so
    # we load once even when the body is empty.
    nlp = _load_spacy_model()

    # ---- Sender-name from "Name <addr>" header at medium confidence.
    sender_display = _parse_sender_display_name(message.from_addr or "")
    if sender_display and _sender_name_is_personal(nlp, sender_display):
        cleaned = _clean_person_span(sender_display)
        if cleaned and not _in_stoplist(cleaned):
            key, label = _canon_name(cleaned)
            out.append(Entity(
                kind="person", key=key, label=label, confidence="medium",
            ))

    # ---- Body NER -- ALL PERSON entities passing stoplists.
    if body and body.strip():
        doc = nlp(body)

        # Collect non-PERSON labels for cross-label overlap check.
        non_person_text_labels: dict[str, set[str]] = defaultdict(set)
        for ent in doc.ents:
            if ent.label_ in _NON_PERSON_NER_LABELS:
                non_person_text_labels[ent.text.lower()].add(ent.label_)

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            # Require at least one PROPN token.
            if not any(tok.pos_ == "PROPN" for tok in ent):
                continue
            cleaned = _clean_person_span(ent.text)
            if not cleaned:
                continue
            if _in_stoplist(cleaned):
                continue
            # Cross-label overlap check.
            if ent.text.lower() in non_person_text_labels:
                continue
            if cleaned.lower() in non_person_text_labels:
                continue
            # Sentence-level org context disambiguation.
            tokens = cleaned.split()
            if len(tokens) >= 2 and _ORG_CONTEXT_RE.search(ent.sent.text):
                continue
            confidence = "high" if len(tokens) >= 2 else "low"
            key, label = _canon_name(cleaned)
            out.append(Entity(
                kind="person", key=key, label=label, confidence=confidence,
            ))

    return out


def extract_from_corpus(
    corpus: Corpus,
    message_ids: Iterable[MessageId] | None = None,
    user_emails: set[str] | None = None,
) -> EntityExtraction:
    """Extract entities from a (subset of a) corpus.

    `message_ids` defaults to all messages. Reply-quote stripping is applied
    on-demand and cached on the Message instance. Bulk-flagged messages are
    skipped wholesale by `extract_entities`.

    All spaCy PERSON entities passing stoplists are emitted. No kinship-word
    gating. The calling model classifies.
    """
    result = EntityExtraction()
    if message_ids is None:
        message_ids = corpus.list_ids()

    for mid in message_ids:
        msg = corpus.get(mid)
        if msg is None:
            continue
        if msg.body_clean is None:
            msg.body_clean = strip_quotes_and_signatures(msg.body)
        ents = extract_entities(
            msg,
            user_emails=user_emails,
        )
        if not ents:
            continue
        ent_set = set(ents)
        result.entities |= ent_set
        result.by_message[msg.message_id] |= ent_set

    return result


# ---------------------------------------------------------------------------
# Canonicalization (unchanged from the regex era)
# ---------------------------------------------------------------------------


# Cosine threshold for treating two entity names as the same person/org.
# Empirically with bge-base-en-v1.5 + the "A person named X." context prompt
# (see _format_for_canon below):
#   cos("Jane Smith", "Jane")        = 0.905
#   cos("Bob Smith",  "Bob")         = 0.900
#   cos("Jane Smith", "Bob Smith")   = 0.775
#   cos("Jane Smith", "Jane Foundation" via org template) = 0.82-0.83
# 0.85 sits comfortably between the same-person (0.90) and different-person
# (0.78) bands. Without the context prompt the raw cosines all collapse into
# the 0.65-0.74 range and become indistinguishable from random pairs.
# (The query-dedupe threshold over in frontier.py is HIGHER — query strings
# are already sentences so the model has enough context; we want a stricter
# match there to avoid collapsing legitimately distinct expansions.)
ENTITY_CANON_THRESHOLD = 0.85

# Names this short get noisy embeddings (1-2 char tokens like "Bo" or "Mi"
# carry essentially no signal). 3+ char names — Bob, Mia, Leo, Jim, Sam — DO
# participate; the "A person named X" prompt template gives them enough
# context for the model to discriminate same-family from different-family.
# Skip canonicalization for sub-threshold labels — they self-map.
ENTITY_CANON_MIN_LEN = 3


# Kinds that participate in embedding-based canonicalization. Email
# addresses and relation words ("mom"/"dad"/"sister") have small fixed
# vocabularies where bge embeddings collapse everything into the same
# tight region — at our threshold every relation word would merge into
# one cluster, and every email address into another. For those kinds
# canonicalization is identity (each label maps to itself).
_CANON_KINDS = frozenset({"person", "org"})


# Context-prompt templates per supported kind. The model needs SOME
# surrounding text to ground a short proper noun — embedding bare "Jane"
# vs "Jane Smith" directly produces cosines around 0.74 (indistinguishable
# from "Jane Smith" vs "Bob Smith" at 0.65). Wrapping the name in a stock
# phrase shifts the embedding into a region where the name token dominates,
# which is what a name-equality decision needs.
_CANON_TEMPLATES = {
    "person": "A person named {label}.",
    "org": "An organization named {label}.",
}


def _format_for_canon(label: str, kind: str) -> str:
    """Wrap a raw entity label in a kind-specific context prompt."""
    tpl = _CANON_TEMPLATES.get(kind, "{label}")
    return tpl.format(label=label)


def _tokens(label: str) -> tuple[str, ...]:
    """Lower-cased whitespace-split tokens. Used for the prefix-containment
    safety check; "Jane" -> ("jane",), "Jane Smith" -> ("jane", "smith")."""
    return tuple(label.lower().split())


def _is_token_prefix(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """True if `short` is a token-prefix of `long` OR vice-versa.

    Used to reject the "surname-only merges with full name" failure mode
    we saw with bge-base-en-v1.5: cos("Smith", "Helen Smith") = 0.87 because
    "Smith" alone embeds close to any "X Smith" — but "Smith" is the SECOND
    token, not the first, so the prefix check rejects the merge. Conversely
    "Jane" IS a prefix of "Jane Smith" so the merge is allowed.
    """
    if not short or not long:
        return False
    if len(short) > len(long):
        short, long = long, short
    return long[: len(short)] == short


def canonicalize(
    entities: Iterable["Entity"],
    embedder: "Embedder",
    store: "EmbeddingStore | None" = None,
) -> dict[str, str]:
    """Cluster entity display names by embedding cosine and pick a canonical
    label per cluster. Returns a `display_name -> canonical_name` map.

    Rules:
      * Only same-`kind` entities can merge (person stays apart from org).
      * Pairs with cosine >= ENTITY_CANON_THRESHOLD cluster (union-find).
      * Within a cluster the canonical name is the LONGEST display name;
        ties broken by lexicographic order (so output is deterministic).
      * Names shorter than ENTITY_CANON_MIN_LEN bypass the embedding step
        entirely and self-map (avoids "Bay"/"Leo" type noise).
      * An empty/None input returns an empty dict.

    The function is pure modulo the embedder. If a `store` is supplied,
    per-entity vectors are cached (key = `entity:{kind}:{display}`) so the
    next iteration doesn't re-embed names we've seen before. Without a
    store the embedder's in-process LRU is the only cache.
    """
    ents = list(entities) if entities is not None else []
    if not ents:
        return {}

    # Group by kind — different kinds never merge.
    by_kind: dict[str, list[Entity]] = defaultdict(list)
    for e in ents:
        by_kind[e.kind].append(e)

    mapping: dict[str, str] = {}

    for kind, group in by_kind.items():
        # De-dup display labels within the kind. Multiple Entity objects can
        # share the same display label (e.g. seen in two messages), and we
        # don't want to embed twice.
        labels = sorted({e.label for e in group})

        # Unsupported kinds (email, relation, ...) get identity mapping.
        # The cosine geometry on those kinds collapses unrelated members
        # into one cluster — see _CANON_KINDS comment.
        if kind not in _CANON_KINDS:
            for lab in labels:
                mapping[lab] = lab
            continue

        # Short labels short-circuit to self.
        short = [lab for lab in labels if len(lab) < ENTITY_CANON_MIN_LEN]
        long = [lab for lab in labels if len(lab) >= ENTITY_CANON_MIN_LEN]
        for lab in short:
            mapping[lab] = lab

        if not long:
            continue
        if len(long) == 1:
            mapping[long[0]] = long[0]
            continue

        # Embed (with store-backed cache).
        vecs = _embed_labels(long, kind, embedder, store)
        toks = [_tokens(lab) for lab in long]

        # Union-find over pairwise (cosine >= threshold AND token-prefix).
        # The prefix check guards against the "Smith" -> "Helen Smith"
        # false merge: bare surnames embed near every "X Smith" but they
        # aren't a token-prefix, so they stay distinct.
        parent = list(range(len(long)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(len(long)):
            for j in range(i + 1, len(long)):
                if not _is_token_prefix(toks[i], toks[j]):
                    continue
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim >= ENTITY_CANON_THRESHOLD:
                    union(i, j)

        # Group by root, pick canonical = longest then lexicographic.
        clusters: dict[int, list[str]] = defaultdict(list)
        for i, lab in enumerate(long):
            clusters[find(i)].append(lab)

        for members in clusters.values():
            # Longest first; lex-smallest breaks ties.
            canonical = sorted(members, key=lambda s: (-len(s), s))[0]
            for lab in members:
                mapping[lab] = canonical

    return mapping


def _embed_labels(
    labels: list[str],
    kind: str,
    embedder: "Embedder",
    store: "EmbeddingStore | None",
) -> list[np.ndarray]:
    """Embed `labels` for canonicalization, consulting `store` first.

    Each label is wrapped in a kind-specific context prompt (see
    `_format_for_canon`) before embedding — bare proper nouns don't carry
    enough signal for the model to discriminate same-person from
    different-person at our threshold. Vectors are unit-normalized
    (the embedder's contract) and returned in the same order as `labels`.

    The store key is `entity:{kind}:{label}` — the LABEL not the formatted
    prompt, because the prompt is an implementation detail of canon and
    callers reading the store get a "vector for this name" semantic.
    """
    if store is None:
        # No on-disk cache; the embedder's in-process LRU covers repeats.
        return [embedder.embed(_format_for_canon(lab, kind)) for lab in labels]

    keys = [f"entity:{kind}:{lab}" for lab in labels]
    cached = store.get_many(keys)
    out: list[np.ndarray] = []
    for lab, key in zip(labels, keys):
        v = cached.get(key)
        if v is None:
            v = embedder.embed(_format_for_canon(lab, kind))
            store.put(key, "entity", v, model_id=embedder.model_id)
        out.append(v)
    return out
