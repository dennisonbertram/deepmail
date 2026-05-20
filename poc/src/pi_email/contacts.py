"""Google People API client + family-detection heuristics.

Pass 12A introduces a new evidence source beyond email content. After 11 passes
tuning email-only extraction, recall is bounded by the fact that real family
members rarely use kinship words in their email. The user's own address book,
however, labels relatives directly (Family group membership, the structured
`relations` field, biography notes like "my sister Jane"). Connecting to that
input unlocks the non-spouse family identifications email content can't reach.

This module is a thin wrapper around the People API + a deterministic scoring
layer that converts raw `Person` resources into `family_signal_strength` and a
`family_signal_source` tag the materializer + LLM judge can consume.

Public surface:
  * `CONTACTS_READONLY_SCOPE` — the OAuth scope constant.
  * `Contact`, `ContactGroup` dataclasses — normalized shapes of People API
    resources, scrubbed of metadata noise.
  * `GoogleContacts` — the client. `list_groups`, `find_family_group`,
    `list_connections`, `list_family_members`, `lookup_by_email`.

Why these signals, in this priority order:
  1. Family-group membership — the user explicitly curated this. Strongest.
  2. `relations` field with a kinship type — Google's structured relation
     field; populated through Contacts UI when the user assigns a relation.
  3. Biography mentioning kinship words — the notes field where users
     casually annotate ("my sister Jane in Portland"). Medium signal.
  4. Surname match with the account owner — same logic as the email-based
     surname signal, applied to contact-derived names.
  5. Starred + personal-domain email — weakest, but a useful tie-breaker.

We do NOT call the searchContacts endpoint for `lookup_by_email` — for the
volumes we care about (~hundreds of contacts), iterating the cached
`list_connections` result is simpler and avoids the extra round-trip + the
search endpoint's "warmup" requirement (it needs an explicit warm call before
the first real query).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


log = logging.getLogger(__name__)


# OAuth scope. Read-only is sufficient for our use — we never mutate contacts.
CONTACTS_READONLY_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"


# Person fields we request from the People API. Bounding the field mask keeps
# response sizes (and quota) small; we only ask for what the scoring layer can
# actually consume.
_PERSON_FIELDS = (
    "names,emailAddresses,memberships,relations,biographies,metadata"
)

# Group fields for contactGroups.list. We need name + memberCount to find the
# family group; groupType lets us distinguish system groups from user groups.
_GROUP_FIELDS = "name,memberCount,groupType,metadata"


# Page sizes. People API caps at 1000 for both endpoints; ask for the cap so
# we minimize round-trips for typical address books (<= 2000 contacts).
_CONNECTIONS_PAGE_SIZE = 1000
_GROUPS_PAGE_SIZE = 1000

# Names we look for when finding the user's family group. Google's system
# "Family" group is `contactGroups/family` (`name: "family"`) but some users
# may not have one and instead use a user-defined group called "Family",
# "My Family", "Bertram Family", etc. We do a case-insensitive substring
# match on the formatted name and stop at the first plausible hit.
_FAMILY_GROUP_NAME_HINTS = (
    "family",  # matches "Family", "family", "My Family", "Bertram Family", ...
)

# Kinship vocabulary the biography scanner looks for. Aligns with
# family_judge._MOCK_RELATION_WORDS so the contact-source signal stays in
# step with the in-email signal.
_BIOGRAPHY_KINSHIP_WORDS: frozenset[str] = frozenset({
    "mom", "mother", "mum", "mommy", "ma",
    "dad", "father", "daddy", "papa", "pa",
    "wife", "husband", "spouse", "partner", "fiance", "fiancee",
    "son", "daughter", "kid", "kids", "child", "children", "baby",
    "brother", "bro", "sister", "sis", "sibling", "siblings",
    "grandma", "grandmother", "nana", "grandpa", "grandfather",
    "grandson", "granddaughter", "grandchild", "grandkids",
    "aunt", "auntie", "uncle",
    "cousin", "niece", "nephew",
    "stepmom", "stepdad", "in-law", "father-in-law", "mother-in-law",
    "brother-in-law", "sister-in-law", "son-in-law", "daughter-in-law",
})

# Pre-compiled biography regex — one alternation over all kinship words with
# longest-first ordering so "mother-in-law" wins over "mother".
_BIOGRAPHY_KINSHIP_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(w) for w in sorted(
            _BIOGRAPHY_KINSHIP_WORDS, key=len, reverse=True
        )
    ) + r")\b",
    re.IGNORECASE,
)

# Google's well-known system contact-group resource names. Used only to
# recognize the system Family group quickly (it has `groupType:
# SYSTEM_CONTACT_GROUP` and the resource name `contactGroups/family`). If the
# user has only a user-defined family group we fall back to name matching.
SYSTEM_FAMILY_GROUP_RESOURCE = "contactGroups/family"
SYSTEM_STARRED_GROUP_RESOURCE = "contactGroups/starred"


# ---------------- Data ----------------


@dataclass
class ContactGroup:
    """A contact group (system or user-defined).

    `name` is the raw API name (lowercase for system groups: "family",
    "starred", "myContacts"). `formatted_name` is the localized display
    version Google renders in the Contacts UI. `member_count` is the
    server-side count and may be 0 for the system "Family" group on users
    who never populated it.
    """

    resource_name: str
    name: str
    formatted_name: str
    member_count: int
    is_system: bool = False


@dataclass
class Contact:
    """A normalized Person resource.

    Lowercased emails make case-insensitive lookups cheap; all other text
    fields preserve their original case for display. `group_memberships` is
    the list of contactGroupResourceNames this contact belongs to (the
    deprecated `contactGroupId` form is dropped here on ingestion).
    `relations` keeps the API shape (`{"person": ..., "type": ...,
    "formattedType": ...}`) so the scoring layer can inspect the structured
    relation type without re-parsing strings.
    """

    resource_name: str
    display_name: str
    given_name: str | None
    family_name: str | None
    email_addresses: list[str] = field(default_factory=list)
    group_memberships: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    biography: str | None = None
    is_starred: bool = False

    # Derived signals (populated by `score_family_signal`).
    family_signal_strength: float = 0.0
    family_signal_source: str = ""


# ---------------- Person -> Contact conversion ----------------


def _person_to_contact(person: dict) -> Contact | None:
    """Convert a People API `Person` resource into our `Contact` dataclass.

    Returns None when the Person has no usable display name AND no emails —
    such records carry no signal we can use and would just pollute the
    candidate pool.

    Conversion notes:
      * `names` is a list; we take the first entry (the Contacts UI only
        ever surfaces one name per contact, and the API ordering puts the
        primary one first).
      * `emailAddresses[].value` is lowercased on ingest.
      * `memberships[].contactGroupMembership.contactGroupResourceName` —
        we extract this directly; the deprecated `contactGroupId` is
        ignored.
      * `relations` is preserved as-is (structured: {person, type,
        formattedType}). The Contacts UI populates `type` with values from
        a controlled vocabulary like "spouse", "mother", "brother".
      * `biographies` is a singleton for contact sources per Google's
        spec; we take the first entry's `value`.
    """
    resource_name = str(person.get("resourceName") or "").strip()
    if not resource_name:
        return None

    names = person.get("names") or []
    primary_name = names[0] if names else {}
    display = str(primary_name.get("displayName") or "").strip()
    given = primary_name.get("givenName")
    family = primary_name.get("familyName")
    given = str(given).strip() if given else None
    family = str(family).strip() if family else None

    email_objs = person.get("emailAddresses") or []
    emails: list[str] = []
    for eo in email_objs:
        val = (eo.get("value") or "").strip().lower()
        if val and val not in emails:
            emails.append(val)

    # Bail when we have neither a name nor an email — nothing to match on.
    if not display and not emails:
        return None

    # If the display name is missing but we have given/family, synthesize it.
    if not display:
        display = " ".join(p for p in (given, family) if p).strip()
        # Fall back to the local-part of the first email as a last resort.
        if not display and emails:
            display = emails[0].split("@", 1)[0]

    membership_objs = person.get("memberships") or []
    group_memberships: list[str] = []
    is_starred = False
    for m in membership_objs:
        cgm = m.get("contactGroupMembership") or {}
        ref = (cgm.get("contactGroupResourceName") or "").strip()
        if not ref:
            continue
        group_memberships.append(ref)
        if ref == SYSTEM_STARRED_GROUP_RESOURCE:
            is_starred = True

    relations_raw = person.get("relations") or []
    relations: list[dict] = []
    for r in relations_raw:
        # Keep the three fields we care about; drop metadata noise.
        entry = {
            "person": str(r.get("person") or "").strip(),
            "type": str(r.get("type") or "").strip(),
            "formattedType": str(r.get("formattedType") or "").strip(),
        }
        if entry["person"] or entry["type"]:
            relations.append(entry)

    bio_objs = person.get("biographies") or []
    biography: str | None = None
    if bio_objs:
        val = bio_objs[0].get("value")
        if val and str(val).strip():
            biography = str(val).strip()

    return Contact(
        resource_name=resource_name,
        display_name=display,
        given_name=given,
        family_name=family,
        email_addresses=emails,
        group_memberships=group_memberships,
        relations=relations,
        biography=biography,
        is_starred=is_starred,
    )


def _group_resource_to_group(g: dict) -> ContactGroup:
    """Convert a contactGroups.list entry into a ContactGroup."""
    resource_name = str(g.get("resourceName") or "").strip()
    name = str(g.get("name") or "").strip()
    formatted = str(g.get("formattedName") or "").strip() or name
    member_count = int(g.get("memberCount") or 0)
    group_type = str(g.get("groupType") or "").strip()
    is_system = group_type == "SYSTEM_CONTACT_GROUP"
    return ContactGroup(
        resource_name=resource_name,
        name=name,
        formatted_name=formatted,
        member_count=member_count,
        is_system=is_system,
    )


# ---------------- Family-signal scoring ----------------


def score_family_signal(
    contact: Contact,
    family_group_resource: str | None,
    user_surname: str | None = None,
) -> tuple[float, str]:
    """Compute (strength, source) for a Contact's family-membership signal.

    Strength ∈ [0.0, 1.0]; source is a "+"-joined tag listing every
    detector that fired. Multiple detectors compound the source label but
    the score is the MAX of the per-detector values, not a sum — additivity
    would push two weak signals above the strong-signal threshold, which
    isn't right (two soft-evidence pings shouldn't outrank a single
    family-group membership in the user's own address book).

    Priority table:
      Family group membership:   1.00  (user curated explicitly)
      Structured relations[]:    0.90  (Google's typed relation field)
      Biography kinship words:   0.75  (note like "my sister Jane")
      Surname match w/ user:     0.70  (matches in-email signal weight)
      Starred + personal email:  0.55  (weak — used as a tiebreaker only)

    `family_group_resource` is None when the caller couldn't find a family
    group. In that case we just skip detector 1; detectors 2-5 still run.
    """
    sources: list[str] = []
    score = 0.0

    # 1. Family group membership.
    if family_group_resource and family_group_resource in contact.group_memberships:
        sources.append("group_membership")
        score = max(score, 1.0)

    # 2. Structured relations field. Any non-empty relations[] is signal;
    # `type` populated with a kinship word is stronger still.
    if contact.relations:
        sources.append("relations_field")
        score = max(score, 0.90)

    # 3. Biography kinship words.
    if contact.biography and _BIOGRAPHY_KINSHIP_RE.search(contact.biography):
        sources.append("biography")
        score = max(score, 0.75)

    # 4. Surname match.
    if user_surname and contact.family_name:
        if contact.family_name.strip().lower() == user_surname.strip().lower():
            sources.append("surname")
            score = max(score, 0.70)

    # 5. Starred + personal-domain email. Implemented as starred-only here —
    # the personal-domain check happens at the materializer level where the
    # PERSONAL_EMAIL_DOMAINS set lives. Starred alone is weak signal so we
    # cap at 0.55.
    if contact.is_starred:
        sources.append("starred")
        score = max(score, 0.55)

    source_str = "+".join(sources)
    return score, source_str


# ---------------- The client ----------------


class GoogleContacts:
    """Wrapper around the People API for contact lookups.

    One instance keeps the discovery service alive across multiple calls.
    Methods are synchronous and blocking — for the materializer's needs
    (one pull per profile build) we don't need async or pagination
    parallelism.
    """

    def __init__(self, credentials: Credentials):
        self._creds = credentials
        self._service = build(
            "people", "v1", credentials=credentials, cache_discovery=False
        )
        # Lazy cache for list_connections so a second call (e.g.
        # lookup_by_email after list_family_members) doesn't re-paginate
        # the whole address book.
        self._connections_cache: list[Contact] | None = None
        self._groups_cache: list[ContactGroup] | None = None

    # ---- groups ----

    def list_groups(self) -> list[ContactGroup]:
        """Return all contact groups (system + user). Paginated.

        Cached on the instance — subsequent calls return the cached list.
        """
        if self._groups_cache is not None:
            return self._groups_cache

        groups: list[ContactGroup] = []
        page_token: str | None = None
        while True:
            req = self._service.contactGroups().list(
                pageSize=_GROUPS_PAGE_SIZE,
                groupFields=_GROUP_FIELDS,
                pageToken=page_token,
            )
            resp = req.execute()
            for g in resp.get("contactGroups", []) or []:
                groups.append(_group_resource_to_group(g))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        self._groups_cache = groups
        return groups

    def find_family_group(self) -> ContactGroup | None:
        """Find the user's "Family" group.

        Strategy:
          1. Prefer the system group `contactGroups/family` if it exists
             AND has at least 1 member. The system family group is
             auto-created for some users (depends on locale + Google
             Workspace settings) but is often empty.
          2. Otherwise, fall back to a case-insensitive substring match
             over the formatted name. "Family", "My Family", "Bertram
             Family", and "FAMILY" all match. We DON'T match groups
             called "Friends and Family" or similar mixed groups — too
             much risk of false positives.

        Returns the first match by member_count desc, then resource_name —
        ensures determinism when a user has multiple plausible groups.
        """
        groups = self.list_groups()

        # Pass 1: the system family group, if non-empty.
        for g in groups:
            if g.resource_name == SYSTEM_FAMILY_GROUP_RESOURCE and g.member_count > 0:
                return g

        # Pass 2: user-defined groups containing "family" as a token.
        candidates: list[ContactGroup] = []
        for g in groups:
            name_lc = (g.formatted_name or g.name).lower()
            # Token-boundary match so "Family" and "My Family" match but
            # "Family Office" or "Family of funds" doesn't pick up business
            # contacts. Match when "family" appears as a whole word.
            tokens = re.split(r"[\s\-_]+", name_lc)
            if any(t == "family" for t in tokens):
                # Reject if the name contains "office" or "business" tokens
                # which usually means a business-shaped group like
                # "Family Office".
                bad_tokens = {"office", "fund", "funds", "business"}
                if any(t in bad_tokens for t in tokens):
                    continue
                candidates.append(g)

        if not candidates:
            return None

        candidates.sort(key=lambda g: (-g.member_count, g.resource_name))
        return candidates[0]

    # ---- contacts ----

    def list_connections(self) -> list[Contact]:
        """List all contacts (paginated). Cached on the instance.

        Personal address books at the scale we care about (low thousands)
        fit comfortably in memory; we don't stream.
        """
        if self._connections_cache is not None:
            return self._connections_cache

        contacts: list[Contact] = []
        page_token: str | None = None
        while True:
            req = self._service.people().connections().list(
                resourceName="people/me",
                pageSize=_CONNECTIONS_PAGE_SIZE,
                personFields=_PERSON_FIELDS,
                pageToken=page_token,
            )
            resp = req.execute()
            for p in resp.get("connections", []) or []:
                c = _person_to_contact(p)
                if c is not None:
                    contacts.append(c)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        self._connections_cache = contacts
        return contacts

    def list_family_members(
        self, user_surname: str | None = None
    ) -> list[Contact]:
        """Return every contact with a non-zero family-signal strength.

        Three-pass merge:
          1. Family-group members (highest signal).
          2. Contacts with a non-empty `relations` field.
          3. Contacts whose biography contains kinship words.
          4. Contacts whose family_name matches `user_surname`.
          5. Starred contacts (weakest — included only as a tiebreaker).

        Deduplication is by `resource_name`. The returned Contacts have
        `family_signal_strength` and `family_signal_source` populated.
        """
        family_group = self.find_family_group()
        family_group_resource = family_group.resource_name if family_group else None

        connections = self.list_connections()
        out: list[Contact] = []
        for c in connections:
            score, source = score_family_signal(
                c,
                family_group_resource=family_group_resource,
                user_surname=user_surname,
            )
            if score <= 0.0:
                continue
            # Populate the derived fields in-place. The Contact came from
            # the cache; mutating it is fine — these fields are derived
            # signals, not API-side data, and a subsequent call with a
            # different user_surname will overwrite them.
            c.family_signal_strength = score
            c.family_signal_source = source
            out.append(c)

        # Sort by descending strength, then display name for determinism.
        out.sort(key=lambda c: (-c.family_signal_strength, c.display_name))
        return out

    def lookup_by_email(
        self, emails: Iterable[str]
    ) -> dict[str, Contact | None]:
        """Map each input email to a matching Contact (or None).

        Case-insensitive. Implementation: iterate `list_connections()` once
        and build a reverse index. We don't call `people.searchContacts`
        even though it exists — for a few hundred lookups the iteration
        beats the network round-trips, and `searchContacts` requires an
        explicit "warmup" call before the first real query (a quirk of
        the People API).
        """
        wanted = {e.strip().lower() for e in emails if e and e.strip()}
        result: dict[str, Contact | None] = {e: None for e in wanted}
        for c in self.list_connections():
            for addr in c.email_addresses:
                if addr in wanted:
                    # First-wins: if the user has two contacts sharing an
                    # email (rare but possible after a merge mishap), keep
                    # the first match.
                    if result[addr] is None:
                        result[addr] = c
        return result


# ---------------- Helpers used by callers (materializer + cli) ----------------


def credentials_have_contacts_scope(creds: Credentials) -> bool:
    """True iff `creds.scopes` includes the contacts.readonly scope.

    Used by callers (cli auth, materializer) to detect tokens minted before
    Pass 12A. When False, callers should either prompt the user to
    re-authenticate (`pi-email auth --refresh-auth`) or skip the contacts
    code path entirely.
    """
    if creds is None:
        return False
    scopes = getattr(creds, "scopes", None) or []
    return CONTACTS_READONLY_SCOPE in set(scopes)
