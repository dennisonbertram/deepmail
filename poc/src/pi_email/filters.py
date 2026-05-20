"""Bulk-mail / marketing / list-managed message detection.

Why this lives here:
  Against a real 121K-message inbox the naive entity extractor turned every
  capitalized phrase in a newsletter body ("Series A", "Y Combinator",
  "Reddit Ads") into a candidate person/family member. ~60-80% of the false
  positives came from list-managed mail. This module is the cheap, defensive
  pre-filter: if a message looks like marketing / newsletter / system mail,
  we still keep it in the corpus (we may want to reference it later) but
  downstream extractors short-circuit on `Message.is_bulk` and refuse to
  emit entities from it.

Two surfaces:

  is_bulk_sender(from_addr) -> bool
      Pattern-match the local-part and domain of the From: address.

  is_bulk_message(headers, from_addr, user_emails=None) -> bool
      Sender check OR RFC-2369/3834-style header signals
      (List-Unsubscribe, List-Id, Precedence, Auto-Submitted).

Both are tolerant of empty/None input — they return False rather than
raise, so callers can wire them in without try/except.

Design notes:
  - Patterns are intentionally conservative. The spec calls out "better to
    let a borderline newsletter through than to silently filter a personal
    email". When in doubt, leave the pattern off.
  - Local-part patterns use `^foo\\b` so they match `noreply@` and
    `noreply+x@` but NOT `nicholas@` or a real person whose first name
    happens to start with `info`. The `\\b` boundary is what saves us from
    "infobright@" or similar false-positives — capture them later if real
    examples surface.
  - Domain patterns mostly target known mass-mail platforms — Substack,
    Mailchimp, SendGrid, Mailgun, Constant Contact, etc. They are matched
    by suffix so subdomains (e.g. `email.list-manage.com`) are covered.
  - All comparisons are case-insensitive — From: lines on the wire are
    canonicalized by mail servers but we don't trust that.

Pass-8C exemptions (loosened against Run-7's 99/100 bulk-filter rate on
seed iters 1+2):

  1. PERSONAL_EMAIL_DOMAINS exemption. A sender at a consumer email provider
     (gmail.com, icloud.com, etc.) is a human, not bulk — overrides BOTH
     the weak local-part patterns (`team@`, `info@`) AND the header signals
     (List-Unsubscribe / List-Id / Auto-Submitted). The exception is the
     strict "system" local-part patterns (noreply, mailer-daemon, ...) which
     still fire even at personal domains.
  2. user_emails exemption. If the From: matches one of the user's known
     addresses, the message is THEIR outbound mail — never bulk regardless
     of headers. Plumbed as an optional parameter; the caller (gmail_searcher /
     loop) is not yet wiring it through in this pass — Fix 1 covers the
     bulk of the recall hit.
  3. List-Unsubscribe alone is no longer sufficient. Modern mail clients
     (Google Calendar, many auto-confirmations) attach `List-Unsubscribe`
     without `List-Id` or `Precedence: bulk|list`. RFC 8058 + RFC 2369
     properly-marked list mail carries multiple headers; require pairing.

Pass-17A exemption:

  4. `calendar-notification@google.com` (and a handful of related Google
     Calendar mailers) are NOT bulk — they carry the user's accepted
     invites and the attendee list of every family event they showed up
     for. We mine the subject + body in `calendar_email_parser.py`. The
     filter check here ensures these messages survive into the corpus
     and reach the parser; without the exemption the bulk filter would
     drop them (the From: local-part starts with `calendar-` which is
     close enough to `calendar`-prefix patterns in some configurations,
     and Google attaches `List-Unsubscribe` headers to many of them).
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Personal-domain list
# ---------------------------------------------------------------------------

# Consumer / free email providers. A `From:` at one of these domains is a
# human, not bulk — Pass 8C exemption.
#
# DESIGN CHOICE — duplicated from `frontier.py` rather than imported. Two
# considerations argued for duplication over a shared import:
#   1. The list is small (~30 entries) and stable; drift risk is low.
#   2. `filters.py` is intentionally a small defensive module that only
#      imports `re`. Pulling in `frontier.py` would chain in numpy (~30MB
#      import cost) and the entire heap/frontier machinery just to read a
#      30-element constant. That coupling would be backwards: `filters.py`
#      is a leaf utility, `frontier.py` is policy machinery.
#
# DRIFT RISK acknowledged: if `frontier.PERSONAL_EMAIL_DOMAINS` ever gains
# a new provider, this copy must be updated in lock-step. A future refactor
# could extract both copies into a shared `pi_email.constants` (or similar)
# module — out-of-scope for Pass 8C.
PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com",
    "icloud.com", "me.com", "mac.com",
    "yahoo.com", "yahoo.co.uk", "yahoo.ca", "yahoo.fr", "yahoo.de",
    "hotmail.com", "outlook.com", "live.com", "msn.com",
    "aol.com", "protonmail.com", "proton.me",
    "fastmail.com", "fastmail.fm",
    "tutanota.com", "tuta.io",
    "zoho.com",
    "duck.com",
}


# ---------------------------------------------------------------------------
# Calendar-notification exemption (Pass 17A)
# ---------------------------------------------------------------------------

# Local-parts that, paired with `CALENDAR_NOTIFICATION_DOMAINS`, mark a
# message as a Google Calendar notification. These messages are never
# bulk — they're the user's accepted invites + family event attendee
# lists, mined by `calendar_email_parser.py`.
CALENDAR_NOTIFICATION_LOCAL_PARTS: set[str] = {
    "calendar-notification",
    "calendar-noreply",
    "calendar",
}

# Suffix-matched domain set for the Calendar mailer exemption. Kept
# narrow on purpose — overly broad coverage would re-introduce bulk
# false-positive risk.
CALENDAR_NOTIFICATION_DOMAINS: set[str] = {
    "google.com",
    "calendar.google.com",
}


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Local-part patterns that signal bulk/marketing/system mail.
#
# Anchored to start (^) with a word-boundary (\b) so we match `noreply@`,
# `no-reply+abc@`, and `info-team@` but NOT names like `infomercial@` (which
# wouldn't be a real local-part anyway) or `hellow@`. A bare `info@` /
# `team@` / `hi@` is a deliberate aggressive flag — companies use them for
# transactional + marketing, almost never for one-to-one human mail.
BULK_LOCAL_PATTERNS = [
    r"^noreply\b",
    r"^no-reply\b",
    r"^donotreply\b",
    r"^newsletter\b",
    r"^newsletters\b",
    r"^digest\b",
    r"^notifications?\b",
    r"^updates?\b",
    r"^announcements?\b",
    r"^marketing\b",
    r"^info\b",
    r"^hello\b",
    r"^team\b",
    r"^hi\b",
    r"^admin\b",
    r"^admins\b",
    r"^reminders?\b",
    r"^support\b",
    r"^contact\b",
    r"^press\b",
    r"^media\b",
    r"^sales\b",
    r"^billing\b",
    r"^accounts?\b",
    r"^security\b",
    r"^onboarding\b",
    r"^editorial\b",
    r"^community\b",
    r"^events?\b",
    r"^postmaster\b",
    r"^mailer-daemon\b",
    r"^bounces?\b",
]

# Subset of BULK_LOCAL_PATTERNS that signal UNAMBIGUOUS automated/system mail —
# these still fire even at a personal-domain sender (Pass 8C Fix 1 exception).
# The rationale: `noreply@gmail.com` (unlikely but possible) is a service
# address; `team@gmail.com` (a small team's shared gmail) is a human group.
# Strict locals include only the ones with no plausible interactive-human use.
STRICT_SYSTEM_LOCAL_PATTERNS = [
    r"^noreply\b",
    r"^no-reply\b",
    r"^donotreply\b",
    r"^postmaster\b",
    r"^mailer-daemon\b",
    r"^bounces?\b",
    r"^newsletter\b",
    r"^newsletters\b",
    r"^mailer\b",
]


# Domain suffix patterns suggesting mass-mail platforms / marketing-cloud
# tenant domains. These are conservative: a hit here is reliable enough that
# we suppress entity extraction even if no list-headers are present.
BULK_DOMAIN_PATTERNS = [
    # Substack — primary author domain + transactional subdomain.
    r"\.substack\.com$",
    r"^substack\.com$",
    # Mailchimp + its sending infrastructure.
    r"\.mailchimpapp\.com$",
    r"\.mailchimp\.com$",
    # Generic transactional / mass-mail senders.
    r"\.sendgrid\.net$",
    r"\.mandrillapp\.com$",
    r"\.mailgun\.org$",
    r"\.amazonses\.com$",
    # Mailchimp's list-rotation tenant domain.
    r"\.list-manage\.com$",
    # Constant Contact.
    r"\.constantcontact\.com$",
    r"\.ccsend\.com$",
    # HubSpot marketing.
    r"\.hubspotmail\.net$",
    # Salesforce Marketing Cloud — careful: this targets the *marketing*
    # tenant domain only; app emails come from {tenant}.my.salesforce.com
    # which the catchall above won't match (suffix differs).
    r"\.salesforce\.com$",
    # Intercom outbound.
    r"\.intercom-mail\.com$",
    # Ad platforms.
    r"^redditads\.com$",
    # Luma + Eventbrite event-mail.
    r"\.luma-mail\.com$",
    r"^luma-mail\.com$",
    r"\.eventbritemail\.com$",
    # Facebook outbound.
    r"\.fbsbx\.com$",
]


# Compile once at import. Each compiled pattern is case-insensitive — the
# raw strings above don't bother with [Aa] alternation.
_LOCAL_RES = [re.compile(p, re.IGNORECASE) for p in BULK_LOCAL_PATTERNS]
_STRICT_SYSTEM_LOCAL_RES = [
    re.compile(p, re.IGNORECASE) for p in STRICT_SYSTEM_LOCAL_PATTERNS
]
_DOMAIN_RES = [re.compile(p, re.IGNORECASE) for p in BULK_DOMAIN_PATTERNS]


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------

# Pull `addr@domain` out of `Some Name <addr@domain>` style headers. We
# deliberately don't use `email.utils.parseaddr` because it's permissive in
# ways that bite us on malformed real-world headers — a stray angle bracket
# with no `@` is handed back as the whole display name. The regex is small
# enough that the trade-off is fine.
_ANGLE_ADDR_RE = re.compile(r"<([^<>@\s]+@[^<>@\s]+)>")
_BARE_ADDR_RE = re.compile(r"([^\s<>,;]+@[^\s<>,;]+)")


def _extract_address(from_addr: str | None) -> str:
    """Pull the `local@domain` substring out of a `From:` header value.

    Handles:
      - `Alice <alice@example.com>` -> `alice@example.com`
      - `alice@example.com`         -> `alice@example.com`
      - `  Alice  Smith  ` (no @)   -> `""`
      - `None` / `""`               -> `""`

    Returns lowercased address; empty string on parse failure.
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


def _split_local_domain(addr: str) -> tuple[str, str]:
    """Split `local@domain` into (local, domain), both lowercased.

    Returns ("", "") for malformed input. We split on the LAST `@` so an
    address with an `@` in the local part (rare but RFC-legal when quoted)
    still picks up the right-hand domain.
    """
    if not addr or "@" not in addr:
        return ("", "")
    local, _, domain = addr.rpartition("@")
    return (local.strip().lower(), domain.strip().lower())


def _is_strict_system_local(local: str) -> bool:
    """True iff the local-part matches an unambiguous system-mailer pattern
    (noreply@, mailer-daemon@, postmaster@, etc.).

    These patterns indicate bulk mail regardless of domain — even at a
    consumer email provider (`noreply@gmail.com`) the address is a service
    address, not a human. Other role-like patterns (`team@`, `info@`,
    `hello@`) are intentionally NOT in this set; they are considered weak
    signals overridable by the personal-domain exemption (Pass 8C Fix 1).
    """
    if not local:
        return False
    for rx in _STRICT_SYSTEM_LOCAL_RES:
        if rx.search(local):
            return True
    return False


def is_calendar_notification_sender(from_addr: str | None) -> bool:
    """True iff `from_addr` is a Google Calendar notification mailer.

    Pass-17A exemption: messages from these mailers are NOT bulk — they
    carry the user's accepted invites + attendee lists, mined by
    `calendar_email_parser.py`. Match logic mirrors the parser's own
    `is_calendar_notification_sender`:

      1. Local-part in `CALENDAR_NOTIFICATION_LOCAL_PARTS`.
      2. Domain (suffix-matched) in `CALENDAR_NOTIFICATION_DOMAINS`.

    Tolerates None / empty / malformed input — returns False.

    Note: we intentionally do NOT import `calendar_email_parser` here —
    the parser already imports from this module's siblings, so the
    one-way constant duplication keeps the dependency graph acyclic.
    """
    addr = _extract_address(from_addr)
    if not addr or "@" not in addr:
        return False
    local, _, domain = addr.rpartition("@")
    local = local.lower()
    domain = domain.lower()
    if local not in CALENDAR_NOTIFICATION_LOCAL_PARTS:
        return False
    for d in CALENDAR_NOTIFICATION_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return True
    return False


def _is_personal_domain(domain: str) -> bool:
    """True iff `domain` is in `PERSONAL_EMAIL_DOMAINS` (case-insensitive).

    Domain matching is exact — `mail.gmail.com` is NOT a personal domain
    (it's a Google infrastructure subdomain that personal addresses never
    route from). We trust the exact-domain match because the set is
    curated to consumer endpoints.
    """
    if not domain:
        return False
    return domain.strip().lower() in PERSONAL_EMAIL_DOMAINS


def _normalize_user_emails(user_emails: set[str] | None) -> set[str]:
    """Pull `user_emails` into a lowercased + parsed-address set.

    Each entry can be a bare `addr@domain` OR a `Name <addr@domain>` header-
    formatted string; we run them through `_extract_address` so callers can
    pass either form. Empty / unparseable entries are dropped.
    Returns an empty set on None / empty input — never raises.
    """
    if not user_emails:
        return set()
    out: set[str] = set()
    for raw in user_emails:
        if not raw:
            continue
        addr = _extract_address(raw)
        if addr:
            out.add(addr)
        else:
            # Allow bare-string entries that survived _extract_address (e.g.
            # a malformed entry); store lowercased for direct compare. This
            # is the substring fallback case mentioned in the docstring.
            stripped = raw.strip().lower()
            if stripped:
                out.add(stripped)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_bulk_sender(from_addr: str | None) -> bool:
    """Return True if the From: address looks like bulk/marketing mail.

    Checks the local-part against `BULK_LOCAL_PATTERNS` and the domain
    against `BULK_DOMAIN_PATTERNS`. Either match wins.

    Pass 8C Fix 1: senders at PERSONAL_EMAIL_DOMAINS are humans by
    definition. The only escape hatch is the strict system local-part set
    (noreply@, mailer-daemon@, ...) — `noreply@gmail.com` is still bulk,
    but `team@gmail.com` and `info@gmail.com` are not.

    Tolerates None, empty strings, and headers without a parseable address
    (returns False in those cases — better to let a borderline message
    through than mis-flag a personal email).
    """
    addr = _extract_address(from_addr)
    if not addr:
        return False
    local, domain = _split_local_domain(addr)

    # Pass-17A — Google Calendar notification mailers are never bulk.
    # We need their subjects + bodies to flow through to
    # `calendar_email_parser.py`. Check BEFORE the strict-system local
    # rule because `calendar-noreply` would otherwise match `noreply`-
    # adjacent patterns. The exemption is narrow (Google domains
    # only).
    if is_calendar_notification_sender(from_addr):
        return False

    # Fix 1 — personal-domain exemption.
    if _is_personal_domain(domain):
        # Strict system local-part patterns still fire (noreply@gmail.com).
        # Weak role patterns (team@, info@, ...) are SKIPPED.
        return _is_strict_system_local(local)

    if local:
        for rx in _LOCAL_RES:
            if rx.search(local):
                return True

    if domain:
        for rx in _DOMAIN_RES:
            if rx.search(domain):
                return True

    return False


# Headers that, when set to certain values, mark a message as bulk/list mail.
# RFC 2369 (List-Unsubscribe / List-Id), RFC 3834 (Auto-Submitted), and the
# pre-RFC `Precedence: bulk|list|junk` convention all converge here.
#
# Pass 8C Fix 3: `List-Unsubscribe` alone is no longer a sufficient
# signal — many legitimate auto-confirmations and calendar invites carry it
# without being list-managed. RFC 8058 + RFC 2369 properly-marked bulk
# mail carries MULTIPLE headers. We now require `List-Id` OR a
# `Precedence: bulk|list|junk` to flag.
_PRECEDENCE_BULK_VALUES = {"bulk", "list", "junk"}


def _header_get_ci(headers: dict[str, str] | None, key: str) -> str | None:
    """Case-insensitive lookup against a header dict.

    RFC 2822 header names are case-insensitive, so we don't trust the
    caller to have normalized them. Returns None if absent.
    """
    if not headers:
        return None
    needle = key.lower()
    for k, v in headers.items():
        if k and k.lower() == needle:
            return v
    return None


def is_bulk_message(
    headers: dict[str, str] | None,
    from_addr: str | None,
    user_emails: set[str] | None = None,
) -> bool:
    """Return True if `headers` or `from_addr` indicate bulk / list / system mail.

    Pass 8C exemption order (each short-circuits to False):

      1. `from_addr` matches one of the user's own emails (`user_emails`).
         The user's own outbound mail is never bulk; their threads are
         load-bearing for family discovery.
      2. `from_addr` is at a personal email provider (gmail.com, icloud.com,
         ...) AND its local-part is NOT a strict system pattern (noreply@,
         mailer-daemon@, ...). Personal-domain humans get an unconditional
         pass — overrides header signals (List-Unsubscribe, Auto-Submitted,
         etc.) because real personal mail (calendar invites, OOO
         auto-replies) routinely carries those headers.

    After exemptions, returns True if any of:
      - `List-Id` header is present (and non-empty) — strong list-managed signal
      - `Precedence` header value is `bulk`, `list`, or `junk`
      - `List-Unsubscribe` is present AND (List-Id OR Precedence: bulk|list|junk)
        is also present (Fix 3 — RFC 8058 spirit: multiple list headers)
      - `Auto-Submitted` header value is anything other than `no`
      - `is_bulk_sender(from_addr)` returns True

    Header lookups are case-insensitive on key. Values are trimmed but
    otherwise taken as-is.

    All inputs are optional / tolerant — missing or None inputs return False.

    `user_emails` is an OPTIONAL parameter. Callers (gmail_searcher) currently
    default to None until a future wire-through; the personal-domain
    exemption (Fix 1) covers the bulk of the recall hit by itself.
    """
    # Pre-parse the sender once for the exemption checks below.
    addr = _extract_address(from_addr)

    # Pass-17A — Google Calendar notification mailers are never bulk.
    # These messages typically carry `List-Unsubscribe` AND look like
    # `noreply`-style senders (`calendar-notification@google.com`), so
    # without this check both the header signal AND the bulk-sender
    # heuristic would flag them. The exemption runs FIRST so neither
    # path fires.
    if is_calendar_notification_sender(from_addr):
        return False

    # Fix 2 — user's own outbound mail is never bulk.
    # Compare the parsed `addr@domain` against the normalized user_emails
    # set. Case-insensitive (both sides lowercased).
    if addr:
        normalized_users = _normalize_user_emails(user_emails)
        if normalized_users and addr in normalized_users:
            return False

    # Fix 1 — personal-domain sender exemption (with strict-local override).
    # This intentionally runs BEFORE the header signals so a personal-domain
    # sender's calendar invite (which carries List-Unsubscribe) and OOO
    # auto-reply (Auto-Submitted: auto-replied) both pass through.
    if addr:
        local, domain = _split_local_domain(addr)
        if _is_personal_domain(domain) and not _is_strict_system_local(local):
            return False

    # Fix 3 — header-based list detection. Three branches:
    #   * `List-Id` alone is strong enough (the explicit RFC 2369 marker).
    #   * `Precedence: bulk|list|junk` alone is strong enough (legacy convention).
    #   * `List-Unsubscribe` alone is NO LONGER enough — calendar invites,
    #     OAuth one-time codes, and many auto-confirmations all carry it.
    #     It only counts as bulk when paired with List-Id or Precedence
    #     above (which would already have fired). Effectively, the standalone
    #     case here is a no-op — left as documentation of the previous
    #     behaviour for the next maintainer.
    list_id = _header_get_ci(headers, "list-id")
    if list_id is not None and list_id.strip():
        return True

    prec = _header_get_ci(headers, "precedence")
    if prec is not None and prec.strip().lower() in _PRECEDENCE_BULK_VALUES:
        return True

    # (Pass 8C Fix 3: `List-Unsubscribe` alone is intentionally NOT checked
    # here. If a message has List-Unsubscribe + List-Id it already returned
    # True above; if it has List-Unsubscribe + Precedence:bulk it already
    # returned True above; if it has List-Unsubscribe alone, that's a weak
    # signal and we let the sender heuristic decide.)

    auto = _header_get_ci(headers, "auto-submitted")
    if auto is not None:
        v = auto.strip().lower()
        # RFC 3834: only `no` means "interactive human"; anything else
        # (`auto-generated`, `auto-replied`, `auto-notified`) is a bot.
        # Personal-domain auto-replies (OOO from a family member) already
        # returned False above via Fix 1, so reaching this branch implies
        # a non-personal-domain sender.
        if v and v != "no":
            return True

    if is_bulk_sender(from_addr):
        return True

    return False
