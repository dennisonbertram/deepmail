"""Command-line entry point for the pi_email POC.

Subcommands:
  auth     — Run the Google OAuth installed-app flow; persist tokens.
  whoami   — Confirm auth by calling `users.getProfile` against Gmail.
  query    — Run a Gmail search query end-to-end (list + batch-fetch).
  run      — Run the full iterative-expansion loop against fixtures or Gmail.

Output discipline (per task spec):
  * Status / log lines go to stderr so they don't pollute pipelines.
  * Subcommand results go to stdout (so `deep-email query "..." | grep` works).
  * `-v / --verbose` enables per-iteration logs in `run`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import click

from .calendar_evidence import (
    CALENDAR_READONLY_SCOPE,
    credentials_have_calendar_scope,
)
from .contacts import (
    CONTACTS_READONLY_SCOPE,
    credentials_have_contacts_scope,
)
from .oauth import OAuthConfig, acquire_credentials, refresh_if_needed
from .token_store import TokenStore


HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent.parent  # src/pi_email/cli.py -> poc/
DEFAULT_FIXTURES_DIR = POC_ROOT / "fixtures" / "family_corpus"
DEFAULT_PROFILES_DIR = POC_ROOT / "profiles"


# ---------------- helpers ----------------


def _echo_err(msg: str) -> None:
    """Status / progress line — goes to stderr so stdout stays parseable."""
    click.echo(msg, err=True)


def _load_creds_or_die() -> "Credentials":  # noqa: F821 - forward-string
    """Load saved creds, refresh if needed, or exit non-zero with a hint."""
    store = TokenStore()
    creds = store.load()
    if creds is None:
        _echo_err("Not authenticated -- run 'deep-email auth' (or 'deep-email setup').")
        sys.exit(2)
    try:
        refresh_if_needed(creds)
    except Exception as exc:
        _echo_err(
            f"Token refresh failed ({exc}). Re-run 'deep-email auth' to re-consent."
        )
        sys.exit(2)
    # Persist any refreshed access token so subsequent commands reuse it.
    try:
        store.save(creds)
    except Exception as exc:  # pragma: no cover - best-effort
        _echo_err(f"warning: could not persist refreshed token: {exc}")
    return creds


def _load_dotenv_quietly() -> None:
    """Best-effort .env loader. .env is searched upward from cwd, matching
    the convention used by `OAuthConfig.from_env()`."""
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


# ---------------- root group ----------------


@click.group(
    help=(
        "pi-email: iterative-expansion email-understanding POC.\n\n"
        "Authenticate with Gmail, run search queries, or kick off the full "
        "expansion loop against your real mailbox or the bundled fixture corpus."
    )
)
@click.version_option("0.2.0", prog_name="pi-email")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable per-iteration log dump (default: terse).",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Top-level CLI group."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _load_dotenv_quietly()


# ---------------- auth ----------------


@main.command(
    help=(
        "Run the Google OAuth installed-app flow and persist the access + "
        "refresh tokens to your platform's user-data directory."
    )
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run the consent flow even if a token already exists.",
)
@click.option(
    "--refresh-auth",
    "refresh_auth",
    is_flag=True,
    help=(
        "Force re-consent to pick up new OAuth scopes. Use after Pass 12A "
        "upgraded the default scope set with contacts.readonly and Pass 14B "
        "added calendar.readonly: existing tokens minted before those passes "
        "are missing the new scopes and the materializer's People API + "
        "Calendar API lookups will be skipped until you re-auth."
    ),
)
@click.pass_context
def auth(ctx: click.Context, force: bool, refresh_auth: bool) -> None:
    """Acquire and persist OAuth credentials for Gmail + Contacts."""
    try:
        config = OAuthConfig.from_env()
    except RuntimeError as exc:
        _echo_err(str(exc))
        sys.exit(2)

    store = TokenStore()

    if not force and not refresh_auth:
        existing = store.load()
        if existing is not None:
            # Pass 12A: check the existing token has the contacts scope. A
            # token minted before Pass 12A is missing it; we trigger a
            # re-consent automatically so the user doesn't have to remember
            # the --refresh-auth flag.
            # Pass 14B: same for calendar.readonly.
            if not credentials_have_contacts_scope(existing):
                _echo_err(
                    "Existing token is missing the contacts.readonly scope. "
                    "Re-running consent flow to upgrade — Pass 12A integrates "
                    "Google Contacts as a family-evidence source."
                )
            elif not credentials_have_calendar_scope(existing):
                _echo_err(
                    "Existing token is missing the calendar.readonly scope. "
                    "Re-running consent flow to upgrade — Pass 14B integrates "
                    "Google Calendar as a family-evidence source."
                )
            else:
                # If the access token is expired we still try to refresh it
                # without re-running the browser flow.
                try:
                    refresh_if_needed(existing)
                    store.save(existing)
                    _echo_err(f"Already authenticated; token at {store.path}")
                    return
                except Exception as exc:
                    _echo_err(
                        f"Existing token could not be refreshed ({exc}); "
                        "re-running consent flow."
                    )

    _echo_err("Opening browser for Google consent...")
    try:
        creds = acquire_credentials(config, open_browser=True)
    except Exception as exc:
        _echo_err(f"OAuth flow failed: {exc}")
        sys.exit(2)

    try:
        store.save(creds)
    except Exception as exc:
        _echo_err(f"Could not persist token: {exc}")
        sys.exit(2)

    _echo_err(f"Token saved to {store.path}")
    _echo_err("Done -- you can now use 'deep-email query' or 'deep-email run'.")


# ---------------- whoami ----------------


@main.command(
    help=(
        "Canary: load saved tokens and call Gmail's users.getProfile. Prints "
        "the authenticated email address plus total message / thread counts."
    )
)
@click.pass_context
def whoami(ctx: click.Context) -> None:
    """Confirm auth works end-to-end against Gmail."""
    creds = _load_creds_or_die()
    # Import here so `deep-email --help` doesn't pay the googleapiclient cost.
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    profile = service.users().getProfile(userId="me").execute()
    email = profile.get("emailAddress", "(unknown)")
    msgs = profile.get("messagesTotal", "?")
    threads = profile.get("threadsTotal", "?")

    click.echo(f"email:           {email}")
    click.echo(f"messages_total:  {msgs}")
    click.echo(f"threads_total:   {threads}")


# ---------------- contacts ----------------


@main.command(
    help=(
        "List family-shaped contacts in your Google address book. Useful for "
        "debugging what the People API client surfaces before kicking off "
        "the full expansion loop."
    )
)
@click.option(
    "--show-emails",
    is_flag=True,
    help="Show the first email of each contact (otherwise emails are redacted).",
)
@click.pass_context
def contacts(ctx: click.Context, show_emails: bool) -> None:
    """List family-group members + relations + biography-tagged contacts."""
    creds = _load_creds_or_die()
    if not credentials_have_contacts_scope(creds):
        _echo_err(
            "Token is missing the contacts.readonly scope. "
            "Run 'deep-email auth --refresh-auth' to re-consent."
        )
        sys.exit(2)

    from .contacts import GoogleContacts

    client = GoogleContacts(creds)

    groups = client.list_groups()
    fam = client.find_family_group()
    _echo_err(f"contact_groups_total: {len(groups)}")
    if fam is not None:
        _echo_err(
            f"family_group: {fam.resource_name} "
            f"name={fam.formatted_name!r} members={fam.member_count} "
            f"is_system={fam.is_system}"
        )
    else:
        _echo_err("family_group: (none found)")

    members = client.list_family_members()
    click.echo(f"family_shaped_contacts: {len(members)}")
    # Aggregate per signal source for the operator.
    by_source: dict[str, int] = {}
    for c in members:
        sources = (c.family_signal_source or "(none)").split("+")
        for s in sources:
            by_source[s] = by_source.get(s, 0) + 1
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        click.echo(f"  by_source {src:>22s}: {n}")

    # Preview, one line per contact.
    click.echo("\npreview:")
    for c in members[:50]:
        first = (
            (c.email_addresses[0] if c.email_addresses else "(no email)")
            if show_emails
            else "(email redacted)"
        )
        rels = ", ".join(
            f"{r.get('type','?')}:{r.get('person','?')}"
            for r in (c.relations or [])
        ) or "-"
        bio = (c.biography or "-").replace("\n", " ")
        if len(bio) > 80:
            bio = bio[:80] + "..."
        click.echo(
            f"  {c.family_signal_strength:.2f} {c.family_signal_source:<40s} "
            f"{c.display_name!r:<32s} email={first} relations={rels} bio={bio!r}"
        )
    if len(members) > 50:
        click.echo(f"  ... ({len(members) - 50} more not shown)")


# ---------------- calendar ----------------


@main.command(
    help=(
        "List family-shaped persons inferred from your Google Calendar — "
        "kinship-word event titles ('Vitus Birthday', 'Mom's anniversary') "
        "and attendees of family-themed events. Useful for debugging what "
        "the Calendar API client surfaces before kicking off the full "
        "expansion loop. Pass 14B."
    )
)
@click.option(
    "--days-back",
    type=int,
    default=365,
    show_default=True,
    help="Lookback window in days for calendar events.",
)
@click.option(
    "--show-emails",
    is_flag=True,
    help=(
        "Show the email address for attendee-derived persons "
        "(otherwise redacted)."
    ),
)
@click.pass_context
def calendar(ctx: click.Context, days_back: int, show_emails: bool) -> None:
    """List family-shaped calendar persons + their scored signal source."""
    creds = _load_creds_or_die()
    if not credentials_have_calendar_scope(creds):
        _echo_err(
            "Token is missing the calendar.readonly scope. "
            "Run 'deep-email auth --refresh-auth' to re-consent."
        )
        sys.exit(2)

    from .calendar_evidence import GoogleCalendar

    client = GoogleCalendar(creds)
    _echo_err(
        f"calendar: fetching events from primary calendar "
        f"(days_back={days_back})..."
    )
    persons = client.list_family_calendar_persons(days_back=days_back)

    click.echo(f"family_shaped_calendar_persons: {len(persons)}")
    # Aggregate per signal source for the operator.
    by_source: dict[str, int] = {}
    for p in persons:
        sources = (p.family_signal_source or "(none)").split("+")
        for s in sources:
            by_source[s] = by_source.get(s, 0) + 1
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        click.echo(f"  by_source {src:>30s}: {n}")

    # Preview, one line per person.
    click.echo("\npreview:")
    for p in persons[:50]:
        email_display = (
            (p.email or "(no email)") if show_emails else "(email redacted)"
        )
        titles = ", ".join(repr(t) for t in p.appears_in_titles[:3])
        if len(p.appears_in_titles) > 3:
            titles += f", ... (+{len(p.appears_in_titles) - 3} more)"
        click.echo(
            f"  {p.family_signal_strength:.2f} {p.family_signal_source:<48s} "
            f"{p.name!r:<32s} attendee={p.is_attendee} "
            f"events={p.attendee_event_count} email={email_display} "
            f"titles=[{titles}]"
        )
    if len(persons) > 50:
        click.echo(f"  ... ({len(persons) - 50} more not shown)")


# ---------------- query ----------------


@main.command(
    help=(
        "Run a Gmail search query and print the matches. Default --max is 50 "
        "to avoid burning quota on a first attempt; the cap can be raised "
        "once you've calibrated expectations."
    )
)
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--max",
    "max_results",
    type=int,
    default=50,
    show_default=True,
    help="Cap on messages fetched (max_results_per_query).",
)
@click.pass_context
def query(ctx: click.Context, query: tuple[str, ...], max_results: int) -> None:
    """Run one Gmail search; print one line per hit."""
    q = " ".join(query).strip()
    if not q:
        _echo_err("query: missing query string")
        sys.exit(2)

    creds = _load_creds_or_die()
    from .gmail_searcher import GmailQuotaExhausted, GmailSearcher

    searcher = GmailSearcher(creds, max_results_per_query=max_results)
    _echo_err(f"querying gmail: {q!r} (max={max_results})")
    try:
        batch = searcher.search_and_fetch(q)
    except GmailQuotaExhausted as exc:
        _echo_err(f"FATAL: {exc}")
        sys.exit(3)

    _echo_err(
        f"hits={len(batch.hits)} quota_units={batch.quota_units_used} "
        f"truncated={'yes' if batch.truncated else 'no'} "
        f"retries={batch.retry_count}"
    )
    if batch.error:
        _echo_err(f"non-fatal error: {batch.error}")

    for msg in batch.hits:
        click.echo(f"{msg.from_addr} | {msg.date} | {msg.subject}")


# ---------------- run ----------------


@main.command(
    help=(
        "Run the iterative-expansion loop end-to-end. Defaults to the bundled "
        "fixture corpus; pass --source gmail to drive the live Gmail searcher."
    )
)
@click.argument("seed", required=True)
@click.option(
    "--source",
    type=click.Choice(["fixture", "gmail"], case_sensitive=False),
    default="fixture",
    show_default=True,
    help="Which Searcher to drive the loop against.",
)
@click.option(
    "--max-corpus",
    type=int,
    default=100,
    show_default=True,
    help="Cap on total messages pulled when --source gmail.",
)
@click.option(
    "--max-per-query",
    "max_per_query",
    type=int,
    default=500,
    show_default=True,
    help=(
        "Per-Gmail-query cap on messages fetched (max_results_per_query). "
        "Lower this to sample wide queries faster and avoid burning quota; "
        "queries that hit the cap are flagged as truncated. "
        "Ignored when --source fixture."
    ),
)
@click.option(
    "--profiles-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_PROFILES_DIR,
    show_default=True,
    help="Output directory for the materialized profile.",
)
@click.option(
    "--fixtures-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_FIXTURES_DIR,
    show_default=True,
    help="Fixture corpus directory (only used when --source fixture).",
)
@click.option(
    "--force-mock",
    is_flag=True,
    help="Force the mock proposer even if ANTHROPIC_API_KEY is set.",
)
@click.option(
    "--skip-judge",
    "skip_judge",
    is_flag=True,
    help=(
        "Skip the LLM-as-final-judge step in the materializer. Every "
        "gathered candidate is written as a family member (legacy "
        "behavior). Useful for offline runs."
    ),
)
@click.option(
    "--strict-extract/--no-strict-extract",
    "strict_extract",
    default=None,
    help=(
        "Force strict or loose entity extraction (Pass 10). Defaults: "
        "--source gmail → loose (open the pipe; LLM judge filters), "
        "--source fixture → strict (preserves fixture demo expectations). "
        "Loose mode uses a 15-token adjacency window and no business-heavy "
        "suppression; strict uses the tighter 8-token window."
    ),
)
@click.pass_context
def run(
    ctx: click.Context,
    seed: str,
    source: str,
    max_corpus: int,
    max_per_query: int,
    profiles_dir: Path,
    fixtures_dir: Path,
    force_mock: bool,
    skip_judge: bool,
    strict_extract: bool | None,
) -> None:
    """Drive the ExpansionLoop end-to-end and write profiles/family.md."""
    verbose: bool = ctx.obj.get("verbose", False)

    on_log: Callable[[str], None] | None = None
    if verbose:
        on_log = _echo_err

    on_event = None
    if verbose:
        from .loop import IterationEvent

        def _on_event(e: IterationEvent) -> None:
            parent = f" (parent={e.parent_entity})" if e.parent_entity else ""
            _echo_err(
                f"  iter {e.n:2d}: q={e.query!r:<40s} "
                f"score={e.score:.2f} hits={e.hits:2d} "
                f"new_msgs={e.new_messages:2d} bulk_filtered={e.bulk_filtered:2d} "
                f"new_ents={e.new_entities:2d} "
                f"recap={e.recapture_rate:.2f} proposed={e.proposed} "
                f"frontier_size={e.frontier_size_after}{parent}"
            )

        on_event = _on_event

    src = source.lower()
    # Pass 10 — per-source default for strict-vs-loose entity extraction.
    # Run 9 evidence: the strict path only surfaced 7 candidates out of 278
    # entities for the LLM judge — too narrow. The live Gmail path now
    # defaults to loose extraction (open the pipe; LLM judge filters
    # business contacts downstream). Fixture stays strict so the bundled
    # demo expectations don't change. Explicit --strict-extract /
    # --no-strict-extract on the CLI overrides either default.
    if strict_extract is None:
        effective_strict = src == "fixture"
    else:
        effective_strict = strict_extract
    if src == "fixture":
        _echo_err(
            f"running loop against fixture corpus: {fixtures_dir} "
            f"(strict_extract={effective_strict})"
        )
        # max_per_query is ignored for fixture source (FilesystemSearcher has
        # no equivalent per-query truncation knob).
        from .loop import run_loop_and_materialize

        result, out_path = run_loop_and_materialize(
            fixtures_dir=fixtures_dir,
            seed=seed,
            profiles_dir=profiles_dir,
            on_event=on_event,
            on_log=on_log,
            force_mock=force_mock,
            skip_judge=skip_judge,
            strict_extract=effective_strict,
        )
    elif src == "gmail":
        _echo_err(
            f"running loop against gmail (max_corpus={max_corpus}, "
            f"max_per_query={max_per_query}, seed={seed!r}, "
            f"strict_extract={effective_strict})"
        )
        result, out_path = _run_loop_gmail(
            seed=seed,
            profiles_dir=profiles_dir,
            max_corpus=max_corpus,
            max_per_query=max_per_query,
            on_event=on_event,
            on_log=on_log,
            force_mock=force_mock,
            skip_judge=skip_judge,
            strict_extract=effective_strict,
        )
    else:  # pragma: no cover - click validates the choice
        _echo_err(f"unknown source: {source}")
        sys.exit(2)

    click.echo(f"iterations:        {len(result.iterations)}")
    click.echo(f"corpus_messages:   {len(result.corpus)}")
    click.echo(f"entities:          {len(result.entities)}")
    click.echo(f"stop_reason:       {result.stop_reason.rule} — {result.stop_reason.detail}")
    click.echo(f"queries_run:       {len(result.queries_run)}")
    click.echo(f"profile:           {out_path}")


def _run_loop_gmail(
    *,
    seed: str,
    profiles_dir: Path,
    max_corpus: int,
    max_per_query: int,
    on_event,
    on_log,
    force_mock: bool,
    skip_judge: bool = False,
    strict_extract: bool = False,
):
    """The `run --source gmail` path. Lives here so the fixture path doesn't
    pay the google-api import cost. Returns (LoopResult, profile-path).

    `max_corpus` is preserved for callers but the per-Gmail-query cap that
    actually bounds quota usage is `max_per_query`; the loop's corpus-byte
    cap is enforced separately by the Frontier.

    Before constructing the loop we call `users.getProfile` to learn the
    authenticated user's email address; this is plumbed through to the
    materializer so the user themselves doesn't appear as a "family
    member" in their own family profile. getProfile does NOT return a
    display name; we leave it as None and let the materializer recover
    the name heuristically from sent-mail From headers in the corpus.
    Follow-up: a clean way to get the canonical display name would be a
    People API lookup (`people.me`) but that requires an extra OAuth
    scope.
    """
    from googleapiclient.discovery import build

    from .gmail_searcher import GmailQuotaExhausted, GmailSearcher
    from .loop import ExpansionLoop
    from .materializer import write_family_profile
    from .proposer import Proposer

    creds = _load_creds_or_die()

    # Identify "you" so we can filter you out of your own family member list.
    user_self: dict | None = None
    try:
        gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile_result = gmail.users().getProfile(userId="me").execute()
        user_email = profile_result.get("emailAddress")
        if user_email:
            user_self = {
                "email": user_email.lower(),
                # getProfile doesn't surface a display name; the materializer
                # recovers it from From headers of the user's own sent mail.
                "display_name": None,
            }
    except Exception as exc:  # pragma: no cover - best-effort
        _echo_err(f"warning: getProfile failed ({exc}); self-filter disabled")

    searcher = GmailSearcher(creds, max_results_per_query=max_per_query)
    proposer = Proposer(force_mock=force_mock, on_log=on_log)
    loop = ExpansionLoop(
        searcher=searcher,
        proposer=proposer,
        on_event=on_event,
        on_log=on_log,
        user_self=user_self,
        strict_extract=strict_extract,
    )
    try:
        result = loop.run(seed)
    except GmailQuotaExhausted as exc:
        _echo_err(f"FATAL: {exc}")
        sys.exit(3)

    # Pass 14B: fetch the user's family-shaped calendar persons from the
    # Google Calendar API. Kinship-word event titles ("Vitus Birthday")
    # and attendees of family-themed events are extracted here BEFORE
    # materializing the profile so the materializer can use them as a
    # parallel evidence source to Pass 12A's contacts. We fetch even when
    # we can't yet plumb them all the way through — the operator log
    # below gives the user immediate visibility into what the calendar
    # source surfaces.
    family_calendar_persons: list | None = None
    if credentials_have_calendar_scope(creds):
        try:
            from .calendar_evidence import GoogleCalendar

            cal_client = GoogleCalendar(creds)
            family_calendar_persons = (
                cal_client.list_family_calendar_persons()
            )
            _echo_err(
                f"calendar: pulled {len(family_calendar_persons)} "
                f"family-shaped persons from the Calendar API"
            )
            # Per-source breakdown — helps the operator see where the
            # signals are coming from at a glance.
            by_source: dict[str, int] = {}
            for p in family_calendar_persons:
                for s in (p.family_signal_source or "").split("+"):
                    if s:
                        by_source[s] = by_source.get(s, 0) + 1
            for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
                _echo_err(f"  calendar by_source {src:>30s}: {n}")
        except Exception as exc:  # pragma: no cover - best-effort
            _echo_err(
                f"warning: Calendar API lookup failed ({exc}); proceeding "
                f"without calendar evidence"
            )
            family_calendar_persons = None
    else:
        _echo_err(
            "calendar: token missing calendar.readonly scope — skipping "
            "Calendar API integration. Run 'deep-email auth --refresh-auth' "
            "to enable calendar-evidence in the materializer."
        )

    # Pass 12A: fetch the user's family-shaped contacts from the People API
    # BEFORE materializing the profile. The materializer integrates them as
    # an evidence source — both attaching ContactEvidence to gathered
    # candidates and adding contact-only candidates the email search missed.
    family_contacts: list | None = None
    if credentials_have_contacts_scope(creds):
        try:
            from .contacts import GoogleContacts

            client = GoogleContacts(creds)
            # Pass the surname (if known) so the contact-side surname signal
            # fires too.
            surname: str | None = None
            if user_self and user_self.get("display_name"):
                toks = str(user_self["display_name"]).strip().split()
                if len(toks) >= 2:
                    surname = toks[-1]
            family_contacts = client.list_family_members(user_surname=surname)
            _echo_err(
                f"contacts: pulled {len(family_contacts)} family-shaped "
                f"contacts from the People API"
            )
        except Exception as exc:  # pragma: no cover - best-effort
            _echo_err(
                f"warning: People API lookup failed ({exc}); proceeding "
                f"without contact evidence"
            )
            family_contacts = None
    else:
        _echo_err(
            "contacts: token missing contacts.readonly scope — skipping "
            "People API integration. Run 'deep-email auth --refresh-auth' to "
            "enable contact-evidence in the materializer."
        )

    # Pass 14B: fold calendar-derived persons into the family_contacts list
    # using the contacts.Contact shape. This is the PRAGMATIC integration
    # path: the materializer's existing `family_contacts` plumbing (Pass 12A)
    # already attaches ContactEvidence to gathered candidates and admits
    # contact-only candidates the email search missed. By converting each
    # CalendarPerson into a Contact-shaped record we get calendar evidence
    # through the judge with zero changes to the materializer.
    if family_calendar_persons:
        calendar_shaped = _calendar_persons_to_contact_shapes(
            family_calendar_persons
        )
        if family_contacts is None:
            family_contacts = list(calendar_shaped)
        else:
            family_contacts = list(family_contacts) + list(calendar_shaped)
        _echo_err(
            f"calendar: folded {len(calendar_shaped)} calendar-derived "
            f"persons into family_contacts (total now "
            f"{len(family_contacts)})"
        )

    from .materializer import _slugify_query

    out_filename = _slugify_query(seed) + ".md"
    out_path = write_family_profile(
        out_path=profiles_dir / out_filename,
        corpus=result.corpus,
        entities=result.entities,
        seen_by_message=result.seen_by_message,
        seed=seed,
        stop_reason=result.stop_reason.rule,
        queries_run=result.queries_run,
        canonical_map=result.canonical_map,
        user_self=user_self,
        on_log=on_log,
        skip_judge=skip_judge,
    )
    return result, out_path


def _calendar_persons_to_contact_shapes(calendar_persons: list) -> list:
    """Convert a list of CalendarPerson into `contacts.Contact`-shaped
    records the materializer's `family_contacts` pipeline accepts.

    Lives in cli.py (not calendar_evidence.py) so calendar_evidence stays
    decoupled from the contacts module — the bridge is operator
    plumbing, not library-level coupling. Pass 14A may later add a
    dedicated `family_calendar_persons` parameter; until then the
    Contact shape is the working integration point.

    The `family_signal_source` is prefixed with `calendar:` so downstream
    code (and the operator looking at debug output) can tell which
    evidence source surfaced each candidate.
    """
    from .contacts import Contact

    out: list = []
    for p in calendar_persons:
        # Split name into given / family for the materializer's surname
        # match. Best-effort: a single-token name has no family_name.
        tokens = p.name.strip().split()
        given = tokens[0] if tokens else None
        family = tokens[-1] if len(tokens) >= 2 else None
        emails = [p.email] if p.email else []
        # Render appears_in_titles as a compact biography. The judge
        # prompt will see "Calendar: '<title1>', '<title2>', ..." and can
        # use it for grounding.
        if p.appears_in_titles:
            titles_str = ", ".join(
                f"'{t}'" for t in p.appears_in_titles[:3]
            )
            if len(p.appears_in_titles) > 3:
                titles_str += f", (+{len(p.appears_in_titles) - 3} more)"
            biography = f"Calendar: {titles_str}"
        else:
            biography = None
        out.append(
            Contact(
                # Synthetic resource name — distinct from
                # `people/<id>` to make calendar-derived records easy
                # to spot in logs.
                resource_name=f"calendar/{p.name.lower().replace(' ', '_')}",
                display_name=p.name,
                given_name=given,
                family_name=family,
                email_addresses=emails,
                group_memberships=[],
                relations=[],
                biography=biography,
                is_starred=False,
                family_signal_strength=p.family_signal_strength,
                family_signal_source=f"calendar:{p.family_signal_source}",
            )
        )
    return out


# Allow `python -m pi_email.cli` to also work.
if __name__ == "__main__":  # pragma: no cover
    main()
