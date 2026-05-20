"""Background build worker. Invoked as a subprocess by the MCP server.

Usage:
    python -m pi_email.build_worker <status_path> [query] [--skip-judge]

Reads the status file path from argv, runs the full Gmail expansion pipeline,
and writes progress updates to the status file as it progresses. Catches
exceptions and writes failure status so the MCP server can report them.

The subprocess is started with ``start_new_session=True`` so it survives if
the MCP server process exits while the build is still running.

When ``--skip-judge`` is passed, the LLM family judge is bypassed and all
gathered candidates are written to the profile with their evidence for the
calling model to evaluate. This is the default when invoked from the MCP
server (the calling AI agent does the judgment instead).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def update_status(status_path: Path, data: dict) -> None:
    """Atomically write status JSON (write to .tmp then rename)."""
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(status_path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m pi_email.build_worker <status_path> [query] [--skip-judge]", file=sys.stderr)
        sys.exit(1)

    status_path = Path(sys.argv[1])
    # Parse positional args and flags. The query is the first non-flag arg
    # after the status_path. --skip-judge can appear anywhere after argv[1].
    remaining = sys.argv[2:]
    skip_judge = "--skip-judge" in remaining
    positional = [a for a in remaining if not a.startswith("--")]
    query = positional[0] if positional else ""
    started_at = datetime.now(timezone.utc).isoformat()

    update_status(status_path, {
        "state": "starting",
        "started_at": started_at,
        "query": query,
        "pid": os.getpid(),
    })

    try:
        _run_pipeline(status_path, query, started_at, skip_judge=skip_judge)
    except Exception as e:
        update_status(status_path, {
            "state": "failed",
            "started_at": started_at,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "pid": os.getpid(),
            "error": str(e),
        })
        sys.exit(1)


def _run_pipeline(status_path: Path, query: str, started_at: str, *, skip_judge: bool = False) -> None:
    """Run the full Gmail expansion pipeline, updating status as we go.

    Mirrors ``cli.py``'s ``_run_loop_gmail`` without modifying it.

    When ``skip_judge`` is True, the LLM family judge is bypassed and all
    gathered candidates are written to the profile with evidence excerpts
    for the calling model to evaluate.
    """
    from googleapiclient.discovery import build

    from .calendar_evidence import (
        GoogleCalendar,
        credentials_have_calendar_scope,
    )
    from .contacts import (
        Contact,
        GoogleContacts,
        credentials_have_contacts_scope,
    )
    from .gmail_searcher import GmailSearcher
    from .loop import ExpansionLoop, IterationEvent
    from .materializer import write_family_profile
    from .oauth import refresh_if_needed
    from .proposer import Proposer
    from .materializer import _slugify_query
    from .token_store import TokenStore

    # ---- Resolve paths ----
    HERE = Path(__file__).resolve().parent
    POC_ROOT = HERE.parent.parent
    profiles_dir = POC_ROOT / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # ---- Auth ----
    store = TokenStore()
    creds = store.load()
    if creds is None:
        raise RuntimeError("Not authenticated. Run: uv run pi-email auth")

    refresh_if_needed(creds)
    try:
        store.save(creds)
    except Exception:
        pass

    update_status(status_path, {
        "state": "running",
        "started_at": started_at,
        "query": query,
        "pid": os.getpid(),
        "progress": {
            "iteration": 0,
            "messages_fetched": 0,
            "entities_found": 0,
            "phase": "initializing",
        },
    })

    # ---- Identify user ----
    user_self: dict | None = None
    try:
        gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile_result = gmail.users().getProfile(userId="me").execute()
        user_email = profile_result.get("emailAddress")
        if user_email:
            user_self = {"email": user_email.lower(), "display_name": None}
    except Exception:
        pass

    # ---- Build the loop with progress callback ----
    searcher = GmailSearcher(creds, max_results_per_query=500)
    proposer = Proposer(force_mock=False)

    # Track progress via the on_event callback
    iteration_count = [0]
    corpus_size = [0]
    entity_count = [0]

    def on_event(e: IterationEvent) -> None:
        iteration_count[0] = e.n
        corpus_size[0] = e.hits
        entity_count[0] = e.new_entities
        update_status(status_path, {
            "state": "running",
            "started_at": started_at,
            "query": query,
            "pid": os.getpid(),
            "progress": {
                "iteration": e.n,
                "messages_fetched": e.hits,
                "entities_found": e.new_entities,
                "frontier_size": e.frontier_size_after,
                "recapture_rate": round(e.recapture_rate, 3),
                "phase": "expansion",
            },
        })

    loop = ExpansionLoop(
        searcher=searcher,
        proposer=proposer,
        on_event=on_event,
        user_self=user_self,
        strict_extract=False,
    )
    result = loop.run(query)

    # ---- Calendar evidence (Pass 14B) ----
    family_calendar_persons: list | None = None
    if credentials_have_calendar_scope(creds):
        try:
            cal_client = GoogleCalendar(creds)
            family_calendar_persons = cal_client.list_family_calendar_persons()
        except Exception:
            family_calendar_persons = None

    # ---- Contacts evidence (Pass 12A) ----
    family_contacts: list | None = None
    if credentials_have_contacts_scope(creds):
        try:
            client = GoogleContacts(creds)
            surname: str | None = None
            if user_self and user_self.get("display_name"):
                toks = str(user_self["display_name"]).strip().split()
                if len(toks) >= 2:
                    surname = toks[-1]
            family_contacts = client.list_family_members(user_surname=surname)
        except Exception:
            family_contacts = None

    # ---- Fold calendar into contacts (Pass 14B) ----
    if family_calendar_persons:
        calendar_shaped = _calendar_persons_to_contact_shapes(
            family_calendar_persons
        )
        if family_contacts is None:
            family_contacts = list(calendar_shaped)
        else:
            family_contacts = list(family_contacts) + list(calendar_shaped)

    # ---- Materialize ----
    update_status(status_path, {
        "state": "running",
        "started_at": started_at,
        "query": query,
        "pid": os.getpid(),
        "progress": {
            "iteration": len(result.iterations),
            "messages_fetched": len(result.corpus),
            "entities_found": len(result.entities),
            "phase": "materializing",
        },
    })

    out_filename = _slugify_query(query) + ".md"
    out_path = write_family_profile(
        out_path=profiles_dir / out_filename,
        corpus=result.corpus,
        entities=result.entities,
        seen_by_message=result.seen_by_message,
        seed=query,
        stop_reason=result.stop_reason.rule,
        queries_run=result.queries_run,
        canonical_map=result.canonical_map,
        user_self=user_self,
        skip_judge=skip_judge,
    )

    # ---- Count accepted/uncertain/rejected from profile ----
    accepted, uncertain, rejected = _count_profile_sections(out_path)
    stop_reason = result.stop_reason.rule if result.stop_reason else "unknown"

    update_status(status_path, {
        "state": "completed",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "pid": os.getpid(),
        "result": {
            "messages_fetched": len(result.corpus),
            "entities_found": len(result.entities),
            "iterations": len(result.iterations),
            "accepted_members": accepted,
            "uncertain_members": uncertain,
            "rejected_members": rejected,
            "profile_path": str(out_path),
            "stop_reason": stop_reason,
        },
    })


def _count_profile_sections(profile_path: Path) -> tuple[int, int, int]:
    """Count accepted/uncertain/rejected members from a profile file.

    Returns (accepted, uncertain, rejected).

    Handles both the judge-based profile format (## Members / ## Uncertain /
    ## Rejected) and the skip-judge format (## Auto-accepted members /
    ## Candidates for review / ## Auto-rejected).
    """
    if not profile_path.exists():
        return (0, 0, 0)
    text = profile_path.read_text(encoding="utf-8")

    accepted = 0
    uncertain = 0
    rejected = 0
    section = ""  # current top-level section

    for line in text.splitlines():
        if line.startswith("## Members") or line.startswith("## Auto-accepted"):
            section = "members"
        elif line.startswith("## Uncertain") or line.startswith("## Possibly family"):
            section = "uncertain"
        elif line.startswith("## Candidates for review"):
            section = "uncertain"
        elif line.startswith("## Rejected") or line.startswith("## Auto-rejected"):
            section = "rejected"
        elif line.startswith("## "):
            section = ""
        elif line.startswith("### ") and section == "members":
            accepted += 1
        elif line.startswith("### ") and section == "uncertain":
            uncertain += 1
        elif line.startswith("- ") and section == "rejected":
            rejected += 1

    return (accepted, uncertain, rejected)


def _calendar_persons_to_contact_shapes(calendar_persons: list) -> list:
    """Convert CalendarPerson records into Contact-shaped records.

    Mirrors the identical function in cli.py and mcp_server.py.
    """
    from .contacts import Contact

    out: list = []
    for p in calendar_persons:
        tokens = p.name.strip().split()
        given = tokens[0] if tokens else None
        family = tokens[-1] if len(tokens) >= 2 else None
        emails = [p.email] if p.email else []
        if p.appears_in_titles:
            titles_str = ", ".join(f"'{t}'" for t in p.appears_in_titles[:3])
            if len(p.appears_in_titles) > 3:
                titles_str += f", (+{len(p.appears_in_titles) - 3} more)"
            biography = f"Calendar: {titles_str}"
        else:
            biography = None
        out.append(
            Contact(
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


if __name__ == "__main__":
    main()
