"""CLI demo: run the iterative-expansion loop end-to-end.

Usage:
    python demo.py "figure out my family"                 # fixture corpus (default)
    python demo.py "figure out my family" --source gmail  # real Gmail

If ANTHROPIC_API_KEY is unset, the proposer falls back to a deterministic mock —
the loop still runs end-to-end, just with simpler proposed queries.

The Gmail path requires `pi-email auth` to have been run first (see README).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd if present; harmless if missing

# Allow `python demo.py ...` without installing the package first.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import click

from pi_email.loop import ExpansionLoop, IterationEvent, run_loop_and_materialize


HERE = Path(__file__).parent
FIXTURES_DIR = HERE / "fixtures" / "family_corpus"
PROFILES_DIR = HERE / "profiles"


@click.command()
@click.argument("seed", required=False, default="figure out my family")
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
    help="Cap on messages pulled per query when --source gmail.",
)
@click.option(
    "--force-mock",
    is_flag=True,
    help="Force the mock proposer even if ANTHROPIC_API_KEY is set.",
)
def main(seed: str, source: str, max_corpus: int, force_mock: bool):
    """Run the iterative-expansion loop and materialize profiles/family.md."""

    print(f"=== pi_email POC: '{seed}' (source={source}) ===")
    print(f"profiles: {PROFILES_DIR}")
    print()

    def on_log(msg: str) -> None:
        print(f"  {msg}")

    def on_event(e: IterationEvent) -> None:
        parent = f" (parent={e.parent_entity})" if e.parent_entity else ""
        print(
            f"  iter {e.n:2d}: q={e.query!r:<40s} "
            f"score={e.score:.2f} hits={e.hits:2d} "
            f"new_msgs={e.new_messages:2d} new_ents={e.new_entities:2d} "
            f"recap={e.recapture_rate:.2f} proposed={e.proposed} "
            f"frontier_size={e.frontier_size_after}{parent}"
        )

    src = source.lower()
    if src == "fixture":
        print(f"fixtures: {FIXTURES_DIR}")
        print()
        result, out_path = run_loop_and_materialize(
            fixtures_dir=FIXTURES_DIR,
            seed=seed,
            profiles_dir=PROFILES_DIR,
            on_event=on_event,
            on_log=on_log,
            force_mock=force_mock,
        )
    else:
        # Gmail path. We mirror the wiring run_loop_and_materialize does for
        # fixtures, but with GmailSearcher + saved OAuth creds.
        from pi_email.gmail_searcher import GmailQuotaExhausted, GmailSearcher
        from pi_email.materializer import write_family_profile
        from pi_email.oauth import refresh_if_needed
        from pi_email.proposer import Proposer
        from pi_email.token_store import TokenStore

        store = TokenStore()
        creds = store.load()
        if creds is None:
            print("Not authenticated — run `pi-email auth` first.", file=sys.stderr)
            sys.exit(2)
        try:
            refresh_if_needed(creds)
        except Exception as exc:
            print(
                f"Token refresh failed ({exc}); re-run `pi-email auth`.",
                file=sys.stderr,
            )
            sys.exit(2)
        store.save(creds)  # persist any refreshed access token

        print(f"gmail: max_results_per_query={max_corpus}")
        print()
        searcher = GmailSearcher(creds, max_results_per_query=max_corpus)
        proposer = Proposer(force_mock=force_mock, on_log=on_log)
        loop = ExpansionLoop(
            searcher=searcher,
            proposer=proposer,
            on_event=on_event,
            on_log=on_log,
        )
        try:
            result = loop.run(seed)
        except GmailQuotaExhausted as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            sys.exit(3)
        out_path = write_family_profile(
            out_path=PROFILES_DIR / "family.md",
            corpus=result.corpus,
            entities=result.entities,
            seen_by_message=result.seen_by_message,
            seed=seed,
            stop_reason=result.stop_reason.rule,
            queries_run=result.queries_run,
            canonical_map=result.canonical_map,
        )

    print()
    print(f"=== Done ===")
    print(f"  iterations: {len(result.iterations)}")
    print(f"  total corpus size: {len(result.corpus)} messages")
    print(f"  total entities: {len(result.entities)}")
    print(f"  stop reason: {result.stop_reason.rule} — {result.stop_reason.detail}")
    print(f"  queries run: {result.queries_run}")
    print(f"  profile written: {out_path}")
    print()


if __name__ == "__main__":
    main()
