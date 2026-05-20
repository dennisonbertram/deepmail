"""Tests for seed queries in the expansion loop.

The pipeline relies exclusively on the proposer to generate seed queries.
No hardcoded deterministic seeds exist — the proposer is the sole source.
These tests verify that SEED_BUDGET accommodates the proposer's output and
that no non-proposer seeds leak into the frontier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.frontier import Frontier, SEED_BUDGET  # noqa: E402
from pi_email.proposer import SEED_SENTINEL, ProposedQuery  # noqa: E402


# ---------------------------------------------------------------------------
# Test: SEED_BUDGET accommodates proposer seeds
# ---------------------------------------------------------------------------


def test_seed_budget_allows_proposer_seeds():
    """With SEED_BUDGET=5, the proposer's typical 3-5 seeds should fit
    without budget exhaustion."""
    assert SEED_BUDGET >= 5, f"SEED_BUDGET={SEED_BUDGET} should be >= 5"

    frontier = Frontier()

    # Proposer seeds (typical output: 3-5 queries).
    proposer_seeds = [
        "family reunion birthday",
        "kids school event",
        "partner anniversary",
        "holiday gathering",
    ]

    pushed = 0
    for q in proposer_seeds:
        if frontier.push(query=q, score=0.80, parent_entity=SEED_SENTINEL,
                         justification="proposer seed"):
            pushed += 1

    # All proposer seeds should land — budget should NOT be the blocker.
    assert pushed == len(proposer_seeds), (
        f"Expected {len(proposer_seeds)} seeds pushed, got {pushed}. "
        f"Budget rejected count: {frontier.budget_rejected_count}"
    )
    assert frontier.budget_rejected_count == 0, (
        f"No seeds should be budget-rejected with SEED_BUDGET={SEED_BUDGET}"
    )


# ---------------------------------------------------------------------------
# Test: only proposer seeds are pushed into the frontier
# ---------------------------------------------------------------------------


def test_only_proposer_seeds_are_pushed():
    """After the seed phase, ALL frontier entries must have queries that came
    from the proposer — no hardcoded deterministic seeds."""
    from pi_email.loop import ExpansionLoop
    from pi_email.frontier import Frontier
    from pi_email.searcher import SearchBatch

    # Mock proposer that returns known queries.
    mock_proposer = MagicMock()
    mock_proposer.is_mock = True
    mock_proposer.banner.return_value = "mock"
    known_queries = [
        ProposedQuery(
            query="family birthday celebration",
            score=0.85,
            parent_entity=SEED_SENTINEL,
            justification="proposer seed 1",
        ),
        ProposedQuery(
            query="kids school activities",
            score=0.80,
            parent_entity=SEED_SENTINEL,
            justification="proposer seed 2",
        ),
        ProposedQuery(
            query="partner wedding anniversary",
            score=0.75,
            parent_entity=SEED_SENTINEL,
            justification="proposer seed 3",
        ),
    ]
    mock_proposer.propose_seed_queries.return_value = known_queries
    mock_proposer.propose_expansion_queries.return_value = []

    # Mock searcher that returns no hits (we only care about seed injection).
    mock_searcher = MagicMock()
    mock_searcher.search_and_fetch.return_value = SearchBatch(query="mock", hits=[])

    # Frontier without an embedder — uses string-similarity fallback for
    # deduplication, which avoids needing the 440MB model in tests.
    frontier = Frontier()

    loop = ExpansionLoop(
        searcher=mock_searcher,
        proposer=mock_proposer,
        frontier=frontier,
    )
    result = loop.run("my family")

    # The queries that were actually run should be a subset of the proposer's
    # known queries — no calendar-acceptance, no personal-domain seeds.
    known_query_strings = {pq.query for pq in known_queries}
    for ran_query in result.queries_run:
        assert ran_query in known_query_strings, (
            f"Query {ran_query!r} was run but is NOT from the proposer. "
            f"Proposer queries: {known_query_strings}"
        )
