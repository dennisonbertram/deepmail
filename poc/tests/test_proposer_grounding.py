"""Tests for the entity-grounding contract between proposer and frontier.

These regression-cover the design fix that makes Rule D / Rule E load-bearing
under live-LLM conditions: every non-seed query must declare a parent_entity
that matches one of the entities the proposer was asked about, and the
frontier enforces that the field is non-empty at push() time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.entities import Entity  # noqa: E402
from pi_email.frontier import Frontier  # noqa: E402
from pi_email.proposer import (  # noqa: E402
    Proposer,
    SEED_SENTINEL,
    SYSTEM_PROMPT,
    _live_propose_expansion,
    _mock_propose_seed,
)


# ---------- Frontier contract ----------


def test_frontier_push_requires_parent_entity_none():
    f = Frontier()
    with pytest.raises(ValueError, match="non-empty parent_entity"):
        f.push("kids", score=0.5, parent_entity=None)


def test_frontier_push_requires_parent_entity_empty_string():
    f = Frontier()
    with pytest.raises(ValueError, match="non-empty parent_entity"):
        f.push("kids", score=0.5, parent_entity="")


def test_frontier_push_requires_parent_entity_whitespace_only():
    f = Frontier()
    with pytest.raises(ValueError, match="non-empty parent_entity"):
        f.push("kids", score=0.5, parent_entity="   ")


def test_frontier_push_accepts_seed_sentinel():
    f = Frontier()
    assert f.push("family", score=0.9, parent_entity=SEED_SENTINEL) is True


def test_frontier_push_accepts_real_entity_string():
    f = Frontier()
    assert f.push("Jane", score=0.7, parent_entity="person:Jane Smith") is True


# ---------- Proposer Phase-2 grounding ----------


class _FakeBlock:
    """Minimal stand-in for an Anthropic response content block."""
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: D401
        self.calls.append(kwargs)
        return _FakeResp(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def test_propose_expansion_queries_drops_invalid_parent():
    """LLM returns one valid + one invalid proposal — only the valid one survives."""
    valid_entity = Entity(kind="person", key="bob smith", label="Bob Smith")
    other_entity = Entity(kind="relation", key="kids", label="kids")
    new_entities = [valid_entity, other_entity]

    # Fake LLM response: one proposal cites a valid parent_entity from the
    # provided list; the other proposes a topic-expansion query rooted in
    # an entity that was NEVER in the input list — must be dropped.
    fake_text = json.dumps({
        "proposals": [
            {
                "query": "from:bob.smith@example.com subject:vacation",
                "parent_entity": "person:Bob Smith",
                "justification": "Drill into Bob's vacation context",
            },
            {
                "query": "wedding OR anniversary OR engagement",
                "parent_entity": "topic:weddings",  # NOT in the entity list
                "justification": "Generic topic expansion (should be rejected)",
            },
        ]
    })
    logged: list[str] = []
    client = _FakeClient(fake_text)

    proposals = _live_propose_expansion(
        client=client,
        new_entities=new_entities,
        ran_queries=[],
        excerpts={
            "person:Bob Smith": "Bob is grilling.",
            "relation:kids": "The kids are picky again.",
        },
        max_queries=3,
        log=logged.append,
    )

    # Only the valid proposal survives.
    assert len(proposals) == 1
    surviving = proposals[0]
    assert surviving.query == "from:bob.smith@example.com subject:vacation"
    assert surviving.parent_entity == "person:Bob Smith"

    # The dropped one is logged.
    assert any("topic:weddings" in line for line in logged), logged
    assert any("dropped" in line for line in logged), logged


def test_propose_expansion_queries_drops_missing_parent():
    """Proposal with no parent_entity field at all is dropped."""
    valid_entity = Entity(kind="person", key="jane smith", label="Jane Smith")
    fake_text = json.dumps({
        "proposals": [
            {
                "query": "subject:dinner",
                # parent_entity intentionally missing
                "justification": "no anchor",
            },
            {
                "query": "Jane",
                "parent_entity": "person:Jane Smith",
                "justification": "anchored",
            },
        ]
    })
    logged: list[str] = []
    client = _FakeClient(fake_text)

    proposals = _live_propose_expansion(
        client=client,
        new_entities=[valid_entity],
        ran_queries=[],
        excerpts={},
        max_queries=3,
        log=logged.append,
    )

    assert len(proposals) == 1
    assert proposals[0].parent_entity == "person:Jane Smith"
    assert any("no parent_entity" in line for line in logged), logged


def test_propose_expansion_queries_empty_entities_returns_empty():
    """No entities -> no expansion proposals (we don't hit the LLM at all)."""
    p = Proposer(force_mock=True)
    out = p.propose_expansion_queries(new_entities=[], ran_queries=[])
    assert out == []


def test_propose_seed_queries_mock_returns_seed_sentinel():
    p = Proposer(force_mock=True)
    seeded = p.propose_seed_queries("figure out my family")
    assert len(seeded) >= 1
    for pq in seeded:
        assert pq.parent_entity == SEED_SENTINEL


def test_propose_expansion_mock_attributes_to_provided_entity():
    """Mock proposer must produce parent_entity values that are in the input set."""
    p = Proposer(force_mock=True)
    entities = [
        Entity(kind="person", key="bob smith", label="Bob Smith"),
        Entity(kind="email", key="jane@example.com", label="jane@example.com"),
    ]
    valid_parents = {str(e) for e in entities}
    out = p.propose_expansion_queries(new_entities=entities, ran_queries=[])
    assert out, "mock should propose at least one expansion query"
    for pq in out:
        assert pq.parent_entity in valid_parents, (
            f"mock proposal {pq.query!r} has parent_entity {pq.parent_entity!r} "
            f"which is not in {valid_parents}"
        )


# ---------- Query-agnostic proposer tests ----------


_FAMILY_WORDS = {"family", "kinship", "relative", "wife", "husband", "mother",
                 "father", "son", "daughter", "sibling", "parent", "child",
                 "mom", "dad", "kids"}


def test_seed_prompt_is_query_agnostic():
    """The system prompt must NOT contain family-specific language.

    Uses word-boundary matching to avoid false positives from compound
    words (e.g. 'parent_entity' should not trigger 'parent', and
    'person' should not trigger 'son').
    """
    import re
    prompt_lower = SYSTEM_PROMPT.lower()
    found = [
        w for w in _FAMILY_WORDS
        if re.search(r'\b' + re.escape(w) + r'\b', prompt_lower)
        # Exclude matches inside compound identifiers like parent_entity
        and not all(
            re.search(r'\b' + re.escape(w) + r'[_a-z]', prompt_lower)
            or re.search(r'[_a-z]' + re.escape(w) + r'\b', prompt_lower)
            for m_obj in [re.search(r'\b' + re.escape(w) + r'\b', prompt_lower)]
        )
    ]
    # Extra check: "parent" appearing ONLY as part of "parent_entity" is OK
    found_clean = []
    for w in found:
        # Find all word-boundary matches
        matches = list(re.finditer(r'\b' + re.escape(w) + r'\b', prompt_lower))
        # Filter out matches that are part of compound identifiers
        standalone = [
            m for m in matches
            if not (m.end() < len(prompt_lower) and prompt_lower[m.end()] == '_')
            and not (m.start() > 0 and prompt_lower[m.start() - 1] == '_')
        ]
        if standalone:
            found_clean.append(w)
    assert not found_clean, (
        f"SYSTEM_PROMPT contains family-specific words: {found_clean}. "
        f"The proposer should be query-agnostic."
    )


def test_mock_seeds_reflect_query_investors():
    """mock_propose_seeds('figure out my investors') should generate queries
    containing 'investors', NOT 'family'."""
    proposals = _mock_propose_seed("figure out my investors")
    all_queries = " ".join(pq.query.lower() for pq in proposals)
    assert "investors" in all_queries, (
        f"Expected 'investors' in mock seed queries, got: "
        f"{[pq.query for pq in proposals]}"
    )
    for pq in proposals:
        for word in _FAMILY_WORDS:
            assert word not in pq.query.lower(), (
                f"Mock seed query {pq.query!r} contains family word {word!r} "
                f"for a non-family seed"
            )


def test_mock_seeds_for_family_query():
    """mock_propose_seeds('figure out my family') should generate queries
    containing 'family' — driven by the seed, not hardcoded."""
    proposals = _mock_propose_seed("figure out my family")
    all_queries = " ".join(pq.query.lower() for pq in proposals)
    assert "family" in all_queries, (
        f"Expected 'family' in mock seed queries for a family seed, got: "
        f"{[pq.query for pq in proposals]}"
    )
    # Every proposal should carry the SEED sentinel
    for pq in proposals:
        assert pq.parent_entity == SEED_SENTINEL
