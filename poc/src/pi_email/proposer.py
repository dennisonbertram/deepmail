"""LLM proposer: given new entities, propose 1-3 follow-up search queries.

The proposer's job is *query generation*, not *answer generation*. Per
research/05, that's the differentiating split — the LLM is good at coming up
with creative angles to search; deterministic code decides when to stop.

API: Anthropic Python SDK, model claude-sonnet-4-5 (per task spec).

Two-phase API:

  - `propose_seed_queries(seed)` — Phase 1, no entities yet. The returned
    proposals are allowed to carry the sentinel `parent_entity = "SEED"`.
    These bootstrap the frontier so the loop has something to pop on iter 1.

  - `propose_expansion_queries(new_entities, ran_queries)` — Phase 2, the
    main expansion. EVERY returned proposal MUST cite a `parent_entity`
    whose string form matches exactly one of the provided entities (via
    `str(Entity)`, i.e. "person:Jane Smith"). Topic-expansion proposals
    (e.g. "wedding OR anniversary") that aren't rooted in a specific
    extracted entity are dropped at parse time and a warning is logged.
    This is what makes the per-entity budget (Rule D) load-bearing under
    live-LLM conditions.

If ANTHROPIC_API_KEY is unset, we fall back to a deterministic mock proposer
that derives queries from the seed string itself with the same parent_entity
contract. The demo prints a clear [MOCK PROPOSER] banner.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from .entities import Entity


MODEL = "claude-sonnet-4-5"
SEED_SENTINEL = "SEED"


# Shared base instructions, applied to both seed and expansion calls. The
# Phase-2 grounding requirement is the load-bearing rule — it's repeated in
# the per-call user message too, but stating it in the system prompt lets
# the cache hit cover the most important rule.
SYSTEM_PROMPT = """You are a Gmail search strategist for an email-understanding agent.

Your role: propose 1-3 Gmail-style search queries per call that would surface
relevant messages from the user's mailbox, guided by the ORIGINAL QUERY the
user provided at session start.

You DO NOT decide when to stop searching. A deterministic frontier algorithm
handles termination. Your only job is to propose creative search angles.

Hard rules:
  1. Use Gmail search operators when useful: from:, to:, subject:, plus plain
     terms. Quote multi-word phrases. `OR` (uppercase) joins disjunctions.
  2. Do NOT repeat queries that have already been run (you'll be shown them).
  3. Bias toward HIGH-RECALL queries — broad terms beat narrow ones. The user
     wants exhaustive coverage, not precision.
  4. Think about: what kinds of emails would contain this information? Who
     would send/receive these emails? What subject lines or keywords would
     appear? What email domains might be involved?
  5. Output ONLY a JSON object. No markdown, no preamble, no explanation
     outside the JSON. The shape is:

     {
       "proposals": [
         {
           "query": "from:bob.smith@example.com subject:vacation",
           "parent_entity": "person:Bob Smith",
           "justification": "Drill into Bob's vacation context"
         },
         ...
       ]
     }

  6. PHASE-2 GROUNDING (when you are given a list of entities): EVERY proposal
     MUST be rooted in ONE specific entity from the provided list. The
     `parent_entity` field must EXACTLY match (case-sensitive) the string
     form of one entity from the list (e.g. "person:Jane Smith",
     "email:bob@example.com"). Do NOT propose topic-expansion queries that
     aren't anchored to a specific extracted entity. Proposals with an
     unknown or missing parent_entity will be REJECTED. Stay focused on the
     ORIGINAL QUERY — don't drift into unrelated topics.

  7. PHASE-1 SEEDING (when no entities are provided, only a seed string): the
     `parent_entity` field should be "SEED". Propose 1-3 initial Gmail
     queries that bootstrap the search from the free-text seed.
"""


@dataclass
class ProposedQuery:
    query: str
    score: float
    justification: str
    parent_entity: str   # required; "SEED" for Phase 1, "kind:label" otherwise


# ---------------- JSON helpers ----------------


def _extract_json(text: str) -> dict | None:
    """Try strict parse; on failure, find the first {...} block. Returns None
    on any failure (the proposer must not crash the loop)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_seed_proposals(
    data: dict,
    log: Callable[[str], None],
) -> list[ProposedQuery]:
    """Parse Phase-1 (seed) proposals. Missing parent_entity is coerced to
    SEED_SENTINEL; anything else (or strict 'SEED') is accepted as-is."""
    out: list[ProposedQuery] = []
    items = data.get("proposals") or data.get("queries") or []
    for item in items:
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        parent = item.get("parent_entity")
        if not parent or not str(parent).strip():
            parent = SEED_SENTINEL
        out.append(ProposedQuery(
            query=q,
            score=float(item.get("score", 0.8)),
            justification=str(item.get("justification", "")),
            parent_entity=str(parent),
        ))
    return out


def _parse_expansion_proposals(
    data: dict,
    valid_parents: set[str],
    log: Callable[[str], None],
) -> list[ProposedQuery]:
    """Parse Phase-2 (expansion) proposals. Drops any proposal whose
    parent_entity isn't in `valid_parents` (exact case-sensitive match)."""
    out: list[ProposedQuery] = []
    items = data.get("proposals") or data.get("queries") or []
    for item in items:
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        parent_raw = item.get("parent_entity")
        if parent_raw is None:
            log(
                f"proposer: dropped proposal with no parent_entity: query={q!r}"
            )
            continue
        parent = str(parent_raw).strip()
        if parent not in valid_parents:
            log(
                f"proposer: dropped proposal with unknown parent_entity "
                f"{parent!r}: query={q!r}"
            )
            continue
        out.append(ProposedQuery(
            query=q,
            score=float(item.get("score", 0.7)),
            justification=str(item.get("justification", "")),
            parent_entity=parent,
        ))
    return out


# ---------------- Mock proposer ----------------


_MOCK_STOPWORDS = frozenset({
    "figure", "out", "my", "the", "a", "an", "all", "about",
    "find", "get", "show", "me", "tell", "who", "what", "are",
    "is", "of", "and", "or", "to", "in", "for", "with",
})


def _mock_propose_seed(seed: str, max_queries: int = 3) -> list[ProposedQuery]:
    """Deterministic stand-in for the seed phase.

    Generates queries from the seed string itself rather than using
    hardcoded terms. This makes the mock proposer work for any topic
    (investors, family, team, etc.) rather than only family discovery.
    """
    words = [w for w in seed.lower().split() if w not in _MOCK_STOPWORDS and len(w) > 1]
    if not words:
        # Fallback: use the entire seed as a single query
        words = [seed.strip()]

    out: list[ProposedQuery] = []
    # First: broad OR-joined query across all meaningful words
    if len(words) > 1:
        out.append(ProposedQuery(
            query=" OR ".join(words),
            score=0.9,
            justification=f"Mock seed: broad keyword search for {', '.join(words)}",
            parent_entity=SEED_SENTINEL,
        ))
    # Then: per-word subject-line searches
    for word in words:
        if len(out) >= max_queries:
            break
        out.append(ProposedQuery(
            query=f"subject:{word}",
            score=0.7,
            justification=f"Mock seed: subject-line search for {word}",
            parent_entity=SEED_SENTINEL,
        ))
    return out[:max_queries]


def _mock_propose_expansion(
    new_entities: list[Entity],
    already_run: list[str],
    max_queries: int = 3,
) -> list[ProposedQuery]:
    """Deterministic stand-in for the expansion phase.

    Picks the most-recently-extracted entities (we treat the END of the input
    list as 'most recent' — the loop sorts deterministically so this is
    stable) and roots each proposal in a single entity. This preserves the
    Rule D / Rule E guarantees under the mock-proposer path.
    """
    if not new_entities:
        return []

    already_lower = {q.strip().lower() for q in already_run}
    out: list[ProposedQuery] = []

    # Walk the input from the END (most recently extracted) so the mock
    # exercises the per-entity attribution against the freshest signal.
    for e in reversed(new_entities):
        if e.kind == "person":
            q = e.label.split()[0]  # first name only — broader
            score = 0.7
        elif e.kind == "email":
            q = f"from:{e.label}"
            score = 0.6
        elif e.kind == "relation":
            q = e.label
            score = 0.5
        else:
            q = e.label
            score = 0.4
        if q.lower() in already_lower:
            continue
        out.append(ProposedQuery(
            query=q,
            score=score,
            justification=f"Mock: drill into {e}",
            parent_entity=str(e),
        ))
        if len(out) >= max_queries:
            break
    return out


# ---------------- Live proposer ----------------


def _format_entity_block(
    entities: list[Entity],
    excerpts: dict[str, str],
    limit: int = 8,
) -> str:
    lines: list[str] = []
    for e in entities[:limit]:
        excerpt = excerpts.get(str(e), "").strip().replace("\n", " ")[:200]
        lines.append(
            f'- parent_entity="{e}"   ({e.kind}={e.label})   excerpt: "{excerpt}"'
        )
    return "\n".join(lines)


def _live_propose_seed(
    client,
    seed: str,
    max_queries: int,
    log: Callable[[str], None],
) -> list[ProposedQuery]:
    user_msg = (
        f"PHASE 1 — SEED.\n\n"
        f"ORIGINAL QUERY: \"{seed}\"\n\n"
        f"No entities have been extracted yet. Propose 1-{max_queries} initial "
        f"Gmail-style search queries that would surface emails relevant to this "
        f"topic.\n\n"
        f"Think about:\n"
        f"- What kinds of emails would contain this information?\n"
        f"- Who would send/receive these emails?\n"
        f"- What subject lines or keywords would appear?\n"
        f"- What email domains might be involved?\n\n"
        f"Set parent_entity = \"SEED\" on every proposal.\n"
        f"Reply ONLY with the JSON object."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            # Prompt caching per the claude-api skill: marking the static
            # system text as ephemeral lets repeated proposer calls reuse it.
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    data = _extract_json(text)
    if data is None:
        log(f"proposer: failed to parse seed JSON; raw={text[:200]!r}")
        return []
    return _parse_seed_proposals(data, log)[:max_queries]


def _live_propose_expansion(
    client,
    new_entities: list[Entity],
    ran_queries: list[str],
    excerpts: dict[str, str],
    max_queries: int,
    log: Callable[[str], None],
    seed: str = "",
) -> list[ProposedQuery]:
    valid_parents = {str(e) for e in new_entities}
    entity_blob = _format_entity_block(new_entities, excerpts)
    ran_blob = "\n".join("  - " + q for q in ran_queries) if ran_queries else "  (none yet)"

    seed_line = f'\nORIGINAL QUERY: "{seed}"\n' if seed else ""
    focus_line = f'\nStay focused on "{seed}" — don\'t drift into unrelated topics.\n' if seed else ""

    user_msg = f"""PHASE 2 — EXPANSION.
{seed_line}
Each of the entities below was newly extracted from the corpus delta. Your job
is to propose 1-{max_queries} Gmail-style search queries that DRILL DEEPER on
ONE SPECIFIC entity from this list, in service of the original query above.
Topic-expansion queries that aren't anchored to one of these entities will be
rejected.

Newly extracted entities (use one of these EXACT strings as parent_entity):
{entity_blob}

Queries already run (do not repeat):
{ran_blob}
{focus_line}
Reply ONLY with the JSON object. Every proposal's parent_entity MUST be an
exact string from the entity list above."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    data = _extract_json(text)
    if data is None:
        log(f"proposer: failed to parse expansion JSON; raw={text[:200]!r}")
        return []
    return _parse_expansion_proposals(data, valid_parents, log)[:max_queries]


# ---------------- Public entry point ----------------


class Proposer:
    """Wraps live + mock backends. Defaults to live; falls back if no API key.

    The class is stateless other than the optional Anthropic client handle —
    one instance can drive both phases across many loop iterations.
    """

    def __init__(
        self,
        force_mock: bool = False,
        on_log: Callable[[str], None] | None = None,
    ):
        self.force_mock = force_mock
        self.is_mock = True
        self._client = None
        self._log = on_log or (lambda s: None)
        self._seed: str = ""  # set by propose_seed_queries, used by expansion
        if not force_mock and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic()
                self.is_mock = False
            except Exception:
                self._client = None
                self.is_mock = True

    def banner(self) -> str:
        return "[MOCK PROPOSER]" if self.is_mock else f"[LIVE PROPOSER: {MODEL}]"

    # -- Phase 1 -------------------------------------------------------

    def propose_seed_queries(
        self,
        seed: str,
        max_queries: int = 3,
    ) -> list[ProposedQuery]:
        """Phase 1: turn a free-text seed into 1-`max_queries` initial Gmail
        search queries. Returned proposals carry parent_entity = "SEED"."""
        self._seed = seed  # remember for expansion-phase context
        if self.is_mock:
            return _mock_propose_seed(seed, max_queries=max_queries)
        try:
            return _live_propose_seed(
                self._client, seed, max_queries, self._log
            )
        except Exception as e:
            self._log(f"proposer: live seed call failed, falling back to mock: {e}")
            return _mock_propose_seed(seed, max_queries=max_queries)

    # -- Phase 2 -------------------------------------------------------

    def propose_expansion_queries(
        self,
        new_entities: list[Entity],
        ran_queries: list[str],
        max_queries: int = 3,
        excerpts: dict[str, str] | None = None,
    ) -> list[ProposedQuery]:
        """Phase 2: given specific new entities, propose queries that drill
        deeper on those SPECIFIC entities. Every returned proposal cites a
        `parent_entity` that exactly matches one of the provided entities;
        proposals without a valid grounding are dropped.
        """
        if not new_entities:
            return []
        excerpts = excerpts or {}
        if self.is_mock:
            return _mock_propose_expansion(new_entities, ran_queries, max_queries=max_queries)
        try:
            return _live_propose_expansion(
                self._client,
                new_entities,
                ran_queries,
                excerpts,
                max_queries,
                self._log,
                seed=self._seed,
            )
        except Exception as e:
            self._log(
                f"proposer: live expansion call failed, falling back to mock: {e}"
            )
            return _mock_propose_expansion(
                new_entities, ran_queries, max_queries=max_queries
            )
