"""Unit tests for the Round-3.5 Rule E / Rule D / stop-reason changes.
Background: three real-Gmail runs (bigrun, bigrun2, bigrun3) terminated on
`iter_cap` rather than `frontier_exhausted` because the LLM proposer added
queries faster than Rule E could down-weight them — even when iterations were
clearly producing nothing new. Round 3.5 tightens Rule E and Rule D so the
frontier actually converges, and splits the stop reasons so post-hoc analysis
can distinguish "we were nearly there" from "the loop is uncontrolled" from
"we ran out of work cleanly".

These tests pin the new behaviors so a regression that re-introduces the
20-iter-iter_cap pathology shows up in the unit suite first.
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.frontier import (  # noqa: E402
    EXHAUST_WINDOW,
    GLOBAL_SATURATION_FRONTIER_SIZE,
    GLOBAL_SATURATION_PERCENTILE,
    GLOBAL_SATURATION_RECAP_THRESHOLD,
    GLOBAL_SATURATION_WINDOW,
    ITER_CAP_ALMOST_DONE_FRONTIER,
    ITER_CAP_UNCONTROLLED_FRONTIER,
    K_PER_ENTITY,
    MAX_ITER,
    NO_NEW_PERSONS_WINDOW,
    PERSONAL_DOMAIN_BUDGET_BONUS,
    PERSONAL_DOMAIN_SCORE_BOOST,
    PERSONAL_EMAIL_DOMAINS,
    RECAPTURE_DOWNWEIGHT_FACTOR,
    RECAPTURE_DOWNWEIGHT_TRIGGER,
    RECAPTURE_IMMEDIATE_PRUNE_RECAP,
    RECAPTURE_PRUNE_AFTER,
    UNPRODUCTIVE_PARENT_POP_LIMIT,
    Frontier,
    QueryCandidate,
)


# ---------- combined_recapture_rate ----------


def test_combined_recapture_takes_max_of_signals():
    # When one channel says "fully known" and the other says "all novel",
    # we MUST take the high signal — otherwise Rule E never fires for the
    # bigrun3 iter-7 case (new_msgs=0 but no entities extracted at all).
    assert Frontier.combined_recapture_rate(0.0, 1.0) == 1.0
    assert Frontier.combined_recapture_rate(1.0, 0.0) == 1.0
    assert Frontier.combined_recapture_rate(0.5, 0.5) == 0.5
    assert Frontier.combined_recapture_rate(0.0, 0.0) == 0.0


# ---------- per-iter down-weight ----------


def test_high_message_recap_downweights_pending_queries():
    """A single high-recap observation must immediately down-weight other
    pending queries from the same parent. Bigrun3 showed that waiting for
    3 consecutive iters above 0.9 (old trigger) almost never fires."""
    f = Frontier()
    parent = "person:Alice"
    # Push two queries from the same parent at score 1.0.
    assert f.push("q1", score=1.0, parent_entity=parent)
    assert f.push("q2", score=1.0, parent_entity=parent)
    # Run one of them, then report a high recapture rate above the trigger.
    cand = f.pop()
    f.mark_ran(cand.query)
    # Use a value clearly above the (Round-3.5) trigger of 0.7.
    high = max(0.8, RECAPTURE_DOWNWEIGHT_TRIGGER + 0.05)
    f.observe_recapture(parent, high)
    # The remaining pending query from `parent` must now have score 0.5.
    pending = list(f._heap)
    assert len(pending) == 1
    assert pending[0].score == 1.0 * RECAPTURE_DOWNWEIGHT_FACTOR, (
        f"expected score={RECAPTURE_DOWNWEIGHT_FACTOR}, got {pending[0].score}"
    )


def test_low_recap_does_not_downweight():
    f = Frontier()
    parent = "person:Bob"
    assert f.push("q1", score=1.0, parent_entity=parent)
    # Below the trigger — nothing should happen.
    low = max(0.0, RECAPTURE_DOWNWEIGHT_TRIGGER - 0.05)
    assert f.observe_recapture(parent, low) is False
    assert f._heap[0].score == 1.0


# ---------- consecutive-high-recap parent prune ----------


def test_three_consecutive_high_recap_zeros_parent_budget():
    """After RECAPTURE_PRUNE_AFTER consecutive high-recap iters, the parent
    entity's remaining budget must be zeroed. Subsequent pushes from that
    parent should be rejected by push() (Rule D gate)."""
    f = Frontier()
    parent = "person:Carol"
    # Push one to register the parent in the budget map and seed the heap.
    assert f.push("seed", score=1.0, parent_entity=parent)
    high = RECAPTURE_DOWNWEIGHT_TRIGGER + 0.05
    for _ in range(RECAPTURE_PRUNE_AFTER):
        f.observe_recapture(parent, high)
    assert f.per_entity_budget[parent] == 0, (
        f"expected budget=0 after {RECAPTURE_PRUNE_AFTER} consecutive high "
        f"recap iters, got {f.per_entity_budget[parent]}"
    )
    # New pushes from the same parent must now be rejected.
    assert f.push("new-q-from-pruned-parent", score=1.0, parent_entity=parent) is False


def test_non_consecutive_high_recap_does_not_prune():
    """A low-recap iter inside the trailing RECAPTURE_PRUNE_AFTER window
    must reset the streak — Rule E's prune is for *sustained* dead
    branches, not transient noise. Pass-5B lowered PRUNE_AFTER to 2 so we
    test with a high-low-high pattern (trailing 2 = [low, high]) rather
    than the old high-high-low pattern (which would now legitimately
    prune)."""
    f = Frontier()
    parent = "person:Dan"
    assert f.push("q", score=1.0, parent_entity=parent)
    high = RECAPTURE_DOWNWEIGHT_TRIGGER + 0.05
    low = RECAPTURE_DOWNWEIGHT_TRIGGER - 0.05
    # high, low, high — trailing PRUNE_AFTER=2 = [low, high]; not all high.
    f.observe_recapture(parent, high)
    f.observe_recapture(parent, low)
    f.observe_recapture(parent, high)
    assert f.per_entity_budget[parent] > 0


# ---------- Rule D zero-yield decrement ----------


def test_decrement_budget_zero_messages():
    f = Frontier(k_per_entity=3)
    parent = "person:Eve"
    # First push registers the parent and pays the up-front -1.
    assert f.push("q1", score=1.0, parent_entity=parent)
    assert f.per_entity_budget[parent] == 2
    # Query returns 0 new messages but some new entities -> -1.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] == 1
    # Another zero-message iter -> -1.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] == 0


def test_decrement_budget_zero_messages_and_zero_entities():
    """The fully-dead query (no messages AND no entities) costs -2, not -1.
    Pairs with the up-front -1 from push() for a net -3 per dead query."""
    f = Frontier(k_per_entity=3)
    parent = "person:Frank"
    assert f.push("q1", score=1.0, parent_entity=parent)
    assert f.per_entity_budget[parent] == 2
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=0)
    assert f.per_entity_budget[parent] == 0


def test_decrement_budget_productive_query_no_penalty():
    """Productive queries (new_messages > 0) pay nothing extra in Rule D;
    they already paid -1 in push(). Without this guarantee, every healthy
    iteration would chew through a parent's budget."""
    f = Frontier(k_per_entity=3)
    parent = "person:Grace"
    assert f.push("q1", score=1.0, parent_entity=parent)
    before = f.per_entity_budget[parent]
    f.decrement_budget_for_yield(parent, new_messages=5, new_entities=10)
    assert f.per_entity_budget[parent] == before


def test_dedupe_does_not_consume_budget():
    """Rule C dedupe of a candidate must NOT decrement Rule D budget — the
    research doc treats them as orthogonal."""
    f = Frontier()
    parent = "person:Henry"
    assert f.push("alpha OR beta", score=1.0, parent_entity=parent)
    budget_after_first = f.per_entity_budget[parent]
    # Same token-bag — dedupe path, should NOT charge budget.
    assert f.push("beta OR alpha", score=1.0, parent_entity=parent) is False
    assert f.per_entity_budget[parent] == budget_after_first, (
        "deduped push must not consume per-entity budget"
    )


# ---------- Stop-reason classification ----------


def _fill_frontier(f: Frontier, n: int) -> None:
    """Pack n candidates onto the frontier, bypassing Rule C dedupe.

    These tests target `stop()` and don't care about push()'s dedupe / Rule D
    gates — we just need the heap to have a specific size when stop fires.
    The public push() path applies SequenceMatcher dedupe (0.80) in the no-
    embedder configuration, which makes generating 50+ distinct-enough
    queries fiddly. Going through `_push` keeps the test focused.
    """
    from pi_email.frontier import QueryCandidate
    for i in range(n):
        f._push(
            QueryCandidate(
                sort_key=(0.0, 0),
                query=f"synthetic_query_{i}",
                score=1.0,
                parent_entity=f"synthetic_parent_{i}",
            )
        )


def test_stop_iter_cap_uncontrolled_when_frontier_large():
    f = Frontier(max_iter=10)
    _fill_frontier(f, ITER_CAP_UNCONTROLLED_FRONTIER + 5)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap_uncontrolled", sr


def test_stop_iter_cap_almost_done_when_frontier_small():
    f = Frontier(max_iter=10)
    # Frontier with 1 pending — well under ITER_CAP_ALMOST_DONE_FRONTIER.
    _fill_frontier(f, 1)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap_almost_done", sr


def test_stop_iter_cap_plain_when_frontier_medium():
    """Frontier size between the two thresholds keeps the bare `iter_cap`
    reason, so existing tooling that switches on it still works."""
    f = Frontier(max_iter=10)
    medium = (ITER_CAP_ALMOST_DONE_FRONTIER + ITER_CAP_UNCONTROLLED_FRONTIER) // 2
    _fill_frontier(f, medium)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap", sr


def test_stop_frontier_exhausted_clean_after_yielding_iter():
    f = Frontier(max_iter=10)
    # Observe one good iter then exhaust the queue.
    f.observe_iteration_yield(new_messages=5, new_entities=3)
    sr = f.stop(iteration=1, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "frontier_exhausted_clean", sr


def test_stop_frontier_exhausted_no_yield_after_dead_iters():
    """Empty frontier following EXHAUST_WINDOW dead iters → _no_yield."""
    f = Frontier(max_iter=10)
    for _ in range(EXHAUST_WINDOW):
        f.observe_iteration_yield(new_messages=0, new_entities=0)
    sr = f.stop(iteration=EXHAUST_WINDOW, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "frontier_exhausted_no_yield", sr


def test_stop_frontier_exhausted_clean_no_iters_observed():
    """Edge case: stop check before any iter ran (empty seed list)."""
    f = Frontier(max_iter=10)
    sr = f.stop(iteration=0, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    # No iters observed yet — call this _clean by convention; there's no
    # dead-branch signal to flag.
    assert sr.rule == "frontier_exhausted_clean", sr


def test_recent_iter_yields_window_is_bounded():
    """The yield window must not grow without bound — otherwise long runs
    leak memory and the classifier looks at ancient history."""
    f = Frontier()
    for i in range(EXHAUST_WINDOW * 4):
        f.observe_iteration_yield(new_messages=i, new_entities=i)
    assert len(f._recent_iter_yields) == EXHAUST_WINDOW


# ---------- Sanity / regression on the round-3.5 tunable changes ----------


def test_iter_cap_lowered_to_25():
    """Pass 12B Fix 3 — MAX_ITER lowered 35 -> 25.

    With the Pass 12B `no_new_persons_after_N` graceful stop carrying
    the load, the hard cap should sit lower than before so a converging
    run terminates on the family-yield signal rather than burning the
    full 35 iters. Below 25 starts to clip legitimate real-Gmail runs
    (Run 11 was visibly converging by iter 11 but had 24 more wasted
    iters before iter_cap fired).
    """
    assert MAX_ITER == 25, f"MAX_ITER should be 25 (Pass 12B Fix 3); got {MAX_ITER}"


def test_recapture_trigger_is_at_most_0_8():
    """Round 3.5 lowered the trigger 0.9 -> 0.7. Bound from above so a
    revert to >=0.9 trips this test."""
    assert RECAPTURE_DOWNWEIGHT_TRIGGER <= 0.8, (
        f"RECAPTURE_DOWNWEIGHT_TRIGGER regressed to "
        f"{RECAPTURE_DOWNWEIGHT_TRIGGER}; should be <=0.8"
    )


def test_k_per_entity_unchanged():
    """K_PER_ENTITY stays at 3 for Round 3.5 — Rule D tightens by adding
    zero-yield decrement, not by lowering the baseline."""
    assert K_PER_ENTITY == 3


# ---------- Pass-5B: Rule E aggressive tightening ----------


def test_recapture_downweight_factor_at_most_0_3():
    """Pass-5B lowered RECAPTURE_DOWNWEIGHT_FACTOR 0.5 -> 0.3. Bound from
    above so a revert to >=0.5 trips this test. A weaker factor was one
    cause of the Round-4 bigrun4 frontier-still-growing-at-cap symptom."""
    assert RECAPTURE_DOWNWEIGHT_FACTOR <= 0.3, (
        f"RECAPTURE_DOWNWEIGHT_FACTOR regressed to {RECAPTURE_DOWNWEIGHT_FACTOR}; "
        f"should be <=0.3"
    )


def test_recapture_prune_after_at_most_2():
    """Pass-5B lowered RECAPTURE_PRUNE_AFTER 3 -> 2 (consecutive high-recap
    iters needed before zeroing a parent's budget). bigrun4 showed parents
    rarely sustain three consecutive high-recap iters because the loop hops
    between them; two is the minimum that still requires a streak."""
    assert RECAPTURE_PRUNE_AFTER <= 2, (
        f"RECAPTURE_PRUNE_AFTER regressed to {RECAPTURE_PRUNE_AFTER}; should "
        f"be <=2"
    )


def test_immediate_prune_on_recap_95_zero_new_msgs():
    """Pass-5B Fix 1 — single iter with recap >= 0.95 AND new_msgs == 0
    must zero the parent's budget immediately, skipping the
    RECAPTURE_PRUNE_AFTER streak requirement. The parent already covered
    the neighborhood; further expansion is wasted."""
    f = Frontier()
    parent = "person:Igor"
    assert f.push("q1", score=1.0, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    assert budget_before > 0
    # One high-recap observation feeds the history.
    f.observe_recapture(parent, 0.95)
    # The streak check alone wouldn't zero budget after a single iter
    # (RECAPTURE_PRUNE_AFTER >= 2). Sanity-check that assumption before
    # exercising the new path.
    assert f.per_entity_budget[parent] > 0
    # Now pair with new_messages=0 — immediate prune should fire.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] == 0, (
        f"expected budget=0 after recap=0.95 + new_msgs=0, got "
        f"{f.per_entity_budget[parent]}"
    )


def test_immediate_prune_does_not_fire_below_threshold():
    """recap=0.9 (below 0.95) + new_msgs=0 should fall back to the standard
    Rule D zero-yield decrement, NOT the immediate prune."""
    f = Frontier(k_per_entity=5)
    parent = "person:Ivy"
    assert f.push("q1", score=1.0, parent_entity=parent)
    # k=5, push -1 -> 4 remaining
    assert f.per_entity_budget[parent] == 4
    f.observe_recapture(parent, 0.9)
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    # Standard zero-yield: -1 for new_msgs=0. Budget should be 3, not 0.
    assert f.per_entity_budget[parent] == 3


def test_immediate_prune_requires_new_msgs_zero():
    """recap=0.97 with new_msgs=2 must NOT trigger the immediate prune.
    Productive iters never pay extra in Rule D — that's the invariant."""
    f = Frontier(k_per_entity=5)
    parent = "person:Ian"
    assert f.push("q1", score=1.0, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    f.observe_recapture(parent, 0.97)
    f.decrement_budget_for_yield(parent, new_messages=2, new_entities=0)
    assert f.per_entity_budget[parent] == budget_before


# ---------- Pass-5B: global saturation panic button ----------


def _fill_with_parents(
    f: Frontier,
    parents: list[str],
    queries_per_parent: int,
) -> None:
    """Push `queries_per_parent` synthetic candidates for each parent via
    `_push`, bypassing Rule C dedupe and Rule D budget-charge. Used to set
    up a frontier with a known parent distribution for saturation tests.
    """
    for p in parents:
        for j in range(queries_per_parent):
            f._push(
                QueryCandidate(
                    sort_key=(0.0, 0),
                    query=f"{p}-q{j}",
                    score=1.0,
                    parent_entity=p,
                )
            )
        # Mirror the budget map that push() would have populated.
        f.per_entity_budget[p] = f.k_per_entity


def test_global_saturation_panic_button():
    """Pass-5B Fix 1 (tightened in Pass-6C) — when frontier_size >
    GLOBAL_SATURATION_FRONTIER_SIZE AND avg recap over the last
    GLOBAL_SATURATION_WINDOW iters > GLOBAL_SATURATION_RECAP_THRESHOLD,
    parents at/above the configured percentile (by their own mean recap)
    should have their budgets zeroed.

    Pass-6C raised the trigger thresholds and percentile (50 / 0.75 / 0.8)
    so this test had to grow its frontier and bump its recap signal to
    actually hit the trigger. Construct a 60-candidate frontier with 15
    parents, 4 candidates each; recap 0.85 globally. With percentile 0.8
    the top 20% (3 of 15) should prune.
    """
    f = Frontier()
    parents = [f"person:p{i:02d}" for i in range(15)]
    _fill_with_parents(f, parents, queries_per_parent=4)
    assert len(f._heap) > GLOBAL_SATURATION_FRONTIER_SIZE
    # Per-parent recap profile: p00=0.07, p01=0.13, ..., p14=1.00 (linear
    # spread over [0.07, 1.00]). Mean recap per parent equals the single
    # value stored.
    for i, p in enumerate(parents):
        f._recapture_history[p] = [(i + 1) / len(parents)]
    # Feed GLOBAL_SATURATION_WINDOW high-recap observations to the global
    # window without touching per-parent history (use a sentinel parent
    # that's not on the heap so it doesn't get pruned).
    sentinel = "person:sentinel-not-in-heap"
    for _ in range(GLOBAL_SATURATION_WINDOW):
        # Above the Pass-6C 0.75 threshold.
        f._recent_recap.append(0.85)
    pruned_count = f._global_saturation_check()
    # 15 parents, 80th percentile cut → at least the top 20% (3) pruned.
    assert pruned_count >= 3, (
        f"expected >=3 parents pruned, got {pruned_count}"
    )
    # The highest-recap parents must be zeroed; the lowest-recap parents
    # must still have budget.
    assert f.per_entity_budget[parents[-1]] == 0  # highest recap
    assert f.per_entity_budget[parents[0]] > 0    # lowest recap
    # Avoid sentinel use warning.
    assert sentinel not in f.per_entity_budget


def test_global_saturation_no_op_when_frontier_small():
    """A 10-candidate frontier should never trigger saturation prune even at
    high recap — the panic button only fires when the queue is bloated."""
    f = Frontier()
    parents = [f"person:p{i}" for i in range(5)]
    _fill_with_parents(f, parents, queries_per_parent=2)  # 10 candidates total
    for i, p in enumerate(parents):
        f._recapture_history[p] = [0.9]
    for _ in range(GLOBAL_SATURATION_WINDOW):
        f._recent_recap.append(0.9)
    pruned = f._global_saturation_check()
    assert pruned == 0
    for p in parents:
        assert f.per_entity_budget[p] > 0


def test_global_saturation_no_op_when_recap_low():
    """If recent recap is mostly low, the frontier isn't choking even when
    big — leave budgets alone."""
    f = Frontier()
    parents = [f"person:p{i:02d}" for i in range(10)]
    _fill_with_parents(f, parents, queries_per_parent=4)
    for i, p in enumerate(parents):
        f._recapture_history[p] = [0.5]
    for _ in range(GLOBAL_SATURATION_WINDOW):
        f._recent_recap.append(0.4)  # <= GLOBAL_SATURATION_RECAP_THRESHOLD
    pruned = f._global_saturation_check()
    assert pruned == 0


def test_recent_recap_window_bounded():
    """`_recent_recap` must not grow without bound — bounded to
    GLOBAL_SATURATION_WINDOW entries after each observation."""
    f = Frontier()
    parent = "person:Q"
    assert f.push("q", score=1.0, parent_entity=parent)
    for i in range(GLOBAL_SATURATION_WINDOW * 4):
        f.observe_recapture(parent, 0.0)
    assert len(f._recent_recap) == GLOBAL_SATURATION_WINDOW


# ---------- Pass-5B: stop-reason classification reaches the loop ----------


def test_stop_reason_classified_in_loop_exit():
    """Pass-5B Fix 2 — frontier_size=40 must yield `iter_cap_uncontrolled`,
    NOT bare `iter_cap`. The Round-3.5 threshold of 50 left 35..50 in the
    bare-iter_cap band, hiding the bigrun4 "frontier still growing" case.
    Pass-5B drops the threshold to 10 so 40 lands in uncontrolled."""
    f = Frontier(max_iter=10)
    _fill_frontier(f, 40)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap_uncontrolled", sr
    assert "frontier_size=40" in sr.detail


def test_stop_reason_classified_clean():
    """Pass-5B Fix 2 — a loop that exits with empty frontier AND at least
    one of the recent EXHAUST_WINDOW iters produced new messages must
    classify as `frontier_exhausted_clean`. Mirrors the existing
    `_yielding_iter` test but pins the labelling explicitly for the new
    integration path that funnels everything through _classify_*."""
    f = Frontier(max_iter=10)
    f.observe_iteration_yield(new_messages=4, new_entities=2)
    f.observe_iteration_yield(new_messages=0, new_entities=0)
    sr = f.stop(iteration=2, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "frontier_exhausted_clean", sr


def test_classify_iter_cap_helper_exposed():
    """The `_classify_iter_cap` helper must be callable independently so
    loop.py can use it at exit time to ensure the materialized YAML's
    `derivation.stop_reason` always sees the classified variant."""
    f = Frontier(max_iter=10)
    _fill_frontier(f, 40)
    sr = f._classify_iter_cap(iteration=10)
    assert sr.rule == "iter_cap_uncontrolled"


# ---------- Pass-5B: proposer-time soft cap ----------


class _Ent:
    """Tiny stand-in for entities.Entity — has `kind` and `key` attrs and
    a `__str__` that mimics the production format. The frontier doesn't
    import Entity (no circular dep) so this duck-typed fake is fine."""

    def __init__(self, kind: str, label: str):
        self.kind = kind
        self.key = label.lower()
        self.label = label

    def __str__(self) -> str:
        return f"{self.kind}:{self.label}"


def test_has_budget_default_for_new_parent():
    """A parent we've never seen before should report has_budget=True so
    the loop doesn't filter out entities the loop has just extracted."""
    f = Frontier(k_per_entity=3)
    assert f.has_budget("person:NewlyExtracted")


def test_has_budget_false_for_exhausted_parent():
    """A parent at budget=0 should report has_budget=False — that's the
    signal loop.py uses to skip handing the entity to the proposer."""
    f = Frontier(k_per_entity=3)
    parent = "person:Exhausted"
    f.per_entity_budget[parent] = 0
    assert not f.has_budget(parent)


def test_proposer_skips_zero_budget_parents():
    """Pass-5B Fix 3 — when loop.py filters sorted_new with
    `frontier.has_budget(str(e))`, entities at budget=0 must drop out.

    This test exercises the filter EXPRESSION used by loop.py (not the
    loop itself, which requires a Searcher / Proposer / Embedder stack).
    The expression must produce the same effect: entities with remaining
    budget pass through, those at zero are dropped."""
    f = Frontier(k_per_entity=3)
    keep = _Ent("person", "Alice")
    drop = _Ent("person", "Bob")
    f.per_entity_budget[str(keep)] = 2  # has budget
    f.per_entity_budget[str(drop)] = 0  # exhausted
    candidates = [keep, drop]
    filtered = [e for e in candidates if f.has_budget(str(e))]
    assert keep in filtered
    assert drop not in filtered


# ---------- Pass-5B: stop diagnostic ----------


def test_stop_diagnostic_renders_expected_lines():
    """The diagnostic block must include all the keys the spec calls for so
    a downstream operator can grep for them — and the counts must reflect
    actual frontier state."""
    from pi_email.frontier import StopReason
    f = Frontier()
    parent = "person:Diag"
    # One real push to populate ran_queries + budget; one duplicate to
    # bump deduped_count.
    assert f.push("alpha OR beta", score=1.0, parent_entity=parent)
    f.mark_ran("alpha OR beta")
    assert f.push("beta OR alpha", score=1.0, parent_entity=parent) is False
    lines = f.stop_diagnostic(
        iterations=5,
        stop_reason=StopReason("frontier_exhausted_clean", "test"),
        total_queries_proposed=12,
    )
    blob = "\n".join(lines)
    assert "=== Stop diagnostic ===" in blob
    assert "iterations: 5" in blob
    assert "frontier_exhausted: yes" in blob
    assert "stop_reason: frontier_exhausted_clean" in blob
    assert "total_queries_proposed: 12" in blob
    assert "total_queries_executed: 1" in blob   # one mark_ran call
    assert "total_queries_deduped: 1" in blob    # one dupe push
    assert "total_queries_pruned" in blob
    assert "avg_recap_last_" in blob
    assert "budget_remaining_at_exit" in blob
    assert "person:Diag" in blob


def test_stop_diagnostic_frontier_exhausted_no_when_iter_cap():
    """If we stopped on iter_cap_uncontrolled, the diagnostic should say
    'frontier_exhausted: no' so an automated alert pipeline can fan out
    differently on the bad-exit case."""
    from pi_email.frontier import StopReason
    f = Frontier()
    lines = f.stop_diagnostic(
        iterations=35,
        stop_reason=StopReason("iter_cap_uncontrolled", "too many"),
        total_queries_proposed=95,
    )
    blob = "\n".join(lines)
    assert "frontier_exhausted: no" in blob


# ---------- Pass-6C: personal-domain score/budget bias ----------


def test_personal_domain_score_boost():
    """Pass-6C — pushing a query with an `email:` parent whose domain is in
    PERSONAL_EMAIL_DOMAINS must bump the candidate's stored score by
    PERSONAL_DOMAIN_SCORE_BOOST. The heap orders by score so this directly
    promotes the candidate up the queue.
    """
    assert "gmail.com" in PERSONAL_EMAIL_DOMAINS  # invariant for this test
    f = Frontier()
    parent = "email:jane@gmail.com"
    assert f.push("from:jane@gmail.com", score=0.6, parent_entity=parent)
    # Score should have been boosted from 0.6 -> 0.85.
    pending = list(f._heap)
    assert len(pending) == 1
    assert abs(pending[0].score - (0.6 + PERSONAL_DOMAIN_SCORE_BOOST)) < 1e-9, (
        f"expected score=0.85, got {pending[0].score}"
    )


def test_business_domain_no_boost():
    """Pass-6C — pushing a query whose `email:` parent is at a NON-personal
    domain must NOT receive the personal-domain BOOST.

    Pass-7B extended this: business parents now receive a symmetric
    PENALTY (BUSINESS_DOMAIN_SCORE_PENALTY = 0.15), so the stored score
    drops from 0.6 to 0.45. The invariant this test enforces is still
    "personal-domain boost does NOT apply to business parents"; we just
    assert the post-penalty value instead of the unchanged one.
    """
    from pi_email.frontier import BUSINESS_DOMAIN_SCORE_PENALTY

    assert "bigco.com" not in PERSONAL_EMAIL_DOMAINS
    f = Frontier()
    parent = "email:bob@bigco.com"
    assert f.push("from:bob@bigco.com", score=0.6, parent_entity=parent)
    pending = list(f._heap)
    assert len(pending) == 1
    expected = 0.6 - BUSINESS_DOMAIN_SCORE_PENALTY
    assert abs(pending[0].score - expected) < 1e-9, (
        f"expected score={expected} (0.6 with Pass-7B penalty), "
        f"got {pending[0].score}"
    )
    # And explicitly: the personal-domain boost did NOT apply.
    assert pending[0].score < 0.6 + 1e-9, (
        "personal-domain boost should NOT apply to business-domain parents"
    )


def test_personal_domain_budget_bonus():
    """Pass-6C — the per-entity budget for a personal-domain `email:` parent
    must seed at K + PERSONAL_DOMAIN_BUDGET_BONUS rather than the bare K.
    With default K=3 and bonus=2 the initial allocation is 5; after one
    push() consumes a slot the remaining budget is 4.
    """
    f = Frontier(k_per_entity=3)
    parent = "email:jane@gmail.com"
    # Direct helper check — pre-push allocation.
    assert f._initial_budget_for(parent) == 3 + PERSONAL_DOMAIN_BUDGET_BONUS
    assert f._initial_budget_for(parent) == 5
    # Push consumes one slot; remaining budget should be 4.
    assert f.push("from:jane@gmail.com", score=0.5, parent_entity=parent)
    assert f.per_entity_budget[parent] == 5 - 1 == 4


def test_business_domain_no_budget_bonus():
    """Sibling to the bonus test — business-domain parents stay at K, not
    K + bonus. Without this, the personal-domain boost would be undermined
    by a no-op-but-wide allocation everywhere."""
    f = Frontier(k_per_entity=3)
    parent = "email:bob@bigco.com"
    assert f._initial_budget_for(parent) == 3


def test_saturation_does_not_prune_personal_domain():
    """Pass-6C — when the saturation panic button fires, parents at personal
    domains must be SKIPPED even if their mean recap is in the top
    percentile. The whole point of Pass-6C is to keep the loop pointed at
    family contacts; saturation prune was the mechanism that killed those
    paths in bigrun5.
    """
    f = Frontier()
    # 14 business-domain parents + 1 personal-domain parent on the heap.
    business = [f"email:p{i:02d}@bigco.com" for i in range(14)]
    personal = "email:jane@gmail.com"
    all_parents = business + [personal]
    _fill_with_parents(f, all_parents, queries_per_parent=4)
    assert len(f._heap) > GLOBAL_SATURATION_FRONTIER_SIZE

    # Give EVERY parent a top-percentile recap so the personal one would
    # otherwise be guaranteed to land in the prune set. The business ones
    # get a slightly lower recap floor so the percentile cut is
    # well-defined and pruning hits the business set.
    for i, p in enumerate(business):
        f._recapture_history[p] = [0.85 + 0.01 * i]
    # Personal parent at the absolute top — would normally prune first.
    f._recapture_history[personal] = [1.0]
    for _ in range(GLOBAL_SATURATION_WINDOW):
        f._recent_recap.append(0.9)
    pruned = f._global_saturation_check()
    assert pruned > 0, "expected at least one business parent pruned"
    # The personal-domain parent must NOT have been zeroed despite the
    # 1.0 recap — that's the load-bearing assertion for Pass-6C.
    assert f.per_entity_budget[personal] > 0, (
        f"personal-domain parent {personal!r} was incorrectly pruned by the "
        f"saturation check (budget={f.per_entity_budget[personal]})"
    )


def test_immediate_prune_skipped_for_personal_domain():
    """Pass-6C — even with recap=0.95 and new_msgs=0, a personal-domain
    parent must NOT be immediately zeroed. Standard Rule D zero-yield
    decrement still applies, so the parent eventually drains — just not
    in one shot. Family contacts often have only one or two threads;
    one high-recap iter shouldn't kill the path.
    """
    f = Frontier(k_per_entity=3)
    parent = "email:jane@gmail.com"
    assert f.push("q1", score=1.0, parent_entity=parent)
    # Personal-domain budget is 5; after push -1 = 4.
    assert f.per_entity_budget[parent] == 4
    # Observe a near-certain recap.
    f.observe_recapture(parent, 0.96)
    # Now call decrement_budget_for_yield with new_msgs=0. For a non-
    # personal parent this would zero the budget immediately. Pass-6C
    # must keep the personal parent's budget > 0.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] > 0, (
        f"personal-domain parent {parent!r} was hit by immediate prune "
        f"(budget={f.per_entity_budget[parent]})"
    )
    # Standard Rule D zero-yield STILL applied: new_msgs=0 with
    # new_entities>0 = -1. From 4 -> 3.
    assert f.per_entity_budget[parent] == 3


# ---------- Pass-7B: symmetric business-domain penalty ----------


def test_business_domain_score_penalty():
    """Pass-7B Fix 1 — pushing a query with an `email:` parent whose
    domain is NOT in PERSONAL_EMAIL_DOMAINS must subtract
    BUSINESS_DOMAIN_SCORE_PENALTY from the candidate's stored score,
    clamping to a 0.0 minimum. Symmetric to the personal-domain boost.

    Pass-9B raised the penalty from 0.15 -> 0.35; the assertion uses the
    constant so the test tracks the tunable rather than pinning a stale
    numeric.
    """
    from pi_email.frontier import BUSINESS_DOMAIN_SCORE_PENALTY

    assert BUSINESS_DOMAIN_SCORE_PENALTY > 0, (
        "penalty is stored as a positive magnitude and SUBTRACTED in push()"
    )
    f = Frontier()
    parent = "email:research@messari.io"
    assert "messari.io" not in PERSONAL_EMAIL_DOMAINS
    assert f.push("from:research@messari.io", score=0.7, parent_entity=parent)
    pending = list(f._heap)
    assert len(pending) == 1
    expected = 0.7 - BUSINESS_DOMAIN_SCORE_PENALTY
    assert abs(pending[0].score - expected) < 1e-9, (
        f"expected score={expected:.2f}, got {pending[0].score}"
    )


def test_business_domain_score_penalty_clamps_to_zero():
    """Pass-7B Fix 1 — penalty must clamp to 0.0 minimum so a near-zero
    proposer score doesn't go negative."""
    f = Frontier()
    parent = "email:notifications@coinbase.com"
    assert f.push("from:notifications@coinbase.com", score=0.05, parent_entity=parent)
    pending = list(f._heap)
    assert pending[0].score == 0.0, (
        f"expected clamp to 0.0, got {pending[0].score}"
    )


def test_personal_domain_no_penalty():
    """Pass-7B Fix 1 — personal-domain parents must STILL receive the
    Pass-6C boost (+0.25), NOT the Pass-7B penalty. The two branches are
    mutually exclusive in push() so there's no double-apply risk; this
    test pins that invariant.
    """
    f = Frontier()
    parent = "email:jane@gmail.com"
    assert f.push("from:jane@gmail.com", score=0.7, parent_entity=parent)
    pending = list(f._heap)
    assert len(pending) == 1
    # 0.7 + 0.25 = 0.95 (boost still applies, no penalty)
    assert abs(pending[0].score - 0.95) < 1e-9, (
        f"expected score=0.95 (Pass-6C boost, no Pass-7B penalty), "
        f"got {pending[0].score}"
    )


def test_business_domain_tighter_immediate_prune():
    """Pass-7B Fix 2 — business-domain parents trigger immediate-prune at
    recap >= 0.85 (vs the general 0.95 threshold) with new_msgs == 0.
    A single observation at 0.85 must zero the budget for a business
    parent — would NOT have fired before Pass-7B."""
    from pi_email.frontier import BUSINESS_IMMEDIATE_PRUNE_RECAP

    f = Frontier()
    parent = "email:research@messari.io"
    assert f.push("q1", score=1.0, parent_entity=parent)
    assert f.per_entity_budget[parent] > 0
    # Observe recap exactly at the business threshold (0.85). The
    # general 0.95 threshold would NOT fire here.
    f.observe_recapture(parent, BUSINESS_IMMEDIATE_PRUNE_RECAP)
    # Sanity: the streak prune wouldn't have fired either (only one obs).
    assert f.per_entity_budget[parent] > 0
    # Now the zero-yield decrement at new_msgs=0 should fire the
    # Pass-7B business-tighter immediate-prune branch.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] == 0, (
        f"expected budget=0 after recap=0.85 + new_msgs=0 (business), got "
        f"{f.per_entity_budget[parent]}"
    )


def test_business_consecutive_zero_msgs_prunes():
    """Pass-7B Fix 2 — a business-domain parent with new_msgs=0 for two
    consecutive iters must be pruned, even if recap is below the 0.85
    immediate-prune threshold. Tighter than Pass-5B's
    RECAPTURE_PRUNE_AFTER (which still requires high recap)."""
    from pi_email.frontier import BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE

    assert BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE == 2
    f = Frontier(k_per_entity=5)
    parent = "email:notifications@coinbase.com"
    assert f.push("q1", score=1.0, parent_entity=parent)
    # Use a LOW recap so neither immediate prune (0.85+) nor streak
    # prune (>0.7) fires. The only mechanism that should zero this
    # parent is the consecutive-zero-msgs path.
    low_recap = 0.3
    # Iter 1: zero msgs. counter=1. Standard Rule D applies (-1).
    f.observe_recapture(parent, low_recap)
    before = f.per_entity_budget[parent]
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] > 0, (
        f"after 1 zero-msg iter the consecutive prune should NOT have "
        f"fired (counter=1 < {BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE}); "
        f"budget was {before} -> {f.per_entity_budget[parent]}"
    )
    # Iter 2: zero msgs again. counter=2 >= 2 -> consecutive prune.
    f.observe_recapture(parent, low_recap)
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] == 0, (
        f"expected budget=0 after 2 consecutive zero-msg iters (business), "
        f"got {f.per_entity_budget[parent]}"
    )


def test_business_consecutive_zero_msgs_reset_by_productive_iter():
    """Productive iter (new_messages > 0) must reset the consecutive
    counter so a one-off dead iter inside an otherwise productive parent
    isn't penalized."""
    f = Frontier(k_per_entity=10)
    parent = "email:bob@bigco.com"
    assert f.push("q1", score=1.0, parent_entity=parent)
    f.observe_recapture(parent, 0.3)
    # Iter 1: zero msgs.
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    # Iter 2: productive — counter resets.
    f.observe_recapture(parent, 0.3)
    f.decrement_budget_for_yield(parent, new_messages=5, new_entities=3)
    # Iter 3: zero msgs again. Counter should be 1, not 2.
    f.observe_recapture(parent, 0.3)
    f.decrement_budget_for_yield(parent, new_messages=0, new_entities=2)
    assert f.per_entity_budget[parent] > 0, (
        "consecutive counter should have reset after the productive iter"
    )


def test_business_high_recap_zero_entities_drops_parent():
    """Pass-7B Fix 5 — business parent with recap >= 0.7 AND
    new_entities == 0 must have budget zeroed AND any pending heap
    entries' scores dropped to 0 (so they pop LAST, after personal
    parents). Reverse cap."""
    from pi_email.frontier import BUSINESS_DROP_RECAP_THRESHOLD

    f = Frontier()
    parent = "email:research@messari.io"
    # Push two queries from this parent so we can observe the pending
    # score-drop side effect.
    assert f.push("q1", score=0.8, parent_entity=parent)
    assert f.push("q2", score=0.9, parent_entity=parent)
    # Pre-condition: pending scores > 0 after the Pass-7B penalty.
    pending = list(f._heap)
    assert all(c.score > 0 for c in pending), (
        f"expected pending scores > 0 pre-drop, got "
        f"{[c.score for c in pending]}"
    )
    assert f.per_entity_budget[parent] > 0
    # Fire Fix 5: recap=0.75 (>= 0.7), new_entities=0.
    acted = f.business_high_recap_drop(
        parent, recap=BUSINESS_DROP_RECAP_THRESHOLD + 0.05, new_entities=0
    )
    assert acted is True
    # Budget zeroed.
    assert f.per_entity_budget[parent] == 0
    # Pending scores dropped to 0 — they pop LAST.
    for c in f._heap:
        if c.parent_entity == parent:
            assert c.score == 0.0, (
                f"expected pending score=0 after Fix 5 drop, got {c.score}"
            )


def test_business_high_recap_drop_no_op_below_threshold():
    """Fix 5 must NOT fire when recap is below 0.7."""
    f = Frontier()
    parent = "email:bob@bigco.com"
    assert f.push("q1", score=0.8, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    acted = f.business_high_recap_drop(parent, recap=0.65, new_entities=0)
    assert acted is False
    assert f.per_entity_budget[parent] == budget_before


def test_business_high_recap_drop_no_op_with_new_entities():
    """Fix 5 must NOT fire when new_entities > 0 — a productive iter
    earned its budget even at high recap."""
    f = Frontier()
    parent = "email:bob@bigco.com"
    assert f.push("q1", score=0.8, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    acted = f.business_high_recap_drop(parent, recap=0.95, new_entities=3)
    assert acted is False
    assert f.per_entity_budget[parent] == budget_before


def test_iter_cap_throttled_classification():
    """Pass-7B Fix 4 — iter_cap with total_pruned > total_executed * 0.5
    must be labeled `iter_cap_throttled` rather than `iter_cap_uncontrolled`.
    Communicates that the run was working hard to terminate.
    """
    f = Frontier(max_iter=10)
    # Populate frontier with > ITER_CAP_UNCONTROLLED_FRONTIER entries —
    # without the throttled override this would be `iter_cap_uncontrolled`.
    _fill_frontier(f, 20)
    # Set up: ran_queries with 10 entries, budget_rejected_count = 6.
    # 6 > 10 * 0.5 = 5 -> throttled override fires.
    f.ran_queries = [f"q{i}" for i in range(10)]
    f.budget_rejected_count = 6
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap_throttled", (
        f"expected iter_cap_throttled, got {sr.rule}: {sr.detail}"
    )
    # Detail must mention the throttle signal.
    assert "executed=10" in sr.detail
    assert "pruned=6" in sr.detail


def test_iter_cap_throttled_does_not_fire_when_pruned_below_half():
    """Fix 4 — pruned == executed * 0.5 (exactly half) should NOT fire
    throttled; the threshold is strict `>`."""
    f = Frontier(max_iter=10)
    _fill_frontier(f, 20)
    f.ran_queries = [f"q{i}" for i in range(10)]
    f.budget_rejected_count = 5  # exactly 50%, NOT strictly greater
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    # Falls through to iter_cap_uncontrolled (frontier_size=20 > 10).
    assert sr.rule == "iter_cap_uncontrolled", (
        f"expected iter_cap_uncontrolled, got {sr.rule}"
    )


def test_personal_domain_unaffected_by_business_rules():
    """Pass-7B — all new business-domain rules must skip personal-domain
    parents. Pins the symmetric structure: personal gets boost+bonus,
    business gets penalty+tighter-prune, and the two paths never cross.
    """
    from pi_email.frontier import (
        BUSINESS_DROP_RECAP_THRESHOLD,
        BUSINESS_IMMEDIATE_PRUNE_RECAP,
        _is_business_domain_email_parent,
    )

    personal = "email:jane@gmail.com"
    assert not _is_business_domain_email_parent(personal)

    # Rule 1: score penalty does NOT apply (still gets boost).
    f = Frontier()
    assert f.push("q1", score=0.7, parent_entity=personal)
    assert abs(f._heap[0].score - 0.95) < 1e-9, (
        "personal parent should get +0.25 boost, NOT -0.15 penalty"
    )

    # Rule 2: business-tighter immediate prune at 0.85 does NOT zero
    # the budget for a personal parent.
    f2 = Frontier(k_per_entity=3)
    assert f2.push("q1", score=1.0, parent_entity=personal)
    budget_before = f2.per_entity_budget[personal]
    f2.observe_recapture(personal, BUSINESS_IMMEDIATE_PRUNE_RECAP)
    f2.decrement_budget_for_yield(personal, new_messages=0, new_entities=2)
    assert f2.per_entity_budget[personal] > 0, (
        f"personal parent must NOT be hit by the Pass-7B business "
        f"immediate-prune (budget={f2.per_entity_budget[personal]})"
    )

    # Rule 3: consecutive zero-msgs prune does NOT fire for personal.
    f3 = Frontier(k_per_entity=10)
    assert f3.push("q1", score=1.0, parent_entity=personal)
    for _ in range(5):  # WELL beyond BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE
        f3.observe_recapture(personal, 0.3)
        f3.decrement_budget_for_yield(personal, new_messages=0, new_entities=2)
        if f3.per_entity_budget[personal] == 0:
            break
    # The personal parent will eventually drain via standard Rule D
    # zero-yield decrement, but NOT via the Pass-7B consecutive path
    # (which would fire at iter 2). Since we can't easily separate
    # those, we instead assert the counter itself is not tracked.
    assert personal not in f3._consecutive_zero_msgs, (
        "consecutive zero-msg counter must not be tracked for personal "
        "parents (it's a business-domain-only mechanism)"
    )

    # Rule 4: business_high_recap_drop is a no-op for personal parents.
    f4 = Frontier()
    assert f4.push("q1", score=0.8, parent_entity=personal)
    budget_before = f4.per_entity_budget[personal]
    pending_score_before = f4._heap[0].score
    acted = f4.business_high_recap_drop(
        personal, recap=BUSINESS_DROP_RECAP_THRESHOLD + 0.2, new_entities=0
    )
    assert acted is False, (
        "business_high_recap_drop must be a no-op for personal parents"
    )
    assert f4.per_entity_budget[personal] == budget_before
    assert f4._heap[0].score == pending_score_before

    # Rule 5: is_business_saturated is False for personal parents.
    f5 = Frontier()
    assert not f5.is_business_saturated(personal, recap=0.99)
    f5._recapture_history[personal] = [0.99]
    assert not f5.is_business_saturated(personal)


def test_is_business_saturated_helper():
    """Pass-7B Fix 3 — helper used by loop.py to skip the proposer call
    and filter sorted_new. Must respect both explicit recap (current
    iter) and stored history (entities surfaced previously)."""
    from pi_email.frontier import BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD

    f = Frontier()
    business = "email:research@messari.io"
    personal = "email:jane@gmail.com"
    person = "person:Alice"

    # With explicit recap: business above the threshold -> saturated.
    assert f.is_business_saturated(
        business, recap=BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD
    )
    assert not f.is_business_saturated(
        business, recap=BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD - 0.01
    )

    # Personal and person never saturate, no matter the recap.
    assert not f.is_business_saturated(personal, recap=1.0)
    assert not f.is_business_saturated(person, recap=1.0)

    # History-based (no explicit recap arg).
    f._recapture_history[business] = [0.95]
    assert f.is_business_saturated(business)
    f._recapture_history[business] = [0.5]
    assert not f.is_business_saturated(business)


def test_personal_domain_helper_covers_known_providers():
    """Catch the easy regression where the PERSONAL_EMAIL_DOMAINS set
    silently shrinks. Spot-check the handful of providers that bigrun5
    actually showed family addresses on, plus a couple of well-known
    aliases."""
    from pi_email.frontier import _is_personal_domain_email_parent
    for d in ("gmail.com", "icloud.com", "yahoo.com", "hotmail.com",
              "outlook.com", "me.com", "aol.com"):
        assert _is_personal_domain_email_parent(f"email:user@{d}"), (
            f"expected {d} to be in PERSONAL_EMAIL_DOMAINS"
        )
    # Negative cases.
    for d in ("tally.xyz", "coinbase.com", "messari.io", "rippling.com"):
        assert not _is_personal_domain_email_parent(f"email:user@{d}"), (
            f"unexpected match for business domain {d}"
        )
    # Non-email kinds never match.
    assert not _is_personal_domain_email_parent("person:Jane Smith")
    assert not _is_personal_domain_email_parent("SEED")
    assert not _is_personal_domain_email_parent(None)
    assert not _is_personal_domain_email_parent("")
    # Malformed `email:` payloads must not crash.
    assert not _is_personal_domain_email_parent("email:")
    assert not _is_personal_domain_email_parent("email:no-at-sign")


# ---------- Pass-9B: stronger business control ----------


def test_business_domain_strong_penalty():
    """Pass-9B Fix 1 — BUSINESS_DOMAIN_SCORE_PENALTY raised 0.15 -> 0.35.
    A business proposal at score=0.6 now lands at 0.25, well below a
    personal proposal at 0.6 (0.85 after boost). Pins the new magnitude
    so a revert is caught by the test suite.
    """
    from pi_email.frontier import BUSINESS_DOMAIN_SCORE_PENALTY

    assert BUSINESS_DOMAIN_SCORE_PENALTY >= 0.35, (
        f"Pass-9B requires penalty >= 0.35; got {BUSINESS_DOMAIN_SCORE_PENALTY}"
    )
    f = Frontier()
    parent = "email:research@figment.io"
    assert f.push("from:research@figment.io", score=0.6, parent_entity=parent)
    pending = list(f._heap)
    assert len(pending) == 1
    # 0.6 - 0.35 = 0.25 with the Pass-9B penalty (was 0.45 in Pass-7B).
    assert abs(pending[0].score - 0.25) < 1e-9, (
        f"expected score=0.25 with Pass-9B penalty, got {pending[0].score}"
    )


def test_business_bias_enforces_personal_priority():
    """Pass-9B Fix 2 — when 3+ of the last BUSINESS_BIAS_WINDOW iters
    chose a business-domain parent AND avg recap > 0.5, the next pop()
    must skip business candidates and return the first non-business one
    even though business has a higher heap score.
    """
    from pi_email.frontier import (
        BUSINESS_BIAS_RECAP_THRESHOLD,
        BUSINESS_BIAS_WINDOW,
    )

    f = Frontier()
    # Seed the bias window: 4 of the last 5 iters were business.
    f._recent_iter_business_parent = [True, True, True, True, False]
    assert len(f._recent_iter_business_parent) == BUSINESS_BIAS_WINDOW
    # Recent recap avg = 0.55, above the BUSINESS_BIAS_RECAP_THRESHOLD.
    for _ in range(BUSINESS_BIAS_WINDOW):
        f._recent_recap.append(BUSINESS_BIAS_RECAP_THRESHOLD + 0.05)
    # Heap setup: business candidate at score 0.9, personal at 0.4.
    # Use _push to bypass push()'s domain-rewrites so the recorded
    # scores match what the test asserts on.
    biz_parent = "email:research@messari.io"
    personal_parent = "email:jane@gmail.com"
    f._push(QueryCandidate(
        sort_key=(0.0, 0),
        query="biz-q",
        score=0.9,
        parent_entity=biz_parent,
    ))
    f._push(QueryCandidate(
        sort_key=(0.0, 0),
        query="personal-q",
        score=0.4,
        parent_entity=personal_parent,
    ))
    # Sanity: by raw score, business would pop first WITHOUT the bias.
    assert f._business_bias_active()
    cand = f.pop()
    assert cand is not None
    assert cand.parent_entity == personal_parent, (
        f"expected personal parent, got {cand.parent_entity!r} "
        f"(score={cand.score})"
    )
    # Business candidate should remain on the heap.
    remaining_parents = [c.parent_entity for c in f._heap]
    assert biz_parent in remaining_parents


def test_business_bias_inactive_when_low_recap():
    """Bias trigger gate is two-part: business-heavy AND high-recap.
    When recap is low (productive iters), the trigger MUST stay off
    even with 5/5 business iters."""
    from pi_email.frontier import BUSINESS_BIAS_WINDOW

    f = Frontier()
    f._recent_iter_business_parent = [True] * BUSINESS_BIAS_WINDOW
    for _ in range(BUSINESS_BIAS_WINDOW):
        f._recent_recap.append(0.2)  # well below 0.5 threshold
    assert not f._business_bias_active()


def test_business_bias_inactive_without_enough_history():
    """Until BUSINESS_BIAS_WINDOW iters have been observed, the bias
    trigger MUST stay off so the early-run pop order is undisturbed."""
    from pi_email.frontier import (
        BUSINESS_BIAS_RECAP_THRESHOLD,
        BUSINESS_BIAS_WINDOW,
    )

    f = Frontier()
    # Only 2 entries — well short of the window.
    f._recent_iter_business_parent = [True, True]
    for _ in range(BUSINESS_BIAS_WINDOW):
        f._recent_recap.append(BUSINESS_BIAS_RECAP_THRESHOLD + 0.1)
    assert not f._business_bias_active()


def test_business_push_per_iter_cap_drops_excess():
    """Pass-9B Fix 3 — at most BUSINESS_PUSH_PER_ITER_CAP business-domain
    candidates may be admitted within a single iteration. Iter 27 of
    bigrun8 pushed 160 in one shot; the cap rejects the excess.
    """
    from pi_email.frontier import BUSINESS_PUSH_PER_ITER_CAP

    f = Frontier()
    f.begin_iteration()
    # Use 10 lexically-very-different query strings so the no-embedder
    # SequenceMatcher dedupe (threshold 0.80) treats them as distinct.
    # Adjacent strings sharing the parent prefix collapse easily; whole
    # words are the safest pattern.
    queries_and_parents = [
        ("alpha widgets", "email:research@figment.io"),
        ("subscription receipt 2024", "email:notifications@coinbase.com"),
        ("zeta product launch invite", "email:sales@messari.io"),
        ("conference badge tracking", "email:events@tally.xyz"),
        ("yearly contract renewal", "email:legal@rippling.com"),
        ("paper shipping schedule overflow", "email:logistics@stripe.com"),
        ("vendor onboarding portal", "email:partners@plaid.com"),
        ("quarterly board summary", "email:investor@a16z.com"),
        ("compliance audit ticket", "email:risk@circle.com"),
        ("trademark refresh notice", "email:trademark@uspto.com"),
    ]
    assert len(queries_and_parents) == 10
    pushed_count = 0
    for q, p in queries_and_parents:
        if f.push(q, score=0.8, parent_entity=p):
            pushed_count += 1
    assert pushed_count == BUSINESS_PUSH_PER_ITER_CAP, (
        f"expected exactly {BUSINESS_PUSH_PER_ITER_CAP} business pushes "
        f"admitted (cap), got {pushed_count}"
    )
    assert f.business_cap_dropped_count == 10 - BUSINESS_PUSH_PER_ITER_CAP, (
        f"expected {10 - BUSINESS_PUSH_PER_ITER_CAP} cap drops, got "
        f"{f.business_cap_dropped_count}"
    )
    # New iteration resets the counter — business pushes work again.
    f.begin_iteration()
    assert f.push(
        "fresh distinct payload after iter boundary reset",
        score=0.8,
        parent_entity="email:bot99@bigco.com",
    )


def test_business_push_per_iter_cap_skips_non_business():
    """Personal-domain and person: parents must NOT count against the
    business per-iter cap. Otherwise the cap would inadvertently throttle
    legitimate personal-domain expansion."""
    from pi_email.frontier import BUSINESS_PUSH_PER_ITER_CAP

    f = Frontier()
    f.begin_iteration()
    # Burn the business cap first using lexically-distinct queries.
    burn_queries = [
        ("alpha widgets shipping", "email:bot01@bigco.com"),
        ("zeta launch invite", "email:bot02@bigco.com"),
        ("quarterly board summary memo", "email:bot03@bigco.com"),
        ("compliance audit ticket overflow", "email:bot04@bigco.com"),
        ("trademark refresh portal notice", "email:bot05@bigco.com"),
    ]
    assert len(burn_queries) == BUSINESS_PUSH_PER_ITER_CAP
    for q, p in burn_queries:
        assert f.push(q, score=0.8, parent_entity=p)
    # Now a personal push should still succeed in the same iter.
    assert f.push(
        "lovely vacation photos with family",
        score=0.5,
        parent_entity="email:jane@gmail.com",
    )
    # And a person: push too.
    assert f.push(
        "birthday dinner reservation downtown",
        score=0.5,
        parent_entity="person:Alice",
    )


def test_business_explosion_zeroes_and_clears():
    """Pass-9B Fix 4 — when a business-domain parent produces
    BUSINESS_EXPLOSION_NEW_ENTS+ new entities in one iter, zero its
    budget AND clear any pending queries from that parent currently on
    the heap. Iter 27 of bigrun8 (figment.io -> 160 entities) is the
    motivating case.
    """
    from pi_email.frontier import BUSINESS_EXPLOSION_NEW_ENTS

    f = Frontier(k_per_entity=10)
    parent = "email:research@figment.io"
    # Seed the parent in the budget map via a real push.
    assert f.push("from:research@figment.io", score=0.8, parent_entity=parent)
    # Plant several extra pending queries with the same parent (e.g. left
    # over from earlier iterations when this parent had budget).
    for i in range(3):
        f._push(QueryCandidate(
            sort_key=(0.0, 0),
            query=f"child-of-figment-{i}",
            score=0.5,
            parent_entity=parent,
        ))
    # Pre-conditions.
    assert f.per_entity_budget[parent] > 0
    assert sum(1 for c in f._heap if c.parent_entity == parent) >= 4
    # Trigger the explosion handler — productive iter with 60 new ents.
    f.decrement_budget_for_yield(
        parent_entity=parent,
        new_messages=10,
        new_entities=BUSINESS_EXPLOSION_NEW_ENTS + 10,
    )
    # Budget zeroed.
    assert f.per_entity_budget[parent] == 0, (
        f"expected budget=0 after explosion, got {f.per_entity_budget[parent]}"
    )
    # All children cleared from the heap.
    remaining = [c for c in f._heap if c.parent_entity == parent]
    assert not remaining, (
        f"expected heap free of {parent!r}'s children, got "
        f"{[c.query for c in remaining]}"
    )


def test_business_explosion_no_op_below_threshold():
    """Fix 4 must NOT fire when new_entities < BUSINESS_EXPLOSION_NEW_ENTS.
    Tighter parents at modest yield should follow the standard Rule D /
    Rule E paths instead.
    """
    from pi_email.frontier import BUSINESS_EXPLOSION_NEW_ENTS

    f = Frontier()
    parent = "email:research@figment.io"
    assert f.push("q1", score=0.8, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    # 1 below the threshold.
    f.decrement_budget_for_yield(
        parent_entity=parent,
        new_messages=10,
        new_entities=BUSINESS_EXPLOSION_NEW_ENTS - 1,
    )
    # Budget unchanged (productive iter, no penalty applied).
    assert f.per_entity_budget[parent] == budget_before


def test_business_explosion_no_op_for_personal_parent():
    """Fix 4 is scoped to BUSINESS-domain parents. A personal-domain
    parent that surfaces 60 family contacts in one iter (which is the
    desirable outcome of the Pass-6C boost) MUST NOT be pruned.
    """
    from pi_email.frontier import BUSINESS_EXPLOSION_NEW_ENTS

    f = Frontier()
    parent = "email:jane@gmail.com"
    assert f.push("q1", score=0.8, parent_entity=parent)
    budget_before = f.per_entity_budget[parent]
    f.decrement_budget_for_yield(
        parent_entity=parent,
        new_messages=10,
        new_entities=BUSINESS_EXPLOSION_NEW_ENTS + 10,
    )
    # Personal parent untouched — productive iter pays nothing in Rule D.
    assert f.per_entity_budget[parent] == budget_before


def test_uncontrolled_threshold_tightened():
    """Pass-9B Fix 5 — ITER_CAP_UNCONTROLLED_FRONTIER lowered 10 -> 7.
    A frontier_size of 8 at iter_cap must now classify as
    `iter_cap_uncontrolled` (was bare `iter_cap` under the old 10
    threshold).
    """
    assert ITER_CAP_UNCONTROLLED_FRONTIER <= 7, (
        f"Pass-9B requires ITER_CAP_UNCONTROLLED_FRONTIER <= 7; got "
        f"{ITER_CAP_UNCONTROLLED_FRONTIER}"
    )
    f = Frontier(max_iter=10)
    _fill_frontier(f, 8)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "iter_cap_uncontrolled", (
        f"expected iter_cap_uncontrolled at frontier_size=8, got {sr.rule}"
    )


def test_begin_iteration_resets_business_counter():
    """The Pass-9B per-iter business cap depends on begin_iteration()
    being called at the top of each loop iter. Without the reset, the
    cap would carry over and silently throttle later iters."""
    from pi_email.frontier import BUSINESS_PUSH_PER_ITER_CAP

    f = Frontier()
    burn_queries = [
        ("alpha widgets shipping schedule", "email:bot01@bigco.com"),
        ("zeta launch invite calendar", "email:bot02@bigco.com"),
        ("quarterly board memo summary", "email:bot03@bigco.com"),
        ("compliance audit ticket overflow", "email:bot04@bigco.com"),
        ("trademark refresh portal notice", "email:bot05@bigco.com"),
    ]
    assert len(burn_queries) == BUSINESS_PUSH_PER_ITER_CAP
    for q, p in burn_queries:
        assert f.push(q, score=0.8, parent_entity=p)
    # One more attempt is dropped at the cap.
    assert not f.push(
        "yearly contract renewal portal kickoff",
        score=0.8,
        parent_entity="email:overflow@bigco.com",
    )
    # Now reset and verify a business push works again.
    f.begin_iteration()
    assert f._business_push_this_iter == 0
    assert f.push(
        "annual vendor invoice reconciliation memo",
        score=0.8,
        parent_entity="email:fresh@bigco.com",
    )


# ---------- Pass 12B: no_new_persons_after_N graceful stop ----------


def test_no_new_persons_after_N_stops_cleanly():
    """Pass 12B Fix 1 — after NO_NEW_PERSONS_WINDOW consecutive iters
    with zero new family-relation-grounded entities, stop() must return
    `no_new_persons_after_N`. Convergence-without-iter_cap is the
    whole point of Pass 12B."""
    f = Frontier(max_iter=100)  # generous so iter_cap doesn't preempt
    # Frontier must be NON-empty for the new rule to fire (frontier
    # empty would short-circuit to frontier_exhausted_no_yield first).
    _fill_frontier(f, 5)
    # Feed exactly NO_NEW_PERSONS_WINDOW zero-family observations.
    for _ in range(NO_NEW_PERSONS_WINDOW):
        f.observe_person_yield(parent_entity="SEED", new_person_count=0)
    sr = f.stop(iteration=10, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    assert sr.rule == "no_new_persons_after_N", (
        f"expected no_new_persons_after_N, got {sr.rule}: {sr.detail}"
    )
    # Detail must mention the window length so log scrapers know which
    # rule signed off.
    assert str(NO_NEW_PERSONS_WINDOW) in sr.detail


def test_no_family_signal_resets_on_yield():
    """Pass 12B Fix 1 — a single nonzero family-yield observation must
    keep the rule from firing, even if surrounded by zero iters.

    Sequence: 5 zero iters, then 1 with new_family=1, then 7 more zero —
    8 zeros total but not CONSECUTIVE. The deque holds the last 8 entries:
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> last 8 = [1, 0, 0, 0, 0, 0, 0, 0]
    -> sum != 0 -> rule does NOT fire.
    """
    f = Frontier(max_iter=100)
    _fill_frontier(f, 5)
    for _ in range(5):
        f.observe_person_yield(parent_entity="SEED", new_person_count=0)
    # The yield resets the consecutive count.
    f.observe_person_yield(parent_entity="SEED", new_person_count=1)
    for _ in range(7):
        f.observe_person_yield(parent_entity="SEED", new_person_count=0)
    # Sanity: window is full (deque maxlen=NO_NEW_PERSONS_WINDOW=8).
    assert len(f._recent_person_yield) == NO_NEW_PERSONS_WINDOW
    # Last 8 = [1, 0, 0, 0, 0, 0, 0, 0]; sum == 1 != 0 -> rule MUST NOT fire.
    sr = f.stop(iteration=20, spend_tokens=0, corpus_bytes=0)
    # stop() may still return iter_cap or similar if those gates fire,
    # but it MUST NOT be no_new_persons_after_N.
    if sr is not None:
        assert sr.rule != "no_new_persons_after_N", (
            f"rule fired despite a nonzero yield in the window: {sr.rule}"
        )


def test_proposer_followup_cap_blocks_unproductive_parent():
    """Pass 12B Fix 2 — after a parent has been popped
    UNPRODUCTIVE_PARENT_POP_LIMIT times with zero family signal, further
    pushes from that parent must be rejected by push().

    Tracks `unproductive_parent_rejected_count` for the diagnostic.
    """
    f = Frontier()
    parent = "person:UnproductivePerson"
    # Sanity baseline — first push works.
    assert f.push("q1", score=1.0, parent_entity=parent)
    # Two unproductive pops (zero family yield) → counter hits the limit.
    for _ in range(UNPRODUCTIVE_PARENT_POP_LIMIT):
        f.observe_person_yield(
            parent_entity=parent, new_person_count=0
        )
    # is_parent_unproductive must report True at the threshold.
    assert f.is_parent_unproductive(parent), (
        f"parent should be flagged unproductive after "
        f"{UNPRODUCTIVE_PARENT_POP_LIMIT} zero-yield pops"
    )
    # New push from same parent must be rejected.
    rejected_before = f.unproductive_parent_rejected_count
    assert f.push("q2", score=1.0, parent_entity=parent) is False
    assert f.unproductive_parent_rejected_count == rejected_before + 1


def test_proposer_followup_cap_resets_on_family_yield():
    """Fix 2 cap must reset when the parent contributes family signal —
    a parent that surfaced a relation in one of its iters has earned a
    fresh leash, even if subsequent iters are dry."""
    f = Frontier()
    parent = "person:OnceProductive"
    assert f.push("q1", score=1.0, parent_entity=parent)
    f.observe_person_yield(parent_entity=parent, new_person_count=0)
    f.observe_person_yield(parent_entity=parent, new_person_count=1)
    # Counter reset → is_parent_unproductive must be False.
    assert not f.is_parent_unproductive(parent)
    assert f.push("q2", score=1.0, parent_entity=parent)


def test_proposer_followup_cap_skips_personal_domain():
    """Personal-domain parents are exempt from Fix 2 just as they are
    from Pass-5B/7B immediate-prunes. Family contacts often have only
    one or two threads; a dry pop or two isn't a death sentence."""
    f = Frontier()
    parent = "email:jane@gmail.com"
    assert f.push("q1", score=1.0, parent_entity=parent)
    # Far beyond the threshold.
    for _ in range(UNPRODUCTIVE_PARENT_POP_LIMIT + 3):
        f.observe_person_yield(
            parent_entity=parent, new_person_count=0
        )
    assert not f.is_parent_unproductive(parent), (
        "personal-domain parent must be exempt from the unproductive cap"
    )
    # And a follow-up push still works.
    assert f.push("q-followup", score=1.0, parent_entity=parent)


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_diagnostic_includes_family_yield():
    """Pass 12B Fix 5 — `stop_diagnostic` block must surface the new
    convergence-related fields so post-run analysis can verify whether
    the no-family-signal rule fired correctly."""
    from pi_email.frontier import StopReason

    f = Frontier()
    # Feed a handful of family-yield observations so the diagnostic
    # has data to render.
    for c in (1, 0, 0, 2, 0):
        f.observe_person_yield(parent_entity="SEED", new_person_count=c)
    # Force the proposer-skipped counter to be non-zero — exercises the
    # diagnostic for Fix 4.
    f._should_skip_proposer_next_iter = True
    assert f.consume_proposer_skip_signal()
    # And exercise Fix 2's reject counter.
    f.unproductive_parent_rejected_count = 3

    lines = f.stop_diagnostic(
        iterations=10,
        stop_reason=StopReason("no_new_persons_after_N", "test"),
        total_queries_proposed=20,
    )
    blob = "\n".join(lines)
    assert f"person_yield_last_{NO_NEW_PERSONS_WINDOW}" in blob, (
        "diagnostic should include the family-yield window"
    )
    assert "consecutive_zero_family_iters:" in blob
    assert "proposer_skipped_iters: 1" in blob
    assert "unproductive_parent_rejected: 3" in blob


def test_soft_saturation_signals_proposer_skip():
    """Pass 12B Fix 4 — when frontier_size > SOFT_SATURATION_FRONTIER_SIZE
    AND recent recap avg > SOFT_SATURATION_RECAP_THRESHOLD, the soft
    saturation check must arm the proposer-skip flag for the next iter
    WITHOUT mutating budgets.
    """
    from pi_email.frontier import (
        SOFT_SATURATION_FRONTIER_SIZE,
        SOFT_SATURATION_RECAP_THRESHOLD,
    )

    f = Frontier()
    parents = [f"person:p{i:02d}" for i in range(8)]
    _fill_with_parents(f, parents, queries_per_parent=5)
    assert len(f._heap) > SOFT_SATURATION_FRONTIER_SIZE
    # High recap on every parent so the soft check's avg condition fires.
    for p in parents:
        f._recapture_history[p] = [SOFT_SATURATION_RECAP_THRESHOLD + 0.1]
    for _ in range(GLOBAL_SATURATION_WINDOW):
        f._recent_recap.append(SOFT_SATURATION_RECAP_THRESHOLD + 0.1)
    # Pre-condition: budgets are non-zero (soft check doesn't touch them).
    budgets_before = {p: f.per_entity_budget[p] for p in parents}
    fired = f._soft_saturation_check()
    assert fired is True
    # Budgets MUST be unchanged — soft signal doesn't prune.
    for p in parents:
        assert f.per_entity_budget[p] == budgets_before[p], (
            f"soft saturation must not mutate budgets; {p!r} changed"
        )
    # consume_proposer_skip_signal returns True once, then False (one-shot).
    assert f.consume_proposer_skip_signal()
    assert not f.consume_proposer_skip_signal()
    assert f.proposer_skipped_iters == 1


def test_stop_priority_frontier_empty_beats_no_family_signal():
    """When BOTH frontier_exhausted AND no-family-signal would fire,
    frontier_exhausted wins. Frontier exhaustion is the cleaner exit —
    the loop discovered everything it knew how to look for. Pass 12B
    Fix 1 must not preempt that.
    """
    f = Frontier(max_iter=100)
    # Empty frontier AND 8 zero-family iters.
    for _ in range(NO_NEW_PERSONS_WINDOW):
        f.observe_person_yield(parent_entity="SEED", new_person_count=0)
        f.observe_iteration_yield(new_messages=0, new_entities=0)
    sr = f.stop(iteration=NO_NEW_PERSONS_WINDOW, spend_tokens=0, corpus_bytes=0)
    assert sr is not None
    # Frontier-empty takes priority.
    assert sr.rule.startswith("frontier_exhausted"), (
        f"expected frontier_exhausted_* (priority over no_family_signal), "
        f"got {sr.rule}"
    )


def test_no_family_signal_does_not_fire_on_short_run():
    """The window must be FULL before the rule fires. A 3-iter fixture
    that legitimately empties its frontier at iter 3 should never hit
    no_new_persons_after_N — the window only had 3 entries.
    """
    f = Frontier(max_iter=100)
    _fill_frontier(f, 5)
    # Only 3 iters observed.
    for _ in range(3):
        f.observe_person_yield(parent_entity="SEED", new_person_count=0)
    sr = f.stop(iteration=3, spend_tokens=0, corpus_bytes=0)
    # Should return None (loop keeps going) since neither iter_cap nor
    # any other terminal condition fires.
    assert sr is None or sr.rule != "no_new_persons_after_N", (
        f"rule fired with only 3 zero iters in the window (need "
        f"{NO_NEW_PERSONS_WINDOW}); got {sr}"
    )
