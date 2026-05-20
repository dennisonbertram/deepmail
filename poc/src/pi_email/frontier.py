"""Frontier: priority queue + stopping policy.

Implements Rules A, B, C, D, E from research/02-stopping-rule.md. Rule C
(query dedupe) is a two-stage pipeline:

  1. Cheap token-bag exact-match pre-pass (catches LLM permutations like
     `mom OR dad` vs `dad OR mom`).
  2. Cosine similarity over query embeddings (catches semantic dupes the
     pre-pass misses — verbose disjunctions where one token swaps in/out).

If no embedder is supplied to the Frontier, stage 2 falls back to a
SequenceMatcher string-ratio check (the original POC behavior). This keeps
unit tests that construct a bare `Frontier()` working without forcing every
test to spin up a 440MB model.

Design intent from the research report:

  > "Let the LLM *propose and score* candidate queries (cheap, recoverable
  >  mistakes), but let a deterministic frontier algorithm *terminate* (the
  >  load-bearing decision the project's thesis says LLMs get wrong)."

In particular, `stop()` only checks rules in this priority order:

  1. Rule A — hard caps (iter / spend / corpus). Unconditional safety net.
  2. Rule B — frontier empty. The PRIMARY terminator — proves the loop
     finished its work, not that the LLM gave up.

Rules D and E *re-rank or down-weight* but never terminate. That's deliberate:
recapture-rate is a noisy signal in small corpora (see research/02 §"Sparse
family") and using it as a hard stop would prune late-surfacing entities.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .embedder import Embedder
    from .embedding_store import EmbeddingStore


# -- Tunables (defaults from research/02 §"Tuning and empirical evaluation") --
# MAX_ITER lowered 35 -> 25 (Pass 12B Fix 3). 11 consecutive real-Gmail passes
# kept hitting iter_cap_uncontrolled at 35/35; we now rely on the Pass-12B
# Fix 1 graceful stop rule (`no_new_persons_after_N`) to terminate the loop
# before the hard cap. Lowering the cap from 35 to 25 forces the new rule to
# carry the load — if convergence-by-family-yield works, the loop should
# typically exit at ~iter 11..15. A bug that breaks the rule will surface as
# `iter_cap_*` sooner instead of being masked by a generous cap.
MAX_ITER = 25
MAX_SPEND_TOKENS = 500_000
MAX_CORPUS_BYTES = 50 * 1024 * 1024
K_PER_ENTITY = 3

# SEED parent budget. The SEED_SENTINEL parent carries the LLM proposer's
# seed queries (typically 3-5). No hardcoded deterministic seeds — the
# proposer is the sole source. Budget of 5 is tight but sufficient; the
# proposer's max_queries parameter caps generation at this level.
SEED_BUDGET = 5

# Cosine threshold for treating two query embeddings as the same query.
# Higher than the entity-canon threshold (0.85) because queries are short and
# noisy — a swap of one disjunct ("vows" -> "married") can drop semantic
# similarity into the 0.85-0.91 range and we want THOSE to still count as
# distinct queries worth running. Empirically tuned against the
# fixture-corpus proposer output.
QUERY_COSINE_THRESHOLD = 0.92

# Fallback string-similarity threshold when no embedder is wired in (used by
# unit tests; the live demo path always supplies a LocalEmbedder).
QUERY_SIM_THRESHOLD = 0.80

# Rule E (recapture-rate de-prioritization). Round 4 (Pass 5B): Round-3.5
# settings (factor 0.5, prune_after 3) still let the frontier grow on real
# Gmail — bigrun4 hit iter_cap at 35/35 with frontier_size=35 because few
# parents ever saw THREE consecutive high-recap iters (the loop hops between
# parents). Tightening the screws:
#   * factor 0.3 — a single high-recap iter slashes a parent's score by 70%
#   * prune_after 2 — TWO consecutive high-recap iters is enough to zero a
#     parent's budget. Two is the minimum that still requires a streak
#     (one bad iter could be transient).
#   * RECAPTURE_IMMEDIATE_PRUNE_RECAP — bypass the streak entirely when a
#     single iter shows near-total recapture AND zero new messages. The
#     parent has nothing more to give; further expansion is wasted.
# Still NEVER terminates the loop — only re-ranks/prunes individual parents.
RECAPTURE_DOWNWEIGHT_TRIGGER = 0.7
RECAPTURE_DOWNWEIGHT_FACTOR = 0.3
RECAPTURE_PRUNE_AFTER = 2  # consecutive high-recap iters -> budget=0 on parent
RECAPTURE_IMMEDIATE_PRUNE_RECAP = 0.95  # one iter at >=this with new_msgs=0 -> prune

# Pass-5B global saturation panic button. When the frontier is bloated AND
# recent iterations are running at high recapture, we're choking on noise
# the per-parent Rule E can't catch fast enough (different parent each iter,
# so no streak ever forms). Threshold values:
#   * GLOBAL_SATURATION_FRONTIER_SIZE — Pass-6C raised 30 -> 50. The 30
#     threshold from Pass-5B fired too easily and over-pruned legitimate
#     paths toward personal-domain contacts in bigrun5 (frontier sized
#     ~16-20 throughout, but the same threshold logic was treated as the
#     "panic button" entry condition by the rest of the system). 50 means
#     we genuinely wait for the queue to bloat past a comfortable
#     working-set size before considering a global prune.
#   * GLOBAL_SATURATION_RECAP_THRESHOLD — Pass-6C raised 0.6 -> 0.75.
#     bigrun5 ran at avg_recap_last_5=0.93 at exit; 0.6 was too permissive
#     and let the trigger fire during normal expansion. 0.75 is the
#     transition point above which the loop is clearly drowning in known
#     ground vs. discovering new neighborhoods.
#   * GLOBAL_SATURATION_WINDOW — 5. Same as iter_yield window precision but
#     a longer view than EXHAUST_WINDOW (3) — we want a sustained signal
#     before pruning half the frontier, not a 2-iter blip.
#   * GLOBAL_SATURATION_PERCENTILE — Pass-6C raised 0.6 -> 0.8. Now we
#     prune only the top 20% by recap (was top 40%); leaves more median
#     parents alone so legitimate personal-domain expansion paths don't
#     get caught in the dragnet. Combined with the personal-domain
#     exception below, parents pointing at family contacts stay on the
#     frontier even under heavy saturation pressure.
GLOBAL_SATURATION_FRONTIER_SIZE = 50
GLOBAL_SATURATION_RECAP_THRESHOLD = 0.75
GLOBAL_SATURATION_WINDOW = 5
GLOBAL_SATURATION_PERCENTILE = 0.8

# Pass 12B Fix 4 — earlier, softer saturation signal. The Pass-5B prune above
# only fires when frontier_size > 50; bigrun11 saw the frontier hover in the
# 9..19 range yet still kept growing because the proposer added queries every
# iteration. The Pass-12B signal fires at a lower frontier size (30) but
# with a SOFTER action: instead of pruning budgets, we just tell loop.py to
# skip the proposer call on the NEXT iteration so the queue can drain.
# Threshold:
#   * SOFT_SATURATION_FRONTIER_SIZE — 30. Below the Pass-5B prune trigger
#     (50) but well above the 5-7 "almost done" band; gives the soft signal
#     room to fire BEFORE the hard prune does. Empirically: bigrun11 had
#     frontier in 9..19 most of the run, peaking at 19 then ringing back —
#     a 30-threshold won't fire until the queue genuinely bloats while
#     still being earlier than the hard prune.
#   * Same window / recap threshold as the hard prune (5 iters, > 0.75
#     avg recap) so both signals respond to the same underlying "we're
#     drowning in known ground" condition.
SOFT_SATURATION_FRONTIER_SIZE = 30
SOFT_SATURATION_RECAP_THRESHOLD = GLOBAL_SATURATION_RECAP_THRESHOLD

# Pass 12B Fix 1 — "no family signal after N" graceful-stop window. We track
# per-iteration the count of NEW family-relation-grounded entities (a PERSON
# entity that came with a kind="relation" entity in the same message — the
# same signal `_gather_family_members` uses to admit candidates). When the
# last N iterations have produced ZERO new family-grounded entities, the
# loop terminates with `no_new_persons_after_N` — a graceful stop that
# says "the loop has decided further search won't help."
#
# Window size 8:
#   * Large enough to ride out a 1-2 iter dry spell that's just the proposer
#     hopping between business-domain parents before circling back to a
#     personal one.
#   * Small enough to fire well before MAX_ITER=25 (Fix 3).
#   * Run 11 trace evidence: iters 4-11 all produced 0 family-grounded
#     entities; the 8-iter window would have fired around iter 11, saving
#     24 wasted iterations.
NO_NEW_PERSONS_WINDOW = 8

# Pass 12B Fix 2 — "unproductive parent" follow-up cap. After a parent has
# been POPPED this many times WITHOUT contributing a single family-relation
# signal, new proposer follow-ups from that parent are rejected at push()
# time. A sharper version of business-prune that applies regardless of
# domain — catches person:X / relation:X parents that the business-domain
# rules can't see.
#
# K = 2 means: one pop with zero family-signal is a tolerated fluke; the
# SECOND pop with zero family-signal locks out further proposals. This
# matches RECAPTURE_PRUNE_AFTER's "two strikes" cadence.
UNPRODUCTIVE_PARENT_POP_LIMIT = 2

# Pass-6C personal-domain bias. The whole point of "figure out my family"
# is to reach personal contacts; in bigrun5 the loop got trapped expanding
# business-domain entities (@tally.xyz, @coinbase.com, @messari.io) while
# `*@gmail.com` and `*@icloud.com` family contacts existed but never reached
# the head of the heap. Two levers fix this without rewriting the proposer:
#   1. A score boost makes personal-domain parents pop sooner.
#   2. A budget bonus lets them propose more follow-ups before Rule D throttles.
#
# The domain set focuses on consumer/free email providers; corporate domains
# (gmail.com is the obvious exception since it IS the provider — but a
# *personal* gmail.com address vs. a *workspace* gmail.com address is
# indistinguishable from the address alone, so we accept the false-positive
# rate. The downside of boosting an occasional small-business address that
# happens to be on Gmail is minor relative to the upside of finding family).
#
# Caveats / known edge cases:
#   - protonmail.com / proton.me: consumer privacy email; included.
#   - duck.com: DuckDuckGo email forwarding; included since these forward
#     consumer email.
#   - International providers (yandex.ru, gmx.de, web.de, ...): not in the
#     set; we err toward false-negative (no boost) rather than false-positive
#     (boosting a corporate forwarder).
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

# Additive on top of the proposer's score (typically 0.5..0.95). 0.25 is
# tuned so a mid-tier personal proposal (0.6) outranks a top-tier business
# proposal (0.8 -> not boosted) — the loop POPS highest score first, so this
# directly reorders the heap. Capped at 1.0 in `push()`.
PERSONAL_DOMAIN_SCORE_BOOST = 0.25

# Personal-domain parents get +2 expansion budget (K=3 -> K+bonus=5). Two is
# enough to let an extra round of "name OR address" queries fire before Rule
# D throttles; matched to RECAPTURE_PRUNE_AFTER so the bonus also covers
# one tolerated false-positive iter before pruning kicks in.
PERSONAL_DOMAIN_BUDGET_BONUS = 2

# Pass-7B — symmetric BUSINESS-domain penalty. bigrun6 hit iter_cap_uncontrolled
# (35/35) with frontier_size=24 because surviving parents were mostly
# business-domain `email:` entities (e.g. email:research@messari.io,
# email:notifications@coinbase.com) that kept producing zero-yield queries
# but accumulating budget. Pass-6C added a personal-domain boost on one
# side; Pass-7B is the inverse on the other.
#
# Trade-off note: we use the SIMPLER form ("any `email:` parent whose
# domain is NOT in PERSONAL_EMAIL_DOMAINS gets the penalty") rather than
# the heuristic form ("apply the penalty unless the domain has personal-
# shape addresses in the corpus"). The simpler rule:
#
#   - PRO: deterministic; doesn't need corpus access from frontier.py
#     (the frontier carefully avoids importing Corpus/Entity to keep
#     compile-time deps minimal).
#   - PRO: matches the symmetric structure of the personal-domain bias.
#   - CON: occasionally penalizes a small/family-business domain like
#     `cousin@cousindomain.org`. This is the same false-positive risk
#     direction as the personal-domain boost (boosting an occasional
#     small business that happens to be on Gmail) and we accept it on
#     the same logic — the upside on the big cases outweighs the
#     minor downside on edge cases.
#
# Pass-9B raised this from 0.15 -> 0.35 after bigrun8 hit iter_cap_uncontrolled
# with frontier_size=39 at exit. The 0.15 nudge wasn't enough to drain a
# queue dominated by business-domain entities (a single business `email:`
# parent — research@figment.io — produced 160 new entities in one iteration
# in iter 27 of bigrun8). At 0.35, a business proposal at score=0.6 is
# stored at 0.25, well below a personal proposal at 0.6 (stored at 0.85
# after PERSONAL_DOMAIN_SCORE_BOOST). The invariant we want is: any
# personal-domain pending query pops before ANY business-domain pending
# query unless the personal one's score is unusually low — which is what
# 0.35 buys us (a typical business top-tier score of ~0.85 lands at 0.5
# after penalty, still below a typical personal floor of ~0.55 = 0.30+0.25).
# Clamped to 0.0 minimum in push() so a low-score business proposal
# (0.05) doesn't end up negative.
BUSINESS_DOMAIN_SCORE_PENALTY = 0.35

# Pass-7B — tighter immediate-prune threshold for business-domain parents.
# The general Pass-5B RECAPTURE_IMMEDIATE_PRUNE_RECAP = 0.95 is the bar
# for ALL non-personal parents (person: parents, SEED, business email:
# parents). For business-domain email parents specifically we drop the
# bar to 0.85 — they're the documented offender in bigrun6 and they
# rarely-if-ever surface NEW personal contacts at high recap. A single
# 0.85+ iter with zero new messages is enough.
BUSINESS_IMMEDIATE_PRUNE_RECAP = 0.85

# Pass-7B — consecutive zero-msg iters that prune a business-domain
# parent. Pass-5B/6C use the per-parent consecutive-streak Rule E with
# RECAPTURE_PRUNE_AFTER=2 high-recap iters. This is the second escape
# hatch: business parents with two consecutive zero-msg iters at ANY
# recap get pruned. Tighter than the streak rule (which requires high
# recap) because zero-msg-from-business is a clearer dead-branch signal
# than zero-msg-from-person.
BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE = 2

# Pass-7B — recap threshold for the Fix-5 "reverse cap": drop a
# business-domain parent's score to 0 (popped LAST) and zero its budget
# when a single iter sees recap >= 0.7 AND new_entities == 0.
#
# Rationale (cited in spec): business-domain parents at recap >= 0.7 with
# zero new entities mean we've fully covered this work-graph node — no
# new contacts surfaced, no fresh signal. Continued exploration is
# wasted budget. Drop them to give personal-domain parents priority in
# the remaining iterations. 0.7 (not 0.85) because we want this to be
# the AGGRESSIVE drop — the goal is to clear business parents off the
# heap so personal parents can pop first.
BUSINESS_DROP_RECAP_THRESHOLD = 0.7

# Pass-7B — recap threshold for skipping proposer follow-ups (Fix 3) on
# a business-domain parent in the CURRENT iteration. Matches the
# immediate-prune threshold at 0.85 so the two conditions fire together:
# a business parent at recap=0.85+ with new_msgs=0 gets immediate-pruned
# AND its newly-discovered entities don't fan out to next-iter proposer
# queries. Together they cut off business-domain expansion fanout at
# both ends (parent budget AND child proposals).
BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD = 0.85

# Pass-9B Fix 2 — consecutive business-domain pop bias. Hard cap on how
# many of the last N iterations may have chosen a business-domain parent
# before the next pop is FORCED to a non-business candidate (skipping
# higher-scoring business pending queries). bigrun8 chose business
# parents in 18 of 35 iterations; the personal-domain entries were on
# the heap but kept being out-prioritized iteration after iteration.
#   * BUSINESS_BIAS_WINDOW — 5. Same window as GLOBAL_SATURATION_WINDOW
#     so both rules use the same "recent history" cadence.
#   * BUSINESS_BIAS_THRESHOLD — 3. Three of five = a clear majority but
#     not unanimous; tolerates the occasional business pop when the
#     proposer happens to surface a high-signal one.
#   * BUSINESS_BIAS_RECAP_THRESHOLD — 0.5. avg recap over the same
#     window MUST be above this for the trigger to fire — if recent
#     iterations are productive (recap low), even a business-heavy
#     run is doing useful work and we don't want to interrupt it. 0.5
#     is the same neutral midpoint Pass-5B uses elsewhere.
BUSINESS_BIAS_WINDOW = 5
BUSINESS_BIAS_THRESHOLD = 3
BUSINESS_BIAS_RECAP_THRESHOLD = 0.5

# Pass-9B Fix 3 — per-iteration cap on how many business-domain candidates
# the proposer is allowed to push. iter 27 of bigrun8 pushed 160 new
# business-domain candidates in a single iteration off the figment.io
# parent; combined with K_PER_ENTITY=3 default they took several iters
# to drain even after Rule D throttled. 5 is generous: the proposer's
# typical batch size is 3-8 anyway, and clamping at 5 trades a small
# amount of recall on enormous batches for guaranteed convergence.
# Reset to 0 by `Frontier.begin_iteration()` at the top of each loop iter.
BUSINESS_PUSH_PER_ITER_CAP = 5

# Pass-9B Fix 4 — graph-explosion threshold. When a business-domain parent
# produces this many or more new entities in a single iteration, it's
# almost certainly a "list of contacts" event (newsletter audience, vendor
# email list, etc.) rather than a tight expansion of a real graph node.
# Iter 27 of bigrun8 produced 160 from figment.io.
#
# Pass 10 lowers this 50 -> 20 after Run 9 showed the guardrail never fired:
# real-Gmail business-explosion peaks were 32 (iter 1 SEED), 23 (trytopoai),
# 19 (bloccelerate), 16 — all below the old 50 threshold. 20 catches the
# moderate explosions (>= 20 entities from a business parent in one iter
# is now treated as the "list of contacts" pattern) while leaving small
# productive business expansions (<= 19) alone. Zeroes the parent's
# budget AND clears any pending queries from this parent currently on the
# heap.
BUSINESS_EXPLOSION_NEW_ENTS = 20

# Rule B classification window (Round 3.5). When the frontier finally drains,
# look at the last N iterations: if any had >0 new messages it's a CLEAN
# exhaustion (good); if all N had 0 new messages it's a NO_YIELD exhaustion
# (we tunneled into a dead branch). Window of 3 matches the existing
# RECAPTURE_PRUNE_AFTER (old value) for consistency.
EXHAUST_WINDOW = 3

# Rule A classification thresholds. When iter_cap fires, classify the run
# state by remaining frontier size so future MAX_ITER tuning can tell
# "the loop was almost done" apart from "the loop is uncontrolled".
#
# Pass-5B: ITER_CAP_UNCONTROLLED_FRONTIER lowered 50 -> 10. Pass-9B further
# tightened it 10 -> 7 after bigrun8 still hit iter_cap_uncontrolled with
# frontier_size=39 — the band 8..10 is now subsumed into "uncontrolled"
# because any non-trivial leftover frontier is a sharper signal than the
# old 10-element wiggle room allowed. Below 5 still gets `_almost_done`.
# The 5-7 band keeps the bare `iter_cap` label for cases where MAX_ITER
# bump would actually help.
ITER_CAP_UNCONTROLLED_FRONTIER = 7
ITER_CAP_ALMOST_DONE_FRONTIER = 5


def _is_personal_domain_email_parent(parent_entity: str | None) -> bool:
    """True iff `parent_entity` is an `email:<addr>` whose domain is in
    PERSONAL_EMAIL_DOMAINS.

    The check is intentionally narrow: only `email:` kind entities can match.
    Person-kind parents (`person:Jane Smith`) carry no domain signal, even if
    later expansion may surface a personal address — they get the standard
    budget/score. The frontier sees parent strings of the form `kind:label`
    (entities.Entity.__str__), so this can be a string-prefix check without
    importing the Entity class (which would re-introduce the circular dep
    between frontier and entities that the rest of the module carefully
    avoids).
    """
    if not parent_entity or not isinstance(parent_entity, str):
        return False
    if not parent_entity.startswith("email:"):
        return False
    addr = parent_entity[len("email:") :].strip().lower()
    if "@" not in addr:
        return False
    # rsplit so weird local-parts with embedded `@` (rare but legal) parse
    # to the rightmost domain.
    domain = addr.rsplit("@", 1)[1]
    return domain in PERSONAL_EMAIL_DOMAINS


def _is_business_domain_email_parent(parent_entity: str | None) -> bool:
    """Pass-7B — inverse of `_is_personal_domain_email_parent`.

    True iff `parent_entity` is an `email:<addr>` whose domain is NOT in
    PERSONAL_EMAIL_DOMAINS. Mirrors the personal helper's gates: malformed
    payloads, non-`email:` kinds, and bare strings without `@` return False
    (only positively-identified business addresses qualify for the Pass-7B
    penalties, never an "unknown" parent).
    """
    if not parent_entity or not isinstance(parent_entity, str):
        return False
    if not parent_entity.startswith("email:"):
        return False
    addr = parent_entity[len("email:") :].strip().lower()
    if "@" not in addr:
        return False
    domain = addr.rsplit("@", 1)[1]
    if not domain:
        return False
    return domain not in PERSONAL_EMAIL_DOMAINS


def _token_bag(query: str) -> frozenset[str]:
    """Normalize a query into a token bag for deterministic exact-match dedupe.

    Per research/02 §"Stopping Rule C", this is a cheap pre-pass for the
    semantic check. The bag is: lowercased, whitespace-split, deduped via
    frozenset. Two queries with the same bag are token-set-equal regardless
    of operator order.

    IMPORTANT: `OR` / `or` is NOT stripped. `mom OR dad` is a Gmail
    *disjunction* (matches either token) while `mom dad` is a *conjunction*
    (matches both); collapsing them would silently drop disjunctive queries
    when a conjunctive variant had already run, which is the wrong direction
    for a recall-first system. The previous OR-stripping implementation was
    a bug — flagged in the live-LLM review.
    """
    return frozenset(query.lower().split())


@dataclass(order=True)
class QueryCandidate:
    """A pending query in the frontier.

    Ordering uses negative score so heapq (a min-heap) yields highest-score first.
    """

    # Sort key first. We negate score so highest-score pops first; tie-break by
    # insertion order to keep behaviour deterministic.
    sort_key: tuple[float, int] = field(compare=True)
    query: str = field(compare=False)
    score: float = field(compare=False)
    # parent_entity is required at push() time; the dataclass keeps a default
    # for ergonomic construction (e.g. in tests) but no real candidate in the
    # frontier will have None here.
    parent_entity: str | None = field(compare=False, default=None)
    justification: str = field(compare=False, default="")


@dataclass
class StopReason:
    """Why the loop terminated.

    `rule` is one of:
      - frontier_exhausted_clean — queue empty AND at least one of the last
        `EXHAUST_WINDOW` iters returned new messages. The good exit.
      - frontier_exhausted_no_yield — queue empty BUT the last
        `EXHAUST_WINDOW` iters returned zero new messages each. Suggests we
        ran into a dead branch (Rule E pruned aggressively or proposer ran
        dry); coverage may be incomplete.
      - frontier_exhausted — legacy synonym kept for callers that don't care
        about the clean/no-yield split (e.g. fixture tests pre-Round-3.5).
        New code should prefer the more specific variants.
      - iter_cap — MAX_ITER hit. Only emitted for the narrow band
        ITER_CAP_ALMOST_DONE_FRONTIER..ITER_CAP_UNCONTROLLED_FRONTIER
        (post-Pass-5B that's 5..10); most real-Gmail runs land in
        iter_cap_uncontrolled instead. Kept for the rare in-between case
        where neither "almost done" nor "uncontrolled" describes the run.
      - iter_cap_uncontrolled — MAX_ITER fired with frontier_size >
        ITER_CAP_UNCONTROLLED_FRONTIER. Lots of pending work; a future
        MAX_ITER bump or stricter Rule E would help. Pass-5B lowered the
        threshold to 10 so frontier_size>=11 lands here.
      - iter_cap_throttled — Pass-7B: MAX_ITER fired but Rule D/E
        pruned more queries than the loop executed. The loop was
        working hard to terminate and just ran out of iters; the
        opposite signal from `iter_cap_uncontrolled`. Overrides the
        frontier-size-based variants when it fires.
      - iter_cap_almost_done — MAX_ITER fired with frontier_size <
        ITER_CAP_ALMOST_DONE_FRONTIER. A modest MAX_ITER bump would let
        this run complete cleanly.
      - no_new_persons_after_N — Pass-12B Fix 1: the last
        NO_NEW_PERSONS_WINDOW iterations all produced ZERO new
        family-relation-grounded entities. A graceful stop — the loop
        has decided further search won't help. Fires AFTER
        frontier_exhausted (which is the cleaner exit) but BEFORE
        iter_cap (so the loop converges naturally rather than running
        the cap out on unproductive expansion).
      - budget_cap, corpus_cap — Rule A safety nets.
    """

    rule: str
    detail: str = ""


class Frontier:
    """Priority queue of pending queries plus the stopping policy.

    Lifecycle on each iteration:

        q = frontier.pop()                              # Rule B selection
        ids = searcher.search(q.query)
        ... fetch + extract entities ...
        frontier.push_many(new_candidates, parent=...)  # LLM-proposed
        frontier.observe_iteration(...)                 # for Rule E rescore
        if (reason := frontier.stop(...)) is not None:
            break
    """

    def __init__(
        self,
        max_iter: int = MAX_ITER,
        max_spend_tokens: int = MAX_SPEND_TOKENS,
        max_corpus_bytes: int = MAX_CORPUS_BYTES,
        k_per_entity: int = K_PER_ENTITY,
        on_log=None,
        embedder: "Embedder | None" = None,
        embedding_store: "EmbeddingStore | None" = None,
    ):
        self.max_iter = max_iter
        self.max_spend_tokens = max_spend_tokens
        self.max_corpus_bytes = max_corpus_bytes
        self.k_per_entity = k_per_entity
        self._on_log = on_log or (lambda s: None)
        # Optional embedder for the semantic stage of Rule C. When None, we
        # fall back to SequenceMatcher (original POC behavior). The live demo
        # path always wires a LocalEmbedder in.
        self._embedder = embedder
        self._store = embedding_store

        # Min-heap of QueryCandidate (sort_key drives order). Insertion counter
        # is a tiebreaker so behaviour is fully deterministic.
        self._heap: list[QueryCandidate] = []
        self._insert_counter = 0

        # Queries already run (lower-cased) for Rule C dedupe.
        self.ran_queries: list[str] = []

        # Parallel list of (token_bag, original_normed_query) for the Rule C
        # deterministic pre-pass. Kept in sync with ran_queries by mark_ran.
        self._ran_query_bags: list[tuple[frozenset[str], str]] = []

        # Parallel cache of unit-norm embeddings for ran queries (only
        # populated when an embedder is wired in). Avoids re-embedding the
        # same prior query on every push.
        self._ran_query_vecs: list[np.ndarray] = []

        # Per-entity remaining-expansion budget (Rule D).
        self.per_entity_budget: dict[str, int] = {}

        # Per-entity recent recapture rates (Rule E). Stored as a small window
        # so we can detect the >TRIGGER over RECAPTURE_PRUNE_AFTER-iter trigger.
        self._recapture_history: dict[str, list[float]] = {}

        # Round 3.5 — recent per-iter (new_messages, new_entities) for Rule B
        # classification (frontier_exhausted_clean vs _no_yield). A bounded
        # sliding window of size EXHAUST_WINDOW; older entries are dropped.
        self._recent_iter_yields: list[tuple[int, int]] = []

        # Pass-5B: global recap signal across all parents. Used by the
        # saturation panic button (see _global_saturation_check); bounded to
        # GLOBAL_SATURATION_WINDOW entries on the trailing end.
        self._recent_recap: list[float] = []

        # Pass-5B diagnostic counters. Incremented inside push() at the
        # corresponding rejection branches so the end-of-run diagnostic can
        # surface why the frontier rejected proposed queries. Observability
        # only — never feeds back into routing decisions.
        self.deduped_count: int = 0          # Rule C rejections
        self.budget_rejected_count: int = 0  # Rule D budget=0 rejections
        # Pass-5B: count of parents zeroed by the global saturation prune,
        # for end-of-run diagnostics.
        self.saturation_prune_count: int = 0

        # Pass-7B: per-parent count of consecutive zero-new_messages iters.
        # Reset to 0 when a productive iter (new_messages > 0) is observed
        # for that parent; incremented on every zero-msg iter. Used to
        # trigger BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE for business-domain
        # parents independently of recap. Tracked only for business-domain
        # `email:` parents — person/SEED/personal parents are not measured
        # (the spec scopes this prune to the documented bigrun6 offenders).
        self._consecutive_zero_msgs: dict[str, int] = {}

        # Pass-9B Fix 2: sliding window of "was the popped parent for this
        # iteration a business-domain `email:`?". Bounded to
        # BUSINESS_BIAS_WINDOW entries; appended inside `pop()` once per
        # iteration. Read by `_business_bias_active()` together with
        # `_recent_recap` to decide whether to skip business candidates
        # on the next pop.
        self._recent_iter_business_parent: list[bool] = []

        # Pass-9B Fix 3: count of business-domain candidates admitted to
        # the heap in the current iteration. Reset to 0 by
        # `begin_iteration()` at the top of each loop iter; incremented
        # inside `push()` when admitting a business candidate. When this
        # counter is at BUSINESS_PUSH_PER_ITER_CAP, subsequent business
        # pushes in the same iter are dropped with a `[business-cap]` log.
        self._business_push_this_iter: int = 0
        # Diagnostic counter — total business-domain pushes dropped by the
        # Pass-9B per-iter cap across the whole run. Exposed in
        # `stop_diagnostic()` so post-hoc analysis can tell convergence
        # by cap apart from convergence by Rule E.
        self.business_cap_dropped_count: int = 0

        # Pass 12B Fix 1 — bounded window of per-iteration NEW
        # family-relation-grounded entity counts. A family-grounded
        # entity = a PERSON entity that came with a kind="relation"
        # entity in the same message; the materializer uses the same
        # signal to admit candidates. Once the window is full AND
        # every entry is zero, `stop()` returns `no_new_persons_after_N`.
        # Fed by `observe_person_yield()` once per iteration; bounded by
        # `maxlen` so we never look at ancient history.
        self._recent_person_yield: deque[int] = deque(
            maxlen=NO_NEW_PERSONS_WINDOW
        )

        # Pass 12B Fix 2 — per-parent tracking of pops without family
        # signal. Each entry maps parent_entity -> consecutive count of
        # pop()s where the parent's iteration produced zero new family-
        # grounded entities. Incremented by `observe_person_yield()`
        # when the iter contributed zero family-grounded entities; reset
        # to 0 when an iter from that parent contributes at least one.
        # When the count hits UNPRODUCTIVE_PARENT_POP_LIMIT, further
        # pushes from that parent are rejected at push() time.
        self._unproductive_parent_pops: dict[str, int] = {}
        # Diagnostic — total parents-pushes rejected because the parent
        # was deemed unproductive (Fix 2). Surfaced in stop_diagnostic.
        self.unproductive_parent_rejected_count: int = 0

        # Pass 12B Fix 4 — sticky bit asking loop.py to skip the proposer
        # call on the NEXT iteration. Set by the soft saturation check
        # when frontier_size > SOFT_SATURATION_FRONTIER_SIZE AND avg
        # recap > SOFT_SATURATION_RECAP_THRESHOLD. Read (and cleared)
        # by loop.py via `consume_proposer_skip_signal()`. Diagnostic
        # counter tracks how many iters were skipped across the run.
        self._should_skip_proposer_next_iter: bool = False
        self.proposer_skipped_iters: int = 0

        # Pass 12B Fix 1 — track the iter index where the most recent
        # nonzero family-yield observation arrived. Purely diagnostic
        # (rendered in stop_diagnostic); helps post-hoc analysis see
        # whether the loop ran past the family signal by a lot or just
        # missed it by one iter.
        self._consecutive_zero_person_iters: int = 0

    # -- Pass-9B: per-iteration lifecycle hook ---------------------------

    def begin_iteration(self) -> None:
        """Reset per-iteration counters. Called by loop.py at the top of
        each iteration before any pop/search/push for that iter.

        Pass-9B introduced BUSINESS_PUSH_PER_ITER_CAP (Fix 3) — a hard
        cap on how many business-domain candidates the proposer is
        allowed to push in one iteration. The counter has to reset at
        a known boundary; this hook is that boundary. No-op for any
        future caller that doesn't track per-iter state.
        """
        self._business_push_this_iter = 0

    # -- Heap operations -------------------------------------------------

    def _push(self, c: QueryCandidate) -> None:
        # Negative score = highest score pops first; tie-break on insertion order.
        self._insert_counter += 1
        c.sort_key = (-c.score, self._insert_counter)
        heapq.heappush(self._heap, c)

    def push(
        self,
        query: str,
        score: float = 1.0,
        parent_entity: str | None = None,
        justification: str = "",
    ) -> bool:
        """Push a query candidate. Returns False if the query was deduped or
        the parent_entity has run out of budget.

        `parent_entity` MUST be a non-empty string. The whole point of Rule D
        (per-entity budget) is that EVERY query is attributed to a specific
        entity it expands on — so we enforce that contract at the data
        structure level rather than trusting the proposer-side check alone.
        The SEED phase uses the sentinel value "SEED" (see proposer.py).
        Passing None or "" raises ValueError.

        Rule C (dedupe): a candidate is skipped if a string-similar query was
        already RUN. We compare against ran_queries only (not pending) — we
        want the LLM to be able to propose variants of pending queries that
        the user hasn't actually paid for yet.
        """
        if not isinstance(parent_entity, str) or not parent_entity.strip():
            raise ValueError(
                f"Frontier.push requires a non-empty parent_entity (got "
                f"{parent_entity!r}). Use the 'SEED' sentinel for the seed "
                f"phase; otherwise this query is not attributable and Rule D "
                f"(per-entity budget) would not throttle it."
            )

        q_norm = query.strip().lower()
        if not q_norm:
            return False

        # Pass-6C — personal-domain score boost. Apply BEFORE dedupe/budget
        # gates so the boost shows up in the QueryCandidate.score for both
        # the heap ordering AND post-hoc inspection (tests look at
        # f._heap[0].score). Boost is additive on top of the proposer's
        # score; capped at 1.0 so a 0.85+boost doesn't overflow above the
        # implicit ceiling other parts of the system assume.
        if _is_personal_domain_email_parent(parent_entity):
            boosted = min(1.0, score + PERSONAL_DOMAIN_SCORE_BOOST)
            if boosted != score:
                self._on_log(
                    f"[domain-boost] +{PERSONAL_DOMAIN_SCORE_BOOST:.2f} "
                    f"for parent_entity={parent_entity!r} (personal domain); "
                    f"score {score:.2f} -> {boosted:.2f}"
                )
                score = boosted
        # Pass-7B — symmetric business-domain penalty. Subtracted from
        # score (clamped at 0.0 minimum). Personal and business branches
        # are mutually exclusive (a domain is either in PERSONAL_EMAIL_DOMAINS
        # or it isn't), so the two `if`s can't both fire for the same parent.
        # No double-apply concern.
        #
        # Pass-9B Fix 3 — additionally enforce the per-iteration cap on
        # business-domain pushes. Iter 27 of bigrun8 pushed 160 in one
        # shot; the cap drops the (counter+1)th business push of an
        # iteration. Drop-on-overflow is first-come-first-served, but
        # since the proposer is itself bounded in batch size (3-8
        # typically) the only way to blow past the cap is the explosion
        # case we're trying to throttle — so FCFS is fine. The counter
        # increments only once all downstream checks (dedupe, budget)
        # have ALSO passed — see the post-checks block near the end of
        # push(). Otherwise a deduped attempt would waste a cap slot.
        elif _is_business_domain_email_parent(parent_entity):
            if self._business_push_this_iter >= BUSINESS_PUSH_PER_ITER_CAP:
                self.business_cap_dropped_count += 1
                self._on_log(
                    f"[business-cap] iter dropped query {query!r} "
                    f"(parent={parent_entity!r}); over per-iter limit "
                    f"({BUSINESS_PUSH_PER_ITER_CAP})"
                )
                return False
            penalized = max(0.0, score - BUSINESS_DOMAIN_SCORE_PENALTY)
            if penalized != score:
                self._on_log(
                    f"[domain-penalty] -{BUSINESS_DOMAIN_SCORE_PENALTY:.2f} "
                    f"for parent_entity={parent_entity!r} (business domain); "
                    f"score {score:.2f} -> {penalized:.2f}"
                )
                score = penalized

        # Rule D — entity-budget gate. Pass-6C: initial allocation uses
        # `_initial_budget_for` so personal-domain parents seed at K + bonus.
        if parent_entity not in self.per_entity_budget:
            self.per_entity_budget[parent_entity] = self._initial_budget_for(
                parent_entity
            )
        if self.per_entity_budget[parent_entity] <= 0:
            self.budget_rejected_count += 1
            return False

        # Pass 12B Fix 2 — unproductive-parent follow-up cap. If this
        # parent has been popped UNPRODUCTIVE_PARENT_POP_LIMIT times
        # without contributing a family-relation signal, reject the new
        # query. Personal-domain parents are exempt (handled inside
        # `is_parent_unproductive`).
        if self.is_parent_unproductive(parent_entity):
            self.unproductive_parent_rejected_count += 1
            self._on_log(
                f"[unproductive-parent] rejecting push from "
                f"{parent_entity!r}: {UNPRODUCTIVE_PARENT_POP_LIMIT}+ "
                f"unproductive pops; query={query!r}"
            )
            return False

        # Rule C — deterministic token-bag pre-pass (research/02 Rule C).
        # Normalizes whitespace/case/order and strips OR/or keywords, then
        # compares as a token set. Catches obvious permutations like
        # `mom OR dad` vs `dad OR mom` that the LLM proposer commonly emits.
        bag = _token_bag(query)
        if not bag:
            return False
        for prior_bag, prior_q in self._ran_query_bags:
            if bag == prior_bag:
                self._on_log(
                    f"dedupe: token-bag match with prior query {prior_q!r}"
                )
                self.deduped_count += 1
                return False
        for c in self._heap:
            if _token_bag(c.query) == bag:
                self._on_log(
                    f"dedupe: token-bag match with pending query {c.query!r}"
                )
                self.deduped_count += 1
                return False

        # Rule C — semantic dedupe over query embeddings (research/02).
        # We compare the candidate's embedding against each ran-query's
        # cached embedding. Threshold 0.92 — high because queries are short
        # and small wording swaps move the cosine meaningfully.
        # When no embedder is configured (unit tests), fall back to the
        # SequenceMatcher behavior at the looser 0.80 string-ratio threshold.
        if self._embedder is not None:
            q_vec = self._get_or_embed_query(q_norm)
            for prior, prior_vec in zip(self.ran_queries, self._ran_query_vecs):
                sim = float(np.dot(q_vec, prior_vec))
                if sim >= QUERY_COSINE_THRESHOLD:
                    self._on_log(
                        f"dedupe: cosine match ({sim:.2f}) with prior query "
                        f"{prior!r}"
                    )
                    self.deduped_count += 1
                    return False
        else:
            for prior in self.ran_queries:
                if SequenceMatcher(None, q_norm, prior).ratio() >= QUERY_SIM_THRESHOLD:
                    self._on_log(
                        f"dedupe: similarity match with prior query {prior!r}"
                    )
                    self.deduped_count += 1
                    return False
            for c in self._heap:
                if SequenceMatcher(None, q_norm, c.query.lower()).ratio() >= QUERY_SIM_THRESHOLD:
                    self._on_log(
                        f"dedupe: similarity match with pending query {c.query!r}"
                    )
                    self.deduped_count += 1
                    return False

        self.per_entity_budget[parent_entity] -= 1

        # Pass-9B Fix 3 — increment per-iter business push counter ONLY
        # now that all dedupe/budget gates have passed. The pre-check at
        # the top of push() ensures we never admit beyond the cap; the
        # counter increment here keeps deduped attempts from wasting
        # cap slots.
        if _is_business_domain_email_parent(parent_entity):
            self._business_push_this_iter += 1

        self._push(
            QueryCandidate(
                sort_key=(0.0, 0),  # set in _push
                query=query,
                score=score,
                parent_entity=parent_entity,
                justification=justification,
            )
        )
        return True

    def push_many(self, candidates: Iterable[dict]) -> int:
        """Push a batch of candidates from a proposer. Returns count actually pushed."""
        n = 0
        for c in candidates:
            if self.push(
                query=c["query"],
                score=float(c.get("score", 1.0)),
                parent_entity=c.get("parent_entity"),
                justification=c.get("justification", ""),
            ):
                n += 1
        return n

    def pop(self) -> QueryCandidate | None:
        """Return the highest-scoring pending query, or None if empty.

        Marking the query as ran is the caller's job — we record it via
        `mark_ran` so the caller can choose not to mark queries that errored.

        Pass-9B Fix 2 — if the business-bias trigger is active (3+ of the
        last BUSINESS_BIAS_WINDOW iterations chose a business-domain
        parent AND avg recap over the same window > 0.5), walk the heap
        for the first non-business pending candidate and pop THAT one,
        even though heap-by-score would have chosen a business candidate.
        If only business candidates remain, fall through to the standard
        heap pop — the loop has no personal work left and a future
        stop-rule (frontier exhaustion or iter cap) will catch it.
        """
        if not self._heap:
            return None
        cand: QueryCandidate
        if self._business_bias_active():
            # Linear scan for the first non-business candidate. The heap
            # holds at most a few dozen entries at the bias-triggered
            # point so the O(n) scan is cheap.
            picked_idx: int | None = None
            for i, c in enumerate(self._heap):
                if not _is_business_domain_email_parent(c.parent_entity):
                    picked_idx = i
                    break
            if picked_idx is not None:
                cand = self._heap.pop(picked_idx)
                heapq.heapify(self._heap)
                self._on_log(
                    f"[business-bias] forced non-business pop "
                    f"(parent={cand.parent_entity!r}, score={cand.score:.2f}); "
                    f"business iters in last {BUSINESS_BIAS_WINDOW}="
                    f"{sum(self._recent_iter_business_parent)}, "
                    f"avg_recap_last_{BUSINESS_BIAS_WINDOW}="
                    f"{sum(self._recent_recap)/len(self._recent_recap):.2f}"
                )
            else:
                cand = heapq.heappop(self._heap)
        else:
            cand = heapq.heappop(self._heap)
        # Record this iteration's parent type for the next iter's
        # bias-trigger check. Bounded to BUSINESS_BIAS_WINDOW entries
        # on the trailing end.
        self._recent_iter_business_parent.append(
            _is_business_domain_email_parent(cand.parent_entity)
        )
        if len(self._recent_iter_business_parent) > BUSINESS_BIAS_WINDOW:
            del self._recent_iter_business_parent[: -BUSINESS_BIAS_WINDOW]
        return cand

    def _business_bias_active(self) -> bool:
        """Pass-9B Fix 2 — true iff the next pop should be FORCED to a
        non-business candidate (skipping over higher-scoring business
        pending queries).

        Conditions:
          1. At least BUSINESS_BIAS_WINDOW iterations have been observed
             (so the window has signal).
          2. At least BUSINESS_BIAS_THRESHOLD of those iterations chose
             a business-domain `email:` parent.
          3. Average recap over the same window > BUSINESS_BIAS_RECAP_THRESHOLD.

        Condition 3 is the productivity gate — if recent iterations are
        productive (recap low), even a business-heavy run is doing useful
        work and we don't want to interrupt it. The trigger fires only
        when the loop is BOTH business-biased AND chewing on noise.
        """
        if len(self._recent_iter_business_parent) < BUSINESS_BIAS_WINDOW:
            return False
        biz_count = sum(self._recent_iter_business_parent[-BUSINESS_BIAS_WINDOW:])
        if biz_count < BUSINESS_BIAS_THRESHOLD:
            return False
        if len(self._recent_recap) < BUSINESS_BIAS_WINDOW:
            return False
        recent = self._recent_recap[-BUSINESS_BIAS_WINDOW:]
        avg = sum(recent) / len(recent)
        return avg > BUSINESS_BIAS_RECAP_THRESHOLD

    def mark_ran(self, query: str) -> None:
        q_norm = query.strip().lower()
        self.ran_queries.append(q_norm)
        # Keep the bag list in sync for the Rule C token-bag pre-pass.
        self._ran_query_bags.append((_token_bag(query), q_norm))
        # And cache the embedding for the semantic stage so future push()
        # calls don't re-encode this same query.
        if self._embedder is not None:
            self._ran_query_vecs.append(self._get_or_embed_query(q_norm))

    def _get_or_embed_query(self, q_norm: str) -> np.ndarray:
        """Return the unit-norm embedding for a normalized query string.

        Round-trips through the EmbeddingStore (if configured) so embeddings
        survive between runs of the demo. The key scheme is `query:{q_norm}`.
        """
        assert self._embedder is not None  # only called from the embed branch
        key = f"query:{q_norm}"
        if self._store is not None:
            v = self._store.get(key)
            if v is not None:
                return v
        v = self._embedder.embed(q_norm)
        if self._store is not None:
            self._store.put(key, "query", v, model_id=self._embedder.model_id)
        return v

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def empty(self) -> bool:
        return not self._heap

    # -- Rule E: recapture-rate down-weighting --------------------------

    @staticmethod
    def combined_recapture_rate(
        entity_recap: float,
        message_recap: float,
    ) -> float:
        """Fold entity- and message-recapture signals into one number.

        Round 3.5: Rule E originally only tracked entity novelty, but real-
        Gmail logs show queries from an "exhausted" parent often return
        already-known *messages* even when the proposer keeps surfacing one
        or two fresh entities per delta (e.g. iter 7 from bigrun3:
        new_msgs=0, hits=1 but entity_recap=0.0 because the single hit
        message had 0 extracted entities). Taking the MAX of the two
        signals fires Rule E if EITHER novelty channel says "we've seen
        this neighborhood before."

        Why max (not avg, not min):
          - max is monotonic in either signal — adding the message_recap
            check can only make Rule E MORE aggressive, never less. That's
            the property we want for a tightening change.
          - avg would dilute a clean 1.0 message_recap signal with a
            stale 0.0 entity_recap, defeating the point.
          - min would require BOTH channels to agree, which is the
            conservative direction (the opposite of what we need on real
            data).
        """
        return max(entity_recap, message_recap)

    def observe_recapture(
        self,
        parent_entity: str | None,
        recapture_rate: float,
    ) -> bool:
        """Record a recapture-rate observation for `parent_entity`. Returns True
        if the entity's pending queries were just down-weighted.

        Rule E (Round 3.5):
          1. On EACH iter where recapture_rate > RECAPTURE_DOWNWEIGHT_TRIGGER,
             multiply the score of all pending queries from that parent by
             RECAPTURE_DOWNWEIGHT_FACTOR (0.5). One bad iter is enough to
             pull a parent's queries down the heap — the old "wait for 3
             consecutive" was too lax against the live-data flood rate.
          2. After RECAPTURE_PRUNE_AFTER (3) consecutive high-recap iters
             from the same parent, ZERO that parent's remaining per-entity
             budget. New queries pushed from that parent will be rejected
             at push() time — effectively pruning the parent.

        Still NEVER terminates the loop; only re-ranks/prunes individual
        parents. Per research/02 §Rule E: "Inspired by capture-recapture
        and Chao1. Important: never terminate on this alone — only re-rank."

        Pass `recapture_rate` as the combined signal (use
        `combined_recapture_rate()` to compute it from the entity/message
        components). For backward compatibility, callers may also pass the
        raw entity_recap — the function doesn't care where the number
        came from.
        """
        # Pass-5B: track the global recap signal regardless of whether
        # parent_entity is known. The seed phase may emit None as parent
        # before the first push registers it; we still want its recap to
        # count toward the global saturation signal.
        self._recent_recap.append(recapture_rate)
        if len(self._recent_recap) > GLOBAL_SATURATION_WINDOW:
            del self._recent_recap[: -GLOBAL_SATURATION_WINDOW]

        if parent_entity is None:
            # Global saturation still gets a chance to fire — the iter
            # contributed a recap value to the window even without a parent.
            self._global_saturation_check()
            return False
        # Pass-20: skip per-parent recapture downweighting for the SEED
        # parent. SEED queries are heterogeneous by design — the proposer
        # generates diverse queries for different facets of the seed topic.
        # Downweighting all SEED heap entries because one iteration had
        # high recapture is wrong and starves later seeds. SEED queries
        # are already budget-limited by SEED_BUDGET.
        if parent_entity == "SEED":
            self._global_saturation_check()
            self._soft_saturation_check()
            return False
        hist = self._recapture_history.setdefault(parent_entity, [])
        hist.append(recapture_rate)
        downweighted = False
        if recapture_rate > RECAPTURE_DOWNWEIGHT_TRIGGER:
            # Per-iter down-weight (no consecutive-streak required).
            new_heap: list[QueryCandidate] = []
            for c in self._heap:
                if c.parent_entity == parent_entity:
                    c.score *= RECAPTURE_DOWNWEIGHT_FACTOR
                    c.sort_key = (-c.score, c.sort_key[1])
                new_heap.append(c)
            heapq.heapify(new_heap)
            self._heap = new_heap
            downweighted = True
        # Consecutive-streak prune. We check the trailing RECAPTURE_PRUNE_AFTER
        # entries (not the whole history) so a parent that already burned
        # through one streak and then redeemed itself can keep budget.
        tail = hist[-RECAPTURE_PRUNE_AFTER:]
        if (
            len(tail) >= RECAPTURE_PRUNE_AFTER
            and all(r > RECAPTURE_DOWNWEIGHT_TRIGGER for r in tail)
            and self.per_entity_budget.get(parent_entity, 0) > 0
        ):
            self.per_entity_budget[parent_entity] = 0
            self._on_log(
                f"Rule E prune: parent {parent_entity!r} hit "
                f"{RECAPTURE_PRUNE_AFTER} consecutive high-recap iters "
                f"(>{RECAPTURE_DOWNWEIGHT_TRIGGER}); budget -> 0"
            )

        # Pass-5B saturation panic button. Fired after every observation so
        # the frontier can't grow more than one iter past the trigger
        # without action. Idempotent on parents already at budget=0.
        self._global_saturation_check()
        # Pass 12B Fix 4 — softer, earlier signal at frontier_size > 30.
        # Sets a flag the loop reads at the top of the next iter; no
        # budget changes here so this is purely advisory.
        self._soft_saturation_check()
        return downweighted

    def _global_saturation_check(self) -> int:
        """Pass-5B: if the frontier is bloated AND recent iters are running
        at high recap, prune the top GLOBAL_SATURATION_PERCENTILE of pending
        parents by their mean recap. Returns the number of parents zeroed.

        Why this exists: bigrun4 hit iter_cap at 35/35 with frontier_size=35
        even after Round-3.5 tightened Rule E. The loop kept hopping between
        different parents so no SINGLE parent ever saw the streak Rule E's
        per-parent prune required. Aggregating recap across parents catches
        the "frontier is choking on collective noise" case the per-parent
        rule structurally can't see.

        Idempotent: parents already at budget=0 are skipped. Safe to call
        on every observation.
        """
        if len(self._heap) <= GLOBAL_SATURATION_FRONTIER_SIZE:
            return 0
        if len(self._recent_recap) < GLOBAL_SATURATION_WINDOW:
            # Not enough signal yet — fewer than WINDOW iters observed.
            return 0
        recent = self._recent_recap[-GLOBAL_SATURATION_WINDOW:]
        avg_recap = sum(recent) / len(recent)
        if avg_recap <= GLOBAL_SATURATION_RECAP_THRESHOLD:
            return 0
        # Collect (parent, mean_recap) for each parent with PENDING queries.
        # Parents whose only queries already drained get no influence.
        pending_parents: set[str] = set()
        for c in self._heap:
            if c.parent_entity:
                pending_parents.add(c.parent_entity)
        parent_recaps: list[tuple[str, float]] = []
        for p in pending_parents:
            hist = self._recapture_history.get(p)
            if not hist:
                continue
            parent_recaps.append((p, sum(hist) / len(hist)))
        if not parent_recaps:
            return 0
        # Percentile threshold: parents at or above this recap get pruned.
        sorted_recaps = sorted(r for _, r in parent_recaps)
        # `int(0.6 * N)` is the index of the first element in the top 40% —
        # i.e. those AT or above the 60th percentile.
        idx = int(GLOBAL_SATURATION_PERCENTILE * len(sorted_recaps))
        if idx >= len(sorted_recaps):
            idx = len(sorted_recaps) - 1
        threshold = sorted_recaps[idx]
        pruned = 0
        for p, r in parent_recaps:
            if r >= threshold and self.per_entity_budget.get(p, 0) > 0:
                # Pass-6C — personal-domain parents are exempt from the
                # saturation prune even when their recap is in the top
                # percentile. The whole point of Pass-6C is to keep the
                # loop pointed at family contacts; saturation prune was
                # exactly the mechanism that killed those paths in
                # bigrun5. Standard Rule D / consecutive-streak Rule E
                # still apply, so a personal parent that's genuinely
                # exhausted will drain via the slow path.
                if _is_personal_domain_email_parent(p):
                    self._on_log(
                        f"[domain-boost] skipping saturation prune for "
                        f"{p!r} (personal domain); mean_recap={r:.2f} >= "
                        f"p{int(GLOBAL_SATURATION_PERCENTILE * 100)}"
                        f"={threshold:.2f} would have zeroed budget"
                    )
                    continue
                self.per_entity_budget[p] = 0
                pruned += 1
                self._on_log(
                    f"Rule E saturation prune: parent {p!r} budget -> 0 "
                    f"(mean_recap={r:.2f} >= p{int(GLOBAL_SATURATION_PERCENTILE * 100)}"
                    f"={threshold:.2f}; frontier_size={len(self._heap)}, "
                    f"avg_recap_last_{GLOBAL_SATURATION_WINDOW}={avg_recap:.2f})"
                )
        if pruned:
            self.saturation_prune_count += pruned
        return pruned

    # -- Pass-7B: business-domain reverse cap & proposer-skip helpers ----

    def is_business_saturated(
        self,
        parent_entity: str | None,
        recap: float | None = None,
    ) -> bool:
        """Pass-7B Fix 3 — true iff `parent_entity` is a business-domain
        email parent whose most-recent recap is at/above the
        BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD (0.85).

        If `recap` is provided (current iteration's just-computed signal),
        use it directly. Otherwise consult the parent's recap history.
        Used by:
          1. loop.py to skip the proposer call entirely when the
             iteration's parent saturated.
          2. The sorted_new filter in loop.py to drop newly-discovered
             entities that themselves are already-saturated business
             parents from previous iterations.
        """
        if not _is_business_domain_email_parent(parent_entity):
            return False
        if recap is not None:
            return recap >= BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD
        hist = self._recapture_history.get(parent_entity) or []
        if not hist:
            return False
        return hist[-1] >= BUSINESS_PROPOSER_SKIP_RECAP_THRESHOLD

    def business_high_recap_drop(
        self,
        parent_entity: str | None,
        recap: float,
        new_entities: int,
    ) -> bool:
        """Pass-7B Fix 5 — reverse cap on unhelpful business parents.

        When a business-domain `email:` parent's iteration ends with
        recap >= BUSINESS_DROP_RECAP_THRESHOLD (0.7) AND zero new
        entities, do TWO things:

          1. Zero the parent's remaining per-entity budget. Subsequent
             pushes from this parent at push() time will be rejected by
             the Rule D gate.
          2. Set the score on ALL of this parent's pending heap entries
             to 0. heapq pops smallest sort_key first; since the key is
             (-score, insertion_counter), score=0 entries pop LAST. That
             gives personal-domain pending parents (score in 0.05..1.0)
             priority in the remaining iterations.

        Rationale: business-domain parents at recap ≥ 0.7 with zero new
        entities mean we've fully covered this work-graph node and
        continued exploration is wasted. Drop them to give
        personal-domain parents priority in the remaining iterations.

        Returns True if any action was taken (budget zeroed OR pending
        scores dropped). Safe to call from a productive iter; the
        new_entities == 0 gate suppresses the no-op case.
        """
        if not _is_business_domain_email_parent(parent_entity):
            return False
        if recap < BUSINESS_DROP_RECAP_THRESHOLD or new_entities != 0:
            return False
        acted = False
        before = self.per_entity_budget.get(parent_entity, 0)
        if before > 0:
            self.per_entity_budget[parent_entity] = 0
            acted = True
        # Drop scores to 0 on pending heap entries from this parent so
        # they pop AFTER any personal-domain (score > 0) pending entries.
        dropped = 0
        rebuild = False
        for c in self._heap:
            if c.parent_entity == parent_entity and c.score != 0.0:
                c.score = 0.0
                c.sort_key = (-c.score, c.sort_key[1])
                dropped += 1
                rebuild = True
        if rebuild:
            heapq.heapify(self._heap)
            acted = True
        if acted:
            self._on_log(
                f"[business-drop] parent {parent_entity!r} recap={recap:.2f}, "
                f"new_entities=0; budget {before} -> 0; "
                f"dropped scores to 0 on {dropped} pending entries"
            )
        return acted

    # -- Rule D: per-entity budget --------------------------------------

    def decrement_budget_for_yield(
        self,
        parent_entity: str | None,
        new_messages: int,
        new_entities: int,
    ) -> int:
        """Apply Rule D's zero-yield penalty to a parent entity's budget.

        Round 3.5 refinement of research/02 Rule C/D: "After first expansion
        yields zero new messages and zero new entities, set budget→0 for
        that entity" — softened to a graduated decrement so one bad query
        doesn't wipe out a parent that's mostly productive.

          - 0 new_messages              -> budget -= 1
          - 0 new_messages AND 0 new_entities -> budget -= 2

        This is in addition to the -1 already paid in push() when the
        query was admitted. Net effect on a fully dead query: -2 to -3.

        A productive query (new_messages > 0) pays nothing extra here —
        just the standard -1 paid up front. Returns the new remaining
        budget (or 0 if parent isn't tracked, which shouldn't happen).
        """
        if parent_entity is None:
            return 0
        if parent_entity not in self.per_entity_budget:
            return 0

        # Pass-9B Fix 4 — graph-explosion handler. When a business-domain
        # parent produces BUSINESS_EXPLOSION_NEW_ENTS+ new entities in a
        # single iteration, that's almost certainly a "list of contacts"
        # event (bigrun8 iter 27: figment.io -> 160 new entities). Zero
        # the parent's budget AND clear any pending queries from this
        # parent currently on the heap. Fires BEFORE the new_messages > 0
        # early return below so the explosion fires on BOTH productive
        # and zero-msg iters.
        if (
            _is_business_domain_email_parent(parent_entity)
            and new_entities >= BUSINESS_EXPLOSION_NEW_ENTS
        ):
            before_budget = self.per_entity_budget[parent_entity]
            before_heap = len(self._heap)
            self.per_entity_budget[parent_entity] = 0
            self._heap = [
                c for c in self._heap if c.parent_entity != parent_entity
            ]
            cleared = before_heap - len(self._heap)
            if cleared:
                heapq.heapify(self._heap)
            self._on_log(
                f"[business-explosion] parent {parent_entity!r} produced "
                f"{new_entities} new entities (>= "
                f"{BUSINESS_EXPLOSION_NEW_ENTS}); budget {before_budget} "
                f"-> 0; cleared {cleared} pending children from heap"
            )
            # Also reset the consecutive-zero-msgs counter so a stale
            # streak doesn't keep firing on this now-dead parent.
            if parent_entity in self._consecutive_zero_msgs:
                self._consecutive_zero_msgs[parent_entity] = 0
            return 0

        if new_messages > 0:
            # Pass-5B: even a productive iter triggers immediate-prune if
            # recap is at ceiling and we admitted zero NEW messages.
            # `new_messages > 0` is the gate above so this branch can't
            # see that case — left here as a no-op for clarity.
            # Pass-7B: productive iter resets the consecutive zero-msg
            # streak for the business-domain tighter prune below.
            if parent_entity in self._consecutive_zero_msgs:
                self._consecutive_zero_msgs[parent_entity] = 0
            return self.per_entity_budget[parent_entity]

        # Pass-7B: track consecutive zero-new_msgs iters for business
        # parents. Used by the consecutive-prune branch below; updated
        # BEFORE any prune branch runs so the log message reflects the
        # observed streak length.
        is_business = _is_business_domain_email_parent(parent_entity)
        if is_business:
            self._consecutive_zero_msgs[parent_entity] = (
                self._consecutive_zero_msgs.get(parent_entity, 0) + 1
            )

        # Pass-7B: business-domain tighter immediate prune. For business
        # `email:` parents the recap bar drops from the general 0.95
        # (Pass-5B) to 0.85 — one observation at that level with zero new
        # messages is enough to zero the budget. Fires BEFORE the
        # Pass-5B 0.95 check so the right `[business-prune]` log line
        # surfaces.
        hist = self._recapture_history.get(parent_entity) or []
        if (
            is_business
            and hist
            and hist[-1] >= BUSINESS_IMMEDIATE_PRUNE_RECAP
        ):
            before = self.per_entity_budget[parent_entity]
            if before > 0:
                self.per_entity_budget[parent_entity] = 0
                self._on_log(
                    f"[business-prune] zeroed budget for {parent_entity!r} "
                    f"(recap={hist[-1]:.2f}, consecutive zero_msgs="
                    f"{self._consecutive_zero_msgs.get(parent_entity, 0)})"
                )
            return 0

        # Pass-7B: business-domain consecutive zero-msgs prune. Any
        # business parent with BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE
        # consecutive iters at new_msgs=0 — regardless of recap — is
        # zeroed. Tighter than the personal/person consecutive-streak
        # Rule E (which still requires high recap).
        if (
            is_business
            and self._consecutive_zero_msgs.get(parent_entity, 0)
            >= BUSINESS_CONSECUTIVE_ZERO_MSGS_PRUNE
        ):
            before = self.per_entity_budget[parent_entity]
            if before > 0:
                self.per_entity_budget[parent_entity] = 0
                last_recap = hist[-1] if hist else 0.0
                self._on_log(
                    f"[business-prune] zeroed budget for {parent_entity!r} "
                    f"(recap={last_recap:.2f}, consecutive zero_msgs="
                    f"{self._consecutive_zero_msgs[parent_entity]})"
                )
            return 0

        # Pass-5B IMMEDIATE PRUNE. The parent's *most recent* recap reading
        # combined with new_messages=0 is a deterministic signal that the
        # parent has nothing more to give: every hit it produced is already
        # in our corpus. Bypass the consecutive-streak requirement of the
        # standard Rule E prune (RECAPTURE_PRUNE_AFTER) — one observation
        # at >= RECAPTURE_IMMEDIATE_PRUNE_RECAP is sufficient when paired
        # with zero new messages. Logged with its own marker so the
        # diagnostic can tell streak prunes apart from immediate prunes.
        #
        # Pass-6C: personal-domain parents are EXEMPT from the immediate
        # prune. Personal contacts (family) often have only one or two
        # threads in the corpus; a single high-recap iter looks
        # exhausted-by-the-numbers but is exactly the case where we want
        # one more proposer cycle to try a name-rooted query. Standard
        # Rule D zero-yield decrement still applies below, so they
        # eventually drain — just not in one shot.
        hist = self._recapture_history.get(parent_entity) or []
        if (
            hist
            and hist[-1] >= RECAPTURE_IMMEDIATE_PRUNE_RECAP
            and not _is_personal_domain_email_parent(parent_entity)
        ):
            before = self.per_entity_budget[parent_entity]
            if before > 0:
                self.per_entity_budget[parent_entity] = 0
                self._on_log(
                    f"Rule E immediate prune: parent {parent_entity!r} "
                    f"recap={hist[-1]:.2f} >= {RECAPTURE_IMMEDIATE_PRUNE_RECAP} "
                    f"and new_msgs=0; budget {before} -> 0"
                )
            return 0
        # Pass-6C log: surface when we DIDN'T prune because the parent is
        # at a personal domain. Useful when reading run.log to confirm the
        # exemption is firing on the right cases.
        if (
            hist
            and hist[-1] >= RECAPTURE_IMMEDIATE_PRUNE_RECAP
            and _is_personal_domain_email_parent(parent_entity)
        ):
            self._on_log(
                f"[domain-boost] skipping immediate prune for "
                f"{parent_entity!r} (personal domain); recap={hist[-1]:.2f} "
                f"with new_msgs=0 — falling through to standard Rule D decrement"
            )

        # Zero new messages: penalize.
        penalty = 2 if new_entities == 0 else 1
        before = self.per_entity_budget[parent_entity]
        after = max(0, before - penalty)
        self.per_entity_budget[parent_entity] = after
        if penalty > 0:
            self._on_log(
                f"Rule D zero-yield: parent {parent_entity!r} budget "
                f"{before} -> {after} (penalty {penalty}; new_msgs="
                f"{new_messages} new_ents={new_entities})"
            )
        return after

    # -- Pass 12B Fix 1 — family-yield tracking & graceful-stop signal --

    def observe_person_yield(
        self,
        parent_entity: str | None,
        new_person_count: int,
    ) -> None:
        """Record one iteration's family-relation-grounded entity yield.

        Pass 12B Fix 1. `new_person_count` is the count of PERSON
        entities discovered in THIS iter that co-occurred with a
        kind="relation" entity in the same message (the same gating
        signal the materializer uses to admit candidates). Called once
        per iteration by loop.py AFTER push_many and recapture observation.

        Two effects:
          1. Appends the count to a bounded `_recent_person_yield` deque
             (maxlen=NO_NEW_PERSONS_WINDOW). When the window is full
             and sums to zero, `stop()` will fire `no_new_persons_after_N`.
          2. Updates the per-parent unproductive-pop counter for Fix 2:
             zero contribution from the current parent -> counter += 1;
             nonzero contribution -> counter reset to 0.

        Tracks the consecutive-zero count too for the stop_diagnostic.
        """
        # Defensive clamp — observe_person_yield only sees non-negative
        # counts in practice, but a stray -1 from a buggy caller would
        # silently corrupt the rolling sum.
        count = max(0, int(new_person_count))
        self._recent_person_yield.append(count)

        if count == 0:
            self._consecutive_zero_person_iters += 1
        else:
            self._consecutive_zero_person_iters = 0

        # Pass 12B Fix 2 — track per-parent unproductive pops. We can't
        # uniquely identify the parent on a SEED iter (multiple
        # SEED-tagged seeds may produce yield in different iters), so
        # SEED is excluded from this counter: SEED queries are bounded
        # by the seed list size, not by Rule D.
        if parent_entity and parent_entity != "SEED":
            if count == 0:
                self._unproductive_parent_pops[parent_entity] = (
                    self._unproductive_parent_pops.get(parent_entity, 0) + 1
                )
            else:
                # Reset on yield — this parent earned a fresh leash.
                self._unproductive_parent_pops[parent_entity] = 0

    def _no_new_persons_window_exhausted(self) -> bool:
        """True iff the family-yield window is full AND every entry is 0.

        Pass 12B Fix 1 — the gating predicate for the graceful stop
        rule. We require the window to be FULL (maxlen entries) so the
        rule never fires on a short run that just hasn't accumulated
        enough signal — e.g. a 3-iter fixture run that legitimately
        terminates on frontier_exhausted before iter 8.
        """
        if len(self._recent_person_yield) < NO_NEW_PERSONS_WINDOW:
            return False
        return sum(self._recent_person_yield) == 0

    def is_parent_unproductive(self, parent_entity: str | None) -> bool:
        """Pass 12B Fix 2 — true iff this parent should no longer get
        proposer follow-ups admitted.

        Used by `push()` to reject pushes from parents that have been
        popped UNPRODUCTIVE_PARENT_POP_LIMIT times without contributing
        a family signal. Also used by loop.py to filter sorted_new
        BEFORE handing to the proposer (avoids wasting LLM tokens on
        parents whose proposals would be rejected anyway).

        Personal-domain parents are EXEMPT for the same reason they're
        exempt from the Pass-5B / Pass-7B immediate-prunes: a personal
        contact may have only one or two threads in the corpus, and
        a single dry pop doesn't mean the family path is dead.
        """
        if not parent_entity or parent_entity == "SEED":
            return False
        if _is_personal_domain_email_parent(parent_entity):
            return False
        pops = self._unproductive_parent_pops.get(parent_entity, 0)
        return pops >= UNPRODUCTIVE_PARENT_POP_LIMIT

    # -- Pass 12B Fix 4 — soft saturation signal --------------------------

    def _soft_saturation_check(self) -> bool:
        """Pass 12B Fix 4 — softer, earlier saturation signal.

        Fires (sets `_should_skip_proposer_next_iter = True`) when:
          1. Frontier size > SOFT_SATURATION_FRONTIER_SIZE (30)
          2. Average recap over the last GLOBAL_SATURATION_WINDOW iters
             > SOFT_SATURATION_RECAP_THRESHOLD (0.75)

        Unlike `_global_saturation_check`, this DOES NOT prune budgets;
        it just tells loop.py to skip the proposer call on the next iter
        so the queue can drain. Idempotent — flipping the flag from True
        to True is fine. Returns True iff the signal was just set.
        """
        if len(self._heap) <= SOFT_SATURATION_FRONTIER_SIZE:
            return False
        if len(self._recent_recap) < GLOBAL_SATURATION_WINDOW:
            return False
        recent = self._recent_recap[-GLOBAL_SATURATION_WINDOW:]
        avg = sum(recent) / len(recent)
        if avg <= SOFT_SATURATION_RECAP_THRESHOLD:
            return False
        if not self._should_skip_proposer_next_iter:
            self._on_log(
                f"[soft-saturation] flagging proposer-skip for next iter "
                f"(frontier_size={len(self._heap)} > "
                f"{SOFT_SATURATION_FRONTIER_SIZE}; "
                f"avg_recap_last_{GLOBAL_SATURATION_WINDOW}={avg:.2f} > "
                f"{SOFT_SATURATION_RECAP_THRESHOLD})"
            )
        self._should_skip_proposer_next_iter = True
        return True

    def consume_proposer_skip_signal(self) -> bool:
        """Pass 12B Fix 4 — read & clear the soft-saturation signal.

        Returns True if the loop should skip the proposer call this
        iteration (and bumps the diagnostic counter). Always clears the
        flag so the skip applies for exactly ONE iteration; if the
        saturation condition persists, the next observe_recapture will
        re-arm the flag.
        """
        if self._should_skip_proposer_next_iter:
            self._should_skip_proposer_next_iter = False
            self.proposer_skipped_iters += 1
            return True
        return False

    # -- Rule B: iteration yield tracking (for clean / no_yield split) --

    def observe_iteration_yield(self, new_messages: int, new_entities: int) -> None:
        """Record one iteration's yield in a bounded sliding window.

        Used by `stop()` to classify a frontier_exhausted termination as
        either `_clean` (some recent iter produced new_messages) or
        `_no_yield` (every recent iter was dead). Keep at most
        EXHAUST_WINDOW entries — older ones don't influence the classifier.
        """
        self._recent_iter_yields.append((new_messages, new_entities))
        if len(self._recent_iter_yields) > EXHAUST_WINDOW:
            del self._recent_iter_yields[: -EXHAUST_WINDOW]

    # -- Stopping policy -------------------------------------------------

    def stop(
        self,
        iteration: int,
        spend_tokens: int,
        corpus_bytes: int,
    ) -> StopReason | None:
        """Return a StopReason if the loop should terminate, else None.

        Priority order:
          A.1 Hard money/space caps — unconditional safety net.
          B.  Frontier empty — primary terminator (the *good* exit).
          F1. (Pass 12B) No family signal after N iters — graceful stop
              when the loop has clearly stopped finding family contacts.
              Placed BEFORE iter_cap so the loop terminates naturally
              instead of running the cap out on unproductive expansion.
          A.2 iter_cap — last-resort hard safety net.

        Round 3.5 classifies both terminal states so a stop reason is more
        diagnostic. See `StopReason` docstring for the variant list.
        """
        # Rule A safety nets that DON'T depend on the loop's progress
        # quality — these short-circuit everything because they signal
        # a resource exhaustion the loop can't recover from.
        if spend_tokens >= self.max_spend_tokens:
            return StopReason("budget_cap", f"{spend_tokens} >= {self.max_spend_tokens}")
        if corpus_bytes >= self.max_corpus_bytes:
            return StopReason("corpus_cap", f"{corpus_bytes} >= {self.max_corpus_bytes}")

        # Rule B — the *primary* terminator. If the frontier is empty after the
        # iteration's pushes, we've discovered everything we know how to look for.
        if self.empty:
            return self._classify_frontier_exhausted()

        # Pass 12B Fix 1 — graceful stop when the loop has stopped
        # finding NEW family-relation-grounded entities. Placed BEFORE
        # iter_cap so a converging run terminates here instead of
        # exhausting the iteration budget. The window must be FULL
        # (NO_NEW_PERSONS_WINDOW entries) so this never fires on
        # short fixture runs that legitimately exhaust the frontier
        # at iter 3.
        if self._no_new_persons_window_exhausted():
            return StopReason(
                "no_new_persons_after_N",
                f"last {NO_NEW_PERSONS_WINDOW} iters produced 0 new "
                f"family-relation-grounded entities; frontier_size="
                f"{len(self._heap)}",
            )

        # Rule A iter_cap — last-resort safety net. Reached only when the
        # graceful stops above didn't catch a runaway loop.
        if iteration >= self.max_iter:
            return self._classify_iter_cap(iteration)

        return None

    def _classify_iter_cap(self, iteration: int) -> StopReason:
        """Choose between _throttled / _uncontrolled / _almost_done / bare iter_cap.

        Extracted as a method (Pass-5B) so loop.py can also call it at exit
        time to make sure the materializer sees the *classified* variant
        in its `derivation.stop_reason` YAML field. Previously this logic
        was inlined in `stop()`; that's still where it normally fires, but
        the helper means callers building a `StopReason` outside of `stop()`
        get the same classification.

        Pass-7B Fix 4: when iter_cap fires AND Rule D/E pruned more than
        half of what we executed, the loop was clearly WORKING HARD to
        terminate — labeling that as `iter_cap_uncontrolled` is misleading.
        `iter_cap_throttled` says the loop was making real progress and
        just ran out of iteration budget. Observability only — no behavior
        change in the loop itself.
        """
        frontier_size = len(self._heap)
        # Pass-7B Fix 4 — throttled classification trumps size-based
        # classification. If we rejected/pruned over half as many queries
        # as we actually ran, that's a strong signal Rule E was carrying
        # the run; surface that in the stop_reason so future analysis
        # can tell "stuck" from "throttled".
        executed = len(self.ran_queries)
        pruned = self.budget_rejected_count
        if executed > 0 and pruned > executed * 0.5:
            return StopReason(
                "iter_cap_throttled",
                f"iteration {iteration} >= {self.max_iter}; "
                f"frontier_size={frontier_size}; "
                f"executed={executed}, pruned={pruned} "
                f"(pruned > 0.5 * executed — Rule D/E throttling)",
            )
        if frontier_size > ITER_CAP_UNCONTROLLED_FRONTIER:
            return StopReason(
                "iter_cap_uncontrolled",
                f"iteration {iteration} >= {self.max_iter}; "
                f"frontier_size={frontier_size} > "
                f"{ITER_CAP_UNCONTROLLED_FRONTIER} — uncontrolled growth",
            )
        if frontier_size < ITER_CAP_ALMOST_DONE_FRONTIER:
            return StopReason(
                "iter_cap_almost_done",
                f"iteration {iteration} >= {self.max_iter}; "
                f"frontier_size={frontier_size} < "
                f"{ITER_CAP_ALMOST_DONE_FRONTIER} — one more pass would do it",
            )
        return StopReason(
            "iter_cap",
            f"iteration {iteration} >= {self.max_iter} "
            f"(frontier_size={frontier_size})",
        )

    def _classify_frontier_exhausted(self) -> StopReason:
        """Decide between _clean and _no_yield based on recent iter yields.

        Per research/02 §Rule B: an empty frontier is the *primary*
        terminator, but a "clean" exhaustion (something was found recently)
        means very different things from a "no_yield" exhaustion (the loop
        tunneled into a dead branch and then ran out of queries). Both are
        terminal; only the labelling changes — observability, not behavior.
        """
        # Edge case: we never observed any iter (frontier emptied on the
        # very first stop check). Call this _clean by convention — there's
        # no signal of dead-branch tunnelling.
        if not self._recent_iter_yields:
            return StopReason(
                "frontier_exhausted_clean",
                "no pending queries (no iters observed)",
            )
        # _no_yield iff EVERY entry in the (already-bounded) window has 0
        # new_messages. As soon as one recent iter had yield, call it _clean.
        if all(new_msgs == 0 for new_msgs, _ in self._recent_iter_yields):
            return StopReason(
                "frontier_exhausted_no_yield",
                f"no pending queries; last "
                f"{len(self._recent_iter_yields)} iters had 0 new messages",
            )
        return StopReason(
            "frontier_exhausted_clean",
            f"no pending queries; recent window had productive iter "
            f"(yields={self._recent_iter_yields})",
        )

    # -- Pass-5B: proposer-time soft cap + end-of-run diagnostic ----------

    def _initial_budget_for(self, parent_entity: str) -> int:
        """Starting Rule D budget for a parent.

        SEED parent gets SEED_BUDGET (5) so the proposer's seed queries
        all fit without draining budget.
        Personal-domain `email:` parents start with K + PERSONAL_DOMAIN_BUDGET_BONUS
        slots; everyone else gets K. Pulled out as a helper so `has_budget()`
        and any future budget-init paths agree on the same allocation rule.
        """
        if parent_entity == "SEED":
            return SEED_BUDGET
        if _is_personal_domain_email_parent(parent_entity):
            return self.k_per_entity + PERSONAL_DOMAIN_BUDGET_BONUS
        return self.k_per_entity

    def has_budget(self, parent_entity: str) -> bool:
        """True iff this parent still has budget remaining.

        Used by the loop (Pass-5B Fix 3) to filter entities BEFORE they're
        handed to the proposer — pruned parents will have their proposals
        rejected at push() anyway, but skipping the proposer call entirely
        is cleaner: log lines are quieter, LLM tokens aren't wasted, and the
        proposer can spend its 1-3 slot budget on parents that still have
        room to expand. Returns True for parents we've never seen (they
        default to k_per_entity on first push).
        """
        if not parent_entity:
            return False
        if parent_entity not in self.per_entity_budget:
            # Default — first push() against this parent will register it
            # at _initial_budget_for(). Personal-domain parents start at
            # K + bonus; everyone else at K. As long as either is > 0 the
            # parent has budget on first push.
            return self._initial_budget_for(parent_entity) > 0
        return self.per_entity_budget[parent_entity] > 0

    def stop_diagnostic(
        self,
        *,
        iterations: int,
        stop_reason: StopReason,
        total_queries_proposed: int,
        top_n: int = 10,
    ) -> list[str]:
        """Render an end-of-run diagnostic as a list of log lines.

        Lines are returned (not printed) so the caller can route them through
        their own on_log sink. Format mirrors the spec in pass-5B:

            === Stop diagnostic ===
              iterations: 35
              frontier_exhausted: no
              total_queries_proposed: 95
              total_queries_executed: 35
              total_queries_deduped: 8
              total_queries_pruned (Rule D/E): 12
              saturation_prunes: 0
              avg_recap_last_5: 0.58
              budget_remaining_at_exit (top 10 by remaining):
                email:tyler@tally.xyz: 0
                ...

        This is the observability surface for the next round if a real-Gmail
        run still hits iter_cap. Pure read of frontier state; no side effects.
        """
        frontier_exhausted = stop_reason.rule.startswith("frontier_exhausted")
        if self._recent_recap:
            avg_recap = sum(self._recent_recap) / len(self._recent_recap)
            recap_str = f"{avg_recap:.2f}"
        else:
            recap_str = "n/a (no iters)"
        budget_items = sorted(
            self.per_entity_budget.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:top_n]
        # Pass 12B Fix 5 — additional diagnostic surface for the new
        # convergence machinery. `family_yield_last_N` shows the actual
        # per-iter family-grounded counts seen by the new stop rule —
        # invaluable for verifying convergence behavior or diagnosing why
        # the rule didn't fire when expected. `consecutive_zero_family_iters`
        # is the running counter that tipped the window to all-zero;
        # `proposer_skipped_iters` and `unproductive_parent_rejected`
        # surface the two new throttling levers (Fix 4 and Fix 2).
        family_yield_list = list(self._recent_person_yield)
        lines: list[str] = [
            "=== Stop diagnostic ===",
            f"  iterations: {iterations}",
            f"  frontier_exhausted: {'yes' if frontier_exhausted else 'no'}",
            f"  stop_reason: {stop_reason.rule}",
            f"  total_queries_proposed: {total_queries_proposed}",
            f"  total_queries_executed: {len(self.ran_queries)}",
            f"  total_queries_deduped: {self.deduped_count}",
            f"  total_queries_pruned (Rule D/E): {self.budget_rejected_count}",
            f"  saturation_prunes: {self.saturation_prune_count}",
            f"  business_cap_dropped: {self.business_cap_dropped_count}",
            f"  unproductive_parent_rejected: "
            f"{self.unproductive_parent_rejected_count}",
            f"  proposer_skipped_iters: {self.proposer_skipped_iters}",
            f"  family_yield_last_{NO_NEW_PERSONS_WINDOW}: "
            f"{family_yield_list}",
            f"  consecutive_zero_family_iters: "
            f"{self._consecutive_zero_person_iters}",
            f"  avg_recap_last_{GLOBAL_SATURATION_WINDOW}: {recap_str}",
            f"  budget_remaining_at_exit (top {top_n} by remaining):",
        ]
        if not budget_items:
            lines.append("    (none)")
        else:
            for parent, remaining in budget_items:
                lines.append(f"    {parent}: {remaining}")
        return lines
