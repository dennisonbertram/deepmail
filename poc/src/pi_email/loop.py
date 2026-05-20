"""ExpansionLoop: the main control flow wiring searcher + corpus + entities +
frontier + proposer + materializer together.

Reads strictly top-to-bottom; the comments mirror the iteration pseudocode in
research/02-stopping-rule.md §"Rule B".
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .corpus import Corpus, MessageId
from .embedder import Embedder, LocalEmbedder
from .embedding_store import EmbeddingStore
from .entities import Entity, EntityExtraction, canonicalize, extract_from_corpus
from .frontier import Frontier, StopReason
from .materializer import write_family_profile
from .proposer import Proposer, SEED_SENTINEL
from .searcher import Searcher
from .strip_quotes import strip_quotes_and_signatures


@dataclass
class IterationEvent:
    """One iteration's record — for logging and the test smoke check."""

    n: int
    query: str
    score: float
    hits: int
    new_messages: int
    new_entities: int
    frontier_size_after: int
    recapture_rate: float
    parent_entity: str | None
    proposed: int
    # Count of newly-admitted messages flagged by filters.is_bulk_message —
    # they're in the corpus but downstream entity extraction skips them.
    # Surfaces the noise leak per iteration in verbose logs.
    bulk_filtered: int = 0


@dataclass
class LoopResult:
    """Summary of a finished run."""

    corpus: Corpus
    entities: set[Entity]
    seen_by_message: dict[MessageId, set[Entity]]
    iterations: list[IterationEvent]
    stop_reason: StopReason
    queries_run: list[str]
    proposer_banner: str
    # display-name -> canonical-name across all kinds. Populated by the
    # embedding-based canonicalization step; used by the materializer to
    # dedupe member sections and wikilinks.
    canonical_map: dict[str, str] = field(default_factory=dict)


class ExpansionLoop:
    def __init__(
        self,
        searcher: Searcher,
        proposer: Proposer | None = None,
        frontier: Frontier | None = None,
        on_event: Callable[[IterationEvent], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        embedder: Embedder | None = None,
        embedding_store: EmbeddingStore | None = None,
        user_self: dict | None = None,
        strict_extract: bool = False,
    ):
        self.searcher = searcher
        self.on_event = on_event or (lambda e: None)
        self.on_log = on_log or (lambda s: None)
        # Pass 8A — record which addresses identify the end-user so entity
        # extraction can relax adjacency binding to sentence-level for
        # messages the user themselves authored. `None` = unknown user;
        # extraction stays strict (Pass 7A behavior) in that case.
        self.user_self = user_self
        self.user_emails: set[str] = set()
        if user_self and user_self.get("email"):
            self.user_emails.add(str(user_self["email"]))
        # Pass 10 — controls the `strict` flag forwarded to
        # `extract_from_corpus`. Default is False ("loose" mode: adjacency
        # window 15, no business-heavy suppression, looser subject fallback)
        # because Run 9 showed strict extraction only surfaced 7 candidates
        # out of 278 entities for the downstream judge. The LLM judge filters
        # business contacts well, so we open the extraction pipe and let it
        # do the discrimination. Fixture / test callers can pass
        # strict_extract=True to preserve historical behavior.
        self.strict_extract = strict_extract
        # Wire frontier dedupe-log messages into the same on_log sink so the
        # demo surfaces "token-bag match with prior query X" lines.
        self.proposer = proposer or Proposer(on_log=self.on_log)
        # Embedder + on-disk sidecar. Default: LocalEmbedder (bge-base-en-v1.5)
        # backed by ./embeddings.db. Lazy-loaded — `Embedder` import doesn't
        # trigger the 440MB model download until the first embed() call.
        self.embedder: Embedder = embedder or LocalEmbedder()
        self.embedding_store: EmbeddingStore = embedding_store or EmbeddingStore(
            Path.cwd() / "embeddings.db"
        )
        self.frontier = frontier or Frontier(
            on_log=self.on_log,
            embedder=self.embedder,
            embedding_store=self.embedding_store,
        )
        # Running display-name -> canonical-name map. Survives across
        # iterations so previously-merged entities stay merged when the loop
        # re-canonicalizes after each delta.
        self.canonical_map: dict[str, str] = {}
        # Phase tracking — "seed" before any iteration produced messages,
        # "expansion" after the first iteration whose query returned >=1
        # message. Used purely for logging / event annotation; the proposer
        # method called per iteration is what actually drives behavior.
        self.phase: str = "seed"

    # ----------------------------------------------------------------

    def run(self, seed: str) -> LoopResult:
        corpus = Corpus()
        entities: set[Entity] = set()
        seen_by_message: dict[MessageId, set[Entity]] = defaultdict(set)
        events: list[IterationEvent] = []

        self.on_log(f"Banner: {self.proposer.banner()}")
        self.on_log(f"Seed: {seed!r}")

        # ---- Phase 1: seed expansion (no entities yet) ----
        self.phase = "seed"
        seeded = self.proposer.propose_seed_queries(seed=seed)
        seeded_pushed = 0

        # Push the proposer-supplied seeds into the frontier. The proposer
        # is the SOLE source of seed queries — no hardcoded deterministic
        # seeds. The calling model (Claude Code, Cursor, etc.) can use
        # search_emails() to drive its own targeted investigation for any
        # topic — family, investors, team, etc.
        for pq in seeded:
            # parent_entity is guaranteed non-empty by the proposer contract
            # (defaults to SEED_SENTINEL); the frontier validates it again.
            if self.frontier.push(
                query=pq.query,
                score=pq.score,
                parent_entity=pq.parent_entity or SEED_SENTINEL,
                justification=pq.justification,
            ):
                seeded_pushed += 1
        self.on_log(
            f"Seeded frontier with {seeded_pushed} initial queries "
            f"({len(seeded)} from proposer): "
            + ", ".join(
                repr(p.query) for p in seeded
            )
        )

        # ---- Main loop ----
        iteration = 0
        spend_tokens = 0  # rough estimate; mock proposer contributes 0
        # Pass-5B diagnostic: count every proposal the proposer returned (NOT
        # just the ones the frontier accepted). Tied with frontier.deduped_count
        # / frontier.budget_rejected_count to surface where queries went.
        total_queries_proposed = 0
        while True:
            iteration += 1
            # Pass-9B Fix 3: reset per-iteration business-push counter so the
            # cap (BUSINESS_PUSH_PER_ITER_CAP) only applies within one iter.
            self.frontier.begin_iteration()

            # Check stop BEFORE popping — handles the rare case where the
            # seeder returned zero queries.
            stop = self.frontier.stop(iteration - 1, spend_tokens, sum(
                len(m.body) for m in corpus.messages.values()
            ))
            if stop:
                self.on_log(f"Stop ({stop.rule}): {stop.detail}")
                break

            cand = self.frontier.pop()
            if cand is None:
                # Race between stop()'s empty check and pop() — should be
                # extremely rare, but if it happens classify using the same
                # logic as `stop()` so the stop_reason field is consistent.
                stop = self.frontier._classify_frontier_exhausted()
                self.on_log(f"Stop ({stop.rule}): {stop.detail}")
                break

            self.frontier.mark_ran(cand.query)

            # ---- Deterministic search: fetch ALL matches in one batch ----
            # search_and_fetch is the widened protocol — a single call returns
            # fully-materialized messages, so a real Gmail impl can batch the
            # list+get round-trip instead of paying per-message HTTP cost.
            t0 = time.time()
            batch = self.searcher.search_and_fetch(cand.query)
            new_ids = []
            new_msgs_admitted: list = []
            for msg in batch.hits:
                if msg.message_id in corpus.messages:
                    continue
                # Only populate body_clean when the searcher didn't already
                # do it. GmailSearcher pre-computes body_clean from the raw
                # body through clean_html_and_css + strip_quotes_and_signatures
                # so HTML tags / CSS font-stacks never reach the NER step.
                # Unconditionally re-assigning here (the old behavior) would
                # clobber the HTML-cleaned text with a quote-stripped-but-
                # still-HTML-laden version, which is how "Arial" and raw
                # <span>/<div> fragments leaked into the materialized profile.
                # Fixture messages (no HTML) leave body_clean as None and we
                # apply the quote-strip pass here.
                if msg.body_clean is None:
                    msg.body_clean = strip_quotes_and_signatures(msg.body)
                corpus.add(msg)
                new_ids.append(msg.message_id)
                new_msgs_admitted.append(msg)
            # Bulk-mail accounting: messages still admitted to the corpus
            # (citation may matter later) but downstream entity extraction
            # will skip them via `if message.is_bulk: return []`. Surfacing
            # the count per iteration makes the noise leak visible in
            # `pi-email run -v` output.
            bulk_filtered = sum(1 for m in new_msgs_admitted if m.is_bulk)
            if bulk_filtered:
                self.on_log(
                    f"  bulk-filtered {bulk_filtered}/{len(new_ids)} new "
                    "messages (still in corpus, entities suppressed)"
                )
            if batch.error:
                self.on_log(
                    f"  search-batch non-fatal error: {batch.error} "
                    f"(retries={batch.retry_count}, truncated={batch.truncated})"
                )
            elif batch.truncated:
                self.on_log(
                    f"  search-batch truncated (retries={batch.retry_count}, "
                    f"quota_units_used={batch.quota_units_used})"
                )

            # ---- Entity extraction over the delta ----
            entities_before = set(entities)
            delta_extract = extract_from_corpus(
                corpus,
                message_ids=new_ids,
                user_emails=self.user_emails or None,
            )
            # Pass-17A: augment the delta with calendar-notification-derived
            # persons. Each new message that's a Google Calendar
            # confirmation email carries a `calendar_persons` list (see
            # `gmail_searcher._message_from_payload` + `calendar_email_parser`).
            # We synthesize Entity records for each so they flow into the
            # materializer's candidate gather + the family judge — these
            # are family signals that never appear in normal email body
            # text (kids' first names, school events, ...). Without this
            # injection the corpus contains the data but the loop never
            # sees the person.
            _inject_calendar_person_entities(
                corpus=corpus,
                new_ids=new_ids,
                delta=delta_extract,
            )
            new_entities = delta_extract.entities - entities_before
            entities |= delta_extract.entities
            for mid, ents in delta_extract.by_message.items():
                seen_by_message[mid] |= ents

            # ---- Embedding-based canonicalization ----
            # Recompute the canonical map over the FULL entity set after each
            # iteration. The set is small (low hundreds at most) and re-running
            # canonicalize is cheap because per-name embeddings are cached in
            # both the in-process LRU and the sqlite sidecar — only genuinely
            # new names hit the model.
            try:
                self.canonical_map = canonicalize(
                    entities, self.embedder, self.embedding_store
                )
            except Exception as e:
                # If the embedder fails (e.g. no network on first download),
                # don't tank the loop — just fall back to identity mapping.
                self.on_log(f"canonicalize: failed ({e!r}); using identity map")
                self.canonical_map = {e.label: e.label for e in entities}

            # Collapse `new_entities` to one Entity per canonical label.
            # Downstream (proposer, recapture) uses the deduped form.
            new_entities = _dedupe_by_canonical(new_entities, self.canonical_map)

            # ---- Recapture rate (Rule E signal) ----
            # Entity-novelty channel — fraction of entities in the delta
            # that were already in our known set.
            total_in_delta = len(delta_extract.entities) or 1
            recaptured = len(delta_extract.entities & entities_before)
            entity_recap = recaptured / total_in_delta
            # Round 3.5 — message-novelty channel. entity_recap alone misses
            # the case where a query returns mostly-already-known messages
            # but those messages contain 0 or 1 extracted entities. real-
            # Gmail bigrun3 iter 7 shows the failure mode: new_msgs=0,
            # new_ents=0 but entity_recap=0.0 (delta had no entities) — so
            # Rule E never fired despite the iteration having zero novelty.
            # `message_recap=1.0` here would have caught it.
            total_hits = len(batch.hits) or 1
            already_known_hits = total_hits - len(new_ids)
            message_recap = already_known_hits / total_hits
            recapture_rate = self.frontier.combined_recapture_rate(
                entity_recap=entity_recap,
                message_recap=message_recap,
            )
            self.frontier.observe_recapture(cand.parent_entity, recapture_rate)
            # Rule D zero-yield penalty (Round 3.5) — apply BEFORE the
            # proposer runs so a parent that just hit budget=0 has its
            # subsequent pushes from the same iter rejected at push() time.
            self.frontier.decrement_budget_for_yield(
                parent_entity=cand.parent_entity,
                new_messages=len(new_ids),
                new_entities=len(new_entities),
            )
            # Pass-7B Fix 5 — reverse cap on unhelpful business parents.
            # If the iter's parent is business-domain AND recap >= 0.7
            # AND new_entities == 0, zero its remaining budget AND drop
            # any pending heap entries from this parent to score=0 so
            # they pop LAST. Gives personal-domain parents priority in
            # the remaining iterations.
            self.frontier.business_high_recap_drop(
                parent_entity=cand.parent_entity,
                recap=recapture_rate,
                new_entities=len(new_entities),
            )
            # Rule B yield window (Round 3.5) — feed per-iter yield to the
            # frontier so an eventual frontier-empty stop can be classified
            # as either `_clean` (recent iter produced messages) or
            # `_no_yield` (last N iters were dead).
            self.frontier.observe_iteration_yield(
                new_messages=len(new_ids),
                new_entities=len(new_entities),
            )

            # Count new PERSON entities this iteration for the
            # no-new-persons-after-N graceful stop rule. Skip recording
            # for zero-hit iterations (query didn't match anything).
            new_person_count = sum(
                1 for e in new_entities if e.kind == "person"
            )
            if len(batch.hits) > 0:
                self.frontier.observe_person_yield(
                    parent_entity=cand.parent_entity,
                    new_person_count=new_person_count,
                )

            # ---- Phase transition: once an iteration produces messages, we
            # are out of the seed bootstrap and into entity-rooted expansion.
            if len(new_ids) > 0 and self.phase == "seed":
                self.phase = "expansion"
                self.on_log("phase: seed -> expansion")

            # ---- Propose follow-ups, grounded in the newly extracted entities ----
            excerpts = self._build_excerpts(corpus, delta_extract.by_message, new_entities)
            # Sort for deterministic proposer input (sets are unordered) — keeps
            # the smoke test stable across runs.
            sorted_new = sorted(new_entities, key=lambda e: (e.kind, e.key))
            # Pass-5B soft cap: drop entities whose Rule D budget has already
            # been zeroed BEFORE handing them to the proposer. push() would
            # reject these candidates anyway (Rule D gate), but skipping the
            # proposer call entirely:
            #   * avoids the LLM spending its 1-3 slots on guaranteed-DOA
            #     parents,
            #   * makes proposer logs cleaner (no "rejected by push" follow-up
            #     for budget=0 cases),
            #   * keeps frontier.deduped_count and budget_rejected_count
            #     honest as observability signals — they no longer get inflated
            #     by guaranteed-reject parents.
            # Parents not yet in per_entity_budget keep their default
            # k_per_entity allocation via frontier.has_budget().
            #
            # Pass-7B Fix 3 — additionally drop newly-discovered entities
            # that are themselves business-saturated parents (an entity
            # that previously fired the proposer-skip threshold at recap
            # >= 0.85). Prevents the proposer from generating ANOTHER
            # round of business-domain queries off an already-known-dead
            # business parent.
            sorted_new = [
                e
                for e in sorted_new
                if self.frontier.has_budget(str(e))
                and not self.frontier.is_business_saturated(str(e))
                # Pass 12B Fix 2 — also drop entities that are already
                # flagged as unproductive parents (e.g. surfaced again
                # from another iter's delta). Saves an LLM call whose
                # output would be rejected by push() anyway.
                and not self.frontier.is_parent_unproductive(str(e))
            ]
            # Pass-7B Fix 3 — if the iteration's own parent was business-
            # domain AND this iter ran at recap >= 0.85, skip the
            # proposer call entirely. The newly-discovered entities from
            # a saturated business-graph node are very likely to be more
            # business-graph nodes; don't fan them out. Together with
            # the immediate-prune at the same threshold (Fix 2), this
            # cuts off both ends of the business-graph fanout.
            if self.frontier.is_business_saturated(
                cand.parent_entity, recap=recapture_rate
            ):
                self.on_log(
                    f"[business-prune] skipping proposer call: parent "
                    f"{cand.parent_entity!r} recap={recapture_rate:.2f} "
                    f">= business-saturation threshold; dropping "
                    f"{len(sorted_new)} new entities from this iter"
                )
                sorted_new = []
            # Pass 12B Fix 4 — soft saturation signal. If the previous
            # iteration tripped the frontier-size > 30 + high-recap
            # condition, the frontier asks us to skip the proposer call
            # this iter so the queue can drain. Drops `sorted_new` to
            # empty; the rest of the iteration (event emission, stop
            # check) runs normally.
            if self.frontier.consume_proposer_skip_signal():
                self.on_log(
                    f"[soft-saturation] skipping proposer call this iter "
                    f"(frontier asked to drain; "
                    f"dropping {len(sorted_new)} new entities)"
                )
                sorted_new = []
            proposed = self.proposer.propose_expansion_queries(
                new_entities=sorted_new,
                ran_queries=list(self.frontier.ran_queries),
                excerpts=excerpts,
            )
            total_queries_proposed += len(proposed)
            pushed = 0
            for pq in proposed:
                # The proposer contract guarantees pq.parent_entity is a valid
                # str(Entity) for one of the entities in sorted_new (Phase 2)
                # — we pass it straight through to the frontier.
                if self.frontier.push(
                    query=pq.query,
                    score=pq.score,
                    parent_entity=pq.parent_entity,
                    justification=pq.justification,
                ):
                    pushed += 1

            # Token-spend estimate: ~500 tokens per live proposer call
            if not self.proposer.is_mock:
                spend_tokens += 500

            event = IterationEvent(
                n=iteration,
                query=cand.query,
                score=cand.score,
                hits=len(batch.hits),
                new_messages=len(new_ids),
                new_entities=len(new_entities),
                frontier_size_after=len(self.frontier),
                recapture_rate=recapture_rate,
                parent_entity=cand.parent_entity,
                proposed=pushed,
                bulk_filtered=bulk_filtered,
            )
            events.append(event)
            self.on_event(event)

            # Post-iteration stop check (Rule B fires here for the normal case).
            stop = self.frontier.stop(iteration, spend_tokens, sum(
                len(m.body) for m in corpus.messages.values()
            ))
            if stop:
                self.on_log(f"Stop ({stop.rule}): {stop.detail}")
                break

        # Pass-5B: belt-and-suspenders. `frontier.stop()` already classifies
        # iter_cap into _uncontrolled / _almost_done / bare iter_cap, but in
        # case a caller built a bare StopReason elsewhere (or an older
        # checkpoint leaked through) reclassify here from the live frontier
        # state. This is the single point at which `result.stop_reason` is
        # set, so the materializer always sees the classified variant in
        # `derivation.stop_reason`.
        if stop is not None and stop.rule == "iter_cap":
            stop = self.frontier._classify_iter_cap(iteration)
        elif stop is not None and stop.rule == "frontier_exhausted":
            stop = self.frontier._classify_frontier_exhausted()

        # Pass-5B stop diagnostic — one block at run end. Emitting through
        # on_log keeps it consistent with how the loop reports stop reasons
        # and dedupe events; the demo's terse stdout (the last cli.echo
        # lines) is unaffected.
        for line in self.frontier.stop_diagnostic(
            # Use the event count, not the raw `iteration` counter — when the
            # pre-iter stop fires we've incremented iteration but haven't
            # actually run that iter, so events is the accurate measure.
            iterations=len(events),
            stop_reason=stop,
            total_queries_proposed=total_queries_proposed,
        ):
            self.on_log(line)

        return LoopResult(
            corpus=corpus,
            entities=entities,
            seen_by_message=seen_by_message,
            iterations=events,
            stop_reason=stop,
            queries_run=list(self.frontier.ran_queries),
            proposer_banner=self.proposer.banner(),
            canonical_map=dict(self.canonical_map),
        )

    # ----------------------------------------------------------------

    @staticmethod
    def _build_excerpts(
        corpus: Corpus,
        by_message: dict[MessageId, set[Entity]],
        new_entities: set[Entity],
    ) -> dict[str, str]:
        """For each new entity, find a short excerpt where it appears.
        Used by the live proposer for grounding."""
        out: dict[str, str] = {}
        for e in new_entities:
            for mid, ents in by_message.items():
                if e in ents:
                    msg = corpus.get(mid)
                    if msg:
                        text = (msg.body_clean or msg.body)
                        out[str(e)] = text[:200]
                        break
        return out


def _inject_calendar_person_entities(
    corpus: Corpus,
    new_ids: list[MessageId],
    delta: EntityExtraction,
) -> None:
    """Inject calendar-notification-derived persons into the delta in place.

    For each message in `new_ids` that's a Google Calendar notification
    (i.e. has a non-empty `Message.calendar_persons` list, populated by
    `gmail_searcher._message_from_payload`), build synthetic Entity
    records and merge them into `delta.entities` + `delta.by_message`.

    Entity construction:
      * `Entity(kind="person", key=<lowercased-name>, label=<display>,
        confidence=...)` — confidence is "high" when
        `family_signal_strength >= 0.80` (kinship-event / possessive),
        "medium" for personal-attendee tier (0.65), "low" below.
      * `Entity(kind="email", key=<email>, label=<email>,
        confidence="high")` — when the person carries an attendee email.

    Names are stripped through `entities._clean_person_span` for parity
    with the body NER path; if cleanup leaves nothing (e.g. raw was
    pure-punctuation) we drop the candidate rather than emit a garbage
    Entity. Cleaning is wrapped so an unrelated upstream failure can't
    tank the loop.

    No-op when `new_ids` is empty or no message has a non-empty
    `calendar_persons` list (the common case — most inbox mail is not
    calendar notifications).
    """
    # Local import keeps the loop import surface unchanged for callers
    # who don't run the live Gmail searcher (fixture demos, tests).
    from .entities import _canon_name, _clean_person_span, _in_stoplist

    for mid in new_ids:
        msg = corpus.get(mid)
        if msg is None:
            continue
        persons = getattr(msg, "calendar_persons", None) or []
        if not persons:
            continue
        for p in persons:
            name = getattr(p, "name", None) or ""
            email = getattr(p, "email", None)
            strength = float(getattr(p, "family_signal_strength", 0.0) or 0.0)
            # Confidence mapping mirrors the body extractor: 0.80+ = high
            # (a strong calendar signal — kinship-event or possessive),
            # 0.65+ = medium (personal-domain attendee), otherwise low.
            if strength >= 0.80:
                confidence = "high"
            elif strength >= 0.65:
                confidence = "medium"
            else:
                confidence = "low"

            cleaned = _clean_person_span(name) if name else ""
            if cleaned and not _in_stoplist(cleaned):
                key, label = _canon_name(cleaned)
                ent = Entity(
                    kind="person",
                    key=key,
                    label=label,
                    confidence=confidence,
                )
                delta.entities.add(ent)
                delta.by_message[mid].add(ent)

            if email:
                addr = str(email).strip().lower()
                if addr and "@" in addr:
                    ent_email = Entity(
                        kind="email",
                        key=addr,
                        label=addr,
                        confidence="high",
                    )
                    delta.entities.add(ent_email)
                    delta.by_message[mid].add(ent_email)


def _dedupe_by_canonical(
    ents: set[Entity], canonical_map: dict[str, str]
) -> set[Entity]:
    """Collapse `ents` so each canonical name appears at most once per kind.

    We pick the Entity whose own label is the canonical (i.e. the longest
    one in the cluster) when present; otherwise we keep the first one in
    sorted order. This keeps proposer grounding stable — if "Jane" and
    "Jane Smith" both extracted in the same iteration, the proposer sees
    only "person:Jane Smith" and roots its expansion there.
    """
    by_kind_canon: dict[tuple[str, str], Entity] = {}
    for e in ents:
        canon = canonical_map.get(e.label, e.label)
        slot = (e.kind, canon)
        cur = by_kind_canon.get(slot)
        if cur is None:
            by_kind_canon[slot] = e
            continue
        # Prefer the entity whose label already IS the canonical.
        if e.label == canon and cur.label != canon:
            by_kind_canon[slot] = e
        elif cur.label != canon and e.label != canon:
            # Both non-canonical — break ties deterministically by label.
            if e.label < cur.label:
                by_kind_canon[slot] = e
    return set(by_kind_canon.values())


# ---- Convenience entry point ----


def run_loop_and_materialize(
    fixtures_dir: Path,
    seed: str,
    profiles_dir: Path,
    on_event: Callable[[IterationEvent], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    force_mock: bool = False,
    embedder: Embedder | None = None,
    embedding_store: EmbeddingStore | None = None,
    user_self: dict | None = None,
    skip_judge: bool = False,
    strict_extract: bool = True,
) -> tuple[LoopResult, Path]:
    """Convenience wrapper used by demo.py and tests/test_smoke.py.

    `embedder` / `embedding_store` are optional injection points: tests pass
    in a fake `Embedder` so they don't download the 440MB model. The demo
    path leaves them None and gets the default `LocalEmbedder` + sqlite
    sidecar at `<cwd>/embeddings.db`.

    `user_self` (optional) is passed straight through to the materializer
    so the user themselves is excluded from the resulting family member
    list. Fixture mode passes None (the fixture has no notion of "you").

    `strict_extract` (Pass 10, default True for fixture mode) controls
    whether the underlying entity extractor uses strict adjacency (tight
    8-token window, business-heavy suppression) or the loose mode
    (15-token window, no suppression). Fixture default = True preserves
    the existing demo expectations; the live Gmail path in cli.py opts
    in to False so the judge sees a broader candidate pool.
    """
    from .searcher import FilesystemSearcher

    searcher = FilesystemSearcher(fixtures_dir)
    proposer = Proposer(force_mock=force_mock, on_log=on_log)
    loop = ExpansionLoop(
        searcher=searcher,
        proposer=proposer,
        on_event=on_event,
        on_log=on_log,
        embedder=embedder,
        embedding_store=embedding_store,
        user_self=user_self,
        strict_extract=strict_extract,
    )
    result = loop.run(seed)

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
