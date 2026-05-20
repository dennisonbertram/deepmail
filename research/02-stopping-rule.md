# Stopping criteria for an iterative email-search expansion loop

## Executive recommendation

Use a **layered, deterministic stopping policy** rather than any single rule. The right combination is (1) hard caps on total iterations, corpus size, and API cost as an unconditional safety net; (2) an **explicit frontier queue** of candidate queries — derived from extracted entities and scored deterministically — and terminate when the queue is empty; (3) a **per-entity expansion budget** so a single noisy participant (mailing list, calendar bot) can't drag the loop forever; and (4) a **recapture-rate guard** that *de-prioritizes* (not eliminates) queries whose grep neighborhoods are already in-corpus. Crucially, avoid pure "new-entity-rate-decay" rules: by Heaps' law the new-entity curve never goes to zero, and the sparse-mention case (the four-email "mom") is exactly the situation rate-based saturation rules fail. Let the LLM *propose and score* candidate queries (cheap, recoverable mistakes), but let a deterministic frontier algorithm *terminate* (the load-bearing decision the project's thesis says LLMs get wrong).

## Prior-art survey

**Iterative deepening / uninformed search.** IDDFS terminates on two conditions: goal found, or no nodes remain at the current depth limit (frontier exhaustion). [Wikipedia: Iterative deepening DFS](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). The analogue is exact: a depth-bounded queue of pending queries that runs to empty.

**Active learning.** Settles' literature survey identifies three classes: budget exhaustion (labels, money, time), performance plateau (held-out accuracy stops climbing), and query-strategy signal decay (uncertainty falls below τ; committee disagreement collapses; expected error reduction → 0). [Wikipedia: Active learning (ML)](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)). Our analogue: stop when proposed queries have low expected information yield — but, like AL, this requires a *measurable* signal of yield, not the LLM's self-report.

**Pseudo-relevance feedback (PRF).** PRF is almost always run for a single round in practice because iterative PRF suffers severe **query drift** — copper-mines → Chile. [Wikipedia: Relevance feedback](https://en.wikipedia.org/wiki/Relevance_feedback). For our loop this is a warning: each LLM-proposed expansion can yank the corpus off-topic, and termination on drift is implicit through bounded depth, not through "convergence."

**Focused crawlers.** Standard stopping signals are **harvest-rate decay** (relevant pages per fetched page over a sliding window), seed-domain whitelisting, and stale-link detection. [Wikipedia: Focused crawler](https://en.wikipedia.org/wiki/Focused_crawler). The direct email analogue is *new-entities-per-message-downloaded*, but with the same caveat as below.

**Theoretical saturation (grounded theory).** Qualitative researchers stop when no new themes appear in K consecutive interviews — Guest, Bunce & Johnson (2006) found ~12 sufficed for homogeneous populations. The principle is *consecutive*, not cumulative: a streak of zero novelty.

**Species accumulation curves / Chao1.** Ecologists fit a negative-exponential to the cumulative-species-vs-effort curve and stop near the asymptote; Chao1 estimates *unseen* species from singleton/doubleton counts. [Wikipedia: Species discovery curve](https://en.wikipedia.org/wiki/Species_discovery_curve). Useful: gives an *estimate of what you're missing*, not just a "rate is low" heuristic.

**Capture-recapture.** As recapture probability → 1, the population has been adequately sampled. [Wikipedia: Mark and recapture](https://en.wikipedia.org/wiki/Mark_and_recapture). The email analogue is strong: when most entities in a freshly downloaded message are already in your entity set, you're near closure on *this* sub-network.

**Heaps' law.** Vocabulary grows as V ≈ Kn^β with β ∈ [0.4, 0.6]; *new types never stop appearing*. [Wikipedia: Heaps's law](https://en.wikipedia.org/wiki/Heaps%27_law). Implication: a pure rate-based stop will either over- or under-fire depending on β for *this user's* email — you cannot pick a global threshold.

**Agentic loops (ReAct / Reflexion / LangGraph).** ReAct terminates on a model-emitted Finish action; Reflexion on trial budget. Production frameworks bolt on a hard `recursion_limit` that raises `GraphRecursionError` — LangGraph's documented escape hatch. [LangGraph docs — graph recursion limit](https://docs.langchain.com/oss/python/langgraph/use-graph-api). This is the floor every deployed agent has, even when an LLM is allowed to gate termination.

## Proposed rules

```
state:
  corpus            # set of message-ids already downloaded
  entities          # set of canonicalized entities (people, orgs, dates)
  queue             # priority queue of candidate (query, score, parent_entity)
  per_entity_budget # dict: entity -> remaining expansions
  iter              # counter
  spend             # bytes downloaded + LLM tokens
```

### Rule A — Hard budget caps (unconditional)
```
if iter >= MAX_ITER:   stop("iter_cap")
if spend >= MAX_SPEND: stop("budget_cap")
if len(corpus) >= MAX_CORPUS_BYTES: stop("corpus_cap")
```
**Failure modes:** under-explores rich networks; over-shoots on cheap noisy mailing lists. **Tuning:** set MAX_SPEND to ~3× the expected p50 of a "fully explored" run (calibrate on 5–10 hand-labeled networks).

### Rule B — Frontier exhaustion (primary terminator)
```
on each iteration:
  q = queue.pop_highest_score()        # deterministic order
  download_all(q); update corpus
  new_entities = extract(corpus_delta) - entities
  for e in new_entities:
    if per_entity_budget[e] > 0:
      for sub_q in propose_queries(e): # LLM scores, doesn't gate
        queue.push(sub_q, score)
  entities |= new_entities
if queue.empty(): stop("frontier_exhausted")
```
**Failure modes:** queue can grow unboundedly with noisy entities (bounded by Rule D). LLM proposes off-topic queries (mitigated by deterministic score + dedupe in Rule C).

### Rule C — Query novelty dedupe (gates *individual* queries, not the loop)
```
before running q:
  if any prior_q with cosine(embed(q), embed(prior_q)) > 0.95: skip q
  run q; if |corpus_delta_msg_ids \ corpus| / |corpus_delta| < 0.05:
    decrement per_entity_budget[q.parent_entity] by 2  # penalize
```
**Failure modes:** legitimate paraphrases get blocked; raise threshold to 0.97 if false positives observed.

### Rule D — Per-entity expansion budget
```
per_entity_budget[e] = K_PER_ENTITY    # e.g. 3
```
**Failure modes:** caps exploration of genuinely central hubs (mom, manager). Mitigation: scale K with entity centrality in the message-co-occurrence graph (PageRank-style), so important hubs get K=8, periphery K=2.

### Rule E — Recapture-rate de-prioritization (NOT a hard stop)
```
recapture_rate = |new_entities ∩ entities_before| / |entities_in_delta|
queue.rescore(by reducing scores of queries whose parent_entity has recapture_rate > 0.9)
```
Inspired by capture-recapture and Chao1. **Important:** never terminate on this alone — only re-rank.

## Adversarial cases

| Case | Naive rule that fails | Why it fails | Mitigation |
|---|---|---|---|
| **Sparse family (4 mentions of "mom")** | "Stop when entity-introduction rate < 1% for 3 iters" | "Mom" surfaces late, after rate has plateaued from work emails. Rule prunes her. | Frontier queue keeps every named entity with a non-zero budget. Termination depends on queue emptiness, not rate. |
| **Noisy mailing list (every msg has new strangers)** | "Stop when no new entities for K iters" | Never fires. Loop runs to wall-clock. | Per-entity budget (Rule D) + Rule A spend cap + recapture-rate down-weighting of mailing-list-domain entities. |
| **Drift via shared common name ("John")** | LLM proposes queries that pull in unrelated Johns | Each is a "new entity" by string match | Canonicalize entities by (name, email_address) pair; use email graph neighborhood for disambiguation before pushing to queue. |
| **Auto-replies / calendar bots inflate corpus** | Corpus-size cap fires before real work done | Bot traffic eats the budget | Pre-filter messages by sender-domain blocklist + auto-detected templates *before* counting against MAX_CORPUS_BYTES. |
| **Single-message dead-end (entity only appears once)** | Per-entity budget=3 wastes 2 expansions on it | Burns budget | After first expansion yields zero new messages and zero new entities, set budget→0 for that entity. |

## Tuning and empirical evaluation

**Tunables and starting points** (calibrate per user):
- `MAX_ITER`: 50; `MAX_SPEND`: $5 or 500k LLM tokens; `MAX_CORPUS_BYTES`: 50 MB
- `K_PER_ENTITY`: 3 baseline, scaled 2–8 by co-occurrence centrality
- Query-cosine dedupe threshold: 0.95
- Recapture down-weight trigger: 0.9 over a 3-iter window

**Evaluation protocol.**
1. Hand-label "ground-truth" expansion targets on 5–10 of your own seed queries (people you *know* are part of the topic).
2. Metrics per run: **recall@stop** (fraction of ground-truth entities found), **precision-of-corpus** (fraction of downloaded messages a human marks topically relevant), **cost-at-stop**, **iterations-at-stop**.
3. Ablate each rule: turn off Rule D, Rule E, etc., one at a time. Watch for the noisy-mailing-list case to blow up cost when D is off, and the sparse-family case to drop recall when only rate-based stops are on.
4. Track *which rule fired* on every run. If Rule A fires more than ~20% of the time, your frontier scoring is poor (LLM proposing too much junk); if Rule B fires nearly always, you can probably loosen caps.

## Sources

- [Wikipedia — Iterative deepening depth-first search](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search)
- [Wikipedia — Active learning (machine learning)](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))
- [Wikipedia — Relevance feedback](https://en.wikipedia.org/wiki/Relevance_feedback)
- [Wikipedia — Focused crawler](https://en.wikipedia.org/wiki/Focused_crawler)
- [Wikipedia — Species discovery curve](https://en.wikipedia.org/wiki/Species_discovery_curve)
- [Wikipedia — Mark and recapture](https://en.wikipedia.org/wiki/Mark_and_recapture)
- [Wikipedia — Heaps's law](https://en.wikipedia.org/wiki/Heaps%27_law)
- [LangGraph docs — recursion limit / GraphRecursionError](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
