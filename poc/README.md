# pi_email POC

Proof-of-concept for the **"deterministic-retrieval + iterative-LLM-expansion +
materialized-profiles"** email-understanding pipeline.

## The thesis

Existing AI email tools let the LLM decide when it's "done searching" — and so
they miss things. This POC inverts that:

1. **Deterministic search.** Every query runs to completion and downloads ALL
   matches. No top-k, no relevance cutoff, no LLM gating.
2. **Quote-stripped corpus.** Reply quotes are stripped at ingest so entities
   are counted by relevance, not by quoting depth.
3. **Entity extraction.** Cheap regex passes pull people, emails, and oblique
   relation words from the corpus delta.
4. **LLM proposes; code disposes.** The LLM proposes 1-3 follow-up queries
   given the new entities. A deterministic frontier algorithm decides whether
   to push them, dedupe them, or down-weight them.
5. **Layered stopping policy** (per `research/02-stopping-rule.md`):
   - Rule A: hard caps on iteration / spend / corpus size (safety net).
   - Rule B: **frontier exhaustion** is the primary terminator — the loop
     stops because there's nothing left to look for, not because the LLM
     gave up.
   - Rule C: query novelty dedupe (string-similarity stand-in for cosine).
   - Rule D: per-entity expansion budget.
   - Rule E: recapture-rate down-weighting (never a hard stop).
6. **Materialized profiles.** When the loop terminates, write
   `profiles/family.md` with YAML frontmatter + narrative + footnote
   provenance back to source messages (per `research/04-context-artifacts.md`).

Background reading lives in the repo-root `research/` directory; the POC
encodes the decisions from those reports rather than re-litigating them.

## Layout

```
poc/
├── README.md
├── pyproject.toml             # uv-friendly; pip install -e . also works
├── src/pi_email/
│   ├── corpus.py              # markdown corpus on disk: read, write, hash
│   ├── searcher.py            # FilesystemSearcher + GmailSearcher (stub)
│   ├── strip_quotes.py        # mail-parser-reply with regex fallback
│   ├── entities.py            # capitalized-name + email regex extractors
│   ├── frontier.py            # priority queue + Rules A/B/C/D/E
│   ├── proposer.py            # Anthropic SDK call; mock fallback if no key
│   ├── materializer.py        # writes profiles/family.md
│   └── loop.py                # ExpansionLoop — wires it all together
├── fixtures/family_corpus/    # 20 synthetic markdown emails
├── tests/test_smoke.py        # one end-to-end smoke test
└── demo.py                    # CLI: python demo.py "figure out my family"
```

## Quickstart

```bash
cd poc
uv sync                # or: pip install -e .

# Demo (works WITHOUT an API key — uses mock proposer):
python demo.py "figure out my family"

# Or with a real Anthropic key (model: claude-sonnet-4-5):
export ANTHROPIC_API_KEY=sk-ant-...
python demo.py "figure out my family"

# Smoke test:
python -m pytest tests/
```

## Fixture corpus

`fixtures/family_corpus/` contains 20 synthetic markdown emails covering:

- **Direct family** (msgs 001-006, 015-020): clean references to mom, dad,
  partner Sarah, sibling Emma, kids Mia and Leo, aunt Carol, grandma Helen.
- **Oblique mentions** (msgs 007-009): "dropping off Mia at swim practice",
  "kids are sick this weekend", "Leo's age bracket signups". A naive
  single-shot search for "family" will miss these — the iterative loop
  must propose follow-ups on "Mia" / "Leo" / "swim" to catch them.
- **Reply-chain noise** (msgs 010-011): nested `On X, Y wrote:` blocks.
  `strip_quotes.py` removes these before entity extraction.
- **Pure noise** (msgs 012, 013, 014, 019): work OKRs, newsletter,
  calendar bot, Amazon shipping confirmation. These should NOT incorporate
  into the family profile.

The fixture is deliberately designed so that a single search round misses
~3 family-relevant signals; the iterative loop catches them. That's the
proof-of-concept.

## Swapping FilesystemSearcher for real Gmail

`gmail_searcher.py:GmailSearcher` implements the production wiring path
documented in `research/01-gmail-api-mechanics.md`. Summary:

1. OAuth client + `gmail.readonly` scope.
2. Paginate `users.messages.list` with `q=<query>` until `nextPageToken`
   is absent.
3. Batch-fetch with `users.messages.get(format="full")` (50 per batch).
4. Convert Gmail payloads into the same `Message` dataclass that
   `FilesystemSearcher` returns and the rest of the loop works unchanged.

The seam is intentional: the loop never sees a "decide whether to keep
paginating" hook, so the LLM cannot accidentally gate retrieval there.

## Known limitations

- **Entity extraction is regex-only.** A real version would do an LLM
  refine-pass on capitalized-name candidates to filter false positives
  (e.g. "Dear Team" looks like a name). The mock proposer handles this
  acceptably for the fixture but will struggle on real mailboxes.
- **Identity merge is absent.** Two "Jane"s from different threads will
  appear as one entity. Per `research/04-context-artifacts.md`, this
  should be a confirmed-or-auto-merge step driven by email-address
  co-occurrence — not implemented in the POC.
- **Rule C uses string similarity** (`difflib.SequenceMatcher`) as a
  stand-in for cosine-embedding dedupe. Upgrade path: swap in a
  sentence-transformer at threshold 0.95.
- **No incremental refresh.** Each demo run rebuilds the corpus from
  scratch; there's no `history.list` / `last_seen_id` tracking. The
  materializer hashes the message-id list into `source_fingerprint`,
  so the staleness check is wired — just not the incremental fetch.
- **Token-spend accounting is rough** (~500 tokens per live proposer
  call, fixed). A production version would read `response.usage`.

## How to read the code

Start here, in order:

1. `loop.py` — the main control loop. The 60 lines from "Main loop"
   onwards are the whole story.
2. `frontier.py` — the stopping policy. The docstring maps each rule
   to the corresponding section of `research/02-stopping-rule.md`.
3. `searcher.py` — see `FilesystemSearcher.search` for the deterministic
   "return all matches" contract.
4. `proposer.py` — see `SYSTEM_PROMPT` for what the LLM is asked to do
   (and not asked to do).
5. `materializer.py` — output shape lifted from `research/04`.

## Wiring real Gmail

The fixture corpus is the default; once you've validated the loop's behavior
against it, point the same pipeline at your live mailbox.

### 1. Install and run the consent flow

Built-in OAuth credentials are embedded — no GCP setup needed. If you
prefer your own credentials, set `GOOGLE_CLIENT_ID` and optionally
`GOOGLE_CLIENT_SECRET` in your environment or a `.env` file.

```
cd poc
pip install -e .
pi-email auth
```

`pi-email auth` opens a browser, prompts you to consent to the
`gmail.readonly` scope, and saves the resulting access + refresh tokens
to the platformdirs path it prints. Re-run with `--force` to start over.

### 2. Sanity-check

```
pi-email whoami
pi-email query "from:me" --max 5
```

`whoami` is the canary for "auth actually works" — it calls
`users.getProfile` and prints the address + total counts. `query` is the
canary for "I can actually pull email by query" — it lists matches, runs a
batched fetch, and prints `from | date | subject` for each hit.

The default `--max 50` on `query` keeps your first runs cheap. Raise it once
you've calibrated against your mailbox.

### 3. First real expansion run

```
pi-email run "figure out my family" --source gmail --max-corpus 100
```

The `--source` flag is what swaps the fixture corpus for the live
`GmailSearcher`. Add `-v` (before the subcommand) for per-iteration logs.

### Quota notes

`GmailSearcher` tracks cumulative quota usage across its lifetime. Per
`research/01-gmail-api-mechanics.md`, `users.messages.list` = 5 units and
`users.messages.get(format="full")` = 20 units. A 100-message corpus costs
roughly 100 * 20 + a handful of list calls = ~2,050 units; the daily
per-user cap (1,000,000 units) gives headroom for many runs but it's worth
budgeting if you turn `--max-corpus` up.
