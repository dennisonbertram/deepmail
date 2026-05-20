# Materialized context artifacts: schema and architecture for derived profile documents

*Research date: 2026-05-16. Audience: a builder designing the "second layer" of an email-understanding tool — the derived `family.md`, `acme-corp.md`, `kitchen-remodel.md` files that future queries grep against instead of re-scanning the raw corpus.*

## Recommendation

**Hybrid markdown: YAML frontmatter index + narrative body + numbered provenance footnotes.** One file per entity (person, household, organization, project), kept in a flat `profiles/` directory, indexed by a slim `profiles/INDEX.md` that lists canonical names, aliases, and `kind`. Patch on each ingest; full re-derive monthly. Every non-trivial claim in the body carries a footnote-style citation back to one or more Gmail message IDs. This is structurally the same pattern Claude Code uses for `MEMORY.md` + topic files, and it's no accident — the constraints are identical: keep the loaded index small, push detail behind on-demand reads, make everything plain-text greppable.

### Concrete sample: `profiles/family.md`

```markdown
---
kind: household
canonical_name: Lindberg-Tanaka family
aliases: [the family, home, "us"]
members:
  - "[[people/dennison-lindberg]]"
  - "[[people/aya-tanaka]]"
  - "[[people/mira-lindberg-tanaka]]"
  - "[[people/kai-lindberg-tanaka]]"
last_derived: 2026-05-16T09:14:22Z
source_messages: 1847        # count of distinct msg-ids consulted
source_fingerprint: sha256:9c1a...
confidence: high
schema_version: 1
---

# Lindberg-Tanaka family

Four people in a single household in Oakland, CA since 2021 [^1]. Aya
took Dennison's surname hyphenated after marriage in 2019 [^2]; both
kids carry the hyphenated form.

## Members

### Dennison Lindberg
That's the user. See [[people/dennison-lindberg]] for the self-profile.

### Aya Tanaka (-Lindberg-Tanaka in legal contexts)
Spouse. Software architect at Stripe since 2022 [^3]. Goes by "Aya"
in all email; signs work mail with full hyphenated surname [^4].
Birthday: 1988-03-14 [^5].

### Mira Lindberg-Tanaka
Daughter, born 2017-07-09 [^6]. Attends Park Day School (Oakland),
2nd grade as of 2025-2026 school year [^7]. Allergic to peanuts —
EpiPen on file with school [^8].

### Kai Lindberg-Tanaka
Son, born 2020-11-22 [^9]. Pre-K at Park Day starting fall 2025 [^10].

## Recurring patterns

- **School comms** come from `@parkdayschool.org`; flagged as
  high-priority by the user historically [^11][^12][^13].
- **Pediatrician**: Dr. Reyes at Bay Pediatrics, `reyes@baypeds.com` [^14].
- **Family logistics** thread: `family-logistics@<user-domain>` — a
  shared list the user, Aya, and grandparents all read [^15].

## Open questions

- Aya's middle name appears as both "Mei" and "M." — unclear which
  is legal. Not load-bearing for any current query.

## Provenance

[^1]: gmail:18f3a9c2b1e44d7 (lease signing, 2021-06-04)
[^2]: gmail:17a02b...e9 (marriage announcement thread, 2019-09-12)
[^3]: gmail:18e1c4...44 (Stripe offer letter forward, 2022-01-18)
[^4]: aggregate across 412 messages — see derivation log
[^5]: gmail:16b8...77 (calendar invite, recurring)
...
```

The four design decisions packed into this sample:

1. **Frontmatter is for things you'd `jq`/grep mechanically** — names, dates, member lists, freshness metadata. Wrap wikilinks in quotes (Obsidian convention; YAML otherwise mis-parses `[[`).
2. **Body is for everything an LLM needs to reason** — relationships, exceptions, "the user historically flags this." Prose, not bullet-only.
3. **Provenance lives in footnotes**, not inline, so the body reads naturally. Gmail message ID is the durable anchor (`History API` survives label changes; thread IDs do not).
4. **`source_fingerprint`** = hash of the sorted msg-id list this profile was derived from. Cheap staleness check: if any *new* message arrives mentioning a member, the profile is potentially stale even if `last_derived` is recent.

## Prior art

| System | Pattern | What's worth stealing |
|---|---|---|
| **[Claude Code `MEMORY.md` + topic files](https://code.claude.com/docs/en/memory)** | Slim index loaded into every session (200 lines / 25KB cap), detail files read on demand. Auto-written by the agent itself. | The two-tier "always-loaded index, lazy-loaded detail" architecture. Same constraint applies here: profile bodies are too big to fit all of them in a single LLM call. |
| **Cursor `.cursor/rules/*.md` with `paths:` frontmatter globs** | Topic files with YAML frontmatter saying when they apply. | Frontmatter as routing metadata. Our analog: `aliases:` + `kind:` drive which profile loads for which query. |
| **[schema.org `Person`](https://schema.org/Person) / FOAF** | ~100 properties; FOAF explicitly equates `foaf:Person` with `schema:Person`. | A *menu* to pick from for frontmatter keys (`givenName`, `familyName`, `birthDate`, `knows`, `worksFor`, `alumniOf`) — don't adopt the full vocabulary, don't go RDF-native. The semantic web tax isn't worth paying for a personal tool. |
| **vCard** | Contact-card serialization. | The structured-contact subset (email, phone, address). Not the format itself — vCard has no relationship semantics beyond "this contact." |
| **Obsidian `[[wikilinks]]` + dataview frontmatter** | Free-form notes with `aliases:`, `tags:`, `related:` frontmatter; community-converged on bracket-link conventions. | `aliases:` as a first-class field, wikilinks-in-frontmatter syntax (quoted), and the "frontmatter for structured, body for prose" split. Dataview's `Key:: Value` inline syntax is too cute for this use case. |
| **[Microsoft Recall](https://learn.microsoft.com/en-us/windows/ai/apis/recall)** | Local screenshot snapshots, semantic search via on-device model. | Not much. Recall is a search index over raw observations; we're building derived *summaries*. Different abstraction layer. |
| **PARA / Zettelkasten / "second brain"** | Folder-hierarchy methodologies for human note-taking. | Mostly fluff for an automated derivation pipeline. The cognitive overhead they reduce (for humans deciding where to file something) doesn't exist when a script writes the files. Skip. |

The cleanest analogy is Claude Code's own memory system, because the constraints are isomorphic: an agent reads a small always-loaded index, decides which detail files to pull, and writes back updates. Reinvent that wheel as little as possible.

## Format comparison

| Format | Human readability | LLM token cost | Grep-friendliness | Updateability | Provenance |
|---|---|---|---|---|---|
| **Pure markdown narrative** | Best | Low | Best (free-text grep) | Poor — diffs are huge on rewrite; no anchor for structured fields | Awkward; inline links pollute prose |
| **JSON/YAML structured only** | Worst — list of nulls everywhere | Lowest at small scale, but you need *every* field defined | Bad — `jq` works, but the corpus is "facts in fields," not "facts in sentences" | Best — patch a single key | Easy (`source` field per claim), but verbose |
| **YAML frontmatter + narrative** *(recommended)* | High | Low-medium | High — frontmatter is parseable, body is greppable | Good — patch frontmatter mechanically, body via LLM | Footnote convention is natural |
| **Frontmatter + body + provenance section** *(recommended++)* | High | Medium (footnotes add tokens; consider stripping for in-context use) | High | Good | Best of the bunch |
| **SQLite + markdown views** | Low (without view) | Low | Bad without tooling | Best — atomic transactions | Best — foreign-key claims to messages |

SQLite is the "real" answer if you're building a product. For a personal email tool the markdown-on-disk version wins on three things that matter: it's editable by hand when the derivation is wrong, it's diffable in git, and an LLM can read it directly without an intermediate query layer. Revisit if you find yourself writing a query DSL.

## Update strategy

**Default to patch, schedule a full rebuild.** New messages arrive → ingest-time worker identifies the affected profiles (by participant + entity-mention scan), generates a diff against current frontmatter and body, applies it, and appends new footnotes. Once a month, full re-derive from scratch — catches drift, lets you change the schema, and is a forcing function for the prompt to stay good.

Three staleness signals worth tracking:

- **`last_derived` age**: trivial.
- **`source_fingerprint` divergence**: hash of the sorted msg-id list. Cheap. If new messages mention a profile entity, fingerprint changes even if no claim has changed yet.
- **Contradiction flag**: if a patch operation finds a claim that conflicts with an existing footnoted claim, flag it in an `## Open questions` section rather than silently overwriting. Profiles should accumulate uncertainty, not paper over it.

Don't try to be clever about "smart merging." Two passes — patch on arrival, full rebuild monthly — covers ~95% of the value of a fancier change-tracking system, and the monthly rebuild gives you a clean audit trail.

## Index and discovery

`profiles/INDEX.md` is the always-loaded routing table. ~one line per profile:

```markdown
| Canonical | Aliases | Kind | File |
|---|---|---|---|
| Lindberg-Tanaka family | "the family", "home", "us" | household | family.md |
| Aya Tanaka | "Aya", "wife", "Aya T" | person | people/aya-tanaka.md |
| Park Day School | "Park Day", "school" | org | orgs/park-day-school.md |
```

Skip Obsidian-style implicit discovery via `[[wikilinks]]` for top-level routing — wikilinks are great for *between-profile* navigation in the body, but you need an explicit index for "given the string 'Aya' in a new email, which profile do I load?" Alias resolution is the load-bearing query path. Keep it boringly explicit.

A second discovery file, `profiles/RECENT.md`, listing the 10 most-recently-updated profiles, makes the agent's freshness behavior much better at near-zero cost.

## Open questions

- **Identity merge**: when "Joe" and "Joseph M. Smith" first appear to be the same person, does the system merge automatically or surface a confirmation? Probably automatic above some embedding-similarity threshold, with a `merged_from: [...]` audit trail in frontmatter.
- **Group vs individual** granularity: when does "the family" stop being a single profile and become four cross-linked ones? Recommend: write both — the household profile is the high-level lens, individual profiles handle depth.
- **Relationship inverses**: if `family.md` lists Aya as a member, does `aya-tanaka.md` need `member_of: [[family]]`? Yes if you want grep symmetry; risk is the two go out of sync. Mitigation: monthly rebuild is the source of truth, mid-month patches can be one-sided.
- **Confidence representation**: per-claim (footnote-level), per-section, or per-profile? Per-profile is too coarse, per-claim is too noisy. Per-section, with `confidence: high|medium|low` as a section-level inline field, is the right middle.
- **Schema evolution**: `schema_version: 1` in frontmatter is mandatory from day one. The first month you'll regret a field name and want to bulk-rename.
- **Privacy/redaction**: if the user shares a profile dump for debugging, what gets stripped? Worth a `[[REDACT-PII]]` convention before you need it.

Sources: [Schema.org Person](https://schema.org/Person), [FOAF spec](http://xmlns.com/foaf/spec/), [Obsidian properties](https://obsidian.md/help/Editing+and+formatting/Properties), [Claude Code memory](https://code.claude.com/docs/en/memory), [Microsoft Recall overview](https://learn.microsoft.com/en-us/windows/ai/apis/recall), [Dataview README](https://github.com/blacksmithgu/obsidian-dataview).
