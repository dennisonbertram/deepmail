# Prior Art Survey: AI Email Tools vs. the "Deep Context Library" Thesis

## Top-line verdict

**The thesis holds up.** Across roughly two dozen commercial and open-source AI email tools surveyed, none ship the combination the user is proposing: (1) deterministic Gmail-search download of **all** matches with no LLM gating, (2) LLM-driven iterative query expansion as a separate planning loop, and (3) durable, on-disk **materialized derived profile documents** (`family.md`, `investor-X.md`, etc.) that act as reusable artifacts. The closest competitor on retrieval sophistication, Shortwave, explicitly uses LLM-gated tool selection followed by a top-k embedding + cross-encoder rerank that drops most matches before the LLM sees them. The closest competitor on "learn the user from email," Cora, runs a one-shot pass over recent history at signup and persists *style/priorities*, not per-contact knowledge. **Confidence: high** for Shortwave/Superhuman/Gemini/Copilot/Cora; **medium** for Spike's iGPT and Microsoft Copilot Outlook (vendors disclose little); **high** for the open-source long tail (uniformly thin LangChain/n8n wrappers).

## Tool-by-tool table

| Tool | Best-guess architecture | LLM gates retrieval? | Derived per-contact profiles? | Likely struggles on "figure out my family" / "summarize investor X" / "all open threads with my lawyer" |
|---|---|---|---|---|
| **Shortwave AI Assistant** | LLM tool-selector → parallel feature-extractor LLMs → Instructor embeddings in Pinecone (per-user namespace) → heuristic rerank → MS Marco MiniLM cross-encoder → single GPT-4 answer call | Yes (tool selection + query reformulation + extractors) | No (only a single user-style doc for the Compose tool) | Top-k cap (rerank cuts "thousand or more" to dozens) means broad relational queries lose recall; no place to accumulate facts across sessions |
| **Superhuman Go** | Partner-agent / MCP-style architecture; eval blog reveals LLM-as-judge over 500/day sampled convos but not retrieval internals | Almost certainly yes (agentic) | Not disclosed | Same agent-stops-early failure mode; eval post implies "lost-in-the-middle" is an active problem |
| **Gemini in Gmail** | Gmail keyword/metadata search + Drive/Calendar context; non-persistent chat | Yes (LLM picks search terms) | No | Search is sender/date/keyword-based per Google's own doc; semantic recall is weak for relational queries |
| **Microsoft Copilot for Outlook** | Microsoft Graph + Microsoft 365 Copilot semantic index; per-request retrieval scoped to convo | Yes | No | Same scope-per-conversation limitation; no cross-thread accumulation |
| **Hey.com** | No AI in product. User-authored "Contact Notes" exist but are manual | N/A | No (manually written by user) | Doesn't even try |
| **Mem.ai** | General memory layer for ChatGPT/Claude; voice/Chrome/mobile capture; no email connector advertised | Yes (delegated to host LLM) | No (no email ingest path on landing page) | Not an email tool; would need user to paste in context |
| **Sanebox** | Heuristic filtering, pre-LLM era | No (rules) | No | Doesn't attempt semantic understanding |
| **Spike iGPT** | Proprietary "email-native" LLM; marketed as understanding email but architecture undisclosed | Likely yes | Not disclosed | Same as everyone else — no public evidence of materialized profiles |
| **Cora (cora.computer)** | Multi-model (Google/Anthropic/OpenAI); one-shot history scan at signup to learn voice + priorities; twice-daily brief + drafts | Yes for new email triage | No per-contact docs; learns *user* style + preferences only | Designed for screening, not for "tell me everything about person X"; backlog isn't processed |
| **Google CC AI Agent** | Gemini + Gmail/Calendar/Drive/web; morning briefing format | Yes | No | "Day ahead" framing, not relational/historical |
| **Inbox Zero (open source, 10.7k★)** | Webhook-driven rule engine; user-written prompt file is two-way synced into discrete DB rules; LLM picks matching rule | Partial: cold-email blocker has deterministic prior-contact gate, otherwise LLM matches rules | No (materializes *rules*, not profiles) | Per-message classifier; no aggregation across an entity over time |
| **Long tail (~30 GitHub repos < 100 stars)** | Almost uniformly: n8n / LangChain / LangGraph wrappers using Groq / OpenAI / Ollama; thin RAG over Gmail API | Yes | No | Hobby/demo quality; none of the top-10 by stars materialize per-entity artifacts |

Sources: Shortwave RAG post ([productionizing-rag-llms-embeddings-cross-encoders](https://www.shortwave.com/blog/productionizing-rag-llms-embeddings-cross-encoders/), [new-shortwave-ai-email-assistant](https://www.shortwave.com/blog/new-shortwave-ai-email-assistant/), [Tasklet](https://www.shortwave.com/blog/introducing-tasklet-ai-automation/), [MCP](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/)); Superhuman ([eval infra](https://superhuman.com/superhuman-eval-infrastructure/)); Gemini ([Gmail help](https://support.google.com/mail/answer/14199860), [Workspace AI](https://workspace.google.com/solutions/ai/)); CC ([blog.google](https://blog.google/innovation-and-ai/models-and-research/google-labs/cc-ai-agent/)); Cora ([cora.computer](https://www.cora.computer/)); Inbox Zero ([ARCHITECTURE.md](https://github.com/elie222/inbox-zero/blob/main/ARCHITECTURE.md)); Hey ([features](https://hey.com/features/)).

## Deep dives

### 1. Shortwave — the only published architecture that approaches the problem seriously

Shortwave's [productionizing-rag](https://www.shortwave.com/blog/productionizing-rag-llms-embeddings-cross-encoders/) and [deep-dive](https://www.shortwave.com/blog/deep-dive-into-worlds-smartest-email-ai/) posts spell out the full pipeline. The pieces relevant to the thesis:

- **LLM-gated entry**: a GPT-4 "tool selection" call decides whether to even touch email history. The user's hypothesis ("LLM decides when it's done") is exactly the design intent — Shortwave's stated principle is *"all reasoning about how to answer a question should be handled by the LLM itself."*
- **Top-k bottleneck**: their own description says "a thousand or more results" must be cut to "dozens" before the cross-encoder, and then dozens to a final ordered fragment list. For "all open threads with my lawyer," this top-k is exactly the bug — important threads that fall outside the top-N never reach the answer LLM.
- **No materialized derived docs**: the *only* persistent derived artifact they describe is a single "pre-computed textual description of [the user's] style and relevant example emails for few-shot prompting" used by the Compose tool. There is no per-contact, per-project, or per-relationship doc.
- **Single-shot, not iterative**: the 2024 update ([new-shortwave-ai-email-assistant](https://www.shortwave.com/blog/new-shortwave-ai-email-assistant/)) added multi-step planning and backtracking, but the stopping criterion is undisclosed and the broader architecture remains "agent decides when it has enough."

Shortwave is the strongest competitor and still doesn't ship the thesis.

### 2. Superhuman Go — eval infrastructure betrays the failure mode

The most informative Superhuman engineering post ([superhuman-eval-infrastructure](https://superhuman.com/superhuman-eval-infrastructure/)) is about *evaluating* Go, not building it. Two tells:

- They explicitly call out the "lost-in-the-middle" effect as a known problem they're working around with prompt engineering — i.e. when retrieval over-stuffs context, model attention degrades. This is consistent with conventional LLM-gated RAG.
- Their utility scale (-1 to 3) includes "proper execution" as a failure mode flagged by the LLM-judge — they're internally measuring incomplete tool use, which is exactly what the thesis predicts.

No architecture details are public. The [Partner Agents](https://blog.superhuman.com/) framing and [Superhuman Mail MCP](https://blog.superhuman.com/) launch suggest a Claude-style tool-use loop, with the same agent-stops-early failure mode.

### 3. Cora — the closest "build a profile" precedent, but it's the user's profile

[Cora](https://www.cora.computer/) advertises that on signup it "analyzes a selection of your email history" to learn who the user responds to, what they act on, and their writing style. This is the spiritual cousin of the user's `family.md` idea — but applied to the user, not their contacts. There is no public mention of per-contact materialized docs, and the product surface (twice-daily Brief + drafts) is oriented around triage, not deep relational queries. Backlog isn't processed; only new mail after activation. So the design pattern of "do a one-shot pass and materialize an artifact" exists in production, but only at user scope.

### 4. Inbox Zero — the only open-source codebase that materializes anything

[Inbox Zero's ARCHITECTURE.md](https://github.com/elie222/inbox-zero/blob/main/ARCHITECTURE.md) (10.7k stars, by far the most-starred open-source AI email tool) does one thing that rhymes with the thesis: a **two-way sync** between a user-authored prompt file and discrete database rules. Their stated reason — *"In most cases, the AI is only deciding if conditions are matched"* — is essentially the same argument the user is making: materialize the LLM's interpretation into a stable, reusable artifact so you don't re-derive it each query. But the artifacts are *rules*, not *people*. The cold-email blocker also implements one deterministic gate the thesis would endorse: it only invokes the LLM "when the user has never sent us an email before" — a hard prior-correspondence check, no LLM judgment.

### 5. The open-source long tail confirms the gap

The top GitHub results for "email agent" / "gmail agent" / "inbox AI" outside Inbox Zero are uniformly thin wrappers: `Drlordbasil/groq-gmail-assistant` (50★), `dbish/DispatchMail` (91★), `jacob-dietle/Autonomous-Sales-Inbox-and-CRM-Assistant` (51★), then a long tail of <30-star projects mostly built on n8n + Groq/Ollama. None describe materialized derived profiles. None describe deterministic exhaustive retrieval. They're all variations on "LLM picks emails, drafts replies." See the awesome-n8n-templates repo (22.2k★) as the canonical example of the ecosystem — it's automation glue, not an architecture.

## Implications for the user's design

**Confirmed gaps in the market:**

1. **No deterministic exhaustive retrieval.** Every commercial tool top-k-caps. Gmail's native query language is far more expressive than any of them use; downloading all matches and letting the LLM grep is a real differentiator.
2. **No materialized per-entity derived docs.** Cora materializes *user* facts. Inbox Zero materializes *rules*. Shortwave materializes *user style*. Nobody materializes `family.md` / `investor-X.md`. This is open territory.
3. **No iterative query expansion as a separate loop.** Shortwave does parallel feature extraction (one shot) and multi-step backtracking (agent-driven), but not a deliberate "fan out queries until coverage saturates" pattern.

**Features worth copying (don't reinvent):**

- Shortwave's **parallel feature-extractor LLMs** for date ranges / people / labels (each with a confidence score). Good for translating natural-language queries into Gmail's `from:`, `to:`, `after:`, `label:` syntax.
- Inbox Zero's **two-way sync between prompt-file and DB artifacts** — strong UX precedent for letting users edit materialized docs and have the system honor changes.
- Cora's **one-shot history scan on signup** as the bootstrap step before iterative expansion kicks in.
- Shortwave's **cross-encoder rerank** if you ever do need to compress a too-large result set (but only after exhaustive download, not as a top-k filter).

**Differentiate on:**

- Deterministic-first retrieval (LLM proposes queries, code executes, results are not filtered by LLM before grep).
- Materialized profile docs as first-class durable artifacts the user can read, edit, and grep.
- An iterative loop where the LLM's job is *query expansion*, not *answer generation* — the answer comes from grep + a final summarization pass over materialized docs.

**Honest risk:** Shortwave has been at this longer and may already do "materialized profiles" internally without marketing it. The public evidence does not show it — but their Tasklet product and MCP work suggest they're heading toward agent automation, where derived-doc materialization is a natural next move. Worth assuming a 12-month window before this becomes table stakes.

## Sources

- [Shortwave — Productionizing RAG: LLMs, embeddings, cross-encoders](https://www.shortwave.com/blog/productionizing-rag-llms-embeddings-cross-encoders/)
- [Shortwave — The new Shortwave AI Assistant (Sep 2024)](https://www.shortwave.com/blog/new-shortwave-ai-email-assistant/)
- [Shortwave — Introducing Tasklet (Oct 2025)](https://www.shortwave.com/blog/introducing-tasklet-ai-automation/)
- [Shortwave — MCP integration (May 2025)](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/)
- [Shortwave — Meet your AI executive assistant](https://www.shortwave.com/blog/meet-your-ai-email-executive-assistant/)
- [Shortwave — AI Launch Week recap](https://www.shortwave.com/blog/everything-we-shipped-for-ai-launch-week/)
- [Superhuman — Building a Rigorous Conversation Quality Evaluation System for Superhuman Go](https://superhuman.com/superhuman-eval-infrastructure/)
- [Superhuman — blog index](https://blog.superhuman.com/)
- [Google Workspace — Gemini in Gmail help](https://support.google.com/mail/answer/14199860)
- [Google Workspace — AI overview](https://workspace.google.com/solutions/ai/)
- [Google Labs — CC AI agent](https://blog.google/innovation-and-ai/models-and-research/google-labs/cc-ai-agent/)
- [Cora](https://www.cora.computer/)
- [Hey.com — features](https://hey.com/features/)
- [Spike — AI features](https://spikenow.com/ai/)
- [Inbox Zero — ARCHITECTURE.md](https://github.com/elie222/inbox-zero/blob/main/ARCHITECTURE.md)
- [Inbox Zero — repo](https://github.com/elie222/inbox-zero)
- [GitHub topic: email-agent](https://github.com/topics/email-agent?o=desc&s=stars)
- [GitHub search: inbox AI assistant](https://github.com/search?q=inbox+AI+assistant&type=repositories&s=stars&o=desc)
- [HN Algolia — AI email assistant stories, 2025+](https://hn.algolia.com/api/v1/search?query=ai+email+assistant&tags=story&numericFilters=created_at_i%3E1735689600)
