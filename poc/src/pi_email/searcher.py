"""Searcher: abstract interface + filesystem + Gmail stub implementations.

The thesis (research/05) is "deterministic Gmail-search download of ALL matches with
no LLM gating". The Searcher abstraction is the seam where that determinism lives —
nothing in the loop ever sees a "decide whether to keep fetching" hook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .corpus import Message, MessageId, parse_message_file


@dataclass
class SearchBatch:
    """One round-trip's worth of search-and-fetch results.

    This is the unit the loop consumes per iteration. Implementations that
    talk to a quota-metered API (Gmail) populate `quota_units_used` and
    `retry_count` so the loop can observe spend; the filesystem impl just
    leaves them at 0.
    """

    query: str
    # Already-fetched messages — *not* IDs. The whole point of widening the
    # protocol is so the impl can batch list+get into a single round-trip
    # and amortize HTTP overhead; the loop never has to call back for bodies.
    hits: list[Message]
    quota_units_used: int = 0
    retry_count: int = 0
    # True if the impl stopped before exhausting matches (e.g. a real Gmail
    # impl hit a configured page-cap). The loop can surface this in logs.
    truncated: bool = False
    # Non-fatal error description if the batch returned partial results.
    # Fatal errors should raise — this is for "we got 47 of 50, here's why".
    error: str | None = None


class Searcher(Protocol):
    """Pluggable search backend. See FilesystemSearcher for the reference impl.

    The primary entry point is `search_and_fetch`: one call, one batch of
    fully-materialized messages. `search` and `fetch` are kept for backward
    compat and for callers (tests, ad-hoc tooling) that don't need batching;
    real impls (GmailSearcher) MUST implement `search_and_fetch` to avoid
    per-message HTTP and quota cost.
    """

    def search_and_fetch(self, query: str) -> SearchBatch:
        """Run the query and return all matches as a single batch.

        Implementations are free to internally paginate, batch, and retry —
        the loop just sees one SearchBatch per call.
        """
        ...

    def search(self, query: str) -> list[MessageId]:
        """Return ALL message IDs that match the query. No top-k, no LLM filter.

        Kept for backward compat; new code should call `search_and_fetch`.
        """
        ...

    def fetch(self, msg_id: MessageId) -> Message:
        """Fetch the full message body. May raise if not found.

        Kept for backward compat; new code should call `search_and_fetch`.
        """
        ...


# ---------- Query parsing ----------
#
# Tiny Gmail-style query grammar:
#
#   expr        := disjunction
#   disjunction := conjunction (OR conjunction)*
#   conjunction := atom+                      # implicit AND
#   atom        := field | phrase | term
#   field       := (from|to|subject|cc|bcc):value
#   phrase      := "..."
#   term        := bare word
#
# Precedence: AND binds tighter than OR (so `family OR mom dad` means
# `family OR (mom AND dad)`) — matches Gmail's documented behaviour (see
# research/01).
#
# `OR` is case-insensitive: both `OR` and `or` are recognized as the
# disjunction operator. This is option (b) from the task spec — the live
# proposer's queries arrive lowercased downstream (frontier.mark_ran strips
# case for dedupe storage) so we MUST match a bare lowercase `or`. The
# collateral damage — that nobody can search for the literal word "or" as a
# plain term — is acceptable because no real user query needs it; quoting
# (`"or"`) is the documented escape and the live proposer never emits it.

_FIELD_OP_RE = re.compile(r"(?i)\b(from|to|subject|cc|bcc):")


def _tokenize(query: str) -> list[tuple]:
    """Lex a query string into tokens.

    Token shapes:
      ('term',   str)             # bare word, lowercased
      ('phrase', str)              # quoted phrase, lowercased, no outer quotes
      ('field',  name, value)      # from:/to:/subject:/cc:/bcc:, both lowercased
      ('or',)                      # disjunction operator
    """
    tokens: list[tuple] = []
    i = 0
    n = len(query)
    while i < n:
        # Skip whitespace.
        while i < n and query[i].isspace():
            i += 1
        if i >= n:
            break

        # Quoted phrase: "..."
        if query[i] == '"':
            j = query.find('"', i + 1)
            if j == -1:
                tokens.append(("phrase", query[i + 1:].lower()))
                break
            tokens.append(("phrase", query[i + 1:j].lower()))
            i = j + 1
            continue

        # Field operator: from:, to:, subject:, cc:, bcc: — may be combined
        # with a quoted value (`from:"alice bob"`) or a bare value.
        m = _FIELD_OP_RE.match(query, i)
        if m:
            name = m.group(1).lower()
            i = m.end()
            if i < n and query[i] == '"':
                j = query.find('"', i + 1)
                if j == -1:
                    val = query[i + 1:]
                    i = n
                else:
                    val = query[i + 1:j]
                    i = j + 1
            else:
                j = i
                while j < n and not query[j].isspace():
                    j += 1
                val = query[i:j]
                i = j
            tokens.append(("field", name, val.lower()))
            continue

        # Bare token: either the OR keyword or a plain term.
        j = i
        while j < n and not query[j].isspace():
            j += 1
        raw = query[i:j]
        i = j
        if raw.lower() == "or":
            tokens.append(("or",))
        else:
            tokens.append(("term", raw.lower()))

    return tokens


# AST nodes are nested tuples for compactness:
#   ('and', [atoms])
#   ('or',  [conjunctions])
#   atoms re-use the lex tuple shapes ('term'|'phrase'|'field', ...).


def _parse(query: str) -> tuple | None:
    """Parse a query string into a tiny AST. Returns None for empty queries."""
    tokens = _tokenize(query)
    if not tokens:
        return None

    # Split the token stream on every `or` token. Everything between two ORs
    # (or between an OR and an end of stream) is implicitly AND-joined, giving
    # AND-tighter-than-OR precedence for free.
    conjs: list[list[tuple]] = [[]]
    for tok in tokens:
        if tok[0] == "or":
            conjs.append([])
        else:
            conjs[-1].append(tok)

    # Drop empty conjunctions so leading/trailing/duplicate ORs are forgiving.
    conjs = [c for c in conjs if c]
    if not conjs:
        return None

    def _wrap_and(atoms: list[tuple]) -> tuple:
        if len(atoms) == 1:
            return atoms[0]
        return ("and", atoms)

    if len(conjs) == 1:
        return _wrap_and(conjs[0])
    return ("or", [_wrap_and(c) for c in conjs])


def _eval(node: tuple, msg: Message) -> bool:
    """Evaluate a parsed AST node against a single message (case-insensitive)."""
    kind = node[0]
    if kind == "term" or kind == "phrase":
        return node[1] in msg.all_text().lower()
    if kind == "field":
        _, name, val = node
        target = {
            "from": msg.from_addr,
            "to": msg.to_addr,
            "subject": msg.subject,
            "cc": msg.to_addr,   # fixtures don't model cc separately
            "bcc": msg.to_addr,
        }.get(name, "")
        return val in target.lower()
    if kind == "and":
        return all(_eval(child, msg) for child in node[1])
    if kind == "or":
        return any(_eval(child, msg) for child in node[1])
    raise ValueError(f"unknown AST node: {node!r}")


# ---------- FilesystemSearcher ----------


class FilesystemSearcher:
    """Walks fixtures/family_corpus/, matches by query terms in subject/body/headers.

    Supports a small subset of Gmail syntax: from:, to:, subject:, cc:, bcc:
    field operators; quoted phrases; plain terms (implicit AND); and `OR` /
    `or` for disjunction. AND binds tighter than OR per Gmail conventions.
    Matching is permissive (substring, case-insensitive).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        # Eagerly load all messages so search is a single-pass scan.
        self._messages: dict[MessageId, Message] = {}
        for p in sorted(self.root.glob("*.md")):
            m = parse_message_file(p)
            self._messages[m.message_id] = m

    def search(self, query: str) -> list[MessageId]:
        ast = _parse(query)
        if ast is None:
            return []
        return [mid for mid, msg in self._messages.items() if _eval(ast, msg)]

    def fetch(self, msg_id: MessageId) -> Message:
        if msg_id not in self._messages:
            raise KeyError(msg_id)
        return self._messages[msg_id]

    def search_and_fetch(self, query: str) -> SearchBatch:
        """Filesystem case: search() then iterate fetch().

        No quota, no retries, no truncation — this is the offline baseline.
        Real impls (GmailSearcher) populate the metering fields.
        """
        ids = self.search(query)
        hits: list[Message] = []
        for mid in ids:
            try:
                hits.append(self._messages[mid])
            except KeyError:
                # Should be impossible (we just listed it from this dict)
                # but if a future impl decouples search from the dict, this
                # mirrors the loop's old defensive skip.
                continue
        return SearchBatch(
            query=query,
            hits=hits,
            quota_units_used=0,
            retry_count=0,
            truncated=False,
            error=None,
        )


# The real GmailSearcher lives in `pi_email.gmail_searcher` — kept in its
# own module so this file stays focused on the in-memory FilesystemSearcher
# and the query-grammar parser. Import directly from there:
#
#     from pi_email.gmail_searcher import GmailSearcher
