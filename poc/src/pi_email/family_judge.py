"""LLM-as-final-judge: given a candidate name + email excerpts, decide whether
this person is most likely the user's family member.

This module is Pass 8B's architectural shift. After 7 passes of rule-based
extraction tuning, we were stuck in a precision/recall corner — strict rules
either over-pruned (zero recall) or under-pruned (high FP rate). The rule-based
gate is too brittle for messy real-world Gmail.

The strategy: loose extraction admits MORE candidates, then this judge filters
the final candidate list. The LLM is much better than regex at "is this person
likely the user's family member based on these emails?".

Public API:
  * `FamilyVerdict(canonical, decision, relation_guess, confidence, reasoning)`
  * `FamilyJudge.judge(candidate, excerpts, user_email, user_display_name,
                       user_surname)`
  * `FamilyJudge.judge_batch(candidates, user_email, user_display_name,
                             user_surname)`

Backend selection mirrors `proposer.py`:
  * Live: Anthropic SDK, model `claude-sonnet-4-5`, with prompt caching on the
    static system prompt. Sequential per-candidate calls — keeps debugging
    simple and lets the cache hit dominate cost.
  * Mock: deterministic, conservative pattern match against the excerpts.
    Fires when an excerpt contains "my <RELATION> <First>" / "your <RELATION>
    <First>" / "<RELATION> <First>" (case-insensitive). Permissive enough to
    keep the bundled fixture demo working without an API key, conservative
    enough that real-Gmail candidates with no relation-word context are
    classified `not_family`.

The judge is invoked from `materializer.write_family_profile` BEFORE the
profile is written, so only `family` verdicts land in the accepted Members
section. `uncertain` verdicts land in a separate "Possibly family" section
for the user to review; `not_family` verdicts are listed as bare slugs in
"Rejected" so the user can sanity-check we didn't silently drop a real
relative.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal


MODEL = "claude-sonnet-4-5"

Decision = Literal["family", "not_family", "uncertain"]

# Relation words the mock backend searches for. Kept in sync with
# `entities.RELATION_WORDS` in spirit but local so this module doesn't import
# from the parallel-worker file. Mock-only — the live judge gets the full
# relation taxonomy via the system prompt.
_MOCK_RELATION_WORDS: frozenset[str] = frozenset({
    "mom", "mother", "mum", "mommy", "ma",
    "dad", "father", "daddy", "papa", "pa",
    "wife", "husband", "spouse", "partner", "fiance", "fiancee",
    "son", "daughter", "kid", "kids", "child", "children", "baby",
    "brother", "bro", "sister", "sis", "sibling", "siblings",
    "grandma", "grandmother", "nana", "grandpa", "grandfather",
    "grandson", "granddaughter", "grandchild", "grandkids",
    "aunt", "auntie", "uncle",
    "cousin", "niece", "nephew",
    "stepmom", "stepdad",
})


# ---------------- Data ----------------


@dataclass(frozen=True)
class FamilyVerdict:
    """One judging result for one candidate."""

    canonical: str
    decision: Decision
    relation_guess: str | None
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class ContactEvidence:
    """Pass 12A: Google Contacts signal for a single candidate.

    Constructed by the materializer from `contacts.Contact` data and passed
    into the judge alongside email excerpts. The judge sees these fields
    rendered into the user prompt as a `GOOGLE_CONTACTS_SIGNAL` block; the
    system prompt is amended to instruct the model that Family-group
    membership (`family_signal_source` containing "group_membership") is a
    near-decisive signal.

    `family_signal_strength` ∈ [0.0, 1.0] — see
    `contacts.score_family_signal` for the priority table.
    `family_signal_source` is a "+"-joined tag listing detectors that
    fired (e.g., "group_membership+relations_field+biography").
    """

    contact_name: str
    emails: tuple[str, ...]
    family_signal_strength: float
    family_signal_source: str
    relations: tuple[str, ...] = ()  # human-readable e.g. "spouse: Jana"
    biography: str | None = None
    in_family_group: bool = False


# ---------------- System prompt (cached) ----------------
#
# Pass 15A: the system prompt is now COMPOSED per call, not a single static
# string. When `contacts_population` is 0 or None, the contacts-related blocks
# are suppressed entirely — the model never reads about "Google Contacts" or
# the GOOGLE_CONTACTS_SIGNAL block, and the SURNAME signal is reasserted as
# the strongest available authority.
#
# Why: through Pass 14A the system prompt told the model that Google Contacts
# is the STRONGEST signal. Even with a separate hint that absence is expected
# when `contacts_population=0`, the model carried the "contacts is strongest"
# framing and hedged on surname-match family candidates (Jana Bertram,
# user's spouse, was demoted from Accepted 0.90 to Uncertain 0.75 in Run 13/14).
#
# Mitigation: when the user has no curated family contacts, don't mention
# contacts at all. Restate the priority hierarchy as if contacts didn't exist.
#
# Cache implications: Anthropic prompt caching keys on exact system-block
# text. The two variants get separate cache entries; each is hit once per
# batch. The cost increase is negligible and the verdict quality wins
# clearly justify it.


STATIC_BASE_PROMPT = """You are an expert at identifying family relationships from email correspondence.

Given a candidate person's name and a few email excerpts mentioning them, decide if this person is most likely a FAMILY MEMBER of the email account owner.

FAMILY MEMBERS include: spouse, partner, parents (mom/dad/mother/father), siblings (brother/sister), children (son/daughter/kid), grandparents, grandchildren, aunts/uncles, cousins, nieces/nephews, in-laws.

NOT FAMILY: coworkers, business partners (e.g., "general partner", "founding partner" of a fund), professional contacts, lawyers, accountants, vendors, customers, friends (unless explicitly family), public figures in news, organization names.

STRONG SIGNALS OF NOT FAMILY:
- Business titles in the candidate's evidence (CEO, Founder, Partner at a firm).
- The candidate's email domain is the same as the user's work domain
  (suggests coworker, not family — unless surname also matches).
- Newsletter / mailing-list-shaped content in the excerpts.
- Public figures (politicians, founders, VCs in news).

For each candidate, output JSON:
{
  "decision": "family" | "not_family" | "uncertain",
  "relation_guess": <one of: spouse, parent, sibling, child, grandparent, grandchild, aunt_or_uncle, cousin, niece_or_nephew, in_law, other> | null,
  "confidence": <0.0-1.0>,
  "reasoning": "<one-sentence explanation>"
}

Use "uncertain" when evidence is ambiguous (e.g., relation word appears but person could be a colleague). Default to "not_family" when no clear family signal exists.

Reply with ONLY the JSON object. No preamble, no markdown, no explanation outside the JSON.
"""


NO_CONTACTS_PROMPT_SECTION = """PRIORITY SIGNALS (rely on email evidence and identity):
1. SURNAME match — if the candidate's last name matches the USER's surname, treat as 0.85+ confidence family unless email evidence clearly contradicts (e.g., candidate is clearly a coworker who happens to share surname).
2. Direct possessive in email body — "my wife <name>", "my husband <name>", "my mom <name>", etc., where <name> matches the candidate.
3. Repeated calendar/email evidence of cohabitation, shared parenting, joint travel, joint household decisions.
4. Personal email domain (gmail, icloud, yahoo, etc.) combined with frequent bidirectional correspondence.

Without these signals, default to not_family.
"""


WITH_CONTACTS_PROMPT_SECTION = """PRIORITY SIGNALS (Google Contacts data is available):
1. Google Contacts "Family" group membership — STRONGEST signal. The user
   curated this directly in their address book. Treat as 0.95+ confidence
   family unless email evidence explicitly contradicts (e.g., the candidate is
   clearly a coworker who got mis-tagged).
2. Google Contacts `relations` field with a kinship type (spouse, mother,
   brother, etc.) — STRONG signal. Treat as 0.85+ confidence family.
3. Google Contacts biography mentioning kinship words ("my sister", "wife",
   "mom") — MEDIUM-STRONG signal. Treat as 0.75+ confidence family.
4. SURNAME match + spouse-like email evidence — 0.85+ confidence family unless
   email evidence clearly contradicts.
5. Direct possessive in email body — "my wife <name>", "my husband <name>",
   "my mom <name>", etc., where <name> matches the candidate.
6. The user (account owner) sends emails introducing the candidate as family
   ("intro to my wife <name>", "my brother <name>").
7. The candidate is referred to with multiple family relations (e.g., "Bob is
   my brother", "Bob and his wife Sarah").

A GOOGLE_CONTACTS_SIGNAL block may appear above the email evidence. When it
shows `in_family_group: true` OR `family_signal_strength >= 0.85`, the
Contacts signal OVERRIDES weak/absent email evidence — the user's own address
book is more authoritative than absence of explicit kinship words in mail.

CONTACTS INTERPRETATION:
- This user has N curated family contacts in their address book.
- A candidate with NO matching Google Contacts entry is weak negative evidence
  (-0.05 to -0.10 confidence on family verdicts).
- When `contacts_population` is > 0: the user actively curates family contacts.
  Absence of a GOOGLE_CONTACTS_SIGNAL on a candidate IS weak negative evidence
  (~0.1 confidence penalty). Presence is strong positive evidence.

BUT: if the Contacts signal is strong (in_family_group OR
family_signal_strength >= 0.85), prefer family over coworker — the user
curated this contact themselves.
"""


def _build_system_prompt(contacts_population: int | None) -> str:
    """Compose the system prompt for the live judge.

    Pass 15A: when the user has NO curated family contacts (population is 0
    or None), suppress all contacts-related instructions. The model otherwise
    over-weights the "Google Contacts is the strongest signal" message even
    when no signal exists, hedging on surname-match family candidates.

    * `contacts_population in (None, 0)` -> base prompt + NO_CONTACTS section
      (no mention of Google Contacts; surname match is signal #1).
    * `contacts_population > 0` -> base prompt + WITH_CONTACTS section
      (full priority signals including Google Contacts).

    The two variants are separate cache entries under Anthropic's prompt
    caching; each is hit once per batch.
    """
    if contacts_population is None or contacts_population == 0:
        contacts_section = NO_CONTACTS_PROMPT_SECTION
    else:
        contacts_section = WITH_CONTACTS_PROMPT_SECTION
    return STATIC_BASE_PROMPT + "\n" + contacts_section


# ---------------- Mock backend ----------------


def _first_token(name: str) -> str:
    """Lowercased first whitespace-token of the candidate's canonical name.
    Empty string when the input is empty."""
    if not name or not name.strip():
        return ""
    return name.split()[0].lower()


def _mock_relation_match(excerpts: Iterable[dict], candidate: str) -> tuple[bool, str | None]:
    """Conservative pattern search for "my/your <RELATION> <candidate_first>"
    OR an adjacent "<RELATION> <candidate_first>" in the excerpts.

    Returns (matched, relation_token) where relation_token is the matched
    relation word (lowercased) when found, else None.

    The spec calls for `my <relation> <first>` only, but we also accept
    `your <relation> <first>` (emails FROM a family member ADDRESSED to the
    user — common in fixtures: "Your sister Emma here") and the bare
    `<relation> <first>` form ("Aunt Carol here", "Grandma Helen"). Without
    these the bundled fixture demo would have zero accepted members. Still
    conservative against real-Gmail noise: requires a relation word
    IMMEDIATELY adjacent to the candidate's first name.
    """
    first = _first_token(candidate)
    if not first:
        return False, None

    # Build one alternation regex over all relation words. Longest first so
    # multi-token relations like "mother-in-law" win the longest match.
    rel_alt = "|".join(
        re.escape(w) for w in sorted(_MOCK_RELATION_WORDS, key=len, reverse=True)
    )
    # Patterns to try, in priority order:
    #   1. "my <relation> <first>"   (the spec's canonical pattern)
    #   2. "your <relation> <first>" (inverse — sender addressing the user)
    #   3. "<relation> <first>"      (adjacent, e.g., "Aunt Carol")
    patterns = [
        rf"\bmy\s+({rel_alt})\s+{re.escape(first)}\b",
        rf"\byour\s+({rel_alt})\s+{re.escape(first)}\b",
        rf"\b({rel_alt})\s+{re.escape(first)}\b",
    ]
    for ex in excerpts:
        # Search across the meaningful text of each excerpt: snippet/body +
        # subject. We DON'T scan from_addr because relation+name in an
        # address like "Aunt Carol <carol@host>" is the candidate itself
        # showing up in the From header — circular evidence.
        haystack = " ".join(
            str(ex.get(k) or "") for k in ("snippet", "body", "subject")
        )
        if not haystack:
            continue
        haystack_low = haystack.lower()
        for pat in patterns:
            m = re.search(pat, haystack_low, flags=re.IGNORECASE)
            if m:
                return True, m.group(1).lower()
    return False, None


def _mock_relation_to_relation_guess(token: str | None) -> str | None:
    """Map a raw relation token from the mock match to the high-level
    relation_guess vocabulary used by FamilyVerdict.relation_guess."""
    if not token:
        return None
    t = token.lower()
    if t in {"wife", "husband", "spouse", "partner", "fiance", "fiancee"}:
        return "spouse"
    if t in {"mom", "mother", "mum", "mommy", "ma", "dad", "father", "daddy", "papa", "pa", "stepmom", "stepdad"}:
        return "parent"
    if t in {"brother", "bro", "sister", "sis", "sibling", "siblings"}:
        return "sibling"
    if t in {"son", "daughter", "kid", "kids", "child", "children", "baby"}:
        return "child"
    if t in {"grandma", "grandmother", "nana", "grandpa", "grandfather"}:
        return "grandparent"
    if t in {"grandson", "granddaughter", "grandchild", "grandkids"}:
        return "grandchild"
    if t in {"aunt", "auntie", "uncle"}:
        return "aunt_or_uncle"
    if t == "cousin":
        return "cousin"
    if t in {"niece", "nephew"}:
        return "niece_or_nephew"
    return "other"


def _candidate_last_token(candidate: str) -> str:
    """Lowercased last whitespace-token of the candidate's canonical name —
    used by the mock backend to test for surname matches against the user's
    surname. Empty string when the input has no tokens."""
    if not candidate or not candidate.strip():
        return ""
    return candidate.split()[-1].lower()


def _mock_judge_one(
    candidate: str,
    excerpts: list[dict],
    user_surname: str | None = None,
    contact_evidence: ContactEvidence | None = None,
    contacts_population: int | None = None,
) -> FamilyVerdict:
    """Deterministic mock — used when no ANTHROPIC_API_KEY is set OR when the
    caller passes force_mock=True.

    Priority order:
      1. Pass 12A: a strong Google Contacts signal (in_family_group OR
         family_signal_strength >= 0.85) returns family at the signal's
         strength. This mirrors the live prompt — Contacts is authoritative.
      2. Surname-match: a 2+ token candidate whose LAST token equals
         `user_surname` returns family at 0.85 confidence.
      3. Excerpt relation-word pattern match returns family at 0.7.
      4. Default: not_family at 0.5.

    Pass 14A: `contacts_population` modulates the "absent contact signal"
    interpretation:
      * None or 0: no penalty for missing contact_evidence — the user has
        no curated family list (or contacts not consulted at all), so the
        absence is uninformative.
      * > 0: the user actively curates family contacts. A family verdict
        for a candidate that lacks contact_evidence picks up a ~0.1
        confidence penalty (the user's address book disagrees by omission).
    """
    # 1. Contacts signal (Pass 12A). The strongest authority.
    if contact_evidence is not None:
        if (
            contact_evidence.in_family_group
            or contact_evidence.family_signal_strength >= 0.85
        ):
            # Try to derive a relation_guess from the structured relations
            # field if present.
            rel_guess: str | None = "other"
            for rel_str in contact_evidence.relations:
                # "spouse: Jana" -> "spouse"
                token = rel_str.split(":", 1)[0].strip().lower()
                mapped = _mock_relation_to_relation_guess(token)
                if mapped:
                    rel_guess = mapped
                    break
            # Pass 14A: when the user curates contacts AND this candidate
            # is in that curated set with a strong signal, boost to 0.95
            # (the system-prompt's "presence is strong positive evidence"
            # clause).
            base_conf = max(
                0.95, float(contact_evidence.family_signal_strength)
            )
            return FamilyVerdict(
                canonical=candidate,
                decision="family",
                relation_guess=rel_guess,
                confidence=base_conf,
                reasoning=(
                    f"[mock] Google Contacts signal "
                    f"({contact_evidence.family_signal_source or 'unknown'}) "
                    f"identifies candidate as family."
                ),
            )

    # Pass 14A: only apply the absent-signal penalty when the user curates
    # contacts (population > 0). When population is None or 0, absence is
    # expected and uninformative.
    apply_absent_penalty = (
        contact_evidence is None
        and contacts_population is not None
        and contacts_population > 0
    )

    # 2. Surname check — it's the strongest single in-email signal.
    cand_last = _candidate_last_token(candidate)
    cand_tokens = candidate.split() if candidate else []
    if (
        user_surname
        and cand_last
        and len(cand_tokens) >= 2
        and cand_last == user_surname.strip().lower()
    ):
        conf = 0.85
        reasoning_extra = ""
        if apply_absent_penalty:
            conf = round(conf - 0.10, 4)
            reasoning_extra = (
                " (penalty: user curates contacts but candidate has no "
                "contacts signal)"
            )
        return FamilyVerdict(
            canonical=candidate,
            decision="family",
            relation_guess="other",
            confidence=conf,
            reasoning=(
                f"[mock] candidate surname matches user's surname "
                f"'{user_surname}'.{reasoning_extra}"
            ),
        )
    matched, rel = _mock_relation_match(excerpts, candidate)
    if matched:
        conf = 0.7
        reasoning_extra = ""
        if apply_absent_penalty:
            conf = round(conf - 0.10, 4)
            reasoning_extra = (
                " (penalty: user curates contacts but candidate has no "
                "contacts signal)"
            )
        return FamilyVerdict(
            canonical=candidate,
            decision="family",
            relation_guess=_mock_relation_to_relation_guess(rel),
            confidence=conf,
            reasoning=(
                f"[mock] excerpt contains relation cue near "
                f"'{candidate.split()[0]}'.{reasoning_extra}"
            ),
        )
    # 3. Weaker contacts signal — anything below 0.85 lands as uncertain via
    # the materializer's bucket; we render it here at the signal's strength.
    if (
        contact_evidence is not None
        and contact_evidence.family_signal_strength > 0.0
    ):
        return FamilyVerdict(
            canonical=candidate,
            decision="uncertain",
            relation_guess=None,
            confidence=float(contact_evidence.family_signal_strength),
            reasoning=(
                f"[mock] weak Google Contacts signal "
                f"({contact_evidence.family_signal_source or 'unknown'}); "
                f"no email cue."
            ),
        )
    return FamilyVerdict(
        canonical=candidate,
        decision="not_family",
        relation_guess=None,
        confidence=0.5,
        reasoning="[mock] no adjacent relation cue found in excerpts.",
    )


# ---------------- Live backend ----------------


# Maximum number of excerpts shown to the judge per candidate. Bumped from 5
# to 8 in Pass 9A — early runs left family signal sitting in the 6th/7th
# excerpt of high-volume contacts (e.g., a spouse with many threads); 8 is a
# pragmatic upper bound that still keeps the per-call token cost bounded.
# Cache hits on the static system block dominate cost anyway.
_MAX_EXCERPTS_PER_CANDIDATE = 8


def _format_excerpts_for_prompt(excerpts: list[dict]) -> str:
    """Render up to 8 excerpts as a numbered list for the user-prompt body.
    Excerpts beyond the cap are dropped — bounds per-call cost. Bumped from
    5 to 8 in Pass 9A; see `_MAX_EXCERPTS_PER_CANDIDATE`."""
    out: list[str] = []
    for idx, ex in enumerate(excerpts[:_MAX_EXCERPTS_PER_CANDIDATE], start=1):
        frm = str(ex.get("from_addr") or "").strip()
        sub = str(ex.get("subject") or "").strip()
        snip = str(ex.get("snippet") or ex.get("body") or "").strip()
        snip = snip.replace("\n", " ")
        if len(snip) > 400:
            snip = snip[:400].rstrip() + "..."
        out.append(
            f"{idx}. from: {frm or '(unknown)'} | subject: {sub or '(none)'}\n"
            f"   excerpt: \"{snip}\""
        )
    if not out:
        return "(no excerpts available)"
    return "\n".join(out)


def _format_contact_evidence(ce: ContactEvidence) -> str:
    """Render a ContactEvidence as a multi-line GOOGLE_CONTACTS_SIGNAL block.

    Pass 12A: the Contacts signal lives ABOVE the email evidence so the
    model reads the strongest authority first. The block is intentionally
    terse — no narrative — so the live model treats it as structured input
    rather than free-form prose."""
    rels_str = ", ".join(ce.relations) if ce.relations else "-"
    bio_str = ce.biography.strip() if ce.biography else "-"
    # Bound the biography snippet so a long notes field doesn't blow the
    # per-call prompt budget.
    if len(bio_str) > 400:
        bio_str = bio_str[:400].rstrip() + "..."
    emails_str = ", ".join(ce.emails) if ce.emails else "-"
    return (
        f"GOOGLE_CONTACTS_SIGNAL:\n"
        f"  contact_name: {ce.contact_name}\n"
        f"  emails: {emails_str}\n"
        f"  in_family_group: {str(ce.in_family_group).lower()}\n"
        f"  family_signal_source: {ce.family_signal_source or '-'}\n"
        f"  family_signal_strength: {ce.family_signal_strength:.2f}\n"
        f"  relations_field: {rels_str}\n"
        f"  biography_snippet: {bio_str}\n"
    )


def _build_user_prompt(
    candidate: str,
    excerpts: list[dict],
    user_email: str | None,
    user_display_name: str | None,
    user_surname: str | None = None,
    contact_evidence: ContactEvidence | None = None,
    contacts_population: int | None = None,
) -> str:
    """Build the per-candidate user message.

    `user_surname` is rendered as a separate field, AND we append a note
    reinforcing the system-prompt's surname-weight instruction so the model
    sees it both as static context (cached) and at the per-call site (not
    cached but immediately above the candidate).

    Pass 12A: when `contact_evidence` is non-None, a GOOGLE_CONTACTS_SIGNAL
    block is rendered ABOVE the email evidence. The model treats this as a
    high-authority signal per the system prompt.

    Pass 14A: when `contacts_population` is non-None, it is rendered as an
    extra field in the USER block so the model can condition on whether the
    user actively curates a family-contacts list. See the system prompt's
    CONTACTS INTERPRETATION section — `contacts_population: 0` tells the
    model that absence of a GOOGLE_CONTACTS_SIGNAL is EXPECTED and should
    not be held against the candidate."""
    contact_block = ""
    if contact_evidence is not None:
        contact_block = "\n" + _format_contact_evidence(contact_evidence) + "\n"
    pop_line = ""
    if contacts_population is not None:
        pop_line = f"  contacts_population: {int(contacts_population)}\n"
    return (
        f"USER (account owner):\n"
        f"  email: {user_email or 'unknown'}\n"
        f"  name: {user_display_name or 'unknown'}\n"
        f"  surname: {user_surname or 'unknown'}\n"
        f"{pop_line}"
        f"\n"
        f"NOTE: A candidate sharing the user's surname is a STRONG signal of "
        f"family relationship (spouse, parent, sibling, child). Treat surname "
        f"matches as high-confidence family indicators unless the excerpts "
        f"contradict.\n\n"
        f"CANDIDATE: {candidate}\n"
        f"{contact_block}"
        f"\nEMAIL_EVIDENCE (up to {_MAX_EXCERPTS_PER_CANDIDATE} excerpts):\n"
        f"{_format_excerpts_for_prompt(excerpts)}\n\n"
        f"Judge this candidate. Reply with ONLY the JSON object."
    )


def _extract_json(text: str) -> dict | None:
    """Try strict parse; on failure, fall back to first {...} block."""
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


def _parse_verdict(
    candidate: str,
    data: dict | None,
    log: Callable[[str], None],
) -> FamilyVerdict | None:
    """Coerce the parsed JSON into a FamilyVerdict. Returns None on shape
    failure — caller decides whether to fall back to mock or skip."""
    if not isinstance(data, dict):
        return None
    decision_raw = str(data.get("decision", "")).strip().lower()
    if decision_raw not in {"family", "not_family", "uncertain"}:
        log(
            f"family_judge: dropping invalid decision={decision_raw!r} for "
            f"candidate={candidate!r}"
        )
        return None
    relation_guess = data.get("relation_guess")
    if relation_guess is not None:
        relation_guess = str(relation_guess).strip() or None
    confidence_raw = data.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", "")).strip()
    return FamilyVerdict(
        canonical=candidate,
        decision=decision_raw,  # type: ignore[arg-type]
        relation_guess=relation_guess,
        confidence=confidence,
        reasoning=reasoning,
    )


def _live_judge_one(
    client,
    candidate: str,
    excerpts: list[dict],
    user_email: str | None,
    user_display_name: str | None,
    user_surname: str | None,
    model: str,
    log: Callable[[str], None],
    contact_evidence: ContactEvidence | None = None,
    contacts_population: int | None = None,
) -> FamilyVerdict | None:
    user_msg = _build_user_prompt(
        candidate,
        excerpts,
        user_email,
        user_display_name,
        user_surname,
        contact_evidence=contact_evidence,
        contacts_population=contacts_population,
    )
    # Pass 15A: the system prompt is now composed per call based on whether
    # the user has any curated family contacts. With population in (None, 0)
    # the model receives a prompt that does NOT mention Google Contacts at
    # all — surname-match becomes signal #1. With population > 0, the full
    # Contacts-aware priority hierarchy is used. The two variants get
    # separate Anthropic prompt-cache entries; each is hit once per batch.
    system_prompt = _build_system_prompt(contacts_population)
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        # temperature=0 — judge must be reproducible across runs (Pass 11).
        # Run 9 and Run 10 returned different verdicts (Accepted 0.90 vs
        # Uncertain 0.75) for the same candidate and similar evidence;
        # sampling jitter at the model's default temperature was the cause.
        # Pinning to 0 collapses the verdict distribution to a single mode
        # per input so identical evidence -> identical bucket every time.
        temperature=0.0,
        # Prompt caching: the (variant-specific) system block is identical
        # across all candidates in a batch with the same contacts_population
        # state, so marking it ephemeral lets the second and subsequent
        # calls hit the cache. The per-candidate user message stays uncached.
        system=[
            {
                "type": "text",
                "text": system_prompt,
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
        log(f"family_judge: failed to parse JSON for {candidate!r}; raw={text[:200]!r}")
        return None
    return _parse_verdict(candidate, data, log)


# ---------------- Public class ----------------


class FamilyJudge:
    """Wraps live + mock backends. Defaults to live; falls back to mock if no
    API key is set or the SDK can't be constructed.

    One instance can judge many candidates across a single materializer pass.
    """

    def __init__(
        self,
        client=None,
        model: str = MODEL,
        force_mock: bool = False,
        on_log: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.force_mock = force_mock
        self.is_mock = True
        self._client = client
        self._log = on_log or (lambda s: None)
        if client is not None:
            # Caller-injected client (tests) — assume live, the test will
            # supply a fake.
            self.is_mock = False
            return
        if force_mock:
            return
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic()
                self.is_mock = False
            except Exception as exc:  # pragma: no cover - best-effort
                self._log(
                    f"family_judge: anthropic client init failed ({exc!r}); "
                    f"falling back to mock"
                )
                self._client = None
                self.is_mock = True

    def banner(self) -> str:
        return "[MOCK JUDGE]" if self.is_mock else f"[LIVE JUDGE: {self.model}]"

    # -- single-candidate --------------------------------------------------

    def judge(
        self,
        candidate: str,
        excerpts: list[dict],
        user_email: str | None,
        user_display_name: str | None = None,
        user_surname: str | None = None,
        contact_evidence: ContactEvidence | None = None,
        contacts_population: int | None = None,
    ) -> FamilyVerdict:
        """Judge ONE candidate. Returns a FamilyVerdict.

        `user_surname` (Pass 9A) is the user's last-name token. When the
        candidate's surname matches it, both the mock and the live backend
        treat the candidate as a strong family signal.

        `contact_evidence` (Pass 12A), when provided, is rendered as a
        GOOGLE_CONTACTS_SIGNAL block ABOVE the email evidence. The system
        prompt instructs the model that strong Contacts signals are
        near-decisive (especially Family-group membership).

        `contacts_population` (Pass 14A), when provided, tells the judge
        how many family-shaped contacts the user has in their address book.
        See the system prompt's CONTACTS INTERPRETATION section — most
        importantly, `contacts_population=0` informs the model that absence
        of a Contacts signal is EXPECTED (the user has no curated family
        contacts at all) and must not be treated as evidence against the
        candidate. None means contacts were not consulted (legacy / fixture).

        On live-backend failure (network, parse, etc.) we fall back to the
        mock for that candidate so the materializer always gets a verdict
        per input — never None, never raises.
        """
        if self.is_mock:
            return _mock_judge_one(
                candidate,
                excerpts,
                user_surname=user_surname,
                contact_evidence=contact_evidence,
                contacts_population=contacts_population,
            )
        try:
            verdict = _live_judge_one(
                self._client,
                candidate,
                excerpts,
                user_email,
                user_display_name,
                user_surname,
                self.model,
                self._log,
                contact_evidence=contact_evidence,
                contacts_population=contacts_population,
            )
        except Exception as exc:
            self._log(
                f"family_judge: live call failed for {candidate!r} ({exc!r}); "
                f"falling back to mock"
            )
            return _mock_judge_one(
                candidate,
                excerpts,
                user_surname=user_surname,
                contact_evidence=contact_evidence,
                contacts_population=contacts_population,
            )
        if verdict is None:
            self._log(
                f"family_judge: live parse failed for {candidate!r}; "
                f"falling back to mock"
            )
            return _mock_judge_one(
                candidate,
                excerpts,
                user_surname=user_surname,
                contact_evidence=contact_evidence,
                contacts_population=contacts_population,
            )
        return verdict

    # -- batch -------------------------------------------------------------

    def judge_batch(
        self,
        candidates: list[tuple[str, list[dict]]],
        user_email: str | None = None,
        user_display_name: str | None = None,
        user_surname: str | None = None,
        contact_evidence_by_candidate: dict[str, ContactEvidence] | None = None,
        contacts_population: int | None = None,
    ) -> list[FamilyVerdict]:
        """Judge every (candidate, excerpts) pair. One sequential call per
        candidate — prompt caching on the system block covers most of the
        per-call token cost. Output order matches input order.

        `user_surname` (Pass 9A) is forwarded to every per-candidate call so
        the live prompt and the mock backend can weight surname matches as
        a strong family signal.

        `contact_evidence_by_candidate` (Pass 12A): optional dict keyed by
        the candidate canonical name. When a candidate has an entry, the
        corresponding ContactEvidence is rendered into its judge prompt.
        Candidates without an entry skip the GOOGLE_CONTACTS_SIGNAL block
        and the judge runs on email evidence alone (legacy path).

        `contacts_population` (Pass 14A): how many family-shaped contacts
        the user has in their address book. Forwarded unchanged to every
        per-candidate call so the model can interpret an absent
        GOOGLE_CONTACTS_SIGNAL correctly. See `judge()` docstring and the
        system prompt's CONTACTS INTERPRETATION section.

        A single-API-call-with-array-output design was considered and
        rejected: sequential calls give clean per-candidate parse-failure
        isolation (one bad JSON doesn't tank the whole batch) at the cost
        of one extra network round-trip per candidate. Cache hits dominate
        token cost either way.
        """
        contact_evidence_by_candidate = contact_evidence_by_candidate or {}
        verdicts: list[FamilyVerdict] = []
        for canonical, excerpts in candidates:
            verdicts.append(
                self.judge(
                    canonical,
                    excerpts,
                    user_email,
                    user_display_name,
                    user_surname=user_surname,
                    contact_evidence=contact_evidence_by_candidate.get(canonical),
                    contacts_population=contacts_population,
                )
            )
        return verdicts
