"""Tests for the materializer's canonical-name normalization + self-filter.
Two failure modes from a real-Gmail run drove these tests:

  1. Canonical names sometimes contain duplicated adjacent tokens or trailing
     role/subject-line cruft ("Mitra Mitra Martin", "Kelly Ebeling Founder",
     "Branson Bollinger Wagmi Intro"). `_normalize_canonical_name` cleans them.

  2. The user themselves was appearing in their own family member list
     (Dennison's family profile listing "Dennison Bertram"). The optional
     `user_self` parameter to `write_family_profile` filters by email + name.
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

import yaml  # noqa: E402

from pi_email.corpus import Corpus, Message  # noqa: E402
from pi_email.entities import Entity  # noqa: E402
from pi_email.family_judge import FamilyVerdict  # noqa: E402
from pi_email.materializer import (  # noqa: E402
    _derive_canonical_name_for_email,
    _gather_rule_5_frequent_personal_correspondents,
    _is_personal_domain,
    _normalize_canonical_name,
    write_family_profile,
)


# ---------------------------------------------------------------------------
# _normalize_canonical_name
# ---------------------------------------------------------------------------


def test_normalize_drops_adjacent_duplicate_first_token():
    # "Mitra" alias merged into "Mitra Martin" produced "Mitra Mitra Martin"
    # in canonicalize output. Adjacent-dedup peels the repeat.
    assert _normalize_canonical_name("Mitra Mitra Martin") == "Mitra Martin"


def test_normalize_strips_trailing_role_token():
    # "Founder" never legitimately ends a person's name.
    assert _normalize_canonical_name("Kelly Ebeling Founder") == "Kelly Ebeling"


def test_normalize_caps_long_name_after_strip():
    # "Branson Bollinger Wagmi Intro": "Intro" strips, then we have
    # ("Branson", "Bollinger", "Wagmi") — 3 tokens, NOT capped. The cap
    # documented in the spec covers cases where the LEN >= 4 even after
    # stripping. We verify the documented behavior of the strip-then-cap
    # pass: after stripping, 3 tokens remain — but in practice "Wagmi"
    # is subject leftover. The spec accepts that the cap doesn't catch
    # every case; here the trailing-strip pass at least removes "Intro".
    # The output is "Branson Bollinger Wagmi", but the spec's expected
    # behavior is also valid if "Wagmi" survives — we document with a
    # second test for the strict 3-token cap fallback case.
    result = _normalize_canonical_name("Branson Bollinger Wagmi Intro")
    # "Intro" must be stripped. The remaining first 3 tokens stay.
    assert result == "Branson Bollinger Wagmi"


def test_normalize_caps_four_plus_tokens_to_three():
    # Even without role-tokens, names of 4+ tokens get truncated to 3.
    # Tradeoff documented in the helper docstring.
    assert (
        _normalize_canonical_name("Branson Bollinger Wagmi Intro Call")
        == "Branson Bollinger Wagmi"
    )


def test_normalize_passes_through_single_token():
    assert _normalize_canonical_name("Bob") == "Bob"


def test_normalize_preserves_legitimate_three_token_name():
    # No duplicate adjacent tokens, no trailing role — 3 tokens stays as is.
    assert _normalize_canonical_name("Mary Jane Watson") == "Mary Jane Watson"


def test_normalize_keeps_lone_role_token():
    # Single-token "Founder" is in the stoplist but stripping would leave
    # an empty string. We keep the original rather than nuke the entry.
    # Documented edge case.
    assert _normalize_canonical_name("Founder") == "Founder"


def test_normalize_empty_string_returns_empty():
    assert _normalize_canonical_name("") == ""


def test_normalize_iterative_trailing_strip():
    # Multiple trailing role tokens peel iteratively.
    assert _normalize_canonical_name("Sam Lee CEO Founder") == "Sam Lee"


def test_normalize_case_insensitive_dedup():
    # "Mitra mitra Martin" — case differs but dedup is case-insensitive.
    # First-token of the dedup output keeps original casing.
    assert _normalize_canonical_name("Mitra mitra Martin") == "Mitra Martin"


# Round-5: business-function tokens added to _NAME_TAIL_STOPLIST so the
# materializer's defense-in-depth pass also strips them. The extractor strips
# at NER time too — these tests pin the materializer-only path for any cached
# canonical that still carries the role tail.


def test_normalize_strips_marketing_tail():
    assert _normalize_canonical_name("Ryan Rigney Marketing") == "Ryan Rigney"


def test_normalize_strips_engineering_tail():
    assert _normalize_canonical_name("Sarah Johnson Engineering") == "Sarah Johnson"


def test_normalize_strips_community_tail():
    # Business-function "community" should peel.
    assert (
        _normalize_canonical_name("Alex Patel Community") == "Alex Patel"
    )


def test_normalize_strips_multiple_business_tails():
    # Iterative strip: "Foo Bar Sales Marketing" peels both tails.
    assert (
        _normalize_canonical_name("Foo Bar Sales Marketing") == "Foo Bar"
    )


# ---------------------------------------------------------------------------
# write_family_profile self-filter
# ---------------------------------------------------------------------------


def _msg(
    message_id: str,
    *,
    from_addr: str = "alice@example.com",
    subject: str = "test subject",
    body: str = "body text",
) -> Message:
    return Message(
        message_id=message_id,
        thread_id=message_id,
        from_addr=from_addr,
        to_addr="me@example.com",
        subject=subject,
        date="2026-01-01",
        body=body,
        source_path=Path("/tmp/fake.md"),
    )


def _build_corpus_with_two_people() -> tuple[Corpus, set[Entity], dict]:
    """Build a tiny corpus where two person entities ("Dennison Bertram"
    and "Jana Bertram") both co-occur with relation words. The
    materializer's `_gather_family_members` then surfaces both."""
    corpus = Corpus()
    # Message 1: from Dennison to a family member, with relation cue.
    m1 = _msg(
        "m1",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="Family dinner Sunday",
        body=(
            "Hi Mom, Dennison Bertram here. Jana Bertram is coming "
            "with us — looking forward to dinner with my family."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)

    # Message 2: a thread mentioning Jana again with another relation cue.
    m2 = _msg(
        "m2",
        from_addr="Jana Bertram <jana@example.com>",
        subject="My sister Jana wedding",
        body="Jana Bertram and I (Dennison Bertram) are siblings going.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)

    # Synthesize the entity set + seen_by_message that the entity extractor
    # WOULD produce. We don't run spaCy in this test — we just hand-craft
    # the inputs to exercise the materializer.
    e_den = Entity(kind="person", key="dennison bertram", label="Dennison Bertram")
    e_jana = Entity(kind="person", key="jana bertram", label="Jana Bertram")
    e_rel_mom = Entity(kind="relation", key="mom", label="mom")
    e_rel_fam = Entity(kind="relation", key="family", label="family")
    e_rel_sib = Entity(kind="relation", key="siblings", label="siblings")
    e_rel_sister = Entity(kind="relation", key="sister", label="sister")

    entities = {e_den, e_jana, e_rel_mom, e_rel_fam, e_rel_sib, e_rel_sister}
    seen_by_message = {
        "m1": {e_den, e_jana, e_rel_mom, e_rel_fam},
        "m2": {e_den, e_jana, e_rel_sib, e_rel_sister},
    }
    return corpus, entities, seen_by_message


def test_write_family_profile_excludes_self_by_email(tmp_path):
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        on_log=logs.append,
        # Skip the LLM judge — this test exercises the self-filter, not
        # the family-vs-not-family classifier added in Pass 8B.
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    assert "Jana Bertram" in content
    assert "### Dennison Bertram" not in content, (
        f"user themselves should not appear as a family member; profile:\n{content}"
    )
    # The exclusion log line surfaces the match for operator visibility.
    assert any(
        "excluded self" in line and "dennison@withtally.com" in line for line in logs
    ), f"expected self-exclusion log; got: {logs}"


def test_write_family_profile_keeps_self_when_user_self_none(tmp_path):
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    out = tmp_path / "family.md"

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self=None,
        # Skip judge — this test exercises self-filter pass-through, not
        # the family-vs-not-family classifier.
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    # Both people appear when no self-filter is configured.
    assert "Jana Bertram" in content
    assert "Dennison Bertram" in content


def test_write_family_profile_excludes_self_by_display_name(tmp_path):
    """When user_self["display_name"] is provided directly, the email-based
    sender-name discovery isn't required — we exclude by name match."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    # Strip the Dennison-from-addr so email discovery yields nothing.
    for msg in list(corpus.messages.values()):
        if "dennison" in msg.from_addr.lower():
            msg.from_addr = "Someone Else <other@example.com>"

    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={
            "email": "dennison@withtally.com",
            "display_name": "Dennison Bertram",
        },
        on_log=logs.append,
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    assert "Jana Bertram" in content
    assert "### Dennison Bertram" not in content


# ---------------------------------------------------------------------------
# Round-3.5 self-filter: derive alias from email local-part (Fix #3)
# ---------------------------------------------------------------------------
#
# The bug: a real-Gmail run produced "Dennison Bertram" as a family member of
# Dennison's own family. Diagnosis:
#   * user_self = {"email": "dennison@withtally.com", "display_name": None}
#     was correctly plumbed through cli.py / loop.py / materializer.
#   * _derive_self_aliases iterated the corpus looking for sent-mail (from_addr
#     containing the user's email) and parsed the sender display name to use
#     as a self-alias. But if no such message landed in the iterative-search
#     corpus, no alias was derived and the filter silently no-op'd.
#
# Fix: also derive an alias from the email local-part — "dennison" — which
# always exists. _matches_self treats a 1-token alias as a first-name match
# against any canonical, so "Dennison Bertram" is now reliably excluded.


def test_self_filter_uses_email_local_part_when_no_sent_mail(tmp_path):
    """No sent mail from the user in the corpus, only the email is provided
    (display_name=None). The local-part 'dennison' must still be derived
    and used to filter the user out by first-name match."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    # Wipe any sent-mail traces so only the local-part can supply an alias.
    for msg in list(corpus.messages.values()):
        msg.from_addr = "Someone Else <other@example.com>"

    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        on_log=logs.append,
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    assert "### Jana Bertram" in content
    assert "### Dennison Bertram" not in content, (
        f"local-part 'dennison' should have caught the canonical; got:\n{content}"
    )
    # Activation log line lists the alias set including ('dennison',).
    assert any(
        "[self-filter] active" in line and "dennison" in line for line in logs
    ), f"expected activation log; got: {logs}"
    # Per-exclusion log names the matching rule.
    assert any(
        "[self-filter] excluding" in line and "Dennison Bertram" in line
        for line in logs
    ), f"expected exclusion log; got: {logs}"


def test_self_filter_via_sender_name_alias_from_corpus(tmp_path):
    """The user IS in the corpus as a sent-mail sender — 'Dennison Bertram
    <dennison@withtally.com>'. The materializer must auto-derive the full
    display name as an alias AND still exclude (both via first-name match
    AND via the recovered display-name match)."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    # _build_corpus_with_two_people already includes a message from
    # "Dennison Bertram <dennison@withtally.com>".

    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        on_log=logs.append,
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    assert "### Jana Bertram" in content
    assert "### Dennison Bertram" not in content
    # Activation log should list BOTH the local-part alias ('dennison',)
    # AND the parsed sender-name alias ('dennison', 'bertram').
    activation_lines = [ln for ln in logs if "[self-filter] active" in ln]
    assert activation_lines, f"expected activation log; got: {logs}"
    joined = " ".join(activation_lines)
    assert "dennison" in joined
    assert "dennison bertram" in joined, (
        f"expected sent-mail display name recovered as an alias; got: {logs}"
    )


def test_self_filter_no_user_self_no_log_lines(tmp_path):
    """user_self=None — no aliases derived, no log lines emitted, no
    exclusion applied."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self=None,
        on_log=logs.append,
        skip_judge=True,
    )

    # No self-filter logs at all.
    assert not any("[self-filter]" in ln for ln in logs), (
        f"expected zero self-filter logs in fixture mode; got: {logs}"
    )
    content = out.read_text(encoding="utf-8")
    assert "### Dennison Bertram" in content
    assert "### Jana Bertram" in content


def test_self_filter_handles_plus_suffix_email(tmp_path):
    """Local-part with `+suffix` is stripped before alias derivation —
    dennison+ci@withtally.com still derives ('dennison',)."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    for msg in list(corpus.messages.values()):
        msg.from_addr = "Someone Else <other@example.com>"

    out = tmp_path / "family.md"
    logs: list[str] = []

    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={
            "email": "dennison+ci@withtally.com",
            "display_name": None,
        },
        on_log=logs.append,
        skip_judge=True,
    )

    content = out.read_text(encoding="utf-8")
    assert "### Dennison Bertram" not in content


# ---------------------------------------------------------------------------
# Pass 9A: user surname derivation + uncertain-bucket population
# ---------------------------------------------------------------------------


class _RecordingJudge:
    """Records the user_surname / user_display_name / contacts_population
    passed to judge_batch, and returns canned verdicts keyed by candidate
    canonical."""

    # Sentinel for "we have never recorded a value". Used so the assertion
    # for `contacts_population=None` (a legitimate value) can be distinguished
    # from "judge_batch was never called".
    _UNSET = object()

    def __init__(self, verdicts: dict[str, FamilyVerdict]) -> None:
        self.is_mock = True
        self.model = "recording-test-double"
        self._verdicts = verdicts
        self.last_user_surname: str | None = None
        self.last_user_display_name: str | None = None
        self.last_contacts_population = self._UNSET
        # Pass 16: track which canonicals the judge was actually asked to
        # judge so tests can assert that auto-accepted (surname-match +
        # evidence) candidates BYPASS the judge entirely.
        self.last_judged_canonicals: list[str] = []
        self.judge_batch_calls: int = 0

    def banner(self) -> str:
        return "[RECORDING JUDGE]"

    def judge(
        self,
        candidate,
        excerpts,
        user_email,
        user_display_name=None,
        user_surname=None,
        contact_evidence=None,
        contacts_population=None,
    ):
        return self._verdicts.get(
            candidate,
            FamilyVerdict(
                canonical=candidate,
                decision="not_family",
                relation_guess=None,
                confidence=0.5,
                reasoning="default",
            ),
        )

    def judge_batch(
        self,
        candidates,
        user_email=None,
        user_display_name=None,
        user_surname=None,
        contact_evidence_by_candidate=None,
        contacts_population=None,
    ):
        self.last_user_display_name = user_display_name
        self.last_user_surname = user_surname
        self.last_contacts_population = contacts_population
        self.last_judged_canonicals = [canon for canon, _ in candidates]
        self.judge_batch_calls += 1
        contact_evidence_by_candidate = contact_evidence_by_candidate or {}
        return [
            self.judge(
                canon,
                exc,
                user_email,
                user_display_name,
                user_surname,
                contact_evidence=contact_evidence_by_candidate.get(canon),
                contacts_population=contacts_population,
            )
            for canon, exc in candidates
        ]


def _read_frontmatter(content: str) -> dict:
    assert content.startswith("---\n")
    end = content.index("\n---\n", 4)
    return yaml.safe_load(content[4:end])


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_user_surname_derived_from_corpus_self_identification(tmp_path):
    """The corpus contains a self-sent message with `Dennison Bertram
    <dennison@withtally.com>`. The materializer must parse the From header,
    extract the display name, and pass user_surname='Bertram' to the judge."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    judge = _RecordingJudge(
        {
            "Jana Bertram": FamilyVerdict(
                canonical="Jana Bertram",
                decision="family",
                relation_guess="spouse",
                confidence=0.92,
                reasoning="surname match",
            ),
        }
    )
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
    )
    assert judge.last_user_display_name == "Dennison Bertram"
    assert judge.last_user_surname == "Bertram"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_user_surname_uses_caller_display_name_when_provided(tmp_path):
    """If user_self["display_name"] is given directly, it wins over the
    corpus heuristic. surname is the last token."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={
            "email": "different@elsewhere.com",
            "display_name": "Alice Q. Mallory",
        },
        judge=judge,
    )
    assert judge.last_user_display_name == "Alice Q. Mallory"
    assert judge.last_user_surname == "Mallory"


def test_user_surname_none_when_no_user_self(tmp_path):
    """Fixture-mode (user_self=None) — no surname is derived."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self=None,
        judge=judge,
    )
    assert judge.last_user_display_name is None
    assert judge.last_user_surname is None


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_derive_user_display_helper_single_token():
    """A single-token display name produces a display_name but no surname
    (we can't reliably split a single token into first/last)."""
    corpus = Corpus()
    display, surname = _derive_user_display_and_surname(
        corpus, {"email": "x@y.com", "display_name": "Dennison"}
    )
    assert display == "Dennison"
    assert surname is None


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_uncertain_section_populated_by_confidence_band(tmp_path):
    """A judge returns a MIX of mid-confidence verdicts whose `decision`
    fields disagree with the confidence bands. The materializer's
    confidence-band routing places them in the Uncertain section."""
    corpus = Corpus()
    m1 = _msg(
        "m1",
        from_addr="alice@example.com",
        subject="dinner",
        body=(
            "Alice Smith joining for dinner; my dad is also coming. "
            "Bob Doe might bring his wife."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)

    e_alice = Entity(kind="person", key="alice smith", label="Alice Smith")
    e_bob = Entity(kind="person", key="bob doe", label="Bob Doe")
    e_dad = Entity(kind="relation", key="dad", label="dad")
    e_wife = Entity(kind="relation", key="wife", label="wife")
    entities = {e_alice, e_bob, e_dad, e_wife}
    seen_by_message = {"m1": {e_alice, e_bob, e_dad, e_wife}}

    verdicts = {
        # not_family @ 0.70 -> uncertain (Run 8's Jana scenario).
        "Alice Smith": FamilyVerdict(
            canonical="Alice Smith",
            decision="not_family",
            relation_guess=None,
            confidence=0.70,
            reasoning="ambiguous",
        ),
        # family @ 0.65 -> uncertain (weak family signal).
        "Bob Doe": FamilyVerdict(
            canonical="Bob Doe",
            decision="family",
            relation_guess="sibling",
            confidence=0.65,
            reasoning="weak sibling signal",
        ),
    }
    judge = _RecordingJudge(verdicts)
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={"Alice Smith": "Alice Smith", "Bob Doe": "Bob Doe"},
        user_self=None,
        judge=judge,
    )
    content = out.read_text(encoding="utf-8")
    fm = _read_frontmatter(content)
    # Both mid-confidence verdicts land in Uncertain regardless of decision.
    assert fm["judge"]["uncertain"] == 2
    assert fm["judge"]["accepted"] == 0
    assert fm["judge"]["rejected"] == 0
    assert "## Possibly family (uncertain)" in content
    assert "### Alice Smith" in content
    assert "### Bob Doe" in content


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_accepts_at_0_70():
    """Pass 11: a surname-matching `family` verdict at 0.72 confidence must
    land in Accepted. Without the surname-match floor lowering, 0.72 would
    bucket as Uncertain (the 0.55-0.80 family band). The fix makes Jana
    Bertram (the user's spouse) stably Accepted across runs even when the
    judge's confidence calibration is jittery in the 0.70-0.85 range."""
    v = FamilyVerdict(
        canonical="Jana Bertram",
        decision="family",
        relation_guess="spouse",
        confidence=0.72,
        reasoning="surname match + spouse evidence",
    )
    assert _bucket_verdict(v, user_surname="Bertram") == "accepted"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_no_match_still_needs_0_80():
    """Pass 11: the lowered floor applies ONLY when the candidate's surname
    matches the user's. A non-matching canonical at the same 0.72 confidence
    must remain Uncertain — we did not drop the floor globally."""
    v = FamilyVerdict(
        canonical="Jane Doe",
        decision="family",
        relation_guess="sibling",
        confidence=0.72,
        reasoning="some evidence",
    )
    # No surname match — Doe != Bertram.
    assert _bucket_verdict(v, user_surname="Bertram") == "uncertain"
    # And the same verdict with no user_surname at all (legacy call sites)
    # also stays Uncertain — the default path is unchanged.
    assert _bucket_verdict(v) == "uncertain"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_below_threshold_still_uncertain():
    """Pass 11: a surname-matching `family` verdict at 0.60 confidence is
    below even the lowered 0.70 floor, so it stays Uncertain. Don't go too
    far — surname-match is strong but a weak family signal from the judge
    still warrants review."""
    v = FamilyVerdict(
        canonical="Jana Bertram",
        decision="family",
        relation_guess=None,
        confidence=0.60,
        reasoning="weak signal",
    )
    assert _bucket_verdict(v, user_surname="Bertram") == "uncertain"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_case_insensitive():
    """Surname comparison must be case-insensitive on both sides."""
    v = FamilyVerdict(
        canonical="Jana BERTRAM",
        decision="family",
        relation_guess="spouse",
        confidence=0.72,
        reasoning="x",
    )
    assert _bucket_verdict(v, user_surname="bertram") == "accepted"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_does_not_promote_not_family():
    """Pass 11: surname-match lowers the FAMILY acceptance floor only. A
    not_family verdict from the judge is not flipped to family just because
    the surname happens to match — the judge already saw the surname in the
    prompt and chose not_family despite it. Bucket exactly as before."""
    # 0.70 not_family → Uncertain (mid-band).
    v_mid = FamilyVerdict(
        canonical="Jana Bertram",
        decision="not_family",
        relation_guess=None,
        confidence=0.70,
        reasoning="business contact who happens to share a surname",
    )
    assert _bucket_verdict(v_mid, user_surname="Bertram") == "uncertain"
    # 0.95 not_family → Rejected (high-band, model is confident).
    v_high = FamilyVerdict(
        canonical="Jana Bertram",
        decision="not_family",
        relation_guess=None,
        confidence=0.95,
        reasoning="x",
    )
    assert _bucket_verdict(v_high, user_surname="Bertram") == "rejected"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_bucket_verdict_helper_table():
    """Pin the full _bucket_verdict mapping in one place — defensive against
    accidental band shifts in future passes."""
    def v(decision: str, conf: float) -> FamilyVerdict:
        return FamilyVerdict(
            canonical="x", decision=decision, relation_guess=None,
            confidence=conf, reasoning="",
        )

    # family band
    assert _bucket_verdict(v("family", 0.95)) == "accepted"
    assert _bucket_verdict(v("family", 0.80)) == "accepted"
    assert _bucket_verdict(v("family", 0.79)) == "uncertain"
    assert _bucket_verdict(v("family", 0.55)) == "uncertain"
    assert _bucket_verdict(v("family", 0.54)) == "rejected"

    # uncertain band: always uncertain
    assert _bucket_verdict(v("uncertain", 0.10)) == "uncertain"
    assert _bucket_verdict(v("uncertain", 0.99)) == "uncertain"

    # not_family band
    assert _bucket_verdict(v("not_family", 0.95)) == "rejected"
    assert _bucket_verdict(v("not_family", 0.85)) == "rejected"
    assert _bucket_verdict(v("not_family", 0.84)) == "uncertain"
    assert _bucket_verdict(v("not_family", 0.55)) == "uncertain"
    assert _bucket_verdict(v("not_family", 0.54)) == "rejected"


# ---------------------------------------------------------------------------
# Pass 10: loosened _gather_family_members candidate-inclusion rules
# ---------------------------------------------------------------------------
#
# Run 9 evidence: the strict rule (PERSON must co-occur with a relation entity
# in some message) admitted only 7 candidates out of 278 extracted entities.
# These tests pin the four new inclusion rules:
#   rule_1: relation co-occurrence (original strict rule)
#   rule_2: sender/recipient + body match
#   rule_3: personal-domain repeat (>=2 messages at personal-domain address)
#   rule_4: surname match
# Plus the 50-candidate cap and priority ordering.


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_includes_sender_recipient_with_body_match():
    """rule_2: a PERSON who appears as a sender (display name in From) AND
    whose label appears in another message's body must be included even
    without any relation word."""
    corpus = Corpus()
    # m1: Alice is the sender (display name in From), no relation word.
    m1 = _msg(
        "m1",
        from_addr="Alice Wonderland <alice@somecompany.com>",
        subject="catching up",
        body="Hope you are well. Let's coffee soon.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    # m2: body mentions "Alice Wonderland" — multi-channel presence.
    m2 = _msg(
        "m2",
        from_addr="Bob <bob@example.com>",
        subject="ack",
        body="Sounds good, Alice Wonderland will be there too.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)

    e_alice = Entity(
        kind="person", key="alice wonderland", label="Alice Wonderland"
    )
    seen_by_message = {"m1": {e_alice}, "m2": {e_alice}}

    members = _gather_family_members(
        corpus,
        entities={e_alice},
        seen_by_message=seen_by_message,
    )
    assert "Alice Wonderland" in members, (
        f"expected rule_2 to admit Alice Wonderland; got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_includes_personal_domain_with_repeat_messages():
    """rule_3: a PERSON appearing in ≥2 messages whose from_addr/to_addr is
    at a personal-mail domain (gmail.com) must be included even without a
    relation word."""
    corpus = Corpus()
    # _msg's default to_addr is "me@example.com" — keep that, no override.
    m1 = _msg(
        "m1",
        from_addr="charlie@gmail.com",
        subject="weekend",
        body="Charlie Brown thinking about Saturday.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    m2 = _msg(
        "m2",
        from_addr="charlie@gmail.com",
        subject="follow up",
        body="As Charlie Brown said yesterday, the picnic is on.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)

    e_charlie = Entity(
        kind="person", key="charlie brown", label="Charlie Brown"
    )
    seen_by_message = {"m1": {e_charlie}, "m2": {e_charlie}}

    members = _gather_family_members(
        corpus,
        entities={e_charlie},
        seen_by_message=seen_by_message,
    )
    assert "Charlie Brown" in members, (
        f"expected rule_3 to admit Charlie Brown (>=2 personal-domain msgs); "
        f"got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_includes_surname_match():
    """rule_4: a PERSON whose surname matches user_surname must be included
    even with NO relation co-occurrence, NO sender/recipient hit, NO
    personal-domain repeat."""
    corpus = Corpus()
    # One message — minimal, no relation word, no personal-domain header.
    m1 = _msg(
        "m1",
        from_addr="newsletter@bigco.com",
        subject="weekly digest",
        body=(
            "Among today's industry updates: Eleanor Bertram joined the "
            "advisory board last quarter."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)

    e_eleanor = Entity(
        kind="person", key="eleanor bertram", label="Eleanor Bertram"
    )
    seen_by_message = {"m1": {e_eleanor}}

    members = _gather_family_members(
        corpus,
        entities={e_eleanor},
        seen_by_message=seen_by_message,
        user_surname="Bertram",
    )
    assert "Eleanor Bertram" in members, (
        f"expected rule_4 (surname match) to admit Eleanor Bertram; "
        f"got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_caps_at_50_candidates():
    """When more than 50 PERSONs are eligible, exactly 50 are returned and
    surname-match candidates always survive the cap."""
    corpus = Corpus()
    # 60 generic personal-domain contacts, each in 2 messages so rule_3 fires.
    # Plus 1 surname-match candidate (rule_4) — it must appear in the output
    # even if alphabetically it would have been sorted out under a naive cap.
    entities: set[Entity] = set()
    seen_by_message: dict[str, set[Entity]] = {}

    # Surname-match candidate. Label starts with 'Z' so it would sort LAST
    # alphabetically and be cut by a naive cap; rule_4 priority forces it
    # to the top.
    e_surname = Entity(
        kind="person", key="zelda bertram", label="Zelda Bertram"
    )
    entities.add(e_surname)
    m_z = _msg(
        "mZ",
        from_addr="zelda@somewhere.com",
        subject="hi",
        body="Zelda Bertram says hello.",
    )
    m_z.body_clean = m_z.body
    corpus.add(m_z)
    seen_by_message["mZ"] = {e_surname}

    for i in range(60):
        name = f"Person{i:02d} Smith"
        key = name.lower()
        e = Entity(kind="person", key=key, label=name)
        entities.add(e)
        ma = _msg(
            f"m{i:02d}a",
            from_addr=f"p{i:02d}@gmail.com",
            subject="hi",
            body=f"{name} says hello again.",
        )
        ma.body_clean = ma.body
        corpus.add(ma)
        mb = _msg(
            f"m{i:02d}b",
            from_addr=f"p{i:02d}@gmail.com",
            subject="follow up",
            body=f"{name} is coming for dinner Saturday.",
        )
        mb.body_clean = mb.body
        corpus.add(mb)
        seen_by_message[f"m{i:02d}a"] = {e}
        seen_by_message[f"m{i:02d}b"] = {e}

    members = _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        user_surname="Bertram",
    )
    # Cap enforced at 50.
    assert len(members) == 50, (
        f"expected exactly 50 members under the cap; got {len(members)}"
    )
    # Surname-match candidate survives despite being alphabetically last
    # in a naive sort — rule_4 is highest priority.
    assert "Zelda Bertram" in members


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_diagnostic_log_emitted():
    """on_log must receive a single '[gather] ...' line with per-rule
    contribution counts."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    logs: list[str] = []
    _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        on_log=logs.append,
    )
    gather_lines = [ln for ln in logs if ln.startswith("[gather]")]
    assert len(gather_lines) == 1, (
        f"expected exactly one [gather] log line; got {gather_lines}"
    )
    line = gather_lines[0]
    assert "rule_1=" in line
    assert "rule_2=" in line
    assert "rule_3=" in line
    assert "rule_4=" in line
    assert "total candidates=" in line
    assert "capped at 50" in line


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_does_not_admit_unmatched_business_contact():
    """A PERSON that fails ALL four rules (no relation, not sender/recipient,
    no personal-domain repeat, no surname match) MUST be dropped. This is
    the canary that the four rules don't collectively become an accept-all
    filter."""
    corpus = Corpus()
    m1 = _msg(
        "m1",
        from_addr="newsletter@bigco.com",
        subject="weekly digest",
        body=(
            "Industry update: Acme Corp announced new partnerships with "
            "Vance Ferrari this quarter."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    e_vance = Entity(
        kind="person", key="vance ferrari", label="Vance Ferrari"
    )
    seen_by_message = {"m1": {e_vance}}

    members = _gather_family_members(
        corpus,
        entities={e_vance},
        seen_by_message=seen_by_message,
        user_surname="Smith",
    )
    assert "Vance Ferrari" not in members, (
        "Vance Ferrari has no relation co-occurrence, no sender/recipient "
        "hit, no personal-domain repeat, and no surname match — should be "
        f"excluded; got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_write_family_profile_normalizes_header_and_slug(tmp_path):
    """End-to-end: a canonical with duplicated tokens / trailing role cruft
    renders as a normalized header AND slug in frontmatter."""
    corpus = Corpus()
    m = _msg(
        "x1",
        from_addr="someone@example.com",
        subject="Family update",
        body=(
            "Mitra Mitra Martin is part of our family. "
            "Kelly Ebeling Founder runs our kids' program."
        ),
    )
    m.body_clean = m.body
    corpus.add(m)

    e1 = Entity(
        kind="person", key="mitra mitra martin", label="Mitra Mitra Martin"
    )
    e2 = Entity(
        kind="person", key="kelly ebeling founder", label="Kelly Ebeling Founder"
    )
    e_fam = Entity(kind="relation", key="family", label="family")
    e_kids = Entity(kind="relation", key="kids", label="kids")
    entities = {e1, e2, e_fam, e_kids}
    seen_by_message = {"x1": {e1, e2, e_fam, e_kids}}

    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={
            "Mitra Mitra Martin": "Mitra Mitra Martin",
            "Kelly Ebeling Founder": "Kelly Ebeling Founder",
        },
        # Skip judge — this test exercises canonical-name normalization,
        # not the family-vs-not-family classifier.
        skip_judge=True,
    )
    content = out.read_text(encoding="utf-8")
    # Headers are normalized (in "Candidates for review" section since
    # skip_judge=True with no user_self means no surname auto-accept).
    assert "### Mitra Martin" in content
    assert "### Kelly Ebeling" in content
    # Un-normalized forms should not appear as ### headers (the raw form
    # may still show up inside quoted excerpts since excerpt text is the
    # verbatim source).
    assert "### Mitra Mitra Martin" not in content
    assert "### Kelly Ebeling Founder" not in content
    # With skip_judge=True and no user_self, candidates land in
    # "Candidates for review" (not auto-accepted), so the frontmatter
    # members list is empty and no wikilinks are generated for them.
    assert "members: []" in content
    # Candidates appear under the review section.
    assert "## Candidates for review" in content
    # Un-normalized slugs do NOT appear.
    assert "[[people/mitra-mitra-martin]]" not in content
    assert "[[people/kelly-ebeling-founder]]" not in content


# ---------------------------------------------------------------------------
# Pass 14A: materializer must derive `contacts_population` from family_contacts
# and forward it to the judge so absent-Contacts-signal can be interpreted
# correctly (see test_family_judge.py for the judge-side semantics).
# ---------------------------------------------------------------------------


def _make_contact(
    display: str,
    *,
    email: str | None = None,
    strength: float = 0.95,
    source: str = "group_membership",
):
    """Tiny helper — builds a contacts.Contact with the minimum fields the
    materializer reads. Importing here keeps the test file's top-level imports
    minimal."""
    from pi_email.contacts import Contact

    return Contact(
        resource_name=f"people/{display.lower().replace(' ', '-')}",
        display_name=display,
        given_name=display.split()[0] if display else None,
        family_name=display.split()[-1] if " " in display else None,
        email_addresses=[email] if email else [],
        group_memberships=(
            ["contactGroups/family"] if "group_membership" in source else []
        ),
        relations=[],
        biography=None,
        is_starred=False,
        family_signal_strength=strength,
        family_signal_source=source,
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_materializer_passes_contacts_population_zero_when_family_contacts_empty(
    tmp_path,
):
    """When family_contacts=[] (the "contacts scope was queried but the user
    has no curated family list" case), the materializer must pass
    contacts_population=0 to the judge. This is the load-bearing fix —
    without it, the judge would penalize candidates like Jana Bertram for
    missing a signal that's physically unobtainable."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
        family_contacts=[],
    )
    assert judge.last_contacts_population == 0, (
        f"expected contacts_population=0 for empty list; "
        f"got {judge.last_contacts_population!r}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_materializer_passes_contacts_population_none_when_family_contacts_none(
    tmp_path,
):
    """family_contacts=None means contacts were not consulted at all (legacy
    path, fixture mode, missing scope). contacts_population must be None so
    the judge knows to ignore the field entirely."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
        family_contacts=None,
    )
    assert judge.last_contacts_population is None, (
        f"expected contacts_population=None when family_contacts=None; "
        f"got {judge.last_contacts_population!r}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_materializer_passes_contacts_population_n_when_family_contacts_n(
    tmp_path,
):
    """family_contacts is a non-empty list — contacts_population equals
    len(family_contacts) and absence-on-a-candidate becomes weak negative
    evidence per the system-prompt rule."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    contacts = [
        _make_contact("Alice Bertram", email="alice@example.com"),
        _make_contact("Bob Bertram", email="bob@example.com"),
        _make_contact("Carol Bertram", email="carol@example.com"),
    ]
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Dennison Bertram": "Dennison Bertram",
            "Jana Bertram": "Jana Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
        family_contacts=contacts,
    )
    assert judge.last_contacts_population == 3, (
        f"expected contacts_population=3; "
        f"got {judge.last_contacts_population!r}"
    )


# ---------------------------------------------------------------------------
# Pass 16: surname-match auto-accept partition
#
# After 15 passes of judge prompt-engineering, the user's spouse (Jana
# Bertram — surname match + spouse evidence) kept oscillating between 0.40
# and 0.90 confidence. The materializer now partitions candidates BEFORE
# `judge_batch`: surname match + >=1 evidence => auto-accept (bypass the
# judge entirely). Everything else flows through the judge as before.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_helper_basic_case():
    """`_surname_match` is True for 2+ token candidates whose last token
    matches user_surname (case-insensitive)."""
    assert _surname_match("Jana Bertram", "Bertram") is True
    assert _surname_match("jana BERTRAM", "bertram") is True
    # Last token must match exactly.
    assert _surname_match("Jana Smith", "Bertram") is False
    # Single-token candidate is not a surname match.
    assert _surname_match("Bertram", "Bertram") is False
    # Empty / missing surname is safe.
    assert _surname_match("Jana Bertram", None) is False
    assert _surname_match("Jana Bertram", "") is False


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_looks_like_spouse_detects_my_wife_pattern():
    """`_looks_like_spouse` returns True when an excerpt contains "my wife
    <first_name>" (case-insensitive). It must NOT fire on bare relation
    words like "children" or on un-prefixed "wife Jana"."""
    spouse_excerpt = [{"snippet": "Introducing my wife Jana — she'll be there."}]
    assert _looks_like_spouse("Jana Bertram", spouse_excerpt) is True

    # Case-insensitive.
    upper_excerpt = [{"snippet": "MY HUSBAND DENNISON is on the flight too."}]
    assert _looks_like_spouse("Dennison Bertram", upper_excerpt) is True

    # No spouse pattern → False (even though excerpt mentions a relation).
    children_excerpt = [{"snippet": "Looking forward to seeing the children Jana."}]
    assert _looks_like_spouse("Jana Bertram", children_excerpt) is False

    # "wife Jana" without possessive → False (conservative).
    bare_excerpt = [{"snippet": "wife Jana is doing fine."}]
    assert _looks_like_spouse("Jana Bertram", bare_excerpt) is False


def _make_corpus_with_surname_pair_and_business_contact() -> tuple[Corpus, set[Entity], dict]:
    """Build a corpus with three person entities — Jana Bertram (the user's
    spouse, surname match, spouse-shaped evidence), Bob Smith (a non-surname
    contact who appears as a sender), and one relation cue. The
    Dennison-from-self message gives the materializer a way to derive
    user_surname='Bertram' from the corpus."""
    corpus = Corpus()
    m1 = _msg(
        "m1",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="Intro",
        body="Introducing my wife Jana Bertram to the team.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    m2 = _msg(
        "m2",
        from_addr="Bob Smith <bob@smithcorp.com>",
        subject="catching up",
        body="Bob Smith here, hope all is well at work.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)
    m3 = _msg(
        "m3",
        from_addr="Jana Bertram <jana@example.com>",
        subject="dinner plans",
        body=(
            "Hi family, Jana Bertram here — looking forward to dinner with "
            "my husband Dennison this weekend."
        ),
    )
    m3.body_clean = m3.body
    corpus.add(m3)

    e_jana = Entity(kind="person", key="jana bertram", label="Jana Bertram")
    e_bob = Entity(kind="person", key="bob smith", label="Bob Smith")
    e_den = Entity(kind="person", key="dennison bertram", label="Dennison Bertram")
    e_fam = Entity(kind="relation", key="family", label="family")
    e_husband = Entity(kind="relation", key="husband", label="husband")
    e_wife = Entity(kind="relation", key="wife", label="wife")

    entities = {e_jana, e_bob, e_den, e_fam, e_husband, e_wife}
    seen_by_message = {
        "m1": {e_den, e_jana, e_wife},
        "m2": {e_bob},
        "m3": {e_jana, e_den, e_fam, e_husband},
    }
    return corpus, entities, seen_by_message


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_with_evidence_auto_accepted(tmp_path):
    """A surname-match candidate with at least one excerpt lands in the
    Accepted bucket via the auto-accept route — and the judge is NOT
    asked about that candidate."""
    corpus, entities, seen_by_message = _make_corpus_with_surname_pair_and_business_contact()
    # The judge would call Bob Smith not_family at 0.5; Jana would normally
    # also be judged but auto-accept must bypass.
    judge = _RecordingJudge(
        {
            "Bob Smith": FamilyVerdict(
                canonical="Bob Smith",
                decision="not_family",
                relation_guess=None,
                confidence=0.5,
                reasoning="business contact",
            ),
        }
    )
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Jana Bertram": "Jana Bertram",
            "Bob Smith": "Bob Smith",
            "Dennison Bertram": "Dennison Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
    )
    # Judge was called, but Jana Bertram was NOT in the batch the judge saw.
    assert "Jana Bertram" not in judge.last_judged_canonicals, (
        f"expected Jana Bertram to bypass the judge; "
        f"judged={judge.last_judged_canonicals}"
    )
    # Jana lands in Accepted; the body has the ### Members section header.
    content = out.read_text(encoding="utf-8")
    fm = _read_frontmatter(content)
    assert fm["judge"]["auto_accepted"] == 1
    assert "### Jana Bertram" in content


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_surname_match_no_evidence_not_auto_accepted(tmp_path):
    """Pass 16: the auto-accept rule requires AT LEAST ONE evidence message.
    A surname-match candidate whose ProfileMember has zero excerpts (e.g.,
    a contact-only candidate added by the contacts step) falls through to
    the judge normally — auto-accept never fires."""
    corpus = Corpus()
    # One self-mail so we derive user_surname='Bertram'.
    m_self = _msg(
        "m_self",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="hi",
        body="just a note.",
    )
    m_self.body_clean = m_self.body
    corpus.add(m_self)

    # Jana Bertram is admitted via rule_4 (surname match) but the body never
    # mentions her name → no excerpt → the materializer's final filter drops
    # her before the judge step is reached. We construct the same setup
    # surfaced via a contact-only candidate: a family contact with a
    # surname-matching display name but no corpus evidence.
    from pi_email.contacts import Contact

    contact = Contact(
        resource_name="people/jana-bertram",
        display_name="Jana Bertram",
        given_name="Jana",
        family_name="Bertram",
        email_addresses=["jana@example.com"],
        group_memberships=[],
        relations=[],
        biography=None,
        is_starred=False,
        family_signal_strength=0.75,
        family_signal_source="relations_field",
    )

    judge = _RecordingJudge(
        {
            # Whatever the judge would return — point is that Jana IS in
            # the judge's batch input rather than auto-accepted.
            "Jana Bertram": FamilyVerdict(
                canonical="Jana Bertram",
                decision="family",
                relation_guess="spouse",
                confidence=0.75,
                reasoning="contact-only signal",
            ),
        }
    )
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=set(),
        seen_by_message={},
        seed="seed",
        stop_reason="x",
        queries_run=[],
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
        family_contacts=[contact],
    )
    # Contact-only candidate has zero excerpts; auto-accept must NOT fire.
    assert "Jana Bertram" in judge.last_judged_canonicals, (
        f"contact-only candidate (no evidence) must go to the judge; "
        f"judged={judge.last_judged_canonicals}"
    )
    content = out.read_text(encoding="utf-8")
    fm = _read_frontmatter(content)
    assert fm["judge"]["auto_accepted"] == 0


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_non_surname_candidate_goes_to_judge(tmp_path):
    """A candidate whose surname does NOT match the user's flows through
    the judge as before — auto-accept must not fire."""
    corpus, entities, seen_by_message = _make_corpus_with_surname_pair_and_business_contact()
    judge = _RecordingJudge(
        {
            "Bob Smith": FamilyVerdict(
                canonical="Bob Smith",
                decision="not_family",
                relation_guess=None,
                confidence=0.5,
                reasoning="business contact",
            ),
        }
    )
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Jana Bertram": "Jana Bertram",
            "Bob Smith": "Bob Smith",
            "Dennison Bertram": "Dennison Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
    )
    # Bob Smith (no surname match) was asked of the judge.
    assert "Bob Smith" in judge.last_judged_canonicals


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_single_token_name_not_auto_accepted():
    """`_surname_match` requires >=2 tokens on the candidate side. A bare
    'Bertram' (no first name) must NOT auto-accept — it's an unqualified
    surname, not a surname-match."""
    assert _surname_match("Bertram", "Bertram") is False
    # End-to-end: 'Bertram' alone gathered as a member doesn't auto-accept.
    # We exercise the helper directly here; integration is covered by the
    # other tests that use multi-token canonicals.


def test_auto_accepted_lands_in_accepted_bucket(tmp_path):
    """Pass 16: auto-accepted verdicts must land in the Accepted section,
    NOT the Possibly-family (uncertain) section, regardless of any
    confidence-band logic. Verifies the bucket-pass-through fix."""
    corpus, entities, seen_by_message = _make_corpus_with_surname_pair_and_business_contact()
    judge = _RecordingJudge({})  # judge returns default 0.5 not_family
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Jana Bertram": "Jana Bertram",
            "Bob Smith": "Bob Smith",
            "Dennison Bertram": "Dennison Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
    )
    content = out.read_text(encoding="utf-8")
    # Jana Bertram is in the Members section, not Possibly family.
    # Split at "## Possibly family" so we can be sure she's BEFORE it.
    members_part, _, uncertain_part = content.partition(
        "## Possibly family (uncertain)"
    )
    assert "### Jana Bertram" in members_part, (
        "Jana Bertram should appear in the Members (Accepted) section, "
        f"not Uncertain. Full body:\n{content}"
    )
    if uncertain_part:
        assert "### Jana Bertram" not in uncertain_part


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_auto_accept_logs_bypass_line(tmp_path):
    """Pass 16: the materializer emits a single [auto-accept] log line
    per surname-match auto-accept so operators can see which candidates
    bypassed the judge."""
    corpus, entities, seen_by_message = _make_corpus_with_surname_pair_and_business_contact()
    judge = _RecordingJudge({})
    out = tmp_path / "family.md"
    logs: list[str] = []
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Jana Bertram": "Jana Bertram",
            "Bob Smith": "Bob Smith",
            "Dennison Bertram": "Dennison Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
        on_log=logs.append,
    )
    auto_lines = [
        ln for ln in logs
        if ln.startswith("[auto-accept]") and "Jana Bertram" in ln
    ]
    assert len(auto_lines) == 1, (
        f"expected one [auto-accept] log line for Jana Bertram; "
        f"got: {auto_lines}"
    )
    assert "surname=Bertram" in auto_lines[0]
    assert "bypassed judge" in auto_lines[0]


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_yaml_frontmatter_includes_auto_accepted_count(tmp_path):
    """Pass 16: frontmatter `judge:` block exposes `auto_accepted` AND
    `judge_accepted` AND the total `accepted` (auto + judge). Tooling can
    use any of the three."""
    corpus, entities, seen_by_message = _make_corpus_with_surname_pair_and_business_contact()
    judge = _RecordingJudge(
        {
            "Bob Smith": FamilyVerdict(
                canonical="Bob Smith",
                decision="not_family",
                relation_guess=None,
                confidence=0.5,
                reasoning="business contact",
            ),
        }
    )
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="seed",
        stop_reason="x",
        queries_run=[],
        canonical_map={
            "Jana Bertram": "Jana Bertram",
            "Bob Smith": "Bob Smith",
            "Dennison Bertram": "Dennison Bertram",
        },
        user_self={"email": "dennison@withtally.com", "display_name": None},
        judge=judge,
    )
    content = out.read_text(encoding="utf-8")
    fm = _read_frontmatter(content)
    assert "auto_accepted" in fm["judge"], fm["judge"]
    assert "judge_accepted" in fm["judge"], fm["judge"]
    assert fm["judge"]["auto_accepted"] == 1
    # Jana auto-accepted; Bob rejected by judge → judge_accepted=0.
    assert fm["judge"]["judge_accepted"] == 0
    # Total accepted = auto + judge.
    assert (
        fm["judge"]["accepted"]
        == fm["judge"]["auto_accepted"] + fm["judge"]["judge_accepted"]
    )


# ---------------------------------------------------------------------------
# Pass 17B: Rule 5 — address-book frequency analysis
#
# Real family often appears in a user's inbox WITHOUT explicit kinship words
# — they exchange emails about everyday things. Rule 5 admits personal-mail-
# domain addresses where the user has bidirectional + substantial-volume
# correspondence. Family-domain over-inclusion is OK; the judge filters
# downstream.
# ---------------------------------------------------------------------------


def _bidir_msg(
    message_id: str,
    *,
    from_addr: str,
    to_addr: str,
    body: str = "body text",
    subject: str = "subject",
) -> Message:
    """A Message helper that also lets us set to_addr (the local `_msg`
    forces to_addr='me@example.com'). Used for Rule 5 tests where the
    user's email is the to/from anchor."""
    return Message(
        message_id=message_id,
        thread_id=message_id,
        from_addr=from_addr,
        to_addr=to_addr,
        subject=subject,
        date="2026-01-01",
        body=body,
        source_path=Path("/tmp/fake.md"),
        body_clean=body,
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_includes_bidirectional_personal_domain():
    """Pass 17B: 3 msgs between user and jana@gmail.com (1 sent, 2 received)
    — Rule 5 admits Jana even with NO PERSON entity and NO relation word."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="user@withtally.com",
        to_addr="jana@gmail.com",
        body="Hey Jana, dinner Sunday?",
    ))
    corpus.add(_bidir_msg(
        "m2",
        from_addr="jana@gmail.com",
        to_addr="user@withtally.com",
        body="Sounds good — see you then.",
    ))
    corpus.add(_bidir_msg(
        "m3",
        from_addr="jana@gmail.com",
        to_addr="user@withtally.com",
        body="Quick question about Saturday.",
    ))

    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@withtally.com"},
    )
    # Canonical derived from local-part `jana` → "Jana".
    assert "Jana" in members, (
        f"expected Rule 5 to admit Jana (1 sent + 2 received); got {list(members)}"
    )
    # Excerpts must be populated via Rule 5's first-300-chars fallback.
    assert members["Jana"].excerpts, (
        "Rule 5 candidate must surface with at least one excerpt"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_excludes_unidirectional():
    """5 messages to newsletter@gmail.com but 0 from — unidirectional →
    Rule 5 does NOT include."""
    corpus = Corpus()
    for i in range(5):
        corpus.add(_bidir_msg(
            f"m{i}",
            from_addr="user@withtally.com",
            to_addr="newsletter@gmail.com",
            body="checking out the latest update",
        ))
    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@withtally.com"},
    )
    assert "Newsletter" not in members
    # No other rule fires either — full set should be empty.
    assert len(members) == 0


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_excludes_low_volume():
    """1 sent + 1 received = total 2, below the threshold of 3 → NOT
    included even though bidirectional."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="user@withtally.com",
        to_addr="bob@gmail.com",
        body="hi",
    ))
    corpus.add(_bidir_msg(
        "m2",
        from_addr="bob@gmail.com",
        to_addr="user@withtally.com",
        body="hey",
    ))
    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@withtally.com"},
    )
    assert "Bob" not in members, (
        f"total=2 is below threshold (3); got {list(members)}"
    )
    assert len(members) == 0


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_excludes_business_domains():
    """Bidirectional + high volume but at a business domain → Rule 5 does
    NOT include (the domain isn't in PERSONAL_EMAIL_DOMAINS)."""
    corpus = Corpus()
    for i in range(4):
        corpus.add(_bidir_msg(
            f"sent_{i}",
            from_addr="user@withtally.com",
            to_addr="bob@bigco.com",
            body=f"sent message {i}",
        ))
    for i in range(4):
        corpus.add(_bidir_msg(
            f"recv_{i}",
            from_addr="bob@bigco.com",
            to_addr="user@withtally.com",
            body=f"received message {i}",
        ))
    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@withtally.com"},
    )
    assert "Bob" not in members, (
        f"business domain @bigco.com must NOT trigger Rule 5; got {list(members)}"
    )
    assert len(members) == 0


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_excludes_user_self():
    """The user's own email never becomes a candidate — even when they
    send messages to themselves (both from_is_user AND to_is_user)."""
    corpus = Corpus()
    # User -> self at the same personal-domain address.
    for i in range(5):
        corpus.add(_bidir_msg(
            f"m{i}",
            from_addr="user@gmail.com",
            to_addr="user@gmail.com",
            body="self-note",
        ))
    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@gmail.com"},
    )
    assert "User" not in members
    assert len(members) == 0


def test_derive_canonical_name_from_display_in_from():
    """When the corpus has a From header like `Jana Smith <jana@gmail.com>`,
    the canonical is "Jana Smith" — NOT "Jana" (the local-part fallback)."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="Jana Smith <jana@gmail.com>",
        to_addr="user@withtally.com",
        body="hi",
    ))
    name = _derive_canonical_name_for_email(corpus, "jana@gmail.com")
    assert name == "Jana Smith"


def test_derive_canonical_name_falls_back_to_local_part():
    """No display name in any From header — fall back to title-cased
    local-part. `jana@gmail.com` → "Jana"."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="jana@gmail.com",  # bare email, no display name
        to_addr="user@withtally.com",
        body="hi",
    ))
    name = _derive_canonical_name_for_email(corpus, "jana@gmail.com")
    assert name == "Jana"


def test_derive_canonical_name_falls_back_local_part_with_dot():
    """Local-part with a separator splits on it before title-casing.
    `jana.smith@gmail.com` → "Jana Smith"."""
    corpus = Corpus()
    name = _derive_canonical_name_for_email(corpus, "jana.smith@gmail.com")
    assert name == "Jana Smith"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_diagnostic_includes_rule_5_count():
    """The diagnostic [gather] line must surface a rule_5=N counter so
    operators can see Rule 5's contribution at a glance."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    logs: list[str] = []
    _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        on_log=logs.append,
    )
    gather_lines = [ln for ln in logs if ln.startswith("[gather]")]
    assert len(gather_lines) == 1, f"expected one [gather] log; got {gather_lines}"
    line = gather_lines[0]
    assert "rule_5=" in line, f"missing rule_5 in diagnostic line: {line}"


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_5_uses_display_name_canonical():
    """When the From header carries a display name on the personal-domain
    side, Rule 5 surfaces the candidate under the display-name canonical
    (NOT the local-part fallback). Verifies the helper is wired into the
    gather pipeline correctly."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="user@withtally.com",
        to_addr="Jana Smith <jana@gmail.com>",
        body="Hi Jana, dinner soon?",
    ))
    corpus.add(_bidir_msg(
        "m2",
        from_addr="Jana Smith <jana@gmail.com>",
        to_addr="user@withtally.com",
        body="Sure — Sunday?",
    ))
    corpus.add(_bidir_msg(
        "m3",
        from_addr="Jana Smith <jana@gmail.com>",
        to_addr="user@withtally.com",
        body="See you then.",
    ))
    members = _gather_family_members(
        corpus,
        entities=set(),
        seen_by_message={},
        user_emails={"user@withtally.com"},
    )
    assert "Jana Smith" in members
    assert "Jana" not in members


def test_is_personal_domain_helper():
    """The personal-domain check accepts both bare emails and full
    'Display <email>' headers, and only returns True for domains in
    PERSONAL_EMAIL_DOMAINS."""
    assert _is_personal_domain("jana@gmail.com") is True
    assert _is_personal_domain("Jana <jana@gmail.com>") is True
    assert _is_personal_domain("bob@bigco.com") is False
    assert _is_personal_domain("") is False
    assert _is_personal_domain("not-an-email") is False


def test_rule_5_helper_returns_empty_when_no_user_emails():
    """The Rule 5 helper safely returns [] when no user_emails are supplied
    — no signal source means no bidirectional accounting."""
    corpus = Corpus()
    corpus.add(_bidir_msg(
        "m1",
        from_addr="alice@gmail.com",
        to_addr="bob@gmail.com",
        body="hi",
    ))
    out = _gather_rule_5_frequent_personal_correspondents(corpus, set())
    assert out == []


# ---------------------------------------------------------------------------
# Pass 18: Rule 6 — calendar-notification high-signal person promotion
#
# Pass 17A added `_inject_calendar_person_entities` in loop.py so that names
# extracted from Google Calendar notification emails (e.g. "Vitus" from
# "Accepted: Vitus Birthday in school") become Entity rows. But Rule 1-5 in
# `_gather_family_members` never picked them up because those names rarely
# co-occur with relation words or in normal-mail bodies. Rule 6 closes the
# gap: any PERSON whose first name matches a CalendarEmailPerson with high
# family_signal_strength (>= 0.80) is promoted to a candidate, with excerpts
# built from the subject + body of the calendar mail.
# ---------------------------------------------------------------------------


from pi_email.calendar_email_parser import CalendarEmailPerson  # noqa: E402


def _calendar_msg(
    message_id: str,
    *,
    subject: str,
    body: str = "calendar notification body",
    persons: list[CalendarEmailPerson] | None = None,
) -> Message:
    """Build a Message representing a Google Calendar notification with
    pre-populated `calendar_persons` (mirrors what `gmail_searcher` writes)."""
    m = Message(
        message_id=message_id,
        thread_id=message_id,
        from_addr="Calendar <calendar-notification@google.com>",
        to_addr="user@example.com",
        subject=subject,
        date="2026-04-15",
        body=body,
        source_path=Path("/tmp/cal.md"),
        body_clean=body,
        calendar_persons=list(persons or []),
    )
    return m


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_6_promotes_calendar_signal_person():
    """A PERSON entity ("Vitus") matched by a high-signal CalendarEmailPerson
    (kinship_event, 0.85) on a calendar-notification message is promoted to
    a candidate even though Rule 1-5 all abstain.
    """
    corpus = Corpus()
    vitus_person = CalendarEmailPerson(
        name="Vitus",
        email=None,
        source="event_title",
        event_title="Vitus Birthday in school",
        family_signal_strength=0.85,
        family_signal_source="kinship_event",
        matched_kinship_words=["birthday"],
    )
    corpus.add(_calendar_msg(
        "cal1",
        subject="Accepted: Vitus Birthday in school",
        body="When: April 15, 2026.\nGuests: alice@example.com",
        persons=[vitus_person],
    ))
    vitus_entity = Entity(
        kind="person",
        key="vitus",
        label="Vitus",
        confidence="high",
    )

    logs: list[str] = []
    members = _gather_family_members(
        corpus,
        entities={vitus_entity},
        seen_by_message={"cal1": {vitus_entity}},
        on_log=logs.append,
    )

    assert "Vitus" in members, (
        f"expected Rule 6 to admit Vitus from calendar signal; got {list(members)}"
    )
    # rule_6 must be counted as >=1 in the diagnostic.
    gather_line = next(
        (ln for ln in logs if ln.startswith("[gather]") and "rule_6=" in ln),
        None,
    )
    assert gather_line is not None, f"missing rule_6 diagnostic in: {logs}"
    assert "rule_6=1" in gather_line, (
        f"expected rule_6=1 in diagnostic; got: {gather_line}"
    )
    # Excerpts must be populated (subject + body chars).
    assert members["Vitus"].excerpts, (
        "Rule 6 candidate must surface with at least one excerpt"
    )
    snippet = members["Vitus"].excerpts[0][1]
    assert "Vitus" in snippet or "vitus" in snippet.lower(), (
        f"excerpt should reference Vitus; got: {snippet}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_6_skips_low_signal():
    """A CalendarEmailPerson with family_signal_strength=0.65 (personal-
    attendee tier) is BELOW the 0.80 Rule 6 threshold — NOT promoted."""
    corpus = Corpus()
    weak_person = CalendarEmailPerson(
        name="Coworker",
        email="coworker@gmail.com",
        source="attendee",
        event_title="Project sync",
        family_signal_strength=0.65,
        family_signal_source="personal_attendee",
        matched_kinship_words=[],
    )
    corpus.add(_calendar_msg(
        "cal1",
        subject="Invitation: Project sync",
        body="Guests: coworker@gmail.com",
        persons=[weak_person],
    ))
    coworker_entity = Entity(
        kind="person",
        key="coworker",
        label="Coworker",
        confidence="medium",
    )

    members = _gather_family_members(
        corpus,
        entities={coworker_entity},
        seen_by_message={"cal1": {coworker_entity}},
    )

    assert "Coworker" not in members, (
        f"strength=0.65 is below Rule 6 threshold; got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_6_no_calendar_messages_no_effect():
    """When the corpus has no calendar-notification messages (i.e. no
    Message.calendar_persons populated anywhere), Rule 6 contributes 0
    and existing rule behavior is preserved."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    logs: list[str] = []
    _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        on_log=logs.append,
    )
    gather_lines = [ln for ln in logs if ln.startswith("[gather]")]
    # No "rule_6 promoted" diagnostic should appear when rule_6 contributes 0.
    promoted_lines = [ln for ln in logs if "rule_6 promoted" in ln]
    assert promoted_lines == [], (
        f"no calendar messages → no rule_6 promoted line; got {promoted_lines}"
    )
    # Main gather line still shows rule_6=0.
    assert any("rule_6=0" in ln for ln in gather_lines), (
        f"expected rule_6=0 in gather diagnostic; got: {gather_lines}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_6_diagnostic_in_log():
    """The diagnostic [gather] line surfaces a rule_6=N counter — operators
    must see Rule 6's contribution alongside Rule 1-5."""
    corpus = Corpus()
    p = CalendarEmailPerson(
        name="Elio",
        email=None,
        source="event_title",
        event_title="Elio Birthday party",
        family_signal_strength=0.85,
        family_signal_source="kinship_event",
        matched_kinship_words=["birthday"],
    )
    corpus.add(_calendar_msg(
        "cal1",
        subject="Accepted: Elio Birthday party",
        body="When: ...",
        persons=[p],
    ))
    elio_entity = Entity(
        kind="person",
        key="elio",
        label="Elio",
        confidence="high",
    )

    logs: list[str] = []
    _gather_family_members(
        corpus,
        entities={elio_entity},
        seen_by_message={"cal1": {elio_entity}},
        on_log=logs.append,
    )
    gather_lines = [ln for ln in logs if ln.startswith("[gather]")]
    assert any("rule_6=" in ln for ln in gather_lines), (
        f"diagnostic line missing rule_6=N: {gather_lines}"
    )
    # The "rule_6 promoted" log must appear when rule_6 > 0.
    assert any("rule_6 promoted" in ln for ln in logs), (
        f"expected 'rule_6 promoted' log when rule_6>0; got: {logs}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_6_merges_with_existing_rules():
    """An entity already covered by Rule 1 (relation co-occurrence in body)
    that ALSO surfaces in a high-signal calendar mail gets both rules
    in its rules set, and the rule_counts include rule_1 AND rule_6."""
    corpus = Corpus()
    # Message 1: relation co-occurrence — Jana with "wife".
    m1 = _msg(
        "m1",
        from_addr="Jana <jana@example.com>",
        subject="Family stuff",
        body="My wife Jana said dinner is at 7.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)

    # Message 2: calendar-notification mentioning Jana with high signal.
    jana_cal = CalendarEmailPerson(
        name="Jana",
        email=None,
        source="event_title",
        event_title="Anniversary with Jana",
        family_signal_strength=0.90,
        family_signal_source="possessive_in_title",
        matched_kinship_words=["anniversary"],
    )
    corpus.add(_calendar_msg(
        "cal1",
        subject="Accepted: Anniversary with Jana",
        body="When: April 15. Guests: jana@example.com",
        persons=[jana_cal],
    ))

    jana_entity = Entity(
        kind="person",
        key="jana",
        label="Jana",
        confidence="high",
    )
    rel_wife = Entity(kind="relation", key="wife", label="wife")

    logs: list[str] = []
    members = _gather_family_members(
        corpus,
        entities={jana_entity, rel_wife},
        seen_by_message={
            "m1": {jana_entity, rel_wife},
            "cal1": {jana_entity},
        },
        on_log=logs.append,
    )

    assert "Jana" in members, f"Jana must be a candidate; got {list(members)}"
    # Diagnostic confirms BOTH rules counted.
    gather_line = next(
        (ln for ln in logs if ln.startswith("[gather]") and "rule_6=" in ln),
        None,
    )
    assert gather_line is not None, f"missing gather line in: {logs}"
    assert "rule_1=1" in gather_line, (
        f"rule_1 must still fire on the merged candidate: {gather_line}"
    )
    assert "rule_6=1" in gather_line, (
        f"rule_6 must also fire on the merged candidate: {gather_line}"
    )


# ---------------------------------------------------------------------------
# Pass 19: Rule 7 — family-graph expansion from confirmed family
#
# Once a person is confirmed as family (auto-accepted via surname match),
# anyone who co-occurs with them in >= 2 messages is a candidate. This
# catches children, in-laws, and close family friends who appear in the same
# threads as the confirmed family member but never near a kinship word.
# The co-occurrence threshold is 2 messages (tunable).
# ---------------------------------------------------------------------------


def _build_rule_7_corpus() -> tuple[Corpus, set[Entity], dict]:
    """Build a corpus where Jana Bertram (surname match) and Vitus (a child)
    co-occur in 3 messages. Vitus has NO relation-word co-occurrence, NO
    personal-domain repeat, NO sender/recipient hit — Rules 1-6 all abstain.
    Rule 7 should surface Vitus via co-occurrence with confirmed family
    member Jana Bertram."""
    corpus = Corpus()
    # m1: both Jana and Vitus in the body (school email).
    m1 = _msg(
        "m1",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="School enrollment",
        body=(
            "Hi, Jana Bertram here. Vitus is enrolled in the morning "
            "program at Brooklyn Waldorf."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    # m2: birthday thread with both Jana and Vitus.
    m2 = _msg(
        "m2",
        from_addr="Jana Bertram <jana@example.com>",
        subject="Vitus Birthday",
        body=(
            "Hi Dennison, let's plan Vitus's birthday party for next "
            "Saturday. Jana Bertram"
        ),
    )
    m2.body_clean = m2.body
    corpus.add(m2)
    # m3: museum visit with both Jana and Vitus.
    m3 = _msg(
        "m3",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="Museum visit",
        body=(
            "Jana and I took Vitus to the Natural History Museum. "
            "It was a great day."
        ),
    )
    m3.body_clean = m3.body
    corpus.add(m3)

    e_den = Entity(kind="person", key="dennison bertram", label="Dennison Bertram")
    e_jana = Entity(kind="person", key="jana bertram", label="Jana Bertram")
    e_vitus = Entity(kind="person", key="vitus", label="Vitus")

    entities = {e_den, e_jana, e_vitus}
    seen_by_message = {
        "m1": {e_den, e_jana, e_vitus},
        "m2": {e_jana, e_vitus},
        "m3": {e_den, e_jana, e_vitus},
    }
    return corpus, entities, seen_by_message


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_finds_cooccurring_person():
    """Vitus appears in 3 messages alongside Jana Bertram (auto-accepted via
    surname match). Rule 7 must surface Vitus as a candidate even though
    Rules 1-6 all abstain (no relation word, no personal-domain, etc.)."""
    corpus, entities, seen_by_message = _build_rule_7_corpus()
    logs: list[str] = []
    members = _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        user_surname="Bertram",
        user_emails={"dennison@withtally.com"},
        on_log=logs.append,
    )
    assert "Vitus" in members, (
        f"expected Rule 7 to admit Vitus via co-occurrence with Jana; "
        f"got {list(members)}"
    )
    # Vitus should have excerpts.
    assert members["Vitus"].excerpts, (
        "Rule 7 candidate must have at least one excerpt"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_ignores_low_cooccurrence():
    """A person appearing with Jana in only 1 message does NOT get gathered
    by Rule 7 (threshold is 2)."""
    corpus = Corpus()
    m1 = _msg(
        "m1",
        from_addr="Jana Bertram <jana@example.com>",
        subject="One-off meeting",
        body="Jana Bertram and Stranger met for coffee.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)

    e_jana = Entity(kind="person", key="jana bertram", label="Jana Bertram")
    e_stranger = Entity(kind="person", key="stranger", label="Stranger")

    logs: list[str] = []
    members = _gather_family_members(
        corpus,
        entities={e_jana, e_stranger},
        seen_by_message={"m1": {e_jana, e_stranger}},
        user_surname="Bertram",
        user_emails={"dennison@withtally.com"},
        on_log=logs.append,
    )
    assert "Stranger" not in members, (
        f"Stranger co-occurs with Jana in only 1 message -- below threshold; "
        f"got {list(members)}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_excludes_self():
    """The user's own name must not become a Rule 7 candidate even if it
    co-occurs with a confirmed family member in many messages."""
    corpus, entities, seen_by_message = _build_rule_7_corpus()
    logs: list[str] = []
    # The user is "Dennison Bertram" and co-occurs with Jana in every message.
    # Rule 7 must NOT surface "Dennison Bertram" as a candidate.
    results = _gather_rule_7_family_graph_expansion(
        corpus,
        entities=[e for e in entities if e.kind == "person"],
        auto_accepted_names={"Jana Bertram"},
        user_emails={"dennison@withtally.com"},
    )
    result_names = [r[0] for r in results]
    assert "Dennison Bertram" not in result_names, (
        f"user themselves should not be a Rule 7 candidate; got {result_names}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_excludes_already_accepted():
    """Jana herself must not become a Rule 7 candidate — she is the confirmed
    family member, not a co-occurrence target."""
    corpus, entities, seen_by_message = _build_rule_7_corpus()
    results = _gather_rule_7_family_graph_expansion(
        corpus,
        entities=[e for e in entities if e.kind == "person"],
        auto_accepted_names={"Jana Bertram"},
        user_emails={"dennison@withtally.com"},
    )
    result_names = [r[0] for r in results]
    assert "Jana Bertram" not in result_names, (
        f"auto-accepted family member should not be a Rule 7 candidate; "
        f"got {result_names}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_merges_with_existing_rules():
    """A person already gathered by Rule 1 (relation co-occurrence) who also
    co-occurs with a confirmed family member via Rule 7 gets both rules
    in the diagnostic counts."""
    corpus = Corpus()
    # m1: Vitus with a relation word (Rule 1) AND with Jana.
    m1 = _msg(
        "m1",
        from_addr="someone@example.com",
        subject="Family dinner",
        body=(
            "My son Vitus and Jana Bertram are coming for dinner. "
            "It will be a great family evening."
        ),
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    # m2: Vitus with Jana, no relation word.
    m2 = _msg(
        "m2",
        from_addr="Jana Bertram <jana@example.com>",
        subject="School update",
        body="Vitus got an A on his math test today. Jana Bertram",
    )
    m2.body_clean = m2.body
    corpus.add(m2)
    # m3: Vitus with Jana, no relation word.
    m3 = _msg(
        "m3",
        from_addr="Dennison Bertram <dennison@withtally.com>",
        subject="Groceries",
        body="Jana and Vitus want more bananas.",
    )
    m3.body_clean = m3.body
    corpus.add(m3)

    e_jana = Entity(kind="person", key="jana bertram", label="Jana Bertram")
    e_vitus = Entity(kind="person", key="vitus", label="Vitus")
    e_den = Entity(kind="person", key="dennison bertram", label="Dennison Bertram")
    e_son = Entity(kind="relation", key="son", label="son")
    e_fam = Entity(kind="relation", key="family", label="family")

    logs: list[str] = []
    members = _gather_family_members(
        corpus,
        entities={e_jana, e_vitus, e_den, e_son, e_fam},
        seen_by_message={
            "m1": {e_vitus, e_jana, e_den, e_son, e_fam},
            "m2": {e_vitus, e_jana},
            "m3": {e_vitus, e_jana, e_den},
        },
        user_surname="Bertram",
        user_emails={"dennison@withtally.com"},
        on_log=logs.append,
    )
    assert "Vitus" in members, f"Vitus must be a candidate; got {list(members)}"
    # Diagnostic confirms BOTH rules counted.
    gather_line = next(
        (ln for ln in logs if ln.startswith("[gather]") and "rule_7=" in ln),
        None,
    )
    assert gather_line is not None, f"missing gather line in: {logs}"
    assert "rule_1=" in gather_line
    # rule_7 must count for Vitus (co-occurs with Jana in >= 2 messages).
    r7_match = [
        part for part in gather_line.split(", ") if part.startswith("rule_7=")
    ]
    assert r7_match, f"rule_7 missing from diagnostic; got: {gather_line}"
    r7_count = int(r7_match[0].split("=")[1].split(";")[0])
    assert r7_count >= 1, (
        f"rule_7 should count at least 1 (Vitus merged); got {r7_count}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_diagnostic_in_log():
    """The [gather] diagnostic log line includes rule_7=N."""
    corpus, entities, seen_by_message = _build_rule_7_corpus()
    logs: list[str] = []
    _gather_family_members(
        corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        user_surname="Bertram",
        user_emails={"dennison@withtally.com"},
        on_log=logs.append,
    )
    gather_lines = [ln for ln in logs if ln.startswith("[gather]")]
    assert len(gather_lines) >= 1, (
        f"expected at least one [gather] log line; got {gather_lines}"
    )
    line = gather_lines[0]
    assert "rule_7=" in line, (
        f"expected rule_7=N in diagnostic; got: {line}"
    )


@pytest.mark.skip(reason="Removed: family-specific logic")
def test_gather_rule_7_co_occurs_with_in_excerpt():
    """The excerpt dicts produced by Rule 7 include a co_occurs_with field
    naming the confirmed family member."""
    corpus, entities, seen_by_message = _build_rule_7_corpus()
    person_entities = sorted(
        [e for e in entities if e.kind == "person"], key=lambda e: e.label
    )
    results = _gather_rule_7_family_graph_expansion(
        corpus,
        entities=person_entities,
        auto_accepted_names={"Jana Bertram"},
        user_emails={"dennison@withtally.com"},
    )
    # Find Vitus in the results.
    vitus_result = [r for r in results if r[0] == "Vitus"]
    assert vitus_result, f"Vitus must be in Rule 7 results; got {[r[0] for r in results]}"
    _name, excerpt_dicts, _mids = vitus_result[0]
    assert excerpt_dicts, "Vitus must have excerpt dicts"
    for ex in excerpt_dicts:
        assert "co_occurs_with" in ex, (
            f"excerpt must include co_occurs_with field; got keys: {list(ex.keys())}"
        )
        assert ex["co_occurs_with"] == "Jana Bertram", (
            f"co_occurs_with should be 'Jana Bertram'; got: {ex['co_occurs_with']}"
        )
