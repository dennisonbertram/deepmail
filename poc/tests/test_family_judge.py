"""Tests for the LLM-as-final-judge step (Pass 8B).

The judge takes a candidate name + a handful of email excerpts and classifies
the candidate as family / not_family / uncertain. The materializer calls it
BEFORE writing the profile so only `family` verdicts land in the accepted
Members section, `uncertain` verdicts go into a user-review pile, and
`not_family` verdicts get listed as bare slugs in a "Rejected" section.

Two layers under test:

  * The mock backend's pattern-match logic (no API key required).
  * The materializer's partitioning + section-writing + frontmatter block.
"""

from __future__ import annotations
import pytest

import json
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

import yaml  # noqa: E402

from pi_email.corpus import Corpus, Message  # noqa: E402
from pi_email.entities import Entity  # noqa: E402
from pi_email.family_judge import (  # noqa: E402
    FamilyJudge,
    FamilyVerdict,
)
from pi_email.materializer import write_family_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal fake Anthropic client (mirrors test_proposer_grounding._FakeClient)
# ---------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    """Returns canned responses in round-robin order. Each `create` call
    pops the next response, looping when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = (len(self.calls) - 1) % len(self._responses)
        return _FakeResp(self._responses[idx])


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.messages = _FakeMessages(responses)


# ---------------------------------------------------------------------------
# Mock-judge unit tests
# ---------------------------------------------------------------------------


def test_mock_judge_my_wife_jana_classified_family():
    """`my <relation> <first>` is the spec's canonical match pattern."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "friend@example.com",
            "subject": "intro",
            "snippet": "happy to introduce my wife Jana to the team",
        }
    ]
    v = judge.judge("Jana Bertram", excerpts, user_email="me@example.com")
    assert v.decision == "family"
    assert v.relation_guess == "spouse"
    assert v.confidence == 0.7
    assert "[mock]" in v.reasoning


def test_mock_judge_general_partner_classified_not_family():
    """The substring "partner" appears, but only as the business sense
    ("general partner"). The mock requires an immediately-adjacent relation
    word BEFORE the candidate's first name — "general partner Jane" does
    NOT match the `my/your/<relation> <first>` patterns."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "vc@example.com",
            "subject": "introducing our team",
            "snippet": "Jane Doe joins us as a general partner at Acme Capital",
        }
    ]
    v = judge.judge("Jane Doe", excerpts, user_email="me@example.com")
    assert v.decision == "not_family"
    assert v.relation_guess is None
    assert v.confidence == 0.5


def test_mock_judge_no_relation_keyword_returns_not_family():
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "colleague@example.com",
            "subject": "Q3 planning",
            "snippet": "Bob Smith will own the migration this quarter.",
        }
    ]
    v = judge.judge("Bob Smith", excerpts, user_email="me@example.com")
    assert v.decision == "not_family"
    assert v.relation_guess is None


def test_mock_judge_your_sister_matches_family():
    """The mock also accepts `your <relation> <first>` — emails FROM a
    family member ADDRESSED to the user. This is what most fixture messages
    look like."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "emma@example.com",
            "subject": "birthday",
            "snippet": "Your sister Emma here. My birthday is coming up.",
        }
    ]
    v = judge.judge("Emma Bertram", excerpts, user_email="me@example.com")
    assert v.decision == "family"
    assert v.relation_guess == "sibling"


def test_mock_judge_bare_aunt_carol_matches_family():
    """Adjacent `<relation> <first>` form covers "Aunt Carol here"."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "carol@example.com",
            "subject": "thanksgiving",
            "snippet": "Aunt Carol here. Mom said she'd come if Jane drives.",
        }
    ]
    v = judge.judge("Carol", excerpts, user_email="me@example.com")
    assert v.decision == "family"
    assert v.relation_guess == "aunt_or_uncle"


def test_judge_batch_returns_verdict_per_candidate():
    judge = FamilyJudge(force_mock=True)
    inputs = [
        (
            "Jana Bertram",
            [{"msg_id": "m1", "snippet": "introducing my wife Jana to the team"}],
        ),
        (
            "Bob Smith",
            [{"msg_id": "m2", "snippet": "Bob Smith will run engineering"}],
        ),
        (
            "Emma Bertram",
            [{"msg_id": "m3", "snippet": "your sister Emma is visiting"}],
        ),
    ]
    verdicts = judge.judge_batch(inputs, user_email="me@example.com")
    assert len(verdicts) == 3
    # Output order matches input order.
    assert [v.canonical for v in verdicts] == [
        "Jana Bertram",
        "Bob Smith",
        "Emma Bertram",
    ]
    assert verdicts[0].decision == "family"
    assert verdicts[1].decision == "not_family"
    assert verdicts[2].decision == "family"


def test_live_judge_with_injected_fake_client_parses_response():
    """With a fake Anthropic client injected, the judge takes the live path
    (is_mock=False) and parses the JSON response into a verdict."""
    response_json = json.dumps(
        {
            "decision": "family",
            "relation_guess": "spouse",
            "confidence": 0.92,
            "reasoning": "Multiple excerpts refer to Jana as the user's wife.",
        }
    )
    client = _FakeClient([response_json])
    judge = FamilyJudge(client=client)
    assert not judge.is_mock
    v = judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "my wife Jana"}],
        user_email="me@example.com",
        user_display_name="Me Surname",
    )
    assert v.decision == "family"
    assert v.relation_guess == "spouse"
    assert v.confidence == 0.92
    # The fake client recorded the call — confirms cache_control and model
    # were set up correctly.
    call_kwargs = client.messages.calls[0]
    assert call_kwargs["model"]  # set
    sys_block = call_kwargs["system"][0]
    assert sys_block["type"] == "text"
    assert sys_block["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Pass 9A: surname-aware mock judge
# ---------------------------------------------------------------------------


def test_surname_match_with_mock_judge_returns_family():
    """A 2+ token candidate whose LAST token matches the user's surname is
    classified as family at 0.85 confidence — the mock mirrors the live
    prompt's surname-weight instruction so offline runs still surface
    surname matches."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "Jane Bertram <jane@example.com>",
            "subject": "groceries",
            "snippet": "Picked up groceries on the way home.",
        }
    ]
    v = judge.judge(
        "Jane Bertram",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
    )
    assert v.decision == "family"
    assert v.confidence == 0.85
    assert v.relation_guess == "other"
    assert "surname" in v.reasoning.lower()


def test_surname_no_match_no_family_signal():
    """A candidate with a different surname AND no relation cue in excerpts
    falls through to the not_family default."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "Jane Smith <jane@example.com>",
            "subject": "lunch",
            "snippet": "Want to grab lunch next week?",
        }
    ]
    v = judge.judge(
        "Jane Smith",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
    )
    assert v.decision == "not_family"


def test_surname_match_requires_two_token_candidate():
    """A bare-first-name candidate ("Bertram") must NOT pass the surname
    check — without 2+ tokens we can't tell first-name-coincidence from a
    real surname match."""
    judge = FamilyJudge(force_mock=True)
    v = judge.judge(
        "Bertram",
        [{"msg_id": "m1", "snippet": "Bertram is a colleague"}],
        user_email="me@example.com",
        user_surname="Bertram",
    )
    # Falls through to the not_family default; no relation cue in excerpts.
    assert v.decision == "not_family"


def test_judge_prompt_includes_surname_instruction():
    """The live system prompt must instruct the model to weight surname
    matches. And the per-call user message must surface the user's surname
    field. Both layers are critical because the system block is cached and
    the user message lands per-candidate.

    Pass 15A: SYSTEM_PROMPT is now composed dynamically via
    `_build_system_prompt(contacts_population)`. Surname-match must remain
    a documented high-confidence signal in BOTH variants (no-contacts and
    with-contacts) — without curated contacts it's literally signal #1, and
    with contacts it's still a 0.85+ family indicator."""
    from pi_email.family_judge import _build_system_prompt, _build_user_prompt

    # 1. Both system-prompt variants mention SURNAME at 0.85+.
    for pop in (0, 5):
        sysp = _build_system_prompt(pop)
        assert "SURNAME" in sysp, (
            f"SURNAME signal missing from system prompt with population={pop}"
        )
        assert "0.85+" in sysp or "0.85" in sysp, (
            f"0.85 surname confidence missing from prompt with population={pop}"
        )

    # 2. User message renders the surname field AND the inline note.
    msg = _build_user_prompt(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
    )
    assert "surname: Bertram" in msg
    assert "STRONG signal of family" in msg


def test_judge_prompt_renders_unknown_surname_when_none():
    """When user_surname is not provided, the user message renders
    'unknown' — the surname-weight instruction still appears (it's static
    context) but doesn't have a value to compare against."""
    from pi_email.family_judge import _build_user_prompt

    msg = _build_user_prompt(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name=None,
        user_surname=None,
    )
    assert "surname: unknown" in msg


# ---------------------------------------------------------------------------
# Pass 9A: confidence-band routing
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_promotes_mid_to_uncertain():
    """The fix that motivated Pass 9A: a 0.70 not_family verdict must land
    in the Uncertain bucket, not Rejected. Run 8 rejected the user's actual
    wife (Jana Bertram) at exactly this confidence."""
    from pi_email.materializer import _bucket_verdict

    v = FamilyVerdict(
        canonical="Jana Bertram",
        decision="not_family",
        relation_guess=None,
        confidence=0.70,
        reasoning="excerpts are ambiguous business context",
    )
    assert _bucket_verdict(v) == "uncertain"


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_high_not_family_stays_rejected():
    """A 0.95 not_family verdict is the model being confident — bucket as
    rejected."""
    from pi_email.materializer import _bucket_verdict

    v = FamilyVerdict(
        canonical="Acme Capital Newsletter",
        decision="not_family",
        relation_guess=None,
        confidence=0.95,
        reasoning="this is a newsletter, not a person",
    )
    assert _bucket_verdict(v) == "rejected"


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_low_family_demoted_to_uncertain():
    """A 0.60 family verdict is weak — don't promote to accepted; route to
    uncertain for review."""
    from pi_email.materializer import _bucket_verdict

    v = FamilyVerdict(
        canonical="Jane Doe",
        decision="family",
        relation_guess="sibling",
        confidence=0.60,
        reasoning="might be a sibling but evidence is thin",
    )
    assert _bucket_verdict(v) == "uncertain"


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_high_family_accepted():
    """A 0.92 family verdict is high-confidence — accepted."""
    from pi_email.materializer import _bucket_verdict

    v = FamilyVerdict(
        canonical="Jana Bertram",
        decision="family",
        relation_guess="spouse",
        confidence=0.92,
        reasoning="multiple excerpts cite Jana as the user's wife",
    )
    assert _bucket_verdict(v) == "accepted"


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_uncertain_passes_through():
    """The model's own `uncertain` decision routes to the Uncertain bucket
    regardless of confidence."""
    from pi_email.materializer import _bucket_verdict

    v_low = FamilyVerdict(
        canonical="x", decision="uncertain", relation_guess=None,
        confidence=0.1, reasoning="",
    )
    v_high = FamilyVerdict(
        canonical="x", decision="uncertain", relation_guess=None,
        confidence=0.99, reasoning="",
    )
    assert _bucket_verdict(v_low) == "uncertain"
    assert _bucket_verdict(v_high) == "uncertain"


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_confidence_band_low_not_family_rejected():
    """A 0.30 not_family verdict — model is unsure but no positive family
    signal either. Bucket as rejected so we bound noise."""
    from pi_email.materializer import _bucket_verdict

    v = FamilyVerdict(
        canonical="x",
        decision="not_family",
        relation_guess=None,
        confidence=0.30,
        reasoning="no signal at all",
    )
    assert _bucket_verdict(v) == "rejected"


def test_judge_batch_forwards_user_surname():
    """`judge_batch` must forward `user_surname` to every per-candidate
    call. With force_mock=True, surname-matching candidates flip to
    family at 0.85; non-matching, no-cue candidates stay not_family."""
    judge = FamilyJudge(force_mock=True)
    inputs = [
        (
            "Jana Bertram",
            [{"msg_id": "m1", "snippet": "groceries"}],
        ),
        (
            "Bob Smith",
            [{"msg_id": "m2", "snippet": "Bob runs sales at Acme"}],
        ),
    ]
    verdicts = judge.judge_batch(
        inputs,
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
    )
    assert verdicts[0].decision == "family"
    assert verdicts[0].confidence == 0.85
    assert verdicts[1].decision == "not_family"


def test_judge_uses_temperature_zero():
    """Pass 11: the live backend MUST pin temperature=0 so the judge is
    deterministic across runs. Run 9 and Run 10 returned different verdicts
    (Accepted 0.90 vs Uncertain 0.75) for the same candidate with similar
    evidence; sampling jitter at the model's default temperature was the
    cause. This test pins the fix so a future refactor can't silently
    reintroduce non-determinism."""
    response_json = json.dumps(
        {
            "decision": "family",
            "relation_guess": "spouse",
            "confidence": 0.90,
            "reasoning": "wife",
        }
    )
    client = _FakeClient([response_json])
    judge = FamilyJudge(client=client)
    judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "my wife Jana"}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
    )
    call_kwargs = client.messages.calls[0]
    assert "temperature" in call_kwargs, (
        "live judge must explicitly set temperature so future SDK default "
        "shifts don't reintroduce sampling jitter"
    )
    assert call_kwargs["temperature"] == 0.0, (
        f"expected temperature=0.0; got {call_kwargs['temperature']!r}"
    )


def test_live_judge_falls_back_to_mock_on_unparseable_response():
    """Malformed JSON from the LLM must not crash — the judge falls back to
    the deterministic mock for that candidate."""
    client = _FakeClient(["this is not json at all"])
    logs: list[str] = []
    judge = FamilyJudge(client=client, on_log=logs.append)
    v = judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "my wife Jana"}],
        user_email="me@example.com",
    )
    # Fell back to mock, which matched "my wife Jana".
    assert v.decision == "family"
    assert v.relation_guess == "spouse"
    assert any("parse" in log for log in logs)


# ---------------------------------------------------------------------------
# Materializer integration
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
    """Tiny corpus + entity set with Dennison + Jana both co-occurring with
    relation words. Mirrors the helper in test_materializer.py."""
    corpus = Corpus()
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

    m2 = _msg(
        "m2",
        from_addr="Jana Bertram <jana@example.com>",
        subject="My sister Jana wedding",
        body="Jana Bertram and I (Dennison Bertram) are siblings going.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)

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


class _StaticJudge:
    """Test double for FamilyJudge that returns canned verdicts keyed by
    normalized canonical name. Avoids needing to plumb a fake client through
    the live backend just to drive materializer integration tests."""

    def __init__(self, verdicts: dict[str, FamilyVerdict]) -> None:
        self.is_mock = True  # surfaces as "mock" in frontmatter
        self.model = "static-test-double"
        self._verdicts = verdicts

    def banner(self) -> str:
        return "[STATIC TEST JUDGE]"

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
                reasoning="default-not-family",
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
    """Pull the YAML frontmatter off a profile markdown."""
    assert content.startswith("---\n")
    end = content.index("\n---\n", 4)
    yaml_blob = content[4:end]
    return yaml.safe_load(yaml_blob)


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_materializer_writes_uncertain_section(tmp_path):
    """One judge verdict is `uncertain` — the profile must contain a
    `## Possibly family (uncertain)` section with the candidate's excerpts."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    verdicts = {
        "Dennison Bertram": FamilyVerdict(
            canonical="Dennison Bertram",
            decision="family",
            relation_guess="other",
            confidence=0.9,
            reasoning="user themselves",
        ),
        "Jana Bertram": FamilyVerdict(
            canonical="Jana Bertram",
            decision="uncertain",
            relation_guess="sibling",
            confidence=0.4,
            reasoning="ambiguous: could be spouse or sister.",
        ),
    }
    judge = _StaticJudge(verdicts)
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
        judge=judge,
    )

    content = out.read_text(encoding="utf-8")
    assert "## Possibly family (uncertain)" in content
    assert "### Jana Bertram" in content
    # Reasoning surfaces in the uncertain section.
    assert "ambiguous" in content
    # Accepted member is in the Members section, not the uncertain one.
    assert "## Members" in content


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_materializer_writes_judge_yaml_block(tmp_path):
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    verdicts = {
        "Dennison Bertram": FamilyVerdict(
            canonical="Dennison Bertram",
            decision="family",
            relation_guess="other",
            confidence=0.85,
            reasoning="self",
        ),
        "Jana Bertram": FamilyVerdict(
            canonical="Jana Bertram",
            decision="uncertain",
            relation_guess=None,
            confidence=0.4,
            reasoning="ambiguous",
        ),
    }
    judge = _StaticJudge(verdicts)
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
        judge=judge,
    )

    fm = _read_frontmatter(out.read_text(encoding="utf-8"))
    assert "judge" in fm, f"frontmatter missing judge block: {fm}"
    jb = fm["judge"]
    assert jb["accepted"] == 1
    assert jb["uncertain"] == 1
    assert jb["rejected"] == 0
    assert jb["skipped"] is False
    # The model is "mock" for the static test double, or the live model name
    # otherwise — confirm a model field is present.
    assert "model" in jb


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_materializer_writes_rejected_section(tmp_path):
    """When the judge classifies a candidate as `not_family` AT HIGH
    CONFIDENCE, the profile lists them as a bare slug in `## Rejected`
    without excerpts.

    Pass 9A introduced confidence-band routing: not_family verdicts below
    0.85 land in Uncertain, not Rejected. We use 0.95 here to exercise the
    high-confidence rejected path; the in-between band is exercised by
    `test_confidence_band_*`."""
    corpus, entities, seen_by_message = _build_corpus_with_two_people()
    verdicts = {
        "Dennison Bertram": FamilyVerdict(
            canonical="Dennison Bertram",
            decision="family",
            relation_guess="other",
            confidence=0.85,
            reasoning="self",
        ),
        "Jana Bertram": FamilyVerdict(
            canonical="Jana Bertram",
            decision="not_family",
            relation_guess=None,
            confidence=0.95,
            reasoning="no clear family signal.",
        ),
    }
    judge = _StaticJudge(verdicts)
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
        judge=judge,
    )

    content = out.read_text(encoding="utf-8")
    assert "## Rejected" in content
    # Rejected names appear as bullet items, NOT as ### headers.
    assert "- Jana Bertram" in content
    assert "### Jana Bertram" not in content
    # The Members section should NOT contain the rejected candidate.
    members_section = content.split("## Members", 1)[1].split("##", 1)[0]
    assert "Jana Bertram" not in members_section


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_skip_judge_flag_writes_all_candidates_as_members(tmp_path):
    """When skip_judge=True, the materializer bypasses the judge entirely.
    All gathered candidates are written as `family` (legacy behavior)."""
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

    content = out.read_text(encoding="utf-8")
    # Both candidates land as accepted members.
    assert "### Dennison Bertram" in content
    assert "### Jana Bertram" in content
    # No uncertain / rejected sections produced.
    assert "## Possibly family (uncertain)" not in content
    assert "## Rejected" not in content
    # Frontmatter records the skip.
    fm = _read_frontmatter(content)
    assert fm["judge"]["skipped"] is True
    # Log line surfaces the skip.
    assert any("[judge] skipped" in line for line in logs)


@pytest.mark.skip(reason="Removed: family-specific judge logic")
def test_materializer_default_judge_uses_mock_without_api_key(monkeypatch, tmp_path):
    """When no judge is injected AND ANTHROPIC_API_KEY is unset, the
    materializer constructs a default FamilyJudge that uses the mock
    backend. End-to-end: candidates with "my <rel>" in their excerpts are
    accepted; others are rejected. Exercises the default-construction code
    path in `write_family_profile`."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    corpus = Corpus()
    # Excerpt for Jana contains "my wife Jana" — should be accepted.
    m1 = _msg(
        "m1",
        from_addr="someone@example.com",
        subject="dinner",
        body="my wife Jana is making dinner and the kids are excited.",
    )
    m1.body_clean = m1.body
    corpus.add(m1)
    # Excerpt for Bob has no relation word adjacent — should be rejected.
    m2 = _msg(
        "m2",
        from_addr="vc@example.com",
        subject="Acme intro",
        body="Bob Smith joins us as a general partner at Acme. family of funds.",
    )
    m2.body_clean = m2.body
    corpus.add(m2)

    e_jana = Entity(kind="person", key="jana", label="Jana")
    e_bob = Entity(kind="person", key="bob smith", label="Bob Smith")
    e_rel_wife = Entity(kind="relation", key="wife", label="wife")
    e_rel_kids = Entity(kind="relation", key="kids", label="kids")
    e_rel_fam = Entity(kind="relation", key="family", label="family")

    entities = {e_jana, e_bob, e_rel_wife, e_rel_kids, e_rel_fam}
    seen_by_message = {
        "m1": {e_jana, e_rel_wife, e_rel_kids},
        "m2": {e_bob, e_rel_fam},
    }
    out = tmp_path / "family.md"
    write_family_profile(
        out_path=out,
        corpus=corpus,
        entities=entities,
        seen_by_message=seen_by_message,
        seed="figure out my family",
        stop_reason="frontier_exhausted",
        queries_run=["test"],
        canonical_map={"Jana": "Jana", "Bob Smith": "Bob Smith"},
        user_self=None,
    )

    content = out.read_text(encoding="utf-8")
    fm = _read_frontmatter(content)
    # Jana lands as a ### header — in either Members or Possibly family.
    # The mock's relation-cue match returns 0.7 family which, under Pass 9A
    # confidence-band routing, lands in the Uncertain bucket (0.55-0.80
    # family band). Without a surname-match signal we don't get to 0.85 here.
    assert "### Jana" in content
    # Bob has no relation cue AND no surname signal — falls to not_family
    # at 0.5 confidence which buckets as rejected (below 0.55 floor).
    assert "## Rejected" in content
    assert "- Bob Smith" in content
    # Frontmatter records mock model + the partition counts.
    assert fm["judge"]["model"] == "mock"
    # Pre-Pass-9A this was `accepted >= 1`; under confidence-band routing
    # the mock relation-cue match produces uncertain instead.
    assert fm["judge"]["accepted"] + fm["judge"]["uncertain"] >= 1
    assert fm["judge"]["rejected"] >= 1


# ---------------------------------------------------------------------------
# Pass 14A: contacts_population — judge must not penalize absent Contacts
# signal when the user has no curated family contacts at all.
# ---------------------------------------------------------------------------
#
# Background: Pass 12A added contacts-aware judging. The system prompt
# (correctly) treats a strong GOOGLE_CONTACTS_SIGNAL as authoritative. But
# Pass 13 surfaced a regression: when a candidate had NO contact_evidence
# attached AND the user had zero curated family contacts, the model still
# expected to see a signal and demoted the candidate (Jana Bertram, the
# user's actual wife, was Accepted in Pass 11/12 and demoted to Uncertain in
# Pass 13).
#
# Fix: pass `contacts_population` through to the judge so it can distinguish
# "absent signal AND user has no contacts" (uninformative absence — judge on
# email evidence alone) from "absent signal AND user actively curates"
# (weak negative evidence — ~0.1 penalty).


def test_contacts_population_zero_does_not_penalize_absent_signal():
    """contacts_population=0 means the user has NO curated family contacts.
    Absence of contact_evidence on Jana Bertram is EXPECTED and uninformative.
    The mock's surname-match path returns family at 0.85 — confidence must
    NOT be penalized below the 0.80 family-acceptance floor."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "Jana Bertram <jana@example.com>",
            "subject": "groceries",
            "snippet": "Picked up groceries on the way home.",
        }
    ]
    v = judge.judge(
        "Jana Bertram",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
        contacts_population=0,
    )
    assert v.decision == "family"
    # Surname-match base confidence (0.85) must NOT be penalized when the
    # user has zero curated family contacts.
    assert v.confidence >= 0.80, (
        f"contacts_population=0 must not penalize surname-match family; "
        f"got confidence={v.confidence}"
    )


def test_contacts_population_positive_penalizes_absent_signal():
    """contacts_population=5 means the user actively curates family contacts.
    A surname-match family verdict for a candidate with NO contact_evidence
    is weak negative evidence — confidence drops by ~0.1 vs the no-curation
    baseline (0.85 -> 0.75)."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "Jana Bertram <jana@example.com>",
            "subject": "groceries",
            "snippet": "Picked up groceries on the way home.",
        }
    ]
    v = judge.judge(
        "Jana Bertram",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
        contacts_population=5,
    )
    # Mock still calls it family — surname match is strong — but penalty
    # applied. Confidence is roughly 0.10 lower than the no-curation case.
    assert v.decision == "family"
    assert 0.70 <= v.confidence <= 0.80, (
        f"contacts_population>0 must apply ~0.1 absent-signal penalty; "
        f"got confidence={v.confidence}"
    )
    # Baseline (no penalty) returns 0.85; this run must be ~0.1 lower.
    assert v.confidence == pytest_approx(0.75)


def test_contacts_population_none_ignores_signal():
    """contacts_population=None means contacts were not consulted at all
    (legacy / fixture path). Behavior matches Pass 12 exactly — the mock's
    surname-match path returns family at 0.85 with no penalty."""
    judge = FamilyJudge(force_mock=True)
    excerpts = [
        {
            "msg_id": "m1",
            "from_addr": "Jana Bertram <jana@example.com>",
            "subject": "groceries",
            "snippet": "Picked up groceries on the way home.",
        }
    ]
    v_default = judge.judge(
        "Jana Bertram",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
    )
    v_none = judge.judge(
        "Jana Bertram",
        excerpts,
        user_email="me@example.com",
        user_surname="Bertram",
        contacts_population=None,
    )
    # Both code paths must produce the same verdict.
    assert v_none.decision == v_default.decision == "family"
    assert v_none.confidence == v_default.confidence == 0.85


def test_judge_prompt_renders_contacts_population():
    """The per-call user prompt must surface `contacts_population: 0` so the
    live model can apply the CONTACTS INTERPRETATION rule. The system prompt
    is cached and tells the model what 0 means; the user prompt provides
    the value."""
    response_json = json.dumps(
        {
            "decision": "family",
            "relation_guess": "spouse",
            "confidence": 0.9,
            "reasoning": "x",
        }
    )
    client = _FakeClient([response_json])
    judge = FamilyJudge(client=client)
    judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=0,
    )
    # Inspect what the live backend actually sent.
    call_kwargs = client.messages.calls[0]
    user_msg = call_kwargs["messages"][0]["content"]
    assert "contacts_population: 0" in user_msg, (
        f"expected `contacts_population: 0` in user prompt; got:\n{user_msg}"
    )


def test_judge_prompt_omits_contacts_population_when_none():
    """When contacts_population is None, the user prompt must NOT render the
    field — that's the "contacts not consulted" path, and rendering would
    confuse the model about what state we're in."""
    from pi_email.family_judge import _build_user_prompt

    msg = _build_user_prompt(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=None,
    )
    assert "contacts_population" not in msg, (
        f"expected no contacts_population field when None; got:\n{msg}"
    )


def test_judge_prompt_renders_contacts_population_positive():
    """contacts_population=N renders as `contacts_population: N`. The exact
    integer round-trips so the model can see how curated the address book is."""
    from pi_email.family_judge import _build_user_prompt

    msg = _build_user_prompt(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=7,
    )
    assert "contacts_population: 7" in msg


def test_system_prompt_includes_contacts_interpretation_section():
    """The cached system prompt (with-contacts variant) must document the
    contacts_population semantics so the model knows how to interpret a
    present GOOGLE_CONTACTS_SIGNAL.

    Pass 15A: the no-contacts variant suppresses contacts language entirely;
    only the WITH_CONTACTS variant carries the CONTACTS INTERPRETATION
    section. The "population = 0" case is now handled by the no-contacts
    variant simply not mentioning contacts at all (verified by a separate
    test below)."""
    from pi_email.family_judge import _build_system_prompt

    sysp = _build_system_prompt(5)  # positive population -> WITH_CONTACTS variant
    assert "CONTACTS INTERPRETATION" in sysp
    # The positive-population behavior must be explicitly named so the model
    # knows absence is weak negative evidence.
    assert "contacts_population` is > 0" in sysp


# ---------------------------------------------------------------------------
# Pass 15A: dynamic system-prompt composition. When the user has no curated
# family contacts (population in (None, 0)), the contacts-related prompt
# blocks are suppressed entirely — the model must not be told that "Google
# Contacts is the strongest signal" when no such signal exists. Without this
# fix the model carried the contacts framing and hedged on surname-match
# family candidates (Jana Bertram demoted Accepted->Uncertain in Run 13/14).
# ---------------------------------------------------------------------------


def test_system_prompt_no_contacts_mentions_when_population_zero():
    """With contacts_population=0 the system prompt MUST NOT mention Google
    Contacts at all. The user has no curated family contacts, so any
    mention of contacts as "the strongest signal" biases the model toward
    hedging on otherwise-strong family candidates."""
    from pi_email.family_judge import _build_system_prompt

    sysp = _build_system_prompt(0)
    assert "Google Contacts" not in sysp, (
        f"contacts_population=0 prompt must not mention Google Contacts; "
        f"found in:\n{sysp}"
    )
    assert "GOOGLE_CONTACTS_SIGNAL" not in sysp, (
        f"contacts_population=0 prompt must not mention "
        f"GOOGLE_CONTACTS_SIGNAL; found in:\n{sysp}"
    )


def test_system_prompt_no_contacts_mentions_when_population_none():
    """contacts_population=None means contacts were not consulted at all
    (e.g., fixture / legacy path). Behavior must match population=0 —
    no mention of Google Contacts in the prompt."""
    from pi_email.family_judge import _build_system_prompt

    sysp = _build_system_prompt(None)
    assert "Google Contacts" not in sysp
    assert "GOOGLE_CONTACTS_SIGNAL" not in sysp


def test_system_prompt_full_contacts_when_population_positive():
    """With contacts_population>0 the system prompt MUST carry the full
    contacts-aware priority hierarchy — Google Contacts is the strongest
    signal, GOOGLE_CONTACTS_SIGNAL block is described, etc."""
    from pi_email.family_judge import _build_system_prompt

    sysp = _build_system_prompt(5)
    assert "Google Contacts" in sysp, (
        f"positive-population prompt must mention Google Contacts; "
        f"missing in:\n{sysp}"
    )
    assert "GOOGLE_CONTACTS_SIGNAL" in sysp
    # The "Family" group membership is the strongest authority — must appear.
    assert "Family" in sysp


def test_system_prompt_reasserts_surname_priority_when_no_contacts():
    """In the no-contacts variant, SURNAME match must be reasserted as
    priority signal #1. Without curated contacts there's no other curated
    authority — the user's own surname is the strongest available signal
    and the prompt must say so explicitly."""
    from pi_email.family_judge import _build_system_prompt

    sysp = _build_system_prompt(0)
    assert "SURNAME" in sysp, (
        f"no-contacts prompt must reassert SURNAME priority; missing in:\n{sysp}"
    )
    # SURNAME should appear as priority signal #1.
    assert "1. SURNAME" in sysp, (
        f"no-contacts prompt must list SURNAME as priority signal #1; "
        f"got:\n{sysp}"
    )


def test_live_judge_uses_dynamic_system_prompt():
    """End-to-end: when the live judge is called with contacts_population=0,
    the system block sent to the Anthropic client must NOT contain "Google
    Contacts" — the dynamic composition must reach the API call site."""
    response_json = json.dumps(
        {
            "decision": "family",
            "relation_guess": "spouse",
            "confidence": 0.9,
            "reasoning": "surname match",
        }
    )
    client = _FakeClient([response_json])
    judge = FamilyJudge(client=client)
    judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "Picked up groceries."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=0,
    )
    call_kwargs = client.messages.calls[0]
    sys_block = call_kwargs["system"][0]
    assert sys_block["type"] == "text"
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    sys_text = sys_block["text"]
    assert "Google Contacts" not in sys_text, (
        f"live judge with contacts_population=0 must send a system block "
        f"that does not mention Google Contacts; got:\n{sys_text}"
    )
    assert "GOOGLE_CONTACTS_SIGNAL" not in sys_text
    # And SURNAME is still asserted as priority signal #1.
    assert "SURNAME" in sys_text


def test_live_judge_with_positive_contacts_population_sends_contacts_prompt():
    """Mirror of the above for the positive-population path — confirms
    `_build_system_prompt` is plumbed correctly in both directions and the
    dynamic composition isn't accidentally pinned to one variant."""
    response_json = json.dumps(
        {
            "decision": "family",
            "relation_guess": "spouse",
            "confidence": 0.92,
            "reasoning": "x",
        }
    )
    client = _FakeClient([response_json])
    judge = FamilyJudge(client=client)
    judge.judge(
        "Jana Bertram",
        [{"msg_id": "m1", "snippet": "..."}],
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=3,
    )
    sys_text = client.messages.calls[0]["system"][0]["text"]
    assert "Google Contacts" in sys_text
    assert "GOOGLE_CONTACTS_SIGNAL" in sys_text


def test_judge_batch_forwards_contacts_population():
    """judge_batch must thread contacts_population through to every
    per-candidate call. With force_mock=True and contacts_population=0,
    surname-match candidates retain their 0.85 confidence (no penalty)."""
    judge = FamilyJudge(force_mock=True)
    inputs = [
        ("Jana Bertram", [{"msg_id": "m1", "snippet": "groceries"}]),
        ("Bob Smith", [{"msg_id": "m2", "snippet": "Bob runs sales at Acme"}]),
    ]
    verdicts = judge.judge_batch(
        inputs,
        user_email="me@example.com",
        user_display_name="Dennison Bertram",
        user_surname="Bertram",
        contacts_population=0,
    )
    # Jana — surname match, no penalty under population=0.
    assert verdicts[0].decision == "family"
    assert verdicts[0].confidence == 0.85
    # Bob — no surname match, no relation cue. Stays not_family.
    assert verdicts[1].decision == "not_family"


def pytest_approx(value: float, tol: float = 0.01) -> "_ApproxCmp":
    return _ApproxCmp(value, tol)


class _ApproxCmp:
    """Tiny approximate-equality wrapper so the tests can use
    `confidence == pytest_approx(0.75)` without depending on pytest.approx
    (kept local for parity with the project's style of self-contained tests).
    """

    def __init__(self, value: float, tol: float) -> None:
        self.value = value
        self.tol = tol

    def __eq__(self, other) -> bool:
        try:
            return abs(float(other) - self.value) <= self.tol
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return f"~{self.value:.4f}"
