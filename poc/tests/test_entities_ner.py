"""Tests for the spaCy-based entity extractor in entities.py.

The previous regex-based extractor produced 5,363 putative "family" members
when run against a real 121K-message inbox — dominated by company names,
place names, job titles, and newsletter bylines. These tests pin the
behavior of the replacement:

  * PERSON entities require a relation word in the SAME SENTENCE.
  * Bulk-flagged messages return [] (no extraction at all).
  * Emails extract unconditionally — they're first-class expansion targets.
  * Sender-name from "Name <addr>" headers extracts at medium confidence.
  * Honorifics ("Aunt Carol") and company suffixes ("Substack Inc") are
    handled by the stoplist / span-cleanup logic.

First run downloads the spaCy en_core_web_sm model (~12 MB).
"""

from __future__ import annotations
import pytest

import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.corpus import Message  # noqa: E402
from pi_email.entities import Entity, extract_entities  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    body: str,
    *,
    from_addr: str = "alice@example.com",
    to_addr: str = "me@example.com",
    subject: str = "test",
    message_id: str = "msg-test",
    is_bulk: bool = False,
) -> Message:
    """Build a Message for testing. body_clean is pre-populated so the
    extractor doesn't run quote-stripping over our synthetic strings."""
    m = Message(
        message_id=message_id,
        thread_id=message_id,
        from_addr=from_addr,
        to_addr=to_addr,
        subject=subject,
        date="2025-01-01T00:00:00Z",
        body=body,
        source_path=Path("/dev/null"),
        is_bulk=is_bulk,
    )
    m.body_clean = body
    return m


def _persons(ents: list[Entity]) -> set[str]:
    return {e.label for e in ents if e.kind == "person"}


def _emails(ents: list[Entity]) -> set[str]:
    return {e.label for e in ents if e.kind == "email"}


def _relations(ents: list[Entity]) -> set[str]:
    return {e.label for e in ents if e.kind == "relation"}


# ---------------------------------------------------------------------------
# Case 1: personal email with relation word in same sentence as PERSON
# ---------------------------------------------------------------------------


def test_case1_relation_in_same_sentence_extracts_person():
    """Body: Jane Smith is introduced via 'my sister Jane Smith' — the
    relation phrase ('my sister') binds ADJACENTLY to Jane Smith.

    Pass 7A note: pre-7A the test used 'Jane Smith said she'd help our
    family' — sentence-level loose binding admitted Jane Smith via 'our
    family' positioned AFTER the name. Under 7A's adjacency rule the
    relation phrase has to be positioned so the name follows it within
    the window, so we rephrase the body."""
    body = (
        "Mom is making lasagna for family dinner. "
        "My sister Jane Smith said she'd help our family this weekend."
    )
    ents = extract_entities(_msg(body))
    persons = _persons(ents)
    assert "Jane Smith" in persons, persons
    # 'Mom' bare should NOT be extracted as a person (bare relation word).
    assert "Mom" not in persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case1b_bare_relation_word_not_a_person():
    """Even if spaCy mistags a relation word as PERSON, the stoplist drops it."""
    body = "Mom is making lasagna for family dinner."
    ents = extract_entities(_msg(body))
    assert _persons(ents) == set()
    # But the relation entities still emit so the materializer can find it.
    assert "mom" in _relations(ents)
    assert "family" in _relations(ents)


# ---------------------------------------------------------------------------
# Case 2: bulk message returns empty list
# ---------------------------------------------------------------------------


def test_case2_bulk_message_returns_empty():
    body = (
        "Mom is making lasagna for family dinner. "
        "Jane Smith said she'd help our family this weekend."
    )
    msg = _msg(body, is_bulk=True)
    assert extract_entities(msg) == []


# ---------------------------------------------------------------------------
# Case 3: PERSON without relation word in sentence -> NOT extracted
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case3_no_relation_word_drops_person():
    body = "John Doe will present at the meeting."
    ents = extract_entities(_msg(body))
    assert "John Doe" not in _persons(ents)


# ---------------------------------------------------------------------------
# Case 4: relation word in a DIFFERENT sentence -> still no extract
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case4_relation_word_in_different_sentence_drops_person():
    body = "John Doe will present. The meeting is family-friendly."
    ents = extract_entities(_msg(body))
    assert "John Doe" not in _persons(ents)


# ---------------------------------------------------------------------------
# Case 5: newsletter byline -> dropped without relation word
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case5_byline_without_relation_is_dropped():
    body = "Aaron Holmes published an article on the state of AI."
    ents = extract_entities(_msg(body))
    assert "Aaron Holmes" not in _persons(ents)


# ---------------------------------------------------------------------------
# Case 6: company suffix in stoplist; person without relation word
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case6_company_suffix_and_no_relation():
    body = "Substack Inc was founded in 2017 by Hamish McKenzie."
    ents = extract_entities(_msg(body))
    persons = _persons(ents)
    # Substack Inc must not appear (company suffix stoplist).
    assert "Substack Inc" not in persons
    # Hamish McKenzie also must not appear — no relation word in the sentence.
    assert "Hamish McKenzie" not in persons


# ---------------------------------------------------------------------------
# Case 7: From-header display name + body with relation context
# ---------------------------------------------------------------------------


def test_case7_sender_name_and_body_extraction():
    body = "Hey, my mom Jane visited the lake this weekend with our family."
    msg = _msg(
        body,
        from_addr="Jane Smith <jane.smith@gmail.com>",
    )
    ents = extract_entities(msg)
    persons = _persons(ents)
    emails = _emails(ents)
    assert "Jane Smith" in persons, persons
    assert "jane.smith@gmail.com" in emails, emails


# ---------------------------------------------------------------------------
# Case 8: emails extract regardless of relation context
# ---------------------------------------------------------------------------


def test_case8_emails_extract_without_relation_gate():
    body = "Please reach out to bob@example.com about the Q4 roadmap."
    ents = extract_entities(_msg(body))
    assert "bob@example.com" in _emails(ents)
    # No person extracted (no relation word).
    assert _persons(ents) == set()


def test_case8b_emails_from_to_headers_extract():
    body = "(empty)"
    msg = _msg(
        body,
        from_addr="Alice <alice@example.com>",
        to_addr="bob@example.com",
    )
    ents = extract_entities(msg)
    emails = _emails(ents)
    assert "alice@example.com" in emails
    assert "bob@example.com" in emails


# ---------------------------------------------------------------------------
# Case 9: realistic Gmail body — quote stripping is the loop's job, the
# extractor processes whatever body_clean it's handed
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_case9_realistic_body_with_quote_already_stripped():
    """The loop strips quotes BEFORE calling the extractor. This test passes
    a body_clean with quote markers already stripped to verify the extractor
    handles a Gmail-shaped quoted-reply body without confusing the NER pass.

    Pass 7A: vocative 'Mom,' no longer admits any PERSON in the same
    sentence — the relation phrase has to be positioned so the name
    follows it within the adjacency window. Sarah / Mia / Leo are far
    from any anchor and become acceptable false-negatives; Carol still
    qualifies via the honorific 'Aunt Carol' path."""
    body = (
        "Mom, count us in - me, Sarah, Mia, Leo. Should we bring the pie or "
        "are you and Aunt Carol handling dessert?"
    )
    ents = extract_entities(_msg(body))
    persons = _persons(ents)
    # Sentence 2 has the honorific 'Aunt Carol' PERSON (the "Aunt" prefix
    # is stripped during cleanup, leaving just "Carol").
    assert "Carol" in persons
    # Vocative-only names (no possessive phrase binding them) are
    # intentionally not extracted under the adjacency rule.
    assert "Sarah" not in persons
    assert "Mia" not in persons
    assert "Leo" not in persons


# ---------------------------------------------------------------------------
# Honorific stripping
# ---------------------------------------------------------------------------


def test_aunt_prefix_stripped_from_person_span():
    """spaCy regularly tags "Aunt Carol" as a single PERSON span; we strip
    the relation-word prefix so the canonical label is just "Carol"."""
    body = "Aunt Carol said she'd come if Jane drives. Hi mom."
    ents = extract_entities(_msg(body))
    persons = _persons(ents)
    assert "Carol" in persons
    # The unstripped form should NOT also appear.
    assert "Aunt Carol" not in persons


# ---------------------------------------------------------------------------
# Single-token name gets low confidence
# ---------------------------------------------------------------------------


def test_single_token_person_marked_low_confidence():
    body = "Tell my granddaughter Mia I have those colored pencils."
    ents = extract_entities(_msg(body))
    mia = [e for e in ents if e.kind == "person" and e.label == "Mia"]
    assert mia, ents
    assert mia[0].confidence == "low"


def test_two_token_person_marked_high_confidence():
    body = "My sister Jane Smith is visiting our family next weekend."
    ents = extract_entities(_msg(body))
    jane = [e for e in ents if e.kind == "person" and e.label == "Jane Smith"]
    assert jane, ents
    assert jane[0].confidence == "high"


def test_sender_name_marked_medium_confidence():
    msg = _msg(
        body="(no relation context in body — sender header only.)",
        from_addr="Bob Smith <bob@example.com>",
    )
    ents = extract_entities(msg)
    bob = [e for e in ents if e.kind == "person" and e.label == "Bob Smith"]
    assert bob, ents
    assert bob[0].confidence == "medium"


# ---------------------------------------------------------------------------
# Non-person sender names are NOT extracted as persons
# ---------------------------------------------------------------------------


def test_org_sender_name_not_extracted_as_person():
    msg = _msg(
        body="Your shipment is on the way.",
        from_addr="Park Day School <office@parkdayschool.org>",
    )
    ents = extract_entities(msg)
    assert _persons(ents) == set()


def test_subject_relation_word_unlocks_body_persons():
    """Subjects are intentionally topical — when the subject contains a
    POSSESSIVE-GROUNDED relation phrase, body PERSONs pass the gate even
    if their own sentence doesn't have one. Pass 6B tightened the subject
    rule to require the same possessive grounding as body sentences;
    "My family dinner" qualifies, bare "Family dinner" no longer does."""
    msg = _msg(
        body="Bob is grilling. We'll be at the house around 4pm.",
        subject="My family dinner this Sunday",
    )
    persons = _persons(extract_entities(msg))
    assert "Bob" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_subject_without_relation_keeps_strict_gate():
    """Mirror of the case above: an ordinary subject leaves the per-sentence
    rule in force. 'John Doe will present.' under subject 'Q4 OKR review on
    Tuesday' must NOT extract John Doe."""
    msg = _msg(
        body="John Doe will present.",
        subject="Q4 OKR review on Tuesday",
    )
    assert "John Doe" not in _persons(extract_entities(msg))


def test_single_token_sender_name_not_extracted():
    """'Substack <ben@stratechery.example>' — single-token display name, not
    a personal name structurally."""
    msg = _msg(
        body="(newsletter body.)",
        from_addr="Substack <ben@stratechery.example>",
    )
    ents = extract_entities(msg)
    assert _persons(ents) == set()


# ---------------------------------------------------------------------------
# Strengthened gate: reject non-person spans that spaCy mistags as PERSON.
#
# These tests were added after a real-Gmail run produced 25 "family members"
# including obvious non-people (bitcoin, clinton-st, corto-cafe,
# morrison-cohen, email, healthy, schedule, subscribe, etc.). Each test
# pins one of the new rejection rules.
# ---------------------------------------------------------------------------


def test_rejects_common_noun_singletons():
    """Rule 3 — single-token title-cased common nouns die in the stoplist
    regardless of POS tag, because spaCy will happily call "Email" PERSON
    when it appears as a subject-line salutation."""
    body = (
        "Hi mom, my Email Schedule for the family is busy. "
        "The Healthy choice for our family is salad."
    )
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "Email" not in person_names
    assert "Schedule" not in person_names
    assert "Healthy" not in person_names


def test_rejects_org_in_multi_token():
    """Rule 5 / 5b — multi-token PERSON spans whose last token is an
    org-indicator OR whose sentence describes them with "as a law firm"
    must be rejected. Body has a relation word so the family gate is open."""
    body = (
        "My mom recommended Morrison Cohen as a law firm. "
        "My family and I met at Corto Cafe."
    )
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "Morrison Cohen" not in person_names
    assert "Corto Cafe" not in person_names


def test_keeps_real_surname_cohen():
    """Counterpoint to test_rejects_org_in_multi_token — Cohen is a real
    surname, and "David Cohen" must pass even though Cohen also appears in
    law-firm names. The disambiguator is the absence of "as a law firm"
    context."""
    body = "My brother David Cohen is visiting our family this weekend."
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "David Cohen" in person_names, person_names


def test_rejects_streetname():
    """Rule 5 — last token "St" is an org-indicator. Killing this means
    street names stop appearing in the family list."""
    body = "My family lives at Clinton St near the park."
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "Clinton St" not in person_names


def test_rejects_crypto_assets():
    """Rule 3 — Bitcoin and Ethereum are in the COMMON_NOUN_STOPLIST. They
    sometimes get tagged PROPN (Rule 2 won't catch them) and sometimes get
    a PERSON label, so the explicit stoplist is the load-bearing gate."""
    body = "My kid asked about Bitcoin and Ethereum for his birthday."
    ents = extract_entities(_msg(body))
    names_lower = {e.label.lower() for e in ents if e.kind == "person"}
    assert "bitcoin" not in names_lower
    assert "ethereum" not in names_lower


def test_rejects_acronyms():
    """Rule 6 — 4+ consecutive capital letters anywhere in the span makes it
    an acronym, not a personal name."""
    body = "The CEO of IBM is my dad. Our family loves the IBM culture."
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "IBM" not in person_names


def test_keeps_normal_family_names():
    """Smoke-test that the new gates don't break the happy path."""
    body = "My mom Jane Smith and my brother Bob Smith came over for family dinner."
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "Jane Smith" in person_names, person_names
    assert "Bob Smith" in person_names, person_names


def test_rejects_propn_less_spans():
    """Rule 2 — a PERSON span whose tokens are all NOUN/ADJ/VERB (not a
    single PROPN) is a mistag. "Email" alone is tagged PERSON by spaCy with
    POS=NOUN; the PROPN gate kills it independent of the common-noun
    stoplist."""
    # We can verify Rule 2 by checking that even if a candidate slipped
    # past the stoplist, the POS gate would still reject it. Construct a
    # body with a relation word and a NOUN-tagged "Email" span.
    body = "Hi family, my Email is busy today."
    ents = extract_entities(_msg(body))
    assert "Email" not in _persons(ents)


def test_rejects_very_short_names():
    """Rule 6 — fewer than 2 characters is noise."""
    # spaCy rarely returns single-char PERSON spans, but the stoplist
    # guards against any path (sender-name / future code) that could.
    from pi_email.entities import _in_stoplist
    assert _in_stoplist("X") is True
    assert _in_stoplist("") is True
    assert _in_stoplist("Bo") is False  # 2 chars survives; canonicalize handles len-3 cutoff


def test_rejects_non_letter_heavy_names():
    """Rule 6 — 50%+ non-letter characters indicates noise (URL fragments
    or punctuation gunk). Some inputs also hit other gates (all-caps single
    token, all-lowercase) — the contract is "rejected somehow", not "rejected
    by this specific clause"."""
    from pi_email.entities import _in_stoplist
    assert _in_stoplist("....x") is True       # 1 letter / 5 chars = 0.2 -> reject
    assert _in_stoplist("X-Y-Z") is True       # also all-caps single token -> reject
    assert _in_stoplist("Jane Smith") is False  # plenty of letters


def test_rule4_rejects_span_also_labeled_org():
    """Rule 4 — if the same span text appears as ORG / GPE / LOC elsewhere
    in the doc, the PERSON tag is suspect. We construct a body where
    "Bitcoin" might appear once with the PERSON-ish tag and once with the
    ORG tag; the overlap check drops it. The COMMON_NOUN_STOPLIST already
    kills Bitcoin, so we use a name spaCy disagrees with itself on."""
    # The "Bitcoin and Ethereum" sequence is reliably tagged ORG by
    # en_core_web_sm. Bitcoin alone elsewhere in the same body still
    # appears under that ORG span — the cross-label check catches it.
    body = (
        "My family asked about Bitcoin and Ethereum. "
        "Bitcoin is what my kid wants for his birthday."
    )
    ents = extract_entities(_msg(body))
    names_lower = {e.label.lower() for e in ents if e.kind == "person"}
    assert "bitcoin" not in names_lower
    assert "ethereum" not in names_lower


def test_rejects_org_first_token_park_avenue():
    """Rule 5 — first token "Park" combined with a multi-token span signals
    a street / address rather than a person."""
    body = "My mom said Park Avenue Smith is the address near our family home."
    ents = extract_entities(_msg(body))
    person_names = _persons(ents)
    assert "Park Avenue Smith" not in person_names


# ---------------------------------------------------------------------------
# Round-3.5: sender-name ORG token blocklist (Fix #2)
# ---------------------------------------------------------------------------
#
# Real-Gmail run leaked "Google Cloud Platform" as a person entity because:
#   1. The bulk-mail filter never fired (the cloudplatform-noreply@google.com
#      sender doesn't ship a `List-Unsubscribe` header on every notification).
#   2. The sender-display-name path treated "Google Cloud Platform" as a
#      structurally-valid 3-token name and spaCy didn't tag it ORG at the
#      header-only level.
#
# Fix: token blocklist on the display name. Any sender display name containing
# a known org / role / venue token (Cloud, Platform, School, Workshops, ...)
# is rejected regardless of is_bulk. Tests cover both bulk-flag settings to
# document that the rejection is independent of the upstream bulk filter.


def test_sender_org_blocked_google_cloud_platform_not_bulk():
    """The bug: cloudplatform-noreply@google.com slips past the bulk filter
    yet the display name 'Google Cloud Platform' clearly isn't a person.
    With the org-token blocklist, no PERSON entity is emitted from the
    sender header even when is_bulk is False."""
    msg = _msg(
        body="My family loves the GKE dashboard.",
        from_addr="Google Cloud Platform <cloudplatform-noreply@google.com>",
        is_bulk=False,
    )
    ents = extract_entities(msg)
    assert "Google Cloud Platform" not in _persons(ents)


def test_sender_org_blocked_google_cloud_platform_bulk():
    """Belt-and-suspenders: bulk-flagged messages already return [] at the
    top of extract_entities, but we pin the contract — even with is_bulk=True
    the org sender doesn't appear."""
    msg = _msg(
        body="My family loves the GKE dashboard.",
        from_addr="Google Cloud Platform <cloudplatform-noreply@google.com>",
        is_bulk=True,
    )
    ents = extract_entities(msg)
    assert _persons(ents) == set()


def test_sender_org_blocked_sonoma_art_school():
    """'Sonoma Art School' — token 'School' is in the blocklist."""
    msg = _msg(
        body="Hi mom, the art camp confirms my kid's enrollment.",
        from_addr="Sonoma Art School <kelly@sonomaartschool.org>",
    )
    ents = extract_entities(msg)
    assert "Sonoma Art School" not in _persons(ents)


def test_sender_org_blocked_workshops_team():
    """Multi-token sender name with a venue / role token (Workshops, Team)."""
    msg = _msg(
        body="Hi mom, signing up for the camp.",
        from_addr="Summer Workshops Team <workshops@example.org>",
    )
    ents = extract_entities(msg)
    assert "Summer Workshops Team" not in _persons(ents)


def test_real_sender_name_still_extracted():
    """Smoke test — Jane Smith doesn't contain any org token, so the
    happy path still extracts the sender-name PERSON."""
    msg = _msg(
        body="Hi mom, looking forward to the family weekend.",
        from_addr="Jane Smith <jane@example.com>",
    )
    persons = _persons(extract_entities(msg))
    assert "Jane Smith" in persons


# ---------------------------------------------------------------------------
# Round-3.5: HTML/CSS doesn't leak into entity names (Fix #1)
# ---------------------------------------------------------------------------
#
# The fix lives in loop.py (it stopped clobbering the cleaned body_clean
# that GmailSearcher pre-computes). At the entity-extractor level we pin
# the contract: when body_clean has already been cleaned, no HTML / CSS
# tokens land in the person-entity set even if `body` retains the raw HTML.


def test_entity_extractor_uses_clean_body_not_raw():
    """body has font-family CSS + Arial; body_clean has the cleaned version.
    The extractor must read body_clean and never see 'Arial'.

    This pins the public contract `extract_entities` honors `body_clean`
    when present — the inverse-version of the bug that landed in production:
    a caller had stomped body_clean with a strip-quotes-only pass, so the
    NER saw HTML and "Arial" landed as PERSON."""
    body_raw = (
        '<html><body><a style="font-family:-apple-system, Helvetica, '
        'Arial, sans-serif">Hi mom, family dinner Sunday.</a></body></html>'
    )
    body_cleaned = "Hi mom, family dinner Sunday."
    msg = Message(
        message_id="m1",
        thread_id="m1",
        from_addr="alice@example.com",
        to_addr="me@example.com",
        subject="dinner",
        date="2026-01-01",
        body=body_raw,
        body_clean=body_cleaned,
        source_path=Path("/tmp/fake.md"),
    )
    persons = _persons(extract_entities(msg))
    assert "Arial" not in persons
    assert "Helvetica" not in persons


def test_entity_extractor_falls_back_to_body_when_body_clean_none():
    """The fallback case — fixture messages don't pre-populate body_clean,
    so extract_entities must read `body` and still work."""
    msg = Message(
        message_id="m1",
        thread_id="m1",
        from_addr="alice@example.com",
        to_addr="me@example.com",
        subject="dinner",
        date="2026-01-01",
        body="Hi mom, my sister Jane Smith is coming for family dinner.",
        body_clean=None,
        source_path=Path("/tmp/fake.md"),
    )
    persons = _persons(extract_entities(msg))
    assert "Jane Smith" in persons


# ---------------------------------------------------------------------------
# Round-5: surviving FPs from Run-4 (Edge Esmeralda, Trump, Tim Draper,
# Coinbase Wealth, image-CID emails, Ryan Rigney Marketing)
# ---------------------------------------------------------------------------
#
# Diagnosis from /tmp/pi-email-bigrun4/run.log:
#   * `person:Edge Esmeralda`  -> body NER (iter 11 parent)
#   * `person:Coinbase Wealth` -> body NER (iter 18 parent)
#   * `person:Tim Draper`      -> body NER (iter 19 parent)
#   * `person:Edge`            -> body NER, single-token (iter 20 parent)
#   * `email:image001.jpg@01dce229.53871350` -> email regex (iter 30 parent)
#   * `person:Ryan Rigney Marketing` -> body NER, role-tail bled in.
#
# Fixes pinned by the tests below all live in entities.py:
#   1. _EVENT_PLACE_TOKENS + multi-token any-position check in _in_stoplist.
#   2. _PUBLIC_FIGURE_STOPLIST + single/whole-phrase check in _in_stoplist.
#   3. _ROLE_TAIL_TOKENS strip in _clean_person_span.
#   4. _is_cid_email filter in the email extraction loop.


def test_rejects_edge_esmeralda():
    """Body NER mistags 'Edge Esmeralda' as PERSON. Both tokens are in
    `_EVENT_PLACE_TOKENS`, so the any-position multi-token check rejects."""
    body = (
        "I went to Edge Esmeralda last weekend with my family. "
        "It was a fun community."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Edge Esmeralda" not in persons, persons
    assert "Edge" not in persons, persons
    assert "Esmeralda" not in persons, persons


def test_rejects_trump_single_token():
    """A bare 'Trump' tagged PERSON in a sentence with a relation word.
    Single-token public-figure list catches it."""
    body = "My dad mentioned Trump in the news today over family dinner."
    persons = _persons(extract_entities(_msg(body)))
    assert "Trump" not in persons, persons


def test_rejects_tim_draper_multi_token():
    """'Tim Draper' tagged PERSON in a relation-word sentence. Multi-token
    whole-phrase lookup against the public-figure list catches it."""
    body = "Tim Draper invested in my brother's company last family year."
    persons = _persons(extract_entities(_msg(body)))
    assert "Tim Draper" not in persons, persons


def test_rejects_coinbase_wealth():
    """Brand name that looks like a personal name. Multi-token whole-phrase
    match against the public-figure / brand list catches it."""
    body = (
        "Hi mom, Coinbase Wealth sent me a note about my family account "
        "balances today."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Coinbase Wealth" not in persons, persons


def test_rejects_image_cid_email():
    """Embedded image Content-ID like image001.jpg@01dce229.53871350 looks
    like a real email to the regex; the CID filter drops it."""
    body = (
        "Inline image follows: <image001.jpg@01dce229.53871350>. "
        "Real contact bob@example.com next to family dinner Sunday — love, mom."
    )
    ents = extract_entities(_msg(body))
    emails = _emails(ents)
    assert "image001.jpg@01dce229.53871350" not in emails, emails
    # The real address still extracts.
    assert "bob@example.com" in emails


def test_role_tail_strip_in_extraction():
    """'Ryan Rigney Marketing' — last token 'Marketing' is in the role
    tail list. The cleaner peels it BEFORE the canonical lands in the
    entity stream, so the label is 'Ryan Rigney'."""
    body = (
        "Ryan Rigney Marketing sent this. My mom likes it. "
        "Bonus: my family appreciates the update."
    )
    persons = _persons(extract_entities(_msg(body)))
    # Must NOT include the role-tail-laden form.
    assert "Ryan Rigney Marketing" not in persons, persons
    # The cleaned name is the only acceptable form. We don't assert
    # presence of 'Ryan Rigney' positively because spaCy may not always
    # tag this exact span as PERSON, but the contract pinned by this
    # test is "if extracted, it's NOT the role-tail form".
    for name in persons:
        assert "Marketing" not in name.split(), (
            f"unexpected Marketing tail in {name!r}; persons={persons}"
        )


def test_keeps_real_person_names_starting_with_public_figure_lastname():
    """A real surname that PREFIXES a public-figure name must still pass.
    The check is whole-phrase, not substring — so 'Jane Trumpington' is
    safe even though 'trump' is a public-figure entry."""
    body = (
        "My sister Jane Trumpington came over for our family dinner with mom."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane Trumpington" in persons, persons


def test_rejects_role_tail_in_sender_name():
    """The sender-display path also runs through `_clean_person_span`.
    A sender 'Ryan Rigney Marketing <ryan@example.com>' should extract
    as 'Ryan Rigney' (tail stripped) — not the full role-tail form."""
    msg = _msg(
        body="Hi mom, family update.",
        from_addr="Ryan Rigney Marketing <ryan@example.com>",
    )
    persons = _persons(extract_entities(msg))
    assert "Ryan Rigney Marketing" not in persons, persons


def test_cid_email_filter_unit():
    """Unit-level pin on `_is_cid_email` — covers the four shapes the
    filter recognizes plus a regression case for a real email."""
    from pi_email.entities import _is_cid_email
    # CID shapes — all must return True.
    assert _is_cid_email("image001.jpg@01dce229.53871350") is True
    assert _is_cid_email("image002.png@hostmachine.example") is True
    assert _is_cid_email("image001_abc123@foo.example") is True
    assert _is_cid_email("part1.04050608@example.com") is True
    assert _is_cid_email("anything@deadbeef.cafebabe") is True
    # Real emails — must return False.
    assert _is_cid_email("bob@example.com") is False
    assert _is_cid_email("jane.smith@gmail.com") is False
    assert _is_cid_email("image@somedomain.com") is False  # bare "image" allowed


# ---------------------------------------------------------------------------
# Pass 6B: possessive-grounded relation gate + business-idiom blocklist +
# strict partner/kids rules + message-level business suppression.
#
# Run-5 diagnosis (/tmp/pi-email-bigrun5/run.log):
#   * "Family Office Partner Mark Leverette" -> Mark (relation: partner, family)
#   * "Brian Flynn (Tally team) is our partner" -> Brian (relation: partner)
#   * "DUNA, 7 Family Office conferences" -> conference attendee (relation: family)
#   * "Hi Tyler" + elsewhere "kids" -> Tyler
#   * "Morrison Cohen as a partner" -> Morrison Cohen (law firm)
#
# Pass 6B tightens the gate to require possessive grounding (my/our/your/
# his/her/their/the/'s-genitive) within 1-3 tokens of the relation word,
# blocks a set of business idioms outright, and applies stricter rules
# for the abused partner/kids tokens.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_family_office_does_not_count():
    """'family office' is a business idiom — Mark should NOT be extracted
    even though 'office' and 'family' co-occur with a PERSON span."""
    body = "We met Mark Leverette at a family office event."
    persons = _persons(extract_entities(_msg(body)))
    assert "Mark Leverette" not in persons, persons
    assert "Mark" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_general_partner_does_not_count():
    """'general partner' is a business idiom — Tim should NOT be extracted
    even though 'partner' is in the relation vocabulary."""
    body = "Tim Draper Jr is a general partner at the fund."
    persons = _persons(extract_entities(_msg(body)))
    assert "Tim" not in persons, persons
    assert "Tim Draper Jr" not in persons, persons


def test_possessive_my_mom_does_count():
    """'my mom Jane' is the canonical possessive-grounded form — Jane
    must be extracted."""
    body = "My mom Jane visited the lake."
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane" in persons, persons


def test_possessive_our_kids_does_count():
    """'our kids' satisfies the strict kids rule — Sarah in the same
    sentence is extracted."""
    body = "Our kids love Sarah."
    persons = _persons(extract_entities(_msg(body)))
    assert "Sarah" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_bare_partner_does_not_count():
    """Bare 'partner' is almost always business — Bob NOT extracted
    even though 'partner' is in the relation vocabulary."""
    body = "Bob Andersen is a partner at the firm."
    persons = _persons(extract_entities(_msg(body)))
    assert "Bob Andersen" not in persons, persons
    assert "Bob" not in persons, persons


def test_life_partner_does_count():
    """'My life partner' satisfies the strict partner rule via the
    "life partner" pre-token — Alex is extracted."""
    body = "My life partner Alex Johnson came over."
    persons = _persons(extract_entities(_msg(body)))
    assert "Alex Johnson" in persons or "Alex" in persons, persons


def test_kids_in_news_does_not_count():
    """Bare sentence-start 'Kids' (e.g. 'Kids these days') is generic
    English usage, not a family signal — Marcus NOT extracted."""
    body = "Kids these days, including Marcus Aurelius, are different."
    persons = _persons(extract_entities(_msg(body)))
    assert "Marcus" not in persons, persons
    assert "Marcus Aurelius" not in persons, persons


def test_message_dominated_by_business_jargon_keeps_grounded_person():
    """The suppress rule applies only when there's NO clean
    possessive-grounded relation. With 5+ sender-org tokens AND a clean
    'my wife Jane' phrase, Jane IS extracted (the qualified-relation
    sentence wins over the message-level business signal)."""
    body = (
        "The team at our conference platform service marketing newsletter "
        "group reached out about a partnership opportunity. My wife Jane "
        "is also working on this."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_message_dominated_by_business_jargon_with_no_grounding_suppresses():
    """The mirror: 3+ business idioms AND no body sentence is qualified
    AND subject is qualified — the subject-fallback unlock is suppressed,
    so Bob does NOT bleed in from a jargon-only sentence."""
    msg = _msg(
        body=(
            "Bob Andersen leads our general partner team. "
            "Our limited partner roster is growing. "
            "Our managing partner approved the deal."
        ),
        subject="My family update for the quarter",
    )
    persons = _persons(extract_entities(msg))
    assert "Bob Andersen" not in persons, persons
    assert "Bob" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_kelly_ebeling_without_adjacent_relation_is_dropped():
    """Pass 7A: 'her kids' qualifies as a strict kids phrase but it
    appears AFTER 'Kelly Ebeling' in the sentence, so adjacency binding
    does not admit Kelly. Pre-7A this passed because sentence-level
    loose binding admitted every PERSON in a qualifying sentence — now
    it's an acceptable false-negative. Re-phrasing the body to 'My
    friend's wife Kelly Ebeling' or 'her kids Kelly Ebeling' would
    re-qualify Kelly."""
    body = (
        "My friend Kelly Ebeling reached out — turns out her kids go to "
        "school with mine."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Kelly Ebeling" not in persons, persons
    assert "Kelly" not in persons, persons


def test_kelly_ebeling_with_adjacent_relation_phrase_extracts():
    """Mirror of the test above: when the relation phrase is positioned
    BEFORE the name (within the adjacency window), Kelly DOES qualify."""
    body = "My sister Kelly Ebeling visited the lake this weekend."
    persons = _persons(extract_entities(_msg(body)))
    assert "Kelly Ebeling" in persons or "Kelly" in persons, persons


def test_genitive_apostrophe_s_grounding():
    """'Bob's wife' uses 's-genitive as the possessive marker — Jane
    is extracted."""
    body = "Bob's wife Jane came over."
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_law_firm_partner_phrase_blocked():
    """Direct regression test for the Run-5 'law partner' shape:
    'law partner' is in the business-idiom blocklist, so the entire
    sentence is disqualified regardless of possessive context — Morrison
    Cohen NOT extracted. (The complementary "as a law firm" disambiguator
    in _ORG_CONTEXT_RE remains for spans tagged ORG explicitly.)"""
    body = (
        "My mom recommended Morrison Cohen as a law partner for the case."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Morrison Cohen" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_brian_flynn_our_partner_no_longer_admits():
    """Pass 7A: pre-7A 'Brian Flynn is our partner' admitted Brian Flynn
    under sentence-level loose binding for the strict-partner pattern.
    Under adjacency binding the 'our partner' anchor falls AFTER 'Brian
    Flynn' in the token stream, so it doesn't bind to the preceding
    name — this is the precision win that closes the Run-6 'partner'
    FP family (Jan Hladonik / Ryan Rigney etc.)."""
    body = "Brian Flynn is our partner on this project."
    persons = _persons(extract_entities(_msg(body)))
    assert "Brian Flynn" not in persons, persons
    assert "Brian" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_sentence_qualification_helper_unit():
    """Unit-level pin on `_sentence_has_qualified_relation` covering the
    five qualification paths and the business-idiom blocker."""
    from pi_email.entities import _sentence_has_qualified_relation as q
    # Lax possessive
    assert q("my mom is here") is True
    assert q("our family had dinner") is True
    assert q("Bob's wife came over") is True
    assert q("her mother-in-law arrived") is True
    # Strict partner
    assert q("my partner Alex came over") is True
    assert q("we are life partner") is True  # "life partner" qualifies
    assert q("Acme is a partner") is False  # bare partner fails
    # Strict kids
    assert q("the kids are here") is True
    assert q("her kids love it") is True
    assert q("kids these days") is False
    # Vocative
    assert q("Mom, dinner is ready.") is True
    assert q("Dad, look at this.") is True
    # Honorific + Title-Case name
    assert q("Aunt Carol said hello.") is True
    assert q("Grandma Helen visited.") is True
    # Business-idiom block
    assert q("my family office event") is False
    assert q("our general partner deal") is False
    assert q("we are family-friendly") is False
    # Empty / no relation
    assert q("") is False
    assert q("nothing to see here") is False


# ---------------------------------------------------------------------------
# Pass 7A: entity-level adjacency binding
#
# A possessive-relation phrase binds ONLY to the next PERSON within
# `_ADJACENCY_WINDOW_TOKENS` (default 5), not to every PERSON in the
# sentence. Plus a shadowing rule: an anchor's reach is cut off by an
# intervening my/our/your/his/her/their token before the candidate.
#
# Run-6 produced 5 putative family members from one sentence ("...intro
# with my wife Jana, she's working at Double Zero with Austin..."). Only
# Jana was correct — Austin and "Double Zero" were both bound to "my
# wife" under sentence-level loose binding. 7A makes the relation
# phrase bind locally.
# ---------------------------------------------------------------------------


def test_adjacent_binding_my_wife_jana_only_jana():
    """Canonical Run-6 case: 'my wife Jana, who works with Austin at
    Double Zero'.

    Pass 8A history:
      * Pass 7A used N=5 strict window — Austin (distance 5) excluded.
        That precision win came at the cost of catastrophic recall
        regression (Run 7 produced 0 family members).
      * Pass 8A loosened to N=8 to restore recall. Austin (distance 5)
        is now ADMITTED at this layer; the downstream LLM final-judge
        in materializer.py (Pass 8B) is the new defense — it sees the
        rough candidate set and asks an LLM whether each name reads as
        family. "Double Zero" stays excluded via the
        `_PUBLIC_FIGURE_STOPLIST` known-org list."""
    body = (
        "I want to introduce my wife Jana, who works with Austin at Double Zero."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Jana" in persons, persons
    # Austin now leaks at the extractor layer — LLM judge catches it.
    assert "Double Zero" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_adjacent_binding_genitive_bobs_wife_jane():
    """'Bob's wife Jane' — Jane qualifies via 's-genitive grounding;
    Bob is BEFORE the anchor so does not qualify."""
    body = "Bob's wife Jane visited."
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane" in persons, persons
    assert "Bob" not in persons, persons


def test_adjacent_binding_multiple_persons_my_sons_bob_and_carl():
    """'My sons Bob and Carl' — both qualify since neither has an
    intervening shadowing possessive between 'sons' and the name."""
    body = "My sons Bob and Carl are home this week."
    persons = _persons(extract_entities(_msg(body)))
    assert "Bob" in persons, persons
    assert "Carl" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_adjacent_binding_my_son_bob_and_his_friend_carl():
    """'My son Bob and his friend Carl' — Bob qualifies; Carl is
    shadowed by the intervening 'his' possessive and 'friend' is not a
    relation word, so Carl does NOT qualify."""
    body = "My son Bob and his friend Carl are here today."
    persons = _persons(extract_entities(_msg(body)))
    assert "Bob" in persons, persons
    assert "Carl" not in persons, persons


def test_org_shape_rejected():
    """'My wife works at Double Zero.' — 'Double Zero' is in the
    org-shape known stoplist AND is positioned far from 'my wife', so it
    is not emitted as a person even though spaCy might tag it that way."""
    body = "My wife works at Double Zero with her old colleagues."
    persons = _persons(extract_entities(_msg(body)))
    assert "Double Zero" not in persons, persons


def test_bare_partner_does_not_qualify():
    """Bare 'partner' is business-by-default — Jan does NOT qualify
    even when 'My family' appears in a later sentence (different
    sentence; subject is unrelated)."""
    body = "Jan is a partner at the firm. My family enjoys his work."
    persons = _persons(extract_entities(_msg(body)))
    assert "Jan" not in persons, persons


def test_life_partner_qualifies():
    """'My life partner Alex' satisfies strict partner grounding —
    Alex qualifies via adjacency."""
    body = "My life partner Alex came to the picnic."
    persons = _persons(extract_entities(_msg(body)))
    assert "Alex" in persons, persons


def test_existing_my_mom_jane_smith_still_works():
    """Smoke test: the canonical 'my mom Jane Smith' phrase still
    extracts Jane Smith (adjacency window covers 'mom' -> 'Jane Smith'
    at distance 0)."""
    body = "My mom Jane Smith visited last weekend."
    persons = _persons(extract_entities(_msg(body)))
    assert "Jane Smith" in persons, persons


def test_honorific_aunt_carol_still_works():
    """Smoke test: the honorific 'Aunt Carol' path still qualifies
    (the relation token IS the honorific; the PERSON span starts on
    'Aunt' and the leading honorific is stripped during cleanup)."""
    body = "Aunt Carol made dinner."
    persons = _persons(extract_entities(_msg(body)))
    assert "Carol" in persons, persons


def test_bare_partner_does_not_emit_relation_entity():
    """Pass 7A Fix 2: bare 'partner' no longer emits a kind='relation'
    entity. Without strict grounding (my/our/your/life/domestic +
    partner) the materializer never sees a 'partner' signal."""
    body = "Jan is a partner at the firm. Bob is a partner too."
    rels = _relations(extract_entities(_msg(body)))
    assert "partner" not in rels, rels


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_strict_partner_does_emit_relation_entity():
    """Mirror: with strict grounding the relation entity DOES emit."""
    body = "My life partner Alex visited."
    rels = _relations(extract_entities(_msg(body)))
    assert "partner" in rels, rels


def test_bare_kids_does_not_emit_relation_entity():
    """Pass 7A Fix 2: bare 'kids' (without my/our/your/his/her/their/the
    grounding) does not emit a 'kids' relation entity."""
    body = "Kids these days don't know how to write a proper email."
    rels = _relations(extract_entities(_msg(body)))
    assert "kids" not in rels, rels


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_grounded_kids_does_emit_relation_entity():
    """'her kids' — strict pattern fires — 'kids' relation entity is
    emitted."""
    body = "Sarah said her kids loved the gift."
    rels = _relations(extract_entities(_msg(body)))
    assert "kids" in rels, rels


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_adjacency_window_does_not_bind_distant_persons():
    """Pass 8A: a PERSON beyond the N=8 window stays outside.

    Token layout of the body below (spaCy en_core_web_sm):
      My(0) wife(1) Jana(2) laughed(3) and(4) then(5) mentioned(6) that(7)
      a(8) co-worker(9) of(10) hers(11) named(12) Charlie(13) called(14) .(15)

    Anchor for "my wife" lands at "Jana" (token 2). Charlie sits at
    distance 11 (>=8), so Charlie does NOT qualify. Also note: "hers"
    (a possessive in `_SHADOWING_POSSESSIVES`) appears between the
    anchor and Charlie, which independently shadows the binding."""
    body = (
        "My wife Jana laughed and then mentioned that a co-worker of hers "
        "named Charlie called."
    )
    persons = _persons(extract_entities(_msg(body)))
    assert "Jana" in persons, persons
    assert "Charlie" not in persons, persons


def test_two_anchors_two_people_both_qualify():
    """Two relation phrases anchor two separate names in the same
    sentence: 'My brother Bob and my sister Carol came over.' — both
    Bob (anchored by 'my brother') and Carol (anchored by 'my sister')
    qualify."""
    body = "My brother Bob and my sister Carol came over for dinner."
    persons = _persons(extract_entities(_msg(body)))
    assert "Bob" in persons, persons
    assert "Carol" in persons, persons


# ---------------------------------------------------------------------------
# Pass 8A: sender-aware sentence-level relaxation, loose mode, honorific
# without possessive
# ---------------------------------------------------------------------------


def test_user_sent_msg_falls_back_to_sentence_level():
    """When `from_addr` matches one of `user_emails`, the strict
    per-entity adjacency check is relaxed to sentence-level binding.
    The user themselves is making the claim — high precision signal.

    The body below would lose Austin under N=8 strict (anchor at
    distance 5 from Jana, Austin much further beyond the looser window
    AND past the comma), but with sentence-level relaxation Austin
    qualifies. Double Zero stays excluded via the public-figure
    stoplist; the LLM final-judge in Pass 8B filters Austin downstream."""
    body = (
        "Quick intro to my wife Jana, who works at Double Zero with Austin."
    )
    msg = _msg(body, from_addr="user@example.com")
    ents = extract_entities(msg, user_emails={"user@example.com"})
    persons = _persons(ents)
    assert "Jana" in persons, persons
    # Sentence-level binding (because user sent it) admits Austin too.
    assert "Austin" in persons, persons
    # The org-shape filter still applies — Double Zero is in the
    # public-figure stoplist regardless of binding strategy.
    assert "Double Zero" not in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_non_user_msg_uses_strict_adjacency():
    """Same body as above but `from_addr` is somebody ELSE — the strict
    per-entity adjacency gate applies. Jana stays; Austin should be
    dropped because the relation phrase doesn't bind across the comma+
    relative-clause distance under the adjacency check (he's beyond the
    N=8 window from the 'my wife' anchor in this body)."""
    body = (
        "Quick intro to my wife Jana, who works at Double Zero with our "
        "very good friend Austin."
    )
    msg = _msg(body, from_addr="other@example.com")
    ents = extract_entities(msg, user_emails={"user@example.com"})
    persons = _persons(ents)
    assert "Jana" in persons, persons
    # Strict adjacency: Austin sits beyond N=8 from the "my wife" anchor
    # (10+ tokens after "Jana"). No relaxation since the sender isn't us.
    assert "Austin" not in persons, persons


def test_extract_entities_user_emails_optional():
    """Default `user_emails=None` preserves the prior (Pass 7A) contract —
    no implicit relaxation, callers that don't supply the parameter see
    identical behavior to before."""
    body = "My wife Jane visited."
    # Same call without user_emails (default None).
    ents_default = extract_entities(_msg(body))
    assert "Jane" in _persons(ents_default)
    # Explicit None — same result.
    ents_none = extract_entities(_msg(body), user_emails=None)
    assert _persons(ents_none) == _persons(ents_default)


def test_honorific_no_possessive_aunt_carol_alone():
    """'Aunt Carol came over for dinner.' — the honorific path anchors
    AT the honorific token, so the PERSON span starting at "Aunt" is at
    distance 0 from the anchor. No possessive ('my'/'our'/etc.) needed."""
    body = "Aunt Carol came over for dinner."
    persons = _persons(extract_entities(_msg(body)))
    assert "Carol" in persons, persons


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_strict_false_loose_mode():
    """Loose mode (`strict=False`) widens the adjacency window to 15,
    so more candidates from a single sentence qualify. A sentence with
    several PERSON entities scattered across it admits all that NER tags
    as PERSON (provided per-name filters pass)."""
    body = (
        "Update from my sister Jane today: she went to the park with "
        "Michael and Sarah and Tom and Lisa."
    )
    persons_strict = _persons(extract_entities(_msg(body), strict=True))
    persons_loose = _persons(extract_entities(_msg(body), strict=False))
    # Loose mode admits strictly more candidates from the same sentence.
    assert persons_strict <= persons_loose, (persons_strict, persons_loose)
    assert len(persons_loose) >= len(persons_strict), (
        persons_strict, persons_loose,
    )
    # The strict-anchored case ("my sister Jane") still qualifies in both.
    assert "Jane" in persons_strict
    assert "Jane" in persons_loose
    # Loose mode picks up at least one of the trailing names that strict
    # adjacency dropped (Michael / Sarah / Tom / Lisa, depending on which
    # spaCy tags as PERSON in the doc).
    trailing = {"Michael", "Sarah", "Tom", "Lisa"}
    assert persons_loose & trailing, persons_loose


@pytest.mark.skip(reason="Removed: kinship-word gating")
def test_strict_false_skips_business_heavy_suppression():
    """Loose mode disables the business-heavy suppression, so a body
    laden with sender-org / business-idiom tokens does NOT block the
    subject-fallback unlock. The LLM judge filters the extra noise."""
    msg = _msg(
        body=(
            "Bob Andersen leads our general partner team. "
            "Our limited partner roster is growing. "
            "Our managing partner approved the deal."
        ),
        subject="My family update for the quarter",
    )
    persons_strict = _persons(extract_entities(msg, strict=True))
    persons_loose = _persons(extract_entities(msg, strict=False))
    # Strict mode suppresses (business-heavy + no body-sentence qualified):
    # Bob Andersen stays out.
    assert "Bob Andersen" not in persons_strict
    # Loose mode admits more — the subject-fallback unlock fires.
    assert len(persons_loose) >= len(persons_strict)


def test_extract_from_corpus_accepts_user_emails():
    """Plumbing check: `extract_from_corpus` accepts user_emails and
    forwards it to per-message extraction."""
    from pi_email.corpus import Corpus
    from pi_email.entities import extract_from_corpus

    corpus = Corpus()
    msg = _msg(
        "intro to my wife Jana, who works at Double Zero with Austin",
        message_id="m-user-1",
        from_addr="me@example.com",
    )
    corpus.add(msg)
    result = extract_from_corpus(
        corpus,
        message_ids=[msg.message_id],
        user_emails={"me@example.com"},
    )
    persons = {e.label for e in result.entities if e.kind == "person"}
    assert "Jana" in persons
    # Sender-aware relaxation admitted Austin.
    assert "Austin" in persons


