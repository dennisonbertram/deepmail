"""Unit tests for the OR-disjunction parser in searcher.py and the
token-bag dedupe pre-pass in frontier.py.

Both regression-cover bugs surfaced by the first live-LLM run (see PR notes):
queries like `mom OR dad OR mother OR father` were being treated as AND
across all literal tokens (so nothing matched), and the proposer was
flooding the frontier with verbose multi-token near-duplicates that the
0.95 SequenceMatcher threshold let through.
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.corpus import Message  # noqa: E402
from pi_email.frontier import Frontier, _token_bag  # noqa: E402
from pi_email.searcher import _eval, _parse  # noqa: E402


def _msg(subject: str = "", body: str = "", from_addr: str = "", to_addr: str = "") -> Message:
    return Message(
        message_id="m",
        thread_id="t",
        from_addr=from_addr,
        to_addr=to_addr,
        subject=subject,
        date="2026-01-01",
        body=body,
        source_path=Path("/tmp/none"),
    )


# ---------- OR parser ----------


def test_or_disjunction_simple():
    ast = _parse("mom OR dad")
    assert _eval(ast, _msg(body="Talking to mom about dinner."))
    assert _eval(ast, _msg(body="Dad fixed the sink."))
    assert not _eval(ast, _msg(body="Nothing relevant here."))


def test_or_is_case_insensitive():
    # Lowercase 'or' must also work — the live proposer's queries arrive
    # downstream lowercased.
    ast = _parse("mom or dad")
    assert _eval(ast, _msg(body="mom"))
    assert _eval(ast, _msg(body="dad"))
    assert not _eval(ast, _msg(body="brother"))


def test_and_binds_tighter_than_or():
    # `family OR mom dad` parses as `family OR (mom AND dad)`.
    ast = _parse("family OR mom dad")
    # message with just 'family' matches the OR-left.
    assert _eval(ast, _msg(body="Our whole family is here."))
    # message with both 'mom' and 'dad' but no 'family' matches the OR-right.
    assert _eval(ast, _msg(body="mom and dad are coming."))
    # message with just 'mom' should NOT match — needs both for the AND.
    assert not _eval(ast, _msg(body="just mom here"))
    # message with just 'dad' should NOT match either.
    assert not _eval(ast, _msg(body="just dad here"))


def test_or_combines_with_field_operators():
    ast = _parse("from:bob OR from:jane")
    assert _eval(ast, _msg(from_addr="bob@example.com", body="hi"))
    assert _eval(ast, _msg(from_addr="jane@example.com", body="hi"))
    assert not _eval(ast, _msg(from_addr="alice@example.com", body="hi"))


def test_or_combines_with_quoted_phrase():
    ast = _parse('"family dinner" OR holiday')
    assert _eval(ast, _msg(body="we had family dinner last night"))
    assert _eval(ast, _msg(body="holiday plans"))
    # Has both words but not as adjacent phrase.
    assert not _eval(ast, _msg(body="family was at the dinner table"))


def test_field_with_quoted_value():
    ast = _parse('from:"alice smith" OR subject:thanks')
    assert _eval(ast, _msg(from_addr="alice smith@example.com"))
    assert _eval(ast, _msg(subject="Re: thanks for dinner"))
    assert not _eval(ast, _msg(from_addr="bob@example.com"))


def test_empty_query_is_none():
    assert _parse("") is None
    assert _parse("   ") is None
    assert _parse("OR or OR") is None  # nothing but disjunctions → no atoms


# ---------- Token-bag dedupe ----------


def test_token_bag_normalizes_order_and_case():
    assert _token_bag("mom or dad") == _token_bag("dad OR mom")
    assert _token_bag("Mom OR DAD") == _token_bag("dad or mom")


def test_token_bag_strips_or_and_dedupes():
    # OR keyword (any case) stripped; duplicates collapse via set semantics.
    assert _token_bag("mom OR mom OR dad") == _token_bag("dad or mom")


def test_token_bag_distinguishes_distinct_words():
    # The two real frontier-flood queries from the previous live run differ
    # in {marriage,vows} vs {married,reception}: the bags MUST NOT collide
    # so the similarity check is what catches them.
    a = _token_bag("wedding or anniversary or engagement or marriage or vows or ceremony")
    b = _token_bag("wedding or anniversary or engagement or married or ceremony or reception")
    assert a != b


def test_frontier_dedupes_token_bag_permutations():
    logged: list[str] = []
    f = Frontier(on_log=logged.append)
    assert f.push("mom OR dad", score=0.7, parent_entity="SEED") is True
    f.pop()
    f.mark_ran("mom OR dad")
    # Same bag, different surface form — must be deduped.
    assert f.push("dad or mom", score=0.7, parent_entity="SEED") is False
    assert any("token-bag" in line for line in logged), logged


def test_frontier_dedupes_token_bag_against_pending():
    logged: list[str] = []
    f = Frontier(on_log=logged.append)
    assert f.push("alpha OR beta", score=0.5, parent_entity="SEED") is True
    # Pending heap should also catch the bag-equal dup.
    assert f.push("beta or alpha", score=0.5, parent_entity="SEED") is False
    assert any("token-bag" in line for line in logged), logged
