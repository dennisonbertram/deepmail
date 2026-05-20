"""Tests for entity canonicalization via embedding cosine similarity.

These use the real LocalEmbedder; the threshold (0.85) is calibrated to
the actual geometry of bge-base-en-v1.5 so a stubbed embedder would let
real bugs slip past. First run downloads the model; subsequent runs are
fast (LRU + sqlite cache).
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.embedder import LocalEmbedder  # noqa: E402
from pi_email.entities import Entity, canonicalize  # noqa: E402


def _people(*names: str) -> list[Entity]:
    return [Entity(kind="person", key=n.lower(), label=n) for n in names]


def test_canonicalize_merges_jane_and_jane_smith():
    """The headline use case: 'Jane' and 'Jane Smith' must collapse to the
    longer name; ditto for Bob / Bob Smith. Two distinct people stay
    distinct."""
    emb = LocalEmbedder()
    ents = _people("Jane Smith", "Jane", "Bob Smith", "Bob")
    out = canonicalize(ents, emb)
    assert out == {
        "Jane Smith": "Jane Smith",
        "Jane": "Jane Smith",
        "Bob Smith": "Bob Smith",
        "Bob": "Bob Smith",
    }, out


def test_canonicalize_does_not_merge_across_kinds():
    """Person 'Jane Smith' must not merge with org 'Jane Foundation' even if
    cosine is above threshold — different kinds never share a cluster."""
    emb = LocalEmbedder()
    ents = [
        Entity(kind="person", key="jane smith", label="Jane Smith"),
        Entity(kind="org", key="jane foundation", label="Jane Foundation"),
    ]
    out = canonicalize(ents, emb)
    # Each maps to itself.
    assert out["Jane Smith"] == "Jane Smith"
    assert out["Jane Foundation"] == "Jane Foundation"


def test_canonicalize_empty_returns_empty():
    emb = LocalEmbedder()
    assert canonicalize([], emb) == {}
    assert canonicalize(None, emb) == {}  # noqa: PLC0119 - explicit None contract


def test_canonicalize_short_names_passthrough():
    """Sub-threshold labels (len < 3) skip the embedding step and self-map.
    Two-char tokens like 'Bo' or 'Al' carry no usable signal and would
    cluster with anything if embedded."""
    emb = LocalEmbedder()
    ents = _people("Bo", "Bo Smith")
    out = canonicalize(ents, emb)
    # "Bo" is too short to participate, so it self-maps.
    assert out["Bo"] == "Bo"
    # "Bo Smith" has nothing to cluster with (Bo was skipped), so it self-maps.
    assert out["Bo Smith"] == "Bo Smith"


def test_canonicalize_lexicographic_tiebreak():
    """When two names have equal length, the lex-smallest wins for
    determinism — re-running the loop on the same corpus must produce the
    same canonical-map."""
    emb = LocalEmbedder()
    # "Jonathan" and "Jonathon" embed very close (5 chars apart, same length).
    # Whether or not they cluster depends on the model — but if they do, the
    # lex-smallest wins.
    ents = _people("Jonathan", "Jonathon")
    out = canonicalize(ents, emb)
    # Both should map to the same canonical, which is lex-smallest.
    if out["Jonathan"] == out["Jonathon"]:
        assert out["Jonathan"] == "Jonathan"  # lex-smallest of equal-length


def test_canonicalize_with_store_round_trip(tmp_path):
    """When a store is supplied, repeat canonicalizations should be cheap
    (vectors come from sqlite) and produce identical output."""
    from pi_email.embedding_store import EmbeddingStore

    emb = LocalEmbedder()
    store = EmbeddingStore(tmp_path / "canon.db")
    ents = _people("Jane Smith", "Jane")
    out1 = canonicalize(ents, emb, store=store)
    out2 = canonicalize(ents, emb, store=store)
    assert out1 == out2
    # Store should have one row per long name.
    assert store.get("entity:person:Jane Smith") is not None
    assert store.get("entity:person:Jane") is not None
