"""Tests for the local embedding pipeline.

These tests DO download the 440MB bge-base-en-v1.5 model on first run. After
the first cache populate, subsequent runs are <1s. CI without an HF cache
will pay the download cost once.

We could mock the model out, but the canonicalization threshold (0.85)
is calibrated against the *actual* embedder's geometry — mocking it would
let real geometry bugs slip through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.embedder import LocalEmbedder, cosine  # noqa: E402
from pi_email.embedding_store import EmbeddingStore  # noqa: E402


# ---------- LocalEmbedder shape and properties ----------


def test_embed_returns_768_dim_unit_norm():
    emb = LocalEmbedder()
    v = emb.embed("Jane Smith")
    assert v.shape == (768,), f"unexpected shape {v.shape}"
    norm = float(np.linalg.norm(v))
    # Unit length to within float tolerance.
    assert abs(norm - 1.0) < 1e-4, f"expected unit-norm vector, got |v|={norm}"


def test_embed_batch_returns_2d():
    emb = LocalEmbedder()
    vs = emb.embed_batch(["Jane Smith", "Bob Smith"])
    assert vs.shape == (2, 768), f"unexpected shape {vs.shape}"
    # Both rows unit-normalized.
    for row in vs:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-4


def test_dim_and_model_id():
    emb = LocalEmbedder()
    # `dim` triggers a lazy load if needed.
    assert emb.dim == 768
    assert emb.model_id == "BAAI/bge-base-en-v1.5"


# ---------- Semantic geometry: the load-bearing claim ----------


def test_similar_names_have_high_cosine():
    """The whole canonicalization design rests on this: 'Jane' and
    'Jane Smith' must embed CLOSER than 'Jane Smith' and an unrelated
    phrase. Raw bge-base-en-v1.5 on bare names gives ~0.74 same-person and
    ~0.40 different-topic — the canonicalizer in entities.py applies a
    context-prompt prefix to push the same-person signal above 0.85, but
    here we test the raw embedder, so the bar is just "noticeably above
    random.\""""
    emb = LocalEmbedder()
    a = emb.embed("Jane Smith")
    b = emb.embed("Jane")
    sim = cosine(a, b)
    assert sim > 0.7, f"expected cos('Jane Smith', 'Jane') > 0.7, got {sim:.3f}"


def test_unrelated_strings_have_low_cosine():
    emb = LocalEmbedder()
    a = emb.embed("Jane Smith")
    b = emb.embed("Quarterly Sales Report")
    sim = cosine(a, b)
    assert sim < 0.5, (
        f"expected cos('Jane Smith', 'Quarterly Sales Report') < 0.5, "
        f"got {sim:.3f}"
    )


# ---------- EmbeddingStore round-trip ----------


def test_store_roundtrip(tmp_path):
    store = EmbeddingStore(tmp_path / "test.db")
    v = np.linspace(-1.0, 1.0, 768, dtype=np.float32)
    # Renormalize so the round-trip looks like a real entry.
    v = v / np.linalg.norm(v)
    store.put("entity:person:Jane Smith", "entity", v, model_id="test/v1")
    got = store.get("entity:person:Jane Smith")
    assert got is not None
    assert got.shape == v.shape
    assert got.dtype == np.float32
    assert np.allclose(got, v, atol=1e-7)


def test_store_get_many_and_all_of_kind(tmp_path):
    store = EmbeddingStore(tmp_path / "many.db")
    vecs = {
        f"entity:person:p{i}": np.full(8, float(i), dtype=np.float32)
        for i in range(3)
    }
    for k, v in vecs.items():
        store.put(k, "entity", v, model_id="test/v1")
    # Add a row of a different kind to verify all_of_kind filters.
    store.put("query:xyz", "query", np.zeros(8, dtype=np.float32), model_id="test/v1")

    got = store.get_many(list(vecs.keys()) + ["entity:person:missing"])
    assert set(got.keys()) == set(vecs.keys())
    for k, v in vecs.items():
        assert np.allclose(got[k], v)

    all_entities = store.all_of_kind("entity")
    assert set(all_entities.keys()) == set(vecs.keys())
    assert "query:xyz" not in all_entities


def test_store_delete_kind_invalidates(tmp_path):
    store = EmbeddingStore(tmp_path / "del.db")
    store.put("entity:person:Jane", "entity", np.zeros(4, dtype=np.float32))
    store.put("query:family", "query", np.zeros(4, dtype=np.float32))
    n = store.delete_kind("entity")
    assert n == 1
    assert store.get("entity:person:Jane") is None
    assert store.get("query:family") is not None


def test_store_put_overwrites(tmp_path):
    store = EmbeddingStore(tmp_path / "over.db")
    a = np.ones(4, dtype=np.float32)
    b = np.full(4, 2.0, dtype=np.float32)
    store.put("k", "entity", a)
    store.put("k", "entity", b)
    got = store.get("k")
    assert np.allclose(got, b)
