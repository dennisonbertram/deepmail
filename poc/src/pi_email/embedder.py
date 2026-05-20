"""Embedder abstraction + a local `sentence-transformers` implementation.

The Protocol is the seam: today the POC uses a local model so it can run
offline against the fixture corpus; later runs can swap in a hosted provider
(Voyage, Cohere, OpenAI) by implementing the same surface.

Design choices baked in:

  * **Lazy load.** Importing this module does NOT pull a 440MB model. The
    model + tokenizer are loaded on the first `embed()` call. Test discovery
    and `import pi_email.embedder` remain instant.
  * **Unit-normalized vectors.** All vectors returned from `embed*` are L2-
    normalized so cosine similarity == dot product. Callers can use a plain
    `a @ b` without bothering with norms.
  * **Device auto-pick.** MPS (Apple Silicon) > CUDA > CPU. Logged once.
  * **Cache on text.** `embed(text)` is wrapped in an LRU cache (size 1024)
    so repeat calls within an iteration don't re-encode the same string.
    The on-disk `EmbeddingStore` is the cross-run cache; this LRU is the
    in-process one.

The `dim` and `model_id` properties exist for cache invalidation in
`EmbeddingStore` — if the model changes, the store can drop stale rows.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np


logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Embedding backend contract.

    All implementations MUST return unit-norm vectors so callers can treat
    cosine similarity as a dot product.
    """

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a 1-D float32 array of shape (dim,)."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns a 2-D float32 array of shape (N, dim)."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g. 768 for bge-base-en-v1.5)."""
        ...

    @property
    def model_id(self) -> str:
        """Stable identifier for the underlying model (used for cache invalidation)."""
        ...


class LocalEmbedder:
    """`sentence-transformers` implementation, defaulting to BAAI/bge-base-en-v1.5.

    The model is lazy-loaded on the first `embed` / `embed_batch` call. The
    first call may take a while (downloads ~440MB on a fresh machine); from
    then on it's cached in the HF cache dir and warm in process memory.
    """

    DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
    DEFAULT_DIM = 768  # bge-base-en-v1.5; verified once the model loads

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_override = device
        self._model = None  # lazy
        self._device: str | None = None  # populated when model loads
        self._dim: int | None = None     # populated when model loads

    # -- Lazy model load -----------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        """MPS > CUDA > CPU. Import torch lazily so module import stays cheap."""
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        # Defer heavy imports until first use.
        from sentence_transformers import SentenceTransformer  # type: ignore

        device = self._device_override or self._pick_device()
        logger.info(
            "LocalEmbedder: loading %s on device=%s (first call may download ~440MB)",
            self._model_name,
            device,
        )
        self._model = SentenceTransformer(self._model_name, device=device)
        self._device = device
        # bge-base-en-v1.5 is 768-dim; we ask the model rather than hardcode
        # so a future model swap "just works". sentence-transformers >=5 uses
        # `get_embedding_dimension`; older versions exposed the now-deprecated
        # `get_sentence_embedding_dimension`. Try the new name first, fall
        # back for older releases.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        if get_dim is None:
            self._dim = self.DEFAULT_DIM
        else:
            self._dim = int(get_dim())
        return self._model

    # -- Public API ----------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Result is unit-normalized float32 of shape (dim,).

        The LRU cache (declared below as a free function tied to self.model_id)
        sits in front of this — see `_cached_embed`. We dispatch through it so
        repeat strings within a process don't re-encode.
        """
        return _cached_embed(self, text)

    def _embed_uncached(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        vec = model.encode(
            [text],
            normalize_embeddings=True,  # unit-norm so cosine == dot
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32, copy=False)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Result is shape (N, dim), unit-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype(np.float32, copy=False)

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Loading the model populates self._dim. We don't actually need
            # an embedding here; calling _ensure_model is enough.
            self._ensure_model()
        # _dim is guaranteed to be set after _ensure_model, but make the
        # type-checker happy with a fallback to the documented default.
        return self._dim or self.DEFAULT_DIM

    @property
    def model_id(self) -> str:
        return self._model_name


# ---------------- LRU cache (process-local, separate from the sqlite store) ----------------


@functools.lru_cache(maxsize=1024)
def _cached_embed(embedder: "LocalEmbedder", text: str) -> np.ndarray:
    """LRU-cached front for LocalEmbedder._embed_uncached.

    Keyed on (embedder identity, text). Two different LocalEmbedder instances
    won't accidentally share cache entries — which matters if a future test
    instantiates two embedders with different model_ids.

    Returned arrays MUST be treated as read-only by callers; they're shared
    across cache hits.
    """
    return embedder._embed_uncached(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors.

    Assumes BOTH inputs are unit-normalized (which is what `Embedder.embed`
    guarantees). If either is zero-length, returns 0.0 rather than NaN.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))
