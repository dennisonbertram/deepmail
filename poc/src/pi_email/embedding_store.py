"""sqlite-backed embedding cache.

A *sidecar* on the filesystem rather than a vector DB — at POC scale (low-
thousands of vectors) we don't need ANN indices or async clients. sqlite +
numpy on the round-trip is plenty.

Schema (created on first connect):

    CREATE TABLE embeddings (
      key TEXT PRIMARY KEY,            -- stable id, e.g. "entity:person:Jane Smith"
      kind TEXT NOT NULL,              -- 'message' | 'entity' | 'query' | 'excerpt'
      model_id TEXT NOT NULL,          -- e.g. "BAAI/bge-base-en-v1.5"
      dim INTEGER NOT NULL,            -- e.g. 768
      vector BLOB NOT NULL,            -- raw np.float32 bytes
      created_at REAL NOT NULL         -- unix epoch seconds
    );
    CREATE INDEX idx_kind ON embeddings(kind);

The `model_id` column is the cache-invalidation knob: callers that detect
a model change should `delete_kind` (or DROP the table) to avoid mixing
vectors from different embedding spaces.

This module is intentionally NOT thread-safe — the POC's loop is single-
threaded. If we ever need parallelism we'll wrap connections per-thread.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  model_id TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kind ON embeddings(kind);
"""


class EmbeddingStore:
    """sqlite-backed vector cache keyed by stable ID.

    Caller picks the key scheme. Conventions used elsewhere in the POC:

      * `entity:{kind}:{display_name}` for canonicalization vectors
      * `query:{normalized_form}` for Rule C dedupe vectors
      * `message:{message_id}` and `excerpt:{message_id}:{i}` reserved for
        future use; not populated by the current loop.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False would be needed for multi-threaded callers;
        # we deliberately omit it so accidental thread sharing fails loudly.
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- Reads --------------------------------------------------------

    def get(self, key: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT vector, dim FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        blob, dim = row
        return _decode(blob, dim)

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Bulk read. Missing keys are simply absent from the returned dict."""
        if not keys:
            return {}
        # sqlite has no bulk IN-clause for variable lists without ?-fanout;
        # for POC volumes a single fan-out query is fine.
        placeholders = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT key, vector, dim FROM embeddings WHERE key IN ({placeholders})",
            keys,
        ).fetchall()
        return {k: _decode(b, d) for k, b, d in rows}

    def all_of_kind(self, kind: str) -> dict[str, np.ndarray]:
        rows = self._conn.execute(
            "SELECT key, vector, dim FROM embeddings WHERE kind = ?", (kind,)
        ).fetchall()
        return {k: _decode(b, d) for k, b, d in rows}

    # -- Writes -------------------------------------------------------

    def put(self, key: str, kind: str, vec: np.ndarray, model_id: str = "") -> None:
        """Insert or overwrite a vector. `model_id` is stamped on the row for
        later invalidation; pass the embedder's `model_id` property."""
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        if vec.ndim != 1:
            raise ValueError(f"put expects a 1-D vector, got shape {vec.shape}")
        self._conn.execute(
            "INSERT INTO embeddings(key, kind, model_id, dim, vector, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  kind=excluded.kind, model_id=excluded.model_id, "
            "  dim=excluded.dim, vector=excluded.vector, "
            "  created_at=excluded.created_at",
            (
                key,
                kind,
                model_id,
                int(vec.shape[0]),
                vec.tobytes(order="C"),
                time.time(),
            ),
        )
        self._conn.commit()

    def delete_kind(self, kind: str) -> int:
        """Drop every row of a given kind. Returns the row count deleted."""
        cur = self._conn.execute("DELETE FROM embeddings WHERE kind = ?", (kind,))
        self._conn.commit()
        return cur.rowcount

    # -- Lifecycle ----------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _decode(blob: bytes, dim: int) -> np.ndarray:
    """sqlite blob -> 1-D float32 numpy array. The store only ever writes
    float32, so we can hardcode the dtype."""
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()
