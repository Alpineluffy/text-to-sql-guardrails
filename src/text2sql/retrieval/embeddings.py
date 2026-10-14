"""Text embedding backends.

Every embedder returns L2-normalised vectors, so a dot product is cosine
similarity. SentenceTransformerEmbedder runs a local model (downloaded once,
~90 MB for the default); HashingEmbedder needs no model and exists so tests
are fast and deterministic.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, query_prefix: str = ""):
        self.name = model_name
        # Some retrieval models (e.g. BGE) expect an instruction before queries.
        self.query_prefix = query_prefix
        self._model = None

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import

            self._model = SentenceTransformer(self.name)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class HashingEmbedder:
    """Bag of words hashed into a fixed-size vector. No semantics, no download."""

    def __init__(self, dims: int = 4096):
        self.name = f"hashing-{dims}"
        self.dims = dims

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                word = word[:-1] if len(word) > 3 and word.endswith("s") else word
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dims] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class CachedEmbedder:
    """Remembers document embeddings on disk; queries are always computed fresh."""

    def __init__(self, inner: Embedder, cache_dir: Optional[Path] = None):
        self.inner = inner
        self.name = inner.name
        self.path = (
            Path(cache_dir) / (re.sub(r"[^\w.-]", "_", inner.name) + ".npz")
            if cache_dir else None
        )
        self._vectors: Dict[str, np.ndarray] = {}
        if self.path and self.path.exists():
            with np.load(self.path) as data:
                self._vectors = {k: data[k] for k in data.files}

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            return self.inner.embed(texts, is_query=True)
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, vec in zip(missing, fresh):
                self._vectors[keys[i]] = vec
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.path, **self._vectors)
        return np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0))


def default_embedder(cache_dir: Optional[Path] = Path("cache/embeddings")) -> Embedder:
    return CachedEmbedder(SentenceTransformerEmbedder(), cache_dir)


def similarities(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of each (normalised) document vector to the query."""
    if not len(docs):
        return np.zeros(0)
    # numpy 2.0 on Apple Accelerate raises bogus divide/overflow warnings on
    # float32 matmul; the results are correct.
    with np.errstate(all="ignore"):
        return docs @ query
