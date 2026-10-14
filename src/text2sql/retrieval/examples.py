"""Few-shot example selection by meaning rather than shared words.

Drop-in replacement for prompt.examples.LexicalExampleSelector: "best-selling
items" finds the "top products by units sold" example even with no word in
common.
"""
from __future__ import annotations

from typing import List

from ..prompt.knowledge import Example
from .embeddings import Embedder, similarities


class EmbeddingExampleSelector:
    def __init__(self, examples: List[Example], embedder: Embedder):
        self.examples = examples
        self.embedder = embedder
        self._vectors = embedder.embed([e.question for e in examples]) if examples else None

    def select(self, question: str, k: int) -> List[Example]:
        if self._vectors is None or k <= 0:
            return []
        query = self.embedder.embed([question], is_query=True)[0]
        order = similarities(self._vectors, query).argsort()[::-1]
        return [self.examples[i] for i in order[:k]]
