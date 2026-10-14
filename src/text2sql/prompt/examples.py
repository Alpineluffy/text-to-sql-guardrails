"""Pick the few-shot examples closest to the user's question.

LexicalExampleSelector scores by word overlap, which needs no model and works
well with a small library. Step 3 adds an embedding-based selector with the
same select() interface.
"""
from __future__ import annotations

import re
from typing import List, Protocol, Set

from .knowledge import Example

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "each",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "me", "many",
    "much", "of", "on", "or", "per", "show", "that", "the", "there", "this",
    "to", "was", "were", "what", "which", "who", "with",
}


def keywords(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Crude singularisation so "orders" matches "order".
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS}


class ExampleSelector(Protocol):
    def select(self, question: str, k: int) -> List[Example]: ...


class LexicalExampleSelector:
    def __init__(self, examples: List[Example]):
        self.examples = examples
        self._keywords = [keywords(e.question) for e in examples]

    def select(self, question: str, k: int) -> List[Example]:
        query = keywords(question)
        if not query or k <= 0:
            return []
        scored = []
        for i, (example, words) in enumerate(zip(self.examples, self._keywords)):
            overlap = len(query & words)
            if overlap:
                # Overlap normalised by the union (Jaccard); ties keep file order.
                scored.append((overlap / len(query | words), -i, example))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [example for _, _, example in scored[:k]]
