"""Lexical (BM25), dense (embedding) and hybrid (reciprocal rank fusion) retrievers."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
         "by", "with", "from", "so", "can", "when", "where", "what", "how", "it", "that"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


Ranking = list[tuple[str, float]]


class BM25:
    """Okapi BM25 implemented from scratch (k1, b are the standard defaults)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(ids)
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.docs), 1)
        self.k1, self.b = k1, b
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5) -> Ranking:
        terms = tokenize(query)
        scores = []
        for doc_id, tf, dl in zip(self.ids, self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((doc_id, s))
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:top_k]


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Offline embedder: hashed word + character trigram features, L2-normalised.

    Captures partial-word overlap (vibration ~ vibrations) without downloading a
    model. Swap in SentenceTransformerEmbedder for real semantic similarity.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        feats = []
        for tok in tokenize(text):
            feats.append("w:" + tok)
            padded = f"#{tok}#"
            feats += ["c:" + padded[i:i + 3] for i in range(len(padded) - 2)]
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """Real semantic embeddings (pip install sentence-transformers)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class DenseRetriever:
    """Exact cosine-similarity search over normalised embeddings (a flat vector index)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], embedder: Embedder | None = None):
        self.ids = list(ids)
        self.embedder = embedder or HashingEmbedder()
        self.matrix = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 5) -> Ranking:
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]


def reciprocal_rank_fusion(rankings: Sequence[Ranking], k: int = 60, top_k: int = 5) -> Ranking:
    """RRF: score(d) = sum over lists of 1 / (k + rank). Rank-based, so no score calibration needed."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]


class HybridRetriever:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], embedder: Embedder | None = None,
                 candidates: int = 10, rrf_k: int = 60):
        self.bm25 = BM25(ids, texts)
        self.dense = DenseRetriever(ids, texts, embedder)
        self.candidates, self.rrf_k = candidates, rrf_k

    def search(self, query: str, top_k: int = 5) -> Ranking:
        return reciprocal_rank_fusion(
            [self.bm25.search(query, self.candidates), self.dense.search(query, self.candidates)],
            k=self.rrf_k, top_k=top_k,
        )
