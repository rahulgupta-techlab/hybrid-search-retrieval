"""Benchmark BM25 vs dense vs hybrid retrieval on a labelled query set.

    python -m src.evaluate                 # offline hashing embedder
    python -m src.evaluate --st            # sentence-transformers embeddings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from src.retrievers import BM25, DenseRetriever, HybridRetriever, Ranking

DATA = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def evaluate(search: Callable[[str, int], Ranking], queries: list[dict], k: int = 3) -> dict[str, float]:
    recalls, rrs = [], []
    for q in queries:
        ranked = [doc for doc, _ in search(q["query"], 10)]
        rel = set(q["relevant"])
        recalls.append(recall_at_k(ranked, rel, k))
        rrs.append(mrr(ranked, rel))
    return {f"recall@{k}": sum(recalls) / len(recalls), "mrr": sum(rrs) / len(rrs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--st", action="store_true", help="use sentence-transformers embeddings")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    corpus = load_jsonl(DATA / "corpus.jsonl")
    queries = load_jsonl(DATA / "queries.jsonl")
    ids, texts = [d["id"] for d in corpus], [d["text"] for d in corpus]

    embedder = None
    if args.st:
        from src.retrievers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    systems = {
        "bm25": BM25(ids, texts),
        "dense": DenseRetriever(ids, texts, embedder),
        "hybrid (RRF)": HybridRetriever(ids, texts, embedder),
    }
    print(f"{'retriever':<14}{'recall@' + str(args.k):>10}{'MRR':>8}")
    for name, system in systems.items():
        m = evaluate(system.search, queries, args.k)
        print(f"{name:<14}{m[f'recall@{args.k}']:>10.3f}{m['mrr']:>8.3f}")


if __name__ == "__main__":
    main()
