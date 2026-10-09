import numpy as np
import pytest

from src.evaluate import DATA, evaluate, load_jsonl, mrr, recall_at_k
from src.retrievers import (BM25, DenseRetriever, HashingEmbedder, HybridRetriever,
                            reciprocal_rank_fusion, tokenize)

CORPUS = load_jsonl(DATA / "corpus.jsonl")
QUERIES = load_jsonl(DATA / "queries.jsonl")
IDS = [d["id"] for d in CORPUS]
TEXTS = [d["text"] for d in CORPUS]


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("The error E-4711 on the press") == ["error", "e-4711", "press"]


def test_bm25_exact_identifier_match():
    assert BM25(IDS, TEXTS).search("E-4711", 1)[0][0] == "d11"


def test_bm25_no_match_returns_empty():
    assert BM25(IDS, TEXTS).search("zebra quantum", 5) == []


def test_bm25_prefers_rarer_terms():
    bm = BM25(["a", "b"], ["machine machine sensor", "machine calibration"])
    assert bm.search("calibration", 2)[0][0] == "b"


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = emb.encode(["vibration sensor"]), emb.encode(["vibration sensor"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_dense_handles_word_variants():
    assert DenseRetriever(IDS, TEXTS).search("spindle vibrations anomaly", 1)[0][0] == "d2"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[("x", 9.0), ("y", 1.0)], [("y", 0.9), ("z", 0.5)]], k=60)
    assert fused[0][0] == "y"


def test_rrf_ignores_raw_score_scale():
    a = reciprocal_rank_fusion([[("x", 1000.0), ("y", 1.0)]])
    b = reciprocal_rank_fusion([[("x", 0.2), ("y", 0.1)]])
    assert a == b


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5
    assert mrr(["a"], {"b"}) == 0.0


def test_hybrid_is_at_least_as_good_as_bm25():
    bm25 = evaluate(BM25(IDS, TEXTS).search, QUERIES)
    hybrid = evaluate(HybridRetriever(IDS, TEXTS).search, QUERIES)
    assert hybrid["mrr"] >= bm25["mrr"]
    assert hybrid["recall@3"] >= 0.8
