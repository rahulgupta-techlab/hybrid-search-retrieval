# Hybrid Search Retrieval

Lexical, dense and hybrid retrieval for RAG, plus a small benchmark to compare them.

- **BM25** (Okapi), implemented from scratch: strong on exact identifiers such as error codes and part numbers
- **Dense vector search**: cosine similarity over normalised embeddings, good for paraphrases
- **Hybrid**: BM25 and dense candidates merged with **Reciprocal Rank Fusion (RRF)**, which needs no score calibration between retrievers
- **Evaluation**: recall@k and MRR on a labelled query set

Embeddings are pluggable:
- `HashingEmbedder`: offline, deterministic (hashed words + character trigrams), no model download. Used in tests.
- `SentenceTransformerEmbedder`: real semantic embeddings (`all-MiniLM-L6-v2`).

## Results

Corpus: 15 short technical documents (predictive maintenance, computer vision, RAG, process mining, MLOps). Queries: 12 labelled queries, mixing paraphrases ("forecast equipment breakdowns from sensor readings") with exact-match lookups ("E-4711").

| Retriever | Embeddings | recall@3 | MRR |
| --- | --- | --- | --- |
| BM25 | n/a | 0.917 | 0.917 |
| Dense | hashing (offline) | 0.917 | 0.938 |
| Hybrid (RRF) | hashing (offline) | 0.917 | 0.938 |
| Dense | all-MiniLM-L6-v2 | 1.000 | 1.000 |
| Hybrid (RRF) | all-MiniLM-L6-v2 | 1.000 | 1.000 |

BM25 misses paraphrased queries that share no words with the answer. Semantic embeddings fix that. Hybrid search keeps the semantic quality and still guarantees that exact identifiers like `E-4711` are found, which pure dense search can miss on larger corpora.

The dataset is deliberately small so the repo runs in seconds. The point is the pipeline and evaluation method, not the absolute numbers.

## Project layout

```
src/
  retrievers.py   # tokenizer, BM25, embedders, dense retriever, RRF, HybridRetriever
  evaluate.py     # recall@k / MRR benchmark across retrievers
data/
  corpus.jsonl    # documents
  queries.jsonl   # queries with relevant document ids
tests/
  test_retrievers.py
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest                 # 10 tests, offline
python -m src.evaluate           # benchmark with the offline embedder
pip install sentence-transformers
python -m src.evaluate --st      # benchmark with real embeddings
```

## Using it in a RAG pipeline

```python
from src.retrievers import HybridRetriever, SentenceTransformerEmbedder

retriever = HybridRetriever(ids, texts, embedder=SentenceTransformerEmbedder())
context = [doc_id for doc_id, _ in retriever.search("why did press 3 stop?", top_k=5)]
```

`DenseRetriever` does exact (flat) search. For large corpora, put the same embeddings in FAISS, Qdrant or ChromaDB and keep the RRF fusion step unchanged.

## License

MIT
