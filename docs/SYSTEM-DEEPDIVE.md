# Production RAG — Engineering Deep Dive

> A component-level walkthrough of the retrieval pipeline, traced to specific files
> and lines.

---

## 1. What this system is

An end-to-end Retrieval-Augmented Generation pipeline over a Wikipedia (Simple English)
corpus, built as separable stages rather than one script:

```
wiki_dataset/          raw corpus (Kaggle, not in git)
      │
spark-preprocessing/   chunk articles with Spark, write parquet to S3
      │
embedding/             embed chunks (sentence-transformers, multi-GPU), parquet to S3
      │
      ├── chromadb_setup/        build the HNSW vector index
      └── elasticsearch_setup/   build the BM25 keyword index
      │
prod-rag/              hybrid retrieval → rerank → generate, behind FastAPI
      │
prod-rag/evals/        RAGAS + LLM-as-judge scoring, including hallucination traps
```

The staging is the point. Each step writes parquet to S3, so re-running embedding does
not force re-chunking, and the two indexes are built from the same artifact.

---

## 2. The query path

`prod-rag/src/rag.py` — `RAGClient.answer()` runs three logged stages:

| Stage | Component | Default |
|---|---|---|
| 1. Retrieve | `HybridRetriever` — Chroma + Elasticsearch, fused | `RETRIEVAL_TOP_K=50` |
| 2. Rerank | `flashrank` cross-encoder | `RERANK_TOP_K=5` |
| 3. Generate | `ChatGroq` via LangChain | `llama-3.1-8b-instant` |

Retrieve-wide-then-rerank-narrow is the right shape: a bi-encoder is cheap and recalls
broadly, a cross-encoder is expensive and precise, so you spend the expensive model only
on 50 candidates.

Queries are logged by SHA-256 digest at INFO and in the clear only at DEBUG
(`_digest`, line 23) — a small detail that shows someone thought about logging user
input.

---

## 3. Hybrid fusion — the interesting problem

Two retrievers return two ranked lists:

- **Chroma** — cosine similarity over `all-MiniLM-L6-v2` embeddings, bounded to
  roughly [0, 1].
- **Elasticsearch** — BM25, **unbounded**, routinely above 20 and corpus-dependent.

These scores cannot be compared. The original merge sorted on the tuple
`(source_count, semantic_score, keyword_score)`, which compares `semantic_score`
before `keyword_score`. Since every Elasticsearch-only document has
`semantic_score = 0.0`, **any** document Chroma returned outranked **every**
keyword-only document, however strong the keyword match. Measured on a synthetic case,
a BM25 hit scoring 60.0 ranked ninth of nine, behind semantic hits scoring as low as
0.2. The hybrid retriever was semantic-only in effect.

It now fuses by **rank** instead of score — Reciprocal Rank Fusion, where each list
contributes `1 / (k + rank)` with `k` configurable via `RRF_K` (default 60). Documents
both retrievers found accumulate from both lists and rise; position within each list is
preserved; and no cross-scale comparison happens anywhere. The same case now places the
BM25 hit second.

RRF is the standard answer here precisely because score normalisation would need
per-corpus calibration that nobody maintains.

---

## 4. Evaluation

`prod-rag/evals/run_evals.py` is more serious than most projects of this size. It runs
two judge families over categorised datasets:

- **RAGAS** metrics with an evaluator LLM and evaluator embeddings.
- **openevals** LLM-as-judge scorers for `retrieval_relevance`, `correctness`, and
  `hallucination`.

There is a dedicated **hallucination-trap** dataset
(`generate_hallucination_dataset.py`) — questions designed so the corpus cannot support
an answer, testing whether the model declines rather than invents. Measuring refusal
behaviour separately from accuracy is a real evaluation instinct.

---

## 5. Testability — why there were no tests

Every client was a module-level singleton constructed at import, and
`ElasticsearchClient.__init__` pinged the server and raised on failure. So
`from src.retriever import HybridRetriever` required a live Elasticsearch, a Chroma
directory, a downloaded cross-encoder and a `GROQ_API_KEY` — in order to reach fusion
logic that performs no I/O at all.

That is why the repository had zero tests: it was not possible to write one. Each client
now builds its expensive resource behind a property on first use. Public names are
unchanged, and failures surface at the query that needs the resource rather than at
import.

With that unblocked, the fusion bug above became testable, and 10 tests now cover it.

---

## 6. Known gaps

| Gap | Detail |
|---|---|
| **CORS is fully open** | `allow_origins=["*"]` with all methods and headers (`main.py:21`). |
| **No authentication** | `/inference` is unauthenticated; every call costs a Groq request and a cross-encoder pass. |
| **Coverage is narrow** | 10 tests, all on fusion. Retrieval clients, reranker and the eval harness are untested. |
| **Query rewriting is a stub** | `rewrite_query` is the identity function; HyDE and multi-query expansion are named in the docstring but unimplemented. |
| **No LICENSE** | Neither a file nor a declaration. |
| **Corpus not reproducible from the repo** | `wiki_dataset/` comes from Kaggle and is gitignored, so a clean clone cannot rebuild the indexes without that manual step. |

### What I would build next

1. Lock CORS to a known origin and put an API key on `/inference`.
2. Implement `rewrite_query` as HyDE and A/B it with the existing eval harness — the
   measurement infrastructure to prove whether it helps is already here.
3. Tune `RRF_K` against the eval set now that fusion is rank-based and tunable.
4. Contract tests for the Chroma and Elasticsearch clients against ephemeral containers.

---

## 7. Attribution

Original implementation by **aditya-dawadikar** across 26 commits, last pushed
2026-06-15. Full history preserved in this repository.

The lazy-initialization refactor, the RRF fusion fix, the double-rewrite fix, the
10-test fusion suite, and this document are my own work.
