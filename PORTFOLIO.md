# production-rag — portfolio notes

> **⚠️ Fill in every `[FILL IN]` below and delete this line before sharing.**
> *My contributions* is verifiable from `git log`. *My role on the original build* is
> yours to state accurately — left blank rather than guessed.

---

## What this is

An end-to-end Retrieval-Augmented Generation pipeline over Wikipedia: Spark chunking →
multi-GPU embedding → dual indexing (Chroma HNSW + Elasticsearch BM25) → hybrid
retrieval → cross-encoder reranking → LLM generation, behind FastAPI, with a RAGAS and
LLM-as-judge evaluation harness including hallucination traps.

**Stack:** Python · PySpark · sentence-transformers · ChromaDB · Elasticsearch ·
flashrank · LangChain · Groq · FastAPI · S3 · RAGAS

Architecture walkthrough → [`docs/SYSTEM-DEEPDIVE.md`](docs/SYSTEM-DEEPDIVE.md)

---

## Provenance

Original implementation by **aditya-dawadikar**, 26 commits, last pushed 2026-06-15.
This repository preserves that full history; the original is at
[Aditya-Dawadikar/Production-RAG](https://github.com/Aditya-Dawadikar/Production-RAG),
tracked here as `upstream`.

### My role on the original build

[FILL IN — name what you actually worked on. If you did not contribute to the original,
say so; the fusion bug found and fixed below stands on its own.]

---

## My contributions in this repository

`git log --author="Rishidhar Reddy Garlapati"`

### Made the hybrid retriever actually hybrid (`d0b9c9a`)

**The bug.** Fusion sorted on the tuple `(source_count, semantic_score, keyword_score)`.
Cosine similarity is bounded to ~[0, 1]; BM25 is unbounded and routinely exceeds 20. The
tuple compares `semantic_score` before `keyword_score`, and every Elasticsearch-only
document has `semantic_score = 0.0` — so **any** document Chroma returned outranked
**every** keyword-only document, no matter how strong the keyword match.

**The measurement.** A BM25 hit scoring 60.0 ranked **ninth of nine**, behind semantic
hits scoring as low as 0.2. The Elasticsearch half of the pipeline was close to
decorative.

**The fix.** Reciprocal Rank Fusion — each list contributes `1 / (k + rank)`, `k`
configurable via `RRF_K`. Fusing by rank sidesteps the incomparable-scale problem
entirely; documents both retrievers agree on accumulate from both lists. The same case
now places the BM25 hit **second**. 10 tests cover preservation, dedup, agreement,
rank-order fidelity, and invariance to keyword-score scaling.

### Made the package testable at all (`cd09357`)

Every client was a module-level singleton built at import, and
`ElasticsearchClient.__init__` pinged the server and raised. So importing
`HybridRetriever` required a live Elasticsearch, a Chroma directory, a downloaded
cross-encoder and a `GROQ_API_KEY` — to reach fusion logic that does no I/O.

That is *why* the repo had zero tests. Each client now initializes lazily behind a
property; public names are unchanged. This had to land before the fusion bug above could
be written as a test.

### Fixed a latent double-rewrite (`aec6601`)

`process()` normalized and rewrote the query, then passed it to `embed_query()`, which
normalized and rewrote it *again*. Harmless today because `rewrite_query` is the
identity function — but it is documented as the hook for LLM-based rewriting and HyDE,
and either applied twice would distort the query and pay for two LLM calls per search.

---

## What I would build next

1. Lock CORS (`allow_origins=["*"]`) and add auth on `/inference` — currently anyone can
   spend Groq credits.
2. Implement `rewrite_query` as HyDE and A/B it — the eval harness to prove whether it
   helps already exists.
3. Tune `RRF_K` against the eval set now that fusion is rank-based and tunable.
4. Contract tests for the Chroma and Elasticsearch clients against ephemeral containers.
5. A LICENSE file — there is currently none.

---

## Honest limitations

- **I have not run the full pipeline.** No Spark job, no embedding run, no index build.
  The fusion fix is verified by unit tests on the pure merge function and the
  before/after ranking is measured on synthetic inputs, not on the real corpus.
- **The corpus is not reproducible from a clean clone.** `wiki_dataset/` comes from
  Kaggle and is gitignored.
- **Coverage is narrow** — 10 tests, all on fusion. The clients, reranker and eval
  harness remain untested.
