import os
from typing import Any

from src.chromadb_client import chromadb_client
from src.elasticsearch_client import elasticsearch_client
from src.query_processor import query_processor


class HybridRetriever:
    """Fuses semantic (Chroma) and keyword (Elasticsearch) retrieval.

    Fusion uses Reciprocal Rank Fusion, which combines the two result lists by
    *rank* rather than by score. Cosine similarity is bounded to roughly [0, 1]
    while BM25 is unbounded and routinely exceeds 20, so the two scores cannot
    be compared or summed directly without a normalisation step that would
    itself need per-corpus calibration. Ranks are already commensurable.
    """

    def __init__(self):
        self.chroma = chromadb_client
        self.elasticsearch = elasticsearch_client
        self.query_processor = query_processor
        # Standard RRF damping constant. Larger values flatten the weight given
        # to the very top of each list.
        self.rrf_k = int(os.getenv("RRF_K", "60"))

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        processed_query = self.query_processor.process(query)

        semantic_results = self.chroma.retrieve(
            query_embedding=processed_query["embedding"],
            query_text=processed_query["query"],
            top_k=top_k,
        )

        keyword_results = self.elasticsearch.retrieve(
            query=processed_query["query"],
            top_k=top_k,
        )

        merged_results = self._merge_and_dedupe(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
        )

        return merged_results[:top_k]

    def _merge_and_dedupe(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse the two ranked lists with Reciprocal Rank Fusion.

        Each list contributes 1 / (k + rank) to a document's fused score, so a
        document both retrievers found accumulates from both and rises above
        single-source hits, while position within each list is still honoured.

        This replaces a sort on (source_count, semantic_score, keyword_score).
        Because that tuple compared semantic_score before keyword_score and
        every Elasticsearch-only document has semantic_score 0.0, any document
        Chroma returned outranked every keyword-only document no matter how
        strong the keyword match -- a top BM25 hit landed below eight weak
        semantic hits, which made the hybrid retriever semantic-only in effect.
        """
        seen: dict[str, dict[str, Any]] = {}

        for source, results, score_field in (
            ("chroma", semantic_results, "semantic_score"),
            ("elasticsearch", keyword_results, "keyword_score"),
        ):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["id"]
                entry = seen.get(chunk_id)

                if entry is None:
                    entry = {
                        **result,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "retrieval_sources": [],
                        "rrf_score": 0.0,
                    }
                    seen[chunk_id] = entry

                entry[score_field] = result.get("score", 0.0)
                entry["retrieval_sources"].append(source)
                entry["rrf_score"] += 1.0 / (self.rrf_k + rank)

        results = list(seen.values())

        # Sorting on rrf_score alone; Python's sort is stable, so documents that
        # tie keep the order they were inserted in and the result is
        # deterministic run to run.
        results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return results


retriever = HybridRetriever()