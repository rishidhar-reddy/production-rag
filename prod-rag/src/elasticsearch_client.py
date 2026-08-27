import os
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()


class ElasticsearchClient:
    def __init__(self):
        self.es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "wiki_chunks")
        self._client: Elasticsearch | None = None

    @property
    def client(self) -> Elasticsearch:
        """Connect on first use.

        This used to connect in __init__, and the module instantiates a
        singleton at import time, so importing anything under src/ required a
        live Elasticsearch -- including the pure retrieval-fusion logic, which
        touches no I/O at all. Connecting lazily keeps the failure at the point
        of the query, where it is actionable.
        """
        if self._client is None:
            client = Elasticsearch(self.es_url)
            if not client.ping():
                raise RuntimeError(f"Could not connect to Elasticsearch at {self.es_url}")
            self._client = client
        return self._client

    def retrieve(self, query: str, top_k: int = 25) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.index_name,
            size=top_k,
            query={
                "match": {
                    "chunk_text": {
                        "query": query,
                    }
                }
            },
        )

        hits = response.get("hits", {}).get("hits", [])

        retrieved = []

        for hit in hits:
            source = hit.get("_source", {})

            retrieved.append(
                {
                    "id": hit.get("_id"),
                    "text": source.get("chunk_text", ""),
                    "metadata": {
                        "doc_id": source.get("doc_id"),
                        "source_file": source.get("source_file"),
                        "chunk_index": source.get("chunk_index"),
                    },
                    "score": hit.get("_score", 0.0),
                    "source": "elasticsearch",
                }
            )

        return retrieved


elasticsearch_client = ElasticsearchClient()