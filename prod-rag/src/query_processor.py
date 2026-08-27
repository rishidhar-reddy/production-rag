import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class QueryProcessor:
    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self._embedding_model: SentenceTransformer | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Load the sentence-transformer on first use.

        Loading at import cost every consumer a model download and several
        seconds, even ones that never embed anything.
        """
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def normalize_query(self, query: str) -> str:
        return query.strip()

    def rewrite_query(self, query: str) -> str:
        """
        Placeholder for future query rewriting.

        Later this can use:
        - LangChain
        - LLM-based rewriting
        - HyDE
        - multi-query expansion
        """
        return query

    def _encode(self, text: str) -> list[float]:
        """Embed text that has already been normalized and rewritten."""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Normalize, rewrite, and embed a raw query."""
        normalized_query = self.normalize_query(query)
        rewritten_query = self.rewrite_query(normalized_query)
        return self._encode(rewritten_query)

    def process(self, query: str) -> dict:
        normalized_query = self.normalize_query(query)
        rewritten_query = self.rewrite_query(normalized_query)
        # Embed the already-rewritten text directly. Calling embed_query here
        # would normalize and rewrite a second time; rewrite_query is currently
        # the identity function, but it is documented as the hook for LLM-based
        # rewriting and HyDE, and applying either twice would both corrupt the
        # query and pay for two LLM calls.
        embedding = self._encode(rewritten_query)

        return {
            "original_query": query,
            "query": rewritten_query,
            "embedding": embedding,
        }


query_processor = QueryProcessor()