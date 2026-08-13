"""
Cohere dual embedder.

Wraps the Cohere Embed API for both document indexing and query encoding.
The active model (v4 or v3) is read from config.yaml — switching backends
is a one-line config change, no code changes required.

Cohere requires declaring input_type for every call:
  - "search_document" at index time  → model encodes for storage
  - "search_query" at query time     → model encodes for retrieval

Using the wrong input_type silently degrades retrieval quality, so the
embedder enforces the distinction through separate methods.
"""

from __future__ import annotations

import time

import cohere
from cohere.errors import TooManyRequestsError

from rag.config import get_api_key, load_config

_BATCH_SIZE = 96  # Cohere max texts per request
_RETRY_WAIT = 61  # seconds to wait after a 429 (trial limit resets per minute)


class CohereEmbedder:
    def __init__(
        self,
        model: str,
        input_type_doc: str,
        input_type_query: str,
        api_key: str,
    ) -> None:
        self.co = cohere.ClientV2(api_key)
        self.model = model
        self.input_type_doc = input_type_doc
        self.input_type_query = input_type_query

    def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        """Embed one batch with retry on rate limit."""
        while True:
            try:
                response = self.co.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                return list(response.embeddings.float_)
            except TooManyRequestsError:
                print(f"\n  Rate limit hit — waiting {_RETRY_WAIT}s ...", flush=True)
                time.sleep(_RETRY_WAIT)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_vectors.extend(self._embed_batch(batch, input_type))
        return all_vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks for indexing."""
        return self._embed(texts, self.input_type_doc)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""
        return self._embed([text], self.input_type_query)[0]

    @property
    def model_name(self) -> str:
        return self.model


def get_embedder(cfg: dict | None = None, api_key: str | None = None) -> CohereEmbedder:
    """Build a CohereEmbedder from config. The active backend is set in config.yaml."""
    if cfg is None:
        cfg = load_config()
    if api_key is None:
        api_key = get_api_key("COHERE_API_KEY")

    backend = cfg["embedding"]["backend"]
    model = cfg["embedding"]["models"][backend]

    return CohereEmbedder(
        model=model,
        input_type_doc=cfg["embedding"]["input_type"]["document"],
        input_type_query=cfg["embedding"]["input_type"]["query"],
        api_key=api_key,
    )
