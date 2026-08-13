"""
Cohere Rerank v4 wrapper.

Takes the fused candidate list from RRF and re-scores every candidate by
reading the query and each chunk together — a cross-encoder pass that is
more accurate than vector similarity but too slow to run on the full corpus.

Sits at the end of the retrieval pipeline:
  dense + sparse → RRF fusion → reranker → top-5 → generation
"""

from __future__ import annotations

from typing import Any

import cohere

from rag.config import get_api_key, load_config


class CohereReranker:
    def __init__(self, model: str, api_key: str, top_n: int = 5) -> None:
        self.co = cohere.ClientV2(api_key)
        self.model = model
        self.default_top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates against query and return the top_n most relevant.

        Each result dict gets a "rerank_score" field added. Results are
        sorted by rerank_score descending (most relevant first).
        """
        if not candidates:
            return []

        n = top_n if top_n is not None else self.default_top_n
        n = min(n, len(candidates))

        response = self.co.rerank(
            model=self.model,
            query=query,
            documents=[c["document"] for c in candidates],
            top_n=n,
        )

        return [
            {**candidates[r.index], "rerank_score": r.relevance_score}
            for r in response.results
        ]

    @property
    def model_name(self) -> str:
        return self.model


def get_reranker(cfg: dict | None = None, api_key: str | None = None) -> CohereReranker:
    """Build a CohereReranker from config."""
    if cfg is None:
        cfg = load_config()
    if api_key is None:
        api_key = get_api_key("COHERE_API_KEY")

    return CohereReranker(
        model=cfg["reranking"]["model"],
        api_key=api_key,
        top_n=cfg["retrieval"]["final_top_k"],
    )
