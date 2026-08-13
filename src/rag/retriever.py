"""
Hybrid retriever — dense (ChromaDB) + sparse (BM25).

Returns two ranked candidate lists per query. RRF fusion (Step 8) merges
them into a single ranked list. Keeping the two retrievers separate here
makes it easy to inspect which results came from which source.

Result dicts from both retrievers share the same schema:
  id        — chunk ID (<doc_id>_c<chunk_index>)
  document  — chunk text
  metadata  — all parent doc metadata fields
  score     — relevance score (dense: cosine similarity; sparse: BM25 score)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rag.config import load_config
from rag.embedder import CohereEmbedder, get_embedder
from rag.index import VectorIndex, get_index


def _chunk_id(chunk: dict) -> str:
    return f"{chunk['doc_id']}_c{chunk.get('chunk_index', 0)}"


def load_chunks(chunk_dir: Path) -> list[dict]:
    """Load all chunks from data/chunks/ JSONL files."""
    chunks = []
    for jsonl_file in sorted(chunk_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


class HybridRetriever:
    def __init__(
        self,
        index: VectorIndex,
        embedder: CohereEmbedder,
        chunks: list[dict],
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.chunks = chunks

        # BM25 needs all documents tokenised at build time
        corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def retrieve_dense(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Embed the query and find nearest vectors in ChromaDB."""
        query_vec = self.embedder.embed_query(query)
        raw = self.index.query(query_vec, top_k=top_k)
        # ChromaDB returns cosine distance (0=identical, 2=opposite).
        # Convert to similarity so higher = better, consistent with BM25.
        results = []
        for hit in raw:
            results.append({
                "id": hit["id"],
                "document": hit["document"],
                "metadata": hit["metadata"],
                "score": 1.0 - hit["distance"],
            })
        return results

    # ------------------------------------------------------------------
    # Sparse retrieval
    # ------------------------------------------------------------------

    def retrieve_sparse(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Score all chunks with BM25 and return the top_k."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # argsort ascending → reverse for descending
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            meta = {k: v for k, v in chunk.items() if k != "text"}
            results.append({
                "id": _chunk_id(chunk),
                "document": chunk["text"],
                "metadata": meta,
                "score": float(scores[idx]),
            })
        return results

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Run both retrievers and return (dense_results, sparse_results).
        RRF fusion happens in the next step.
        """
        dense = self.retrieve_dense(query, top_k)
        sparse = self.retrieve_sparse(query, top_k)
        return dense, sparse


def get_retriever(
    cfg: dict | None = None,
    chunk_dir: Path | None = None,
) -> HybridRetriever:
    """Build a HybridRetriever from config — loads chunks, embedder, and index."""
    if cfg is None:
        cfg = load_config()
    if chunk_dir is None:
        chunk_dir = Path("data/chunks")

    chunks = load_chunks(chunk_dir)
    embedder = get_embedder(cfg=cfg)
    index = get_index(cfg=cfg)
    return HybridRetriever(index=index, embedder=embedder, chunks=chunks)
