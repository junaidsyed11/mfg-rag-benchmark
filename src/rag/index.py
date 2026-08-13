"""
ChromaDB vector index wrapper.

Manages persistent collections for dense vector storage and retrieval.
Each embedding backend (cohere-v4, cohere-v3) gets its own named collection
so both can coexist on disk and be queried independently during evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.config import load_config

# ChromaDB only accepts str, int, float, bool in metadata dicts
_SCALAR_TYPES = (str, int, float, bool)


def _safe_metadata(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Strip None values and non-scalar types that ChromaDB rejects.
    Returns None for empty dicts (ChromaDB rejects empty metadata)."""
    clean = {k: v for k, v in meta.items() if isinstance(v, _SCALAR_TYPES)}
    return clean if clean else None


def _collection_name(backend: str) -> str:
    """Derive a ChromaDB-safe collection name from the config backend key."""
    return f"mfg_rag_{backend.replace('-', '_')}"


class VectorIndex:
    def __init__(self, collection_name: str, db_path: Path) -> None:
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._name = collection_name

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=[_safe_metadata(m) for m in metadatas],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return top_k results as a list of dicts with keys:
          id, document, metadata, distance
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self.client.delete_collection(self._name)
        self.collection = self.client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )


def get_index(cfg: dict | None = None, db_path: Path | None = None) -> VectorIndex:
    """Build a VectorIndex for the active embedding backend from config."""
    if cfg is None:
        cfg = load_config()
    if db_path is None:
        db_path = Path(cfg["paths"]["chroma_db"])
    backend = cfg["embedding"]["backend"]
    return VectorIndex(_collection_name(backend), db_path)
