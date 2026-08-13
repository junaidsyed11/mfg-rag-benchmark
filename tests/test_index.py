import tempfile
from pathlib import Path

import pytest

from rag.index import VectorIndex, _collection_name, _safe_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(tmp_path: Path, name: str = "test_collection") -> VectorIndex:
    return VectorIndex(collection_name=name, db_path=tmp_path)


def _fake_vectors(n: int, dim: int = 8) -> list[list[float]]:
    return [[float(i) / (dim * n)] * dim for i in range(n)]


# ---------------------------------------------------------------------------
# Metadata sanitisation
# ---------------------------------------------------------------------------

def test_safe_metadata_strips_none():
    raw = {"doc_id": "NCR-001", "supplier": None, "severity": "Major"}
    clean = _safe_metadata(raw)
    assert "supplier" not in clean
    assert clean["doc_id"] == "NCR-001"


def test_safe_metadata_keeps_scalars():
    raw = {"a": "str", "b": 1, "c": 1.5, "d": True}
    assert _safe_metadata(raw) == raw


def test_collection_name_replaces_hyphens():
    assert _collection_name("cohere-v4") == "mfg_rag_cohere_v4"
    assert _collection_name("cohere-v3") == "mfg_rag_cohere_v3"


# ---------------------------------------------------------------------------
# VectorIndex — add and count
# ---------------------------------------------------------------------------

def test_add_and_count(tmp_path):
    idx = _make_index(tmp_path)
    idx.add(
        ids=["a", "b", "c"],
        embeddings=_fake_vectors(3),
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"doc_type": "NCR"}] * 3,
    )
    assert idx.count() == 3


def test_add_is_idempotent(tmp_path):
    idx = _make_index(tmp_path)
    vectors = _fake_vectors(2)
    for _ in range(3):
        idx.add(
            ids=["x", "y"],
            embeddings=vectors,
            documents=["doc x", "doc y"],
            metadatas=[{"doc_type": "NCR"}, {"doc_type": "ECR"}],
        )
    assert idx.count() == 2  # upsert — no duplicates


# ---------------------------------------------------------------------------
# VectorIndex — query
# ---------------------------------------------------------------------------

def test_query_returns_top_k(tmp_path):
    idx = _make_index(tmp_path)
    idx.add(
        ids=[f"doc_{i}" for i in range(10)],
        embeddings=_fake_vectors(10),
        documents=[f"text {i}" for i in range(10)],
        metadatas=[{"chunk_index": i} for i in range(10)],
    )
    results = idx.query(query_embedding=[0.0] * 8, top_k=3)
    assert len(results) == 3


def test_query_result_shape(tmp_path):
    idx = _make_index(tmp_path)
    idx.add(
        ids=["chunk_0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        documents=["porosity detected on X-ray inspection"],
        metadatas=[{"doc_type": "NCR", "severity": "Major"}],
    )
    results = idx.query(query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    hit = results[0]
    assert "id" in hit
    assert "document" in hit
    assert "metadata" in hit
    assert "distance" in hit
    assert hit["metadata"]["doc_type"] == "NCR"


# ---------------------------------------------------------------------------
# VectorIndex — reset
# ---------------------------------------------------------------------------

def test_reset_clears_collection(tmp_path):
    idx = _make_index(tmp_path)
    idx.add(
        ids=["a", "b"],
        embeddings=_fake_vectors(2),
        documents=["x", "y"],
        metadatas=[{"doc_type": "NCR"}, {"doc_type": "NCR"}],
    )
    assert idx.count() == 2
    idx.reset()
    assert idx.count() == 0
