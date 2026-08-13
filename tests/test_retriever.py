import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import HybridRetriever, _chunk_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNKS = [
    {
        "doc_id": "NCR-001", "doc_type": "NCR", "chunk_index": 0,
        "text": "Porosity detected on X-ray inspection of turbine blade casting.",
    },
    {
        "doc_id": "NCR-002", "doc_type": "NCR", "chunk_index": 0,
        "text": "Surface finish non-conformance found on hydraulic actuator bore.",
    },
    {
        "doc_id": "AUD-001", "doc_type": "supplier_audit", "chunk_index": 0,
        "text": "Calibration records incomplete for three gauges during supplier audit.",
    },
    {
        "doc_id": "CAR-001", "doc_type": "corrective_action", "chunk_index": 0,
        "text": "Corrective action required supplier to submit updated PPAP documentation.",
    },
    {
        "doc_id": "INC-001", "doc_type": "incident_report", "chunk_index": 0,
        "text": "Foreign object debris event occurred in assembly bay during shift change.",
    },
]


def _mock_embedder(query_vector=None):
    emb = MagicMock()
    emb.embed_query.return_value = query_vector or [0.1] * 8
    return emb


def _mock_index(hits=None):
    idx = MagicMock()
    idx.query.return_value = hits or [
        {"id": "NCR-001_c0", "document": CHUNKS[0]["text"],
         "metadata": {"doc_type": "NCR"}, "distance": 0.1},
    ]
    return idx


def _retriever(hits=None, query_vec=None):
    return HybridRetriever(
        index=_mock_index(hits),
        embedder=_mock_embedder(query_vec),
        chunks=CHUNKS,
    )


# ---------------------------------------------------------------------------
# chunk_id helper
# ---------------------------------------------------------------------------

def test_chunk_id_format():
    chunk = {"doc_id": "NCR-2024-00001", "chunk_index": 2}
    assert _chunk_id(chunk) == "NCR-2024-00001_c2"


def test_chunk_id_defaults_to_zero():
    chunk = {"doc_id": "AUD-001"}
    assert _chunk_id(chunk) == "AUD-001_c0"


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

def test_sparse_finds_keyword_match():
    r = _retriever()
    results = r.retrieve_sparse("porosity X-ray turbine", top_k=3)
    assert len(results) > 0
    assert results[0]["id"] == "NCR-001_c0"


def test_sparse_result_schema():
    r = _retriever()
    results = r.retrieve_sparse("calibration gauge audit", top_k=2)
    assert len(results) > 0
    hit = results[0]
    assert "id" in hit
    assert "document" in hit
    assert "metadata" in hit
    assert "score" in hit
    assert hit["score"] > 0


def test_sparse_returns_at_most_top_k():
    r = _retriever()
    results = r.retrieve_sparse("inspection", top_k=2)
    assert len(results) <= 2


def test_sparse_zero_score_chunks_excluded():
    r = _retriever()
    # Query with a term that matches nothing
    results = r.retrieve_sparse("zzzznonexistentterm", top_k=5)
    assert all(hit["score"] > 0 for hit in results)


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def test_dense_calls_embedder_with_query():
    r = _retriever()
    r.retrieve_dense("find porosity defects", top_k=5)
    r.embedder.embed_query.assert_called_once_with("find porosity defects")


def test_dense_converts_distance_to_similarity():
    hits = [{"id": "X", "document": "text", "metadata": {}, "distance": 0.3}]
    r = _retriever(hits=hits)
    results = r.retrieve_dense("query", top_k=1)
    assert abs(results[0]["score"] - 0.7) < 1e-6


def test_dense_result_schema():
    r = _retriever()
    results = r.retrieve_dense("query", top_k=1)
    hit = results[0]
    assert "id" in hit
    assert "document" in hit
    assert "metadata" in hit
    assert "score" in hit


# ---------------------------------------------------------------------------
# Combined retrieve()
# ---------------------------------------------------------------------------

def test_retrieve_returns_tuple_of_two_lists():
    r = _retriever()
    dense, sparse = r.retrieve("porosity inspection", top_k=5)
    assert isinstance(dense, list)
    assert isinstance(sparse, list)


def test_retrieve_dense_and_sparse_independently():
    r = _retriever()
    dense, sparse = r.retrieve("calibration gauge", top_k=3)
    # Dense comes from mock index
    assert len(dense) >= 1
    # Sparse comes from real BM25
    assert len(sparse) >= 1
