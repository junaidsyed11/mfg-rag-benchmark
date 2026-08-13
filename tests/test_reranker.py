from unittest.mock import MagicMock, patch

from rag.reranker import CohereReranker, get_reranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rerank_result(index: int, score: float):
    r = MagicMock()
    r.index = index
    r.relevance_score = score
    return r


def _make_response(results):
    resp = MagicMock()
    resp.results = results
    return resp


def _reranker(top_n: int = 5) -> CohereReranker:
    with patch("rag.reranker.cohere.ClientV2"):
        return CohereReranker(model="rerank-english-v4.0", api_key="test", top_n=top_n)


def _candidates(n: int) -> list[dict]:
    return [
        {"id": f"doc_{i}", "document": f"chunk text {i}",
         "metadata": {"doc_type": "NCR"}, "rrf_score": 0.1}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------

def test_rerank_returns_top_n():
    r = _reranker(top_n=3)
    r.co.rerank.return_value = _make_response([
        _make_rerank_result(0, 0.95),
        _make_rerank_result(2, 0.80),
        _make_rerank_result(1, 0.60),
    ])
    results = r.rerank("find NCRs with porosity", _candidates(5))
    assert len(results) == 3


def test_rerank_adds_rerank_score():
    r = _reranker()
    r.co.rerank.return_value = _make_response([_make_rerank_result(0, 0.92)])
    results = r.rerank("query", _candidates(1))
    assert "rerank_score" in results[0]
    assert abs(results[0]["rerank_score"] - 0.92) < 1e-6


def test_rerank_preserves_original_metadata():
    r = _reranker()
    candidates = [{"id": "X", "document": "text", "metadata": {"doc_type": "NCR"}, "rrf_score": 0.5}]
    r.co.rerank.return_value = _make_response([_make_rerank_result(0, 0.9)])
    results = r.rerank("query", candidates)
    assert results[0]["id"] == "X"
    assert results[0]["metadata"]["doc_type"] == "NCR"


def test_rerank_maps_index_to_correct_candidate():
    r = _reranker(top_n=2)
    candidates = _candidates(3)
    # Reranker picks candidate at index 2 first, then index 0
    r.co.rerank.return_value = _make_response([
        _make_rerank_result(2, 0.99),
        _make_rerank_result(0, 0.50),
    ])
    results = r.rerank("query", candidates)
    assert results[0]["id"] == "doc_2"
    assert results[1]["id"] == "doc_0"


def test_rerank_empty_candidates_returns_empty():
    r = _reranker()
    results = r.rerank("query", [])
    assert results == []
    r.co.rerank.assert_not_called()


def test_rerank_top_n_capped_at_candidate_count():
    r = _reranker(top_n=10)
    candidates = _candidates(3)
    r.co.rerank.return_value = _make_response([
        _make_rerank_result(i, 0.9 - i * 0.1) for i in range(3)
    ])
    r.rerank("query", candidates)
    call_kwargs = r.co.rerank.call_args.kwargs
    assert call_kwargs["top_n"] == 3  # capped at len(candidates)


# ---------------------------------------------------------------------------
# get_reranker factory
# ---------------------------------------------------------------------------

def test_get_reranker_reads_config():
    cfg = {
        "reranking": {"model": "rerank-english-v4.0"},
        "retrieval": {"final_top_k": 5},
    }
    with patch("rag.reranker.cohere.ClientV2"):
        rr = get_reranker(cfg=cfg, api_key="test")
    assert rr.model_name == "rerank-english-v4.0"
    assert rr.default_top_n == 5
