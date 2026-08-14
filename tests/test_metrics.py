import math

import pytest

from rag.eval.metrics import dcg, evaluate, hit_rate, ndcg, reciprocal_rank


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

def test_hit_rate_first_result():
    assert hit_rate(["a", "b", "c"], {"a"}, k=5) == 1.0

def test_hit_rate_last_in_k():
    assert hit_rate(["x", "y", "a"], {"a"}, k=3) == 1.0

def test_hit_rate_outside_k():
    assert hit_rate(["x", "y", "a"], {"a"}, k=2) == 0.0

def test_hit_rate_no_relevant():
    assert hit_rate(["a", "b"], {"z"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

def test_rr_rank_1():
    assert reciprocal_rank(["a", "b", "c"], {"a"}, k=5) == pytest.approx(1.0)

def test_rr_rank_2():
    assert reciprocal_rank(["x", "a", "c"], {"a"}, k=5) == pytest.approx(0.5)

def test_rr_rank_5():
    assert reciprocal_rank(["x", "x2", "x3", "x4", "a"], {"a"}, k=5) == pytest.approx(0.2)

def test_rr_outside_k():
    assert reciprocal_rank(["x", "x2", "a"], {"a"}, k=2) == pytest.approx(0.0)

def test_rr_no_relevant():
    assert reciprocal_rank(["a", "b"], {"z"}, k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ndcg
# ---------------------------------------------------------------------------

def test_ndcg_perfect():
    # Only one relevant doc, ranked first → NDCG = 1.0
    assert ndcg(["a", "b", "c"], {"a"}, k=5) == pytest.approx(1.0)

def test_ndcg_zero_when_no_hit():
    assert ndcg(["x", "y", "z"], {"a"}, k=5) == pytest.approx(0.0)

def test_ndcg_partial():
    # Relevant at rank 2: DCG = 1/log2(3), Ideal = 1/log2(2) = 1.0
    score = ndcg(["x", "a", "y"], {"a"}, k=5)
    expected = (1.0 / math.log2(3)) / 1.0
    assert score == pytest.approx(expected)

def test_ndcg_no_relevant_docs():
    assert ndcg(["a", "b"], set(), k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_empty():
    result = evaluate([])
    assert result["n_queries"] == 0
    assert result["mrr"] == 0.0

def test_evaluate_all_hits():
    queries = [
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},
        {"retrieved_ids": ["c", "d"], "relevant_ids": ["c"]},
    ]
    result = evaluate(queries, k=5)
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["n_queries"] == 2

def test_evaluate_no_hits():
    queries = [
        {"retrieved_ids": ["x", "y"], "relevant_ids": ["a"]},
    ]
    result = evaluate(queries, k=5)
    assert result["hit_rate"] == pytest.approx(0.0)
    assert result["mrr"] == pytest.approx(0.0)

def test_evaluate_mixed():
    queries = [
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},  # hit @ 1
        {"retrieved_ids": ["x", "y"], "relevant_ids": ["z"]},  # miss
    ]
    result = evaluate(queries, k=5)
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
