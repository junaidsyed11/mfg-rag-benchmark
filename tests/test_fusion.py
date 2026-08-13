from rag.fusion import fuse, reciprocal_rank_fusion


def _hit(doc_id: str, score: float = 1.0) -> dict:
    return {"id": doc_id, "document": f"text for {doc_id}",
            "metadata": {"doc_type": "NCR"}, "score": score}


# ---------------------------------------------------------------------------
# Core RRF scoring
# ---------------------------------------------------------------------------

def test_document_in_both_lists_scores_higher():
    dense  = [_hit("A"), _hit("B"), _hit("C")]
    sparse = [_hit("A"), _hit("D"), _hit("E")]
    results = fuse(dense, sparse)
    ids = [r["id"] for r in results]
    # A appears in both → should be ranked first
    assert ids[0] == "A"


def test_earlier_rank_scores_higher():
    dense  = [_hit("top"), _hit("bottom")]
    sparse = []
    results = fuse(dense, sparse)
    ids = [r["id"] for r in results]
    assert ids[0] == "top"


def test_rrf_score_formula():
    k = 60
    dense  = [_hit("X")]   # rank 1 in dense
    sparse = [_hit("X")]   # rank 1 in sparse
    results = fuse(dense, sparse, k=k)
    expected = 1.0 / (k + 1) + 1.0 / (k + 1)
    assert abs(results[0]["rrf_score"] - expected) < 1e-9


def test_documents_only_in_one_list_are_included():
    dense  = [_hit("A"), _hit("B")]
    sparse = [_hit("C"), _hit("D")]
    results = fuse(dense, sparse)
    ids = {r["id"] for r in results}
    assert ids == {"A", "B", "C", "D"}


def test_output_sorted_descending_by_rrf_score():
    dense  = [_hit("A"), _hit("B"), _hit("C")]
    sparse = [_hit("B"), _hit("C"), _hit("A")]
    results = fuse(dense, sparse)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_score_field_present():
    results = fuse([_hit("X")], [_hit("Y")])
    for r in results:
        assert "rrf_score" in r


def test_empty_lists_return_empty():
    assert fuse([], []) == []


def test_one_empty_list_still_works():
    dense  = [_hit("A"), _hit("B")]
    results = fuse(dense, [])
    assert len(results) == 2
    assert results[0]["id"] == "A"


# ---------------------------------------------------------------------------
# k constant behaviour
# ---------------------------------------------------------------------------

def test_lower_k_amplifies_top_rank_advantage():
    dense = [_hit("top"), _hit("bottom")]
    r_low  = fuse(dense, [], k=1)
    r_high = fuse(dense, [], k=1000)
    # With low k, gap between rank 1 and rank 2 is larger
    gap_low  = r_low[0]["rrf_score"]  - r_low[1]["rrf_score"]
    gap_high = r_high[0]["rrf_score"] - r_high[1]["rrf_score"]
    assert gap_low > gap_high


# ---------------------------------------------------------------------------
# Multi-list fusion
# ---------------------------------------------------------------------------

def test_three_list_fusion():
    lists = [
        [_hit("A"), _hit("B")],
        [_hit("B"), _hit("C")],
        [_hit("A"), _hit("C")],
    ]
    results = reciprocal_rank_fusion(lists, k=60)
    # A and B and C all appear twice — A at rank 1 twice, C at rank 2 twice
    scores = {r["id"]: r["rrf_score"] for r in results}
    # A: 1/(61)+1/(61), C: 1/(62)+1/(62), B: 1/(61)+1/(62)
    assert scores["A"] > scores["B"] > scores["C"]
