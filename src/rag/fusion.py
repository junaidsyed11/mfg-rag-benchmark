"""
Reciprocal Rank Fusion (RRF).

Merges multiple ranked result lists into a single ranked list using the
formula from Cormack, Clarke & Buettcher (SIGIR 2009):

    score(d) = Σ 1 / (k + rank(d))

where rank is 1-based and the sum is over all result lists. Documents
that appear in multiple lists accumulate score from each — this is the
key property that makes RRF effective for hybrid search.

k=60 is the standard constant from the original paper. It controls how
aggressively top ranks are rewarded vs the tail. Lower k amplifies top
ranks; higher k flattens the curve. 60 generalises well without tuning.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists into one using RRF scoring.

    Each result dict must have an "id" field. All other fields are
    preserved from whichever list first introduced that document.
    An "rrf_score" field is added to every result.

    Args:
        result_lists: e.g. [dense_results, sparse_results]
        k: RRF constant (default 60, from original paper)

    Returns:
        Single list sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**docs[doc_id], "rrf_score": score} for doc_id, score in ranked]


def fuse(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Convenience wrapper for the two-list hybrid search case."""
    return reciprocal_rank_fusion([dense_results, sparse_results], k=k)
