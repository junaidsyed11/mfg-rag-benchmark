"""
Retrieval evaluation metrics.

All metrics operate on ranked result lists (list of retrieved doc IDs)
and a set of relevant doc IDs (ground truth).

Metrics implemented:
  - Hit Rate @ K   : at least one relevant doc appears in the top K
  - MRR @ K        : Mean Reciprocal Rank — how high up is the first hit?
  - NDCG @ K       : Normalised Discounted Cumulative Gain — how well are
                     all relevant docs ranked across the top K?
"""

from __future__ import annotations

import math


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc appears in top-k retrieved, else 0.0."""
    return float(any(doc_id in relevant for doc_id in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Reciprocal rank of the first relevant result in the top-k list.
    Returns 0.0 if no relevant result appears in the top k.
    """
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain — binary relevance (1 if relevant, 0 if not)."""
    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalised DCG. Divides actual DCG by the ideal DCG (all relevant docs
    ranked at the top). Returns 0.0 if there are no relevant docs.
    """
    actual = dcg(retrieved, relevant, k)
    # Ideal: relevant docs ranked 1, 2, 3, ...
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return actual / ideal if ideal > 0 else 0.0


def evaluate(
    queries: list[dict],
    k: int = 5,
) -> dict[str, float]:
    """
    Compute aggregate metrics over a list of evaluated queries.

    Each query dict must have:
      retrieved_ids : list[str]  — ranked list of retrieved chunk/doc IDs
      relevant_ids  : list[str]  — ground-truth relevant doc IDs

    Returns a dict with mean Hit Rate, MRR, and NDCG at k.
    """
    if not queries:
        return {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "k": k, "n_queries": 0}

    hit_rates, mrrs, ndcgs = [], [], []

    for q in queries:
        retrieved = q["retrieved_ids"]
        relevant = set(q["relevant_ids"])

        hit_rates.append(hit_rate(retrieved, relevant, k))
        mrrs.append(reciprocal_rank(retrieved, relevant, k))
        ndcgs.append(ndcg(retrieved, relevant, k))

    n = len(queries)
    return {
        "hit_rate": round(sum(hit_rates) / n, 4),
        "mrr":      round(sum(mrrs) / n, 4),
        "ndcg":     round(sum(ndcgs) / n, 4),
        "k":        k,
        "n_queries": n,
    }
