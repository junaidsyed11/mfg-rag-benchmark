"""
Run the retrieval evaluation harness.

For each backend (cohere-v3, cohere-v4), runs every query in the golden
dataset through the full retrieval pipeline (dense + sparse → RRF → rerank)
and measures Hit Rate, MRR, and NDCG at K=5.

The retrieved IDs are matched against relevant_doc_ids from the golden set.
Since one document can produce multiple chunks, a chunk is considered a hit
if its doc_id prefix matches any relevant doc ID.

Results are written to data/eval_results.json and printed as a comparison
table so you can see v3 vs v4 side by side.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --k 3
    python scripts/run_eval.py --backend cohere-v3   # run one backend only
    python scripts/run_eval.py --no-rerank           # skip reranker (baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_api_key, load_config
from rag.embedder import get_embedder
from rag.eval.metrics import evaluate
from rag.fusion import fuse
from rag.index import get_index
from rag.reranker import get_reranker
from rag.retriever import HybridRetriever, load_chunks

_ALL_BACKENDS = ["cohere-v3", "cohere-v4"]


def _doc_id_from_chunk_id(chunk_id: str) -> str:
    """Extract doc_id from chunk ID format '<doc_id>_c<index>'."""
    return chunk_id.rsplit("_c", 1)[0]


def run_query(
    query: str,
    retriever: HybridRetriever,
    reranker,
    cfg: dict,
    k: int,
    use_rerank: bool,
) -> list[str]:
    """Run one query through the pipeline, return ranked doc IDs."""
    top_k = cfg["retrieval"]["top_k"]
    rrf_k = cfg["retrieval"]["rrf_k"]
    rerank_top_n = cfg["retrieval"]["rerank_top_n"]

    dense, sparse = retriever.retrieve(query, top_k=top_k)
    fused = fuse(dense, sparse, k=rrf_k)
    candidates = fused[:rerank_top_n]

    if use_rerank and candidates:
        results = reranker.rerank(query, candidates, top_n=k)
    else:
        results = candidates[:k]

    # Extract doc_ids from chunk IDs
    seen, doc_ids = set(), []
    for r in results:
        chunk_id = r.get("id", "")
        doc_id = _doc_id_from_chunk_id(chunk_id)
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)

    return doc_ids


def eval_backend(
    backend: str,
    golden: list[dict],
    cfg: dict,
    chunks: list[dict],
    k: int,
    use_rerank: bool,
) -> dict:
    """Run evaluation for one embedding backend."""
    print(f"\n  Backend: {backend}")
    cfg = {**cfg, "embedding": {**cfg["embedding"], "backend": backend}}

    cohere_key = get_api_key("COHERE_API_KEY")
    embedder = get_embedder(cfg=cfg, api_key=cohere_key)
    index = get_index(cfg=cfg)

    if index.count() == 0:
        print(f"  WARNING: collection for {backend} is empty — run build_index.py first")
        return {}

    retriever = HybridRetriever(index=index, embedder=embedder, chunks=chunks)
    reranker = get_reranker(cfg=cfg, api_key=cohere_key) if use_rerank else None

    evaluated = []
    for i, q in enumerate(golden, 1):
        print(f"  [{i:02d}/{len(golden)}] {q['query_id']}", end=" ... ", flush=True)
        try:
            retrieved_ids = run_query(
                query=q["query"],
                retriever=retriever,
                reranker=reranker,
                cfg=cfg,
                k=k,
                use_rerank=use_rerank,
            )
            evaluated.append({
                "query_id":     q["query_id"],
                "query":        q["query"],
                "retrieved_ids": retrieved_ids,
                "relevant_ids": q["relevant_doc_ids"],
            })
            # Check if we got a hit
            relevant = set(q["relevant_doc_ids"])
            hit = any(rid in relevant for rid in retrieved_ids)
            print("HIT" if hit else "miss")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Small delay to avoid rate limits on Cohere embed
        time.sleep(0.3)

    return evaluate(evaluated, k=k)


def print_table(results: dict[str, dict], k: int) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL EVALUATION RESULTS  (k={k})")
    print(f"{'='*60}")
    print(f"  {'Metric':<15}", end="")
    for backend in results:
        print(f"  {backend:>15}", end="")
    print()
    print(f"  {'-'*55}")
    for metric in ["hit_rate", "mrr", "ndcg"]:
        label = {"hit_rate": f"Hit Rate@{k}", "mrr": f"MRR@{k}", "ndcg": f"NDCG@{k}"}[metric]
        print(f"  {label:<15}", end="")
        for backend, res in results.items():
            val = res.get(metric, "-")
            print(f"  {val:>15}", end="")
        print()
    print(f"  {'-'*55}")
    print(f"  {'Queries':<15}", end="")
    for backend, res in results.items():
        print(f"  {res.get('n_queries', '-'):>15}", end="")
    print()
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff (default: 5)")
    parser.add_argument("--backend", choices=_ALL_BACKENDS,
                        help="Run one backend only (default: both)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip reranker — evaluate RRF output directly")
    args = parser.parse_args()

    cfg = load_config()
    chunk_dir = Path("data/chunks")
    golden_path = Path("data/golden.jsonl")

    if not golden_path.exists():
        print("ERROR: data/golden.jsonl not found — run scripts/build_golden.py first")
        sys.exit(1)

    if not chunk_dir.exists() or not list(chunk_dir.glob("*.jsonl")):
        print("ERROR: data/chunks/ is empty — run scripts/ingest.py first")
        sys.exit(1)

    with open(golden_path) as f:
        golden = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(golden)} golden queries")

    chunks = load_chunks(chunk_dir)
    print(f"Loaded {len(chunks)} chunks")

    backends = [args.backend] if args.backend else _ALL_BACKENDS
    use_rerank = not args.no_rerank
    print(f"Reranker: {'enabled' if use_rerank else 'disabled (baseline)'}")

    all_results = {}
    for backend in backends:
        res = eval_backend(
            backend=backend,
            golden=golden,
            cfg=cfg,
            chunks=chunks,
            k=args.k,
            use_rerank=use_rerank,
        )
        if res:
            all_results[backend] = res

    if not all_results:
        print("No results — check that indices are built for both backends.")
        sys.exit(1)

    print_table(all_results, k=args.k)

    out = {
        "k": args.k,
        "reranker": use_rerank,
        "results": all_results,
    }
    out_path = Path("data/eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
