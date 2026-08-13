"""
End-to-end RAG query pipeline.

Runs a query through the full pipeline:
  1. Hybrid retrieval — dense (ChromaDB) + sparse (BM25)
  2. RRF fusion — merge ranked lists
  3. Cohere Rerank v4 — cross-encoder re-scoring
  4. Claude generation — grounded answer from top-5 chunks

Usage:
    python scripts/query.py --query "What NCRs involved porosity defects?"
    python scripts/query.py --query "Which suppliers had corrective actions?" --top-k 3
    python scripts/query.py --query "..." --no-generate   # retrieval only, skip Claude
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_api_key, load_config
from rag.embedder import get_embedder
from rag.fusion import fuse
from rag.generator import get_generator
from rag.index import get_index
from rag.reranker import get_reranker
from rag.retriever import HybridRetriever, load_chunks


def run_pipeline(query: str, cfg: dict, top_k: int | None, generate: bool) -> None:
    print(f"\nQuery : {query}")
    print(f"Backend : {cfg['embedding']['backend']}\n")

    # --- Load components ---
    chunk_dir = Path("data/chunks")
    chunks = load_chunks(chunk_dir)
    embedder = get_embedder(cfg=cfg)
    index = get_index(cfg=cfg)
    retriever = HybridRetriever(index=index, embedder=embedder, chunks=chunks)
    reranker = get_reranker(cfg=cfg)

    # --- Retrieve ---
    rk = cfg["retrieval"]["top_k"]
    print(f"[1/4] Retrieving top-{rk} dense + top-{rk} sparse ...")
    dense, sparse = retriever.retrieve(query, top_k=rk)
    print(f"      Dense hits : {len(dense)}")
    print(f"      Sparse hits: {len(sparse)}")

    # --- Fuse ---
    print(f"[2/4] RRF fusion (k={cfg['retrieval']['rrf_k']}) ...")
    fused = fuse(dense, sparse, k=cfg["retrieval"]["rrf_k"])
    rerank_n = cfg["retrieval"]["rerank_top_n"]
    candidates = fused[:rerank_n]
    print(f"      Fused candidates: {len(fused)} → passing top-{rerank_n} to reranker")

    # --- Rerank ---
    final_k = top_k or cfg["retrieval"]["final_top_k"]
    print(f"[3/4] Reranking with {cfg['reranking']['model']} → top-{final_k} ...")
    results = reranker.rerank(query, candidates, top_n=final_k)

    # --- Show retrieved chunks ---
    print(f"\n{'─'*60}")
    print(f"TOP {len(results)} CHUNKS")
    print(f"{'─'*60}")
    for i, r in enumerate(results, 1):
        doc_id = r.get("doc_id", "?")
        doc_type = r.get("doc_type", "?")
        supplier = r.get("supplier", "")
        score = r.get("rerank_score", 0)
        print(f"\n[{i}] {doc_id} ({doc_type}{' · ' + supplier if supplier else ''})")
        print(f"    Rerank score: {score:.4f}")
        print(f"    {r['document'][:200]}{'...' if len(r['document']) > 200 else ''}")

    # --- Generate ---
    if generate:
        print(f"\n{'─'*60}")
        print("ANSWER")
        print(f"{'─'*60}\n")
        generator = get_generator(cfg=cfg)
        answer = generator.generate(query, results)
        print(answer)

    print(f"\n{'─'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG query end-to-end")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of final chunks (default: config final_top_k)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip Claude generation, show retrieved chunks only")
    args = parser.parse_args()

    cfg = load_config()
    run_pipeline(
        query=args.query,
        cfg=cfg,
        top_k=args.top_k,
        generate=not args.no_generate,
    )


if __name__ == "__main__":
    main()
