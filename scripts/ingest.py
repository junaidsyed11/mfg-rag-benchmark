"""
Document ingest and chunking pipeline.

Reads all JSONL files from data/synthetic/ and data/raw/asrs.jsonl,
chunks each document using the token-aware Chunker, and writes the
results to data/chunks/ — one JSONL file per source doc type.

Chunk files are what the embedding step consumes.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --source synthetic   # only synthetic docs
    python scripts/ingest.py --source asrs        # only NASA ASRS docs
    python scripts/ingest.py --stats              # print stats and exit
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunker import Chunker
from rag.config import load_config


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def ingest_file(src: Path, chunker: Chunker, out_dir: Path) -> dict:
    docs = load_jsonl(src)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))

    out_file = out_dir / src.name
    with open(out_file, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    return {
        "source": src.name,
        "documents": len(docs),
        "chunks": len(all_chunks),
        "avg_chunks_per_doc": round(len(all_chunks) / len(docs), 1) if docs else 0,
    }


def print_stats(chunk_dir: Path) -> None:
    total_chunks = 0
    print(f"\n{'Source':<30} {'Chunks':>8}")
    print("-" * 40)
    for f in sorted(chunk_dir.glob("*.jsonl")):
        count = sum(1 for line in open(f) if line.strip())
        print(f"  {f.name:<28} {count:>8,}")
        total_chunks += count
    print("-" * 40)
    print(f"  {'TOTAL':<28} {total_chunks:>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and chunk documents")
    parser.add_argument(
        "--source",
        choices=["synthetic", "asrs", "all"],
        default="all",
        help="Which data source to process (default: all)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print chunk statistics and exit",
    )
    args = parser.parse_args()

    cfg = load_config()
    chunker = Chunker(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )

    synthetic_dir = Path(cfg["paths"]["synthetic_data"])
    raw_dir = Path(cfg["paths"]["raw_data"])
    chunk_dir = Path("data/chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_stats(chunk_dir)
        return

    results = []

    if args.source in ("synthetic", "all"):
        if not synthetic_dir.exists():
            print(f"WARNING: {synthetic_dir} not found — run generate_synthetic.py first")
        else:
            for jsonl_file in sorted(synthetic_dir.glob("*.jsonl")):
                print(f"  Chunking {jsonl_file.name} ...", end=" ", flush=True)
                r = ingest_file(jsonl_file, chunker, chunk_dir)
                results.append(r)
                print(f"{r['documents']} docs → {r['chunks']} chunks "
                      f"(avg {r['avg_chunks_per_doc']}/doc)")

    if args.source in ("asrs", "all"):
        asrs_file = raw_dir / "asrs.jsonl"
        if not asrs_file.exists():
            print(f"WARNING: {asrs_file} not found — run load_asrs.py first")
        else:
            print(f"  Chunking {asrs_file.name} ...", end=" ", flush=True)
            r = ingest_file(asrs_file, chunker, chunk_dir)
            results.append(r)
            print(f"{r['documents']} docs → {r['chunks']} chunks "
                  f"(avg {r['avg_chunks_per_doc']}/doc)")

    if results:
        total_docs = sum(r["documents"] for r in results)
        total_chunks = sum(r["chunks"] for r in results)
        print(f"\nTotal: {total_docs} documents → {total_chunks} chunks")
        print(f"Chunk files written to: {chunk_dir}/")
    else:
        print("Nothing to process.")


if __name__ == "__main__":
    main()
