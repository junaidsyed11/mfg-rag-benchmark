"""
Build the ChromaDB vector index from chunked documents.

Reads all JSONL files from data/chunks/, embeds every chunk using the
active Cohere model (set in config.yaml), and upserts into a persistent
ChromaDB collection. Run once per backend:

    # Build the v4 index (default)
    python scripts/build_index.py

    # Switch config.yaml backend to cohere-v3, then:
    python scripts/build_index.py

    # Check what's in the index
    python scripts/build_index.py --stats

    # Wipe and rebuild from scratch
    python scripts/build_index.py --reset

The script is idempotent by default — ChromaDB upserts so re-running
won't duplicate documents. Use --reset only if you want to change
chunk sizes or re-generate the source data.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_api_key, load_config
from rag.embedder import get_embedder
from rag.index import get_index

_EMBED_BATCH = 96   # Cohere API max
_UPSERT_BATCH = 500  # ChromaDB comfortable batch size


def load_all_chunks(chunk_dir: Path) -> list[dict]:
    chunks = []
    for jsonl_file in sorted(chunk_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def chunk_id(chunk: dict) -> str:
    return f"{chunk['doc_id']}_c{chunk.get('chunk_index', 0)}"


def build(chunks: list[dict], embedder, index, verbose: bool = True) -> None:
    total = len(chunks)
    print(f"  Embedding {total} chunks with {embedder.model_name} ...")

    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []

    # Embed in batches, collect results
    for start in range(0, total, _EMBED_BATCH):
        batch = chunks[start : start + _EMBED_BATCH]
        texts = [c["text"] for c in batch]
        vectors = embedder.embed_documents(texts)

        for chunk, vec in zip(batch, vectors):
            cid = chunk_id(chunk)
            meta = {k: v for k, v in chunk.items() if k != "text"}
            all_ids.append(cid)
            all_embeddings.append(vec)
            all_documents.append(chunk["text"])
            all_metadatas.append(meta)

        done = min(start + _EMBED_BATCH, total)
        if verbose:
            pct = int(done / total * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {done}/{total}", end="", flush=True)

    print()  # newline after progress bar

    # Upsert into ChromaDB in batches
    print(f"  Writing to collection '{index.name}' ...")
    for start in range(0, total, _UPSERT_BATCH):
        index.add(
            ids=all_ids[start : start + _UPSERT_BATCH],
            embeddings=all_embeddings[start : start + _UPSERT_BATCH],
            documents=all_documents[start : start + _UPSERT_BATCH],
            metadatas=all_metadatas[start : start + _UPSERT_BATCH],
        )

    print(f"  Done — {index.count()} vectors in collection.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the collection before rebuilding")
    parser.add_argument("--stats", action="store_true",
                        help="Print collection stats and exit")
    args = parser.parse_args()

    cfg = load_config()
    api_key = get_api_key("COHERE_API_KEY")

    embedder = get_embedder(cfg=cfg, api_key=api_key)
    index = get_index(cfg=cfg)

    if args.stats:
        print(f"\nCollection : {index.name}")
        print(f"Vectors    : {index.count():,}")
        print(f"Model      : {embedder.model_name}")
        return

    chunk_dir = Path("data/chunks")
    if not chunk_dir.exists() or not list(chunk_dir.glob("*.jsonl")):
        print("ERROR: data/chunks/ is empty — run scripts/ingest.py first.")
        sys.exit(1)

    chunks = load_all_chunks(chunk_dir)
    print(f"\nLoaded {len(chunks)} chunks from {chunk_dir}/")
    print(f"Backend    : {cfg['embedding']['backend']}")
    print(f"Model      : {embedder.model_name}")
    print(f"Collection : {index.name}")

    if args.reset:
        print("  Resetting collection ...")
        index.reset()

    build(chunks, embedder, index)
    print("\nIndex built successfully.")


if __name__ == "__main__":
    main()
