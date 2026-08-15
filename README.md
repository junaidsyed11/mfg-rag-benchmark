# Manufacturing RAG Benchmark

A hybrid RAG system built on top of aerospace manufacturing data. Covers synthetic data generation, token-aware chunking, hybrid dense and sparse retrieval, reranking, grounded generation, and a retrieval evaluation harness.

## What this is

Manufacturing datasets are rarely public. They are proprietary, export-controlled, and sensitive. This project generates 500 synthetic aerospace quality documents using Llama 3.3 70B, then builds a full retrieval pipeline on top of them to explore what an enterprise RAG system looks like in practice.

The pipeline covers:

- Synthetic document generation (NCRs, ECRs, supplier audits, corrective action reports, incident reports)
- Token-aware sentence-boundary chunking with 64-token overlap
- Dense retrieval using Cohere Embed v3 and ChromaDB
- Sparse retrieval using BM25Okapi
- Hybrid fusion with Reciprocal Rank Fusion (RRF)
- Reranking with Cohere Rerank v4
- Grounded response generation with Claude Sonnet via OpenRouter
- Evaluation harness with Hit Rate, MRR, and NDCG@5

## Results

Evaluated against a golden dataset of 27 queries derived from document metadata:

| Metric | Score | What it measures |
|---|---|---|
| Hit Rate@5 | 0.89 | At least one correct document in top 5 (24/27 queries) |
| MRR@5 | 0.74 | Average rank of the first correct result |
| NDCG@5 | 0.62 | Quality of the full top-5 ranking |

An MRR of 0.74 means the first relevant document is landing at rank 1 or 2 on average.

## Architecture

Two phases: index time and query time.

**Index time:** synthetic documents are chunked, embedded with Cohere Embed v3, and stored in ChromaDB. A BM25 index is built in memory from the same chunks.

**Query time:** the query is embedded and used for dense retrieval (cosine similarity via HNSW). BM25 handles sparse retrieval in parallel. RRF merges both ranked lists into one. Cohere Rerank v4 re-scores the top 20 fused candidates and returns the top 5. Claude Sonnet generates a grounded answer from those 5 chunks.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# Fill in: COHERE_API_KEY, OPENROUTER_API_KEY
```

## Running the pipeline

```bash
# Generate synthetic data
python scripts/generate_synthetic.py

# Chunk and ingest into ChromaDB
python scripts/ingest.py

# Ask a question
python scripts/query.py "Which NCRs involved porosity defects?"

# Build the golden dataset and run evals
python scripts/build_golden.py
python scripts/run_eval.py
```

## Project structure

```
src/rag/
    chunker.py      token-aware sentence-boundary chunker
    embedder.py     Cohere Embed v3 wrapper
    index.py        ChromaDB vector index
    retriever.py    hybrid retriever (dense + BM25)
    fusion.py       Reciprocal Rank Fusion
    reranker.py     Cohere Rerank v4 wrapper
    generator.py    Claude Sonnet generation
    eval/           Hit Rate, MRR, NDCG metrics

scripts/
    generate_synthetic.py   LLM-based document generation
    ingest.py               chunk + embed + store
    build_golden.py         generate evaluation queries
    run_eval.py             full evaluation run
    query.py                single-query interface

data/
    synthetic/              generated JSONL documents
    chunks/                 chunked output

config.yaml                 all hyperparameters in one place
```

## Configuration

All pipeline parameters are in `config.yaml`: chunk size, overlap, top-k values, RRF constant, model names, sampling parameters. No magic numbers scattered through the code.

## Document types

| Type | Count | Description |
|---|---|---|
| NCR (Non-Conformance Report) | 150 | Failed inspection records |
| ECR (Engineering Change Request) | 100 | Drawing and specification change requests |
| Supplier Audit | 100 | Quality audits of supplier processes |
| Corrective Action Report | 100 | Root cause analysis and action plans |
| Incident Report | 50 | Safety and process incidents |

## What's next

- Run the `--no-rerank` baseline to measure how much the reranker contributes
- Expand the golden dataset to 100+ queries using LLM-generated questions from document text
- Explore unstructured and multimodal data in future iterations
