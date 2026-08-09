# Manufacturing / Supply Chain RAG Benchmark

Hybrid search + rerank retrieval-augmented generation system for manufacturing and supply chain documents.

Compares **Cohere Embed v4** vs **EmbeddingGemma** as embedding backends, using hybrid dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion, reranked with **Cohere Rerank v4**, and generating answers via the **Anthropic Claude API**.

## Architecture

_Diagram coming in Step 12._

## Results

_Benchmark table coming in Step 11._

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy and fill in your API keys
cp .env.example .env
```

## Project Structure

```
src/rag/
├── embedders/     # Cohere + Gemma embedding backends
├── retrieval/     # Dense index, BM25, RRF fusion
├── generation/    # Anthropic Claude generation step
└── eval/          # Recall@k, MRR eval harness

data/
├── synthetic/     # LLM-generated NCR/ECR/supplier docs
└── raw/           # Public datasets (Kaggle DataCo, etc.)

scripts/           # One-off data generation + ingest scripts
tests/             # pytest test suite
```
