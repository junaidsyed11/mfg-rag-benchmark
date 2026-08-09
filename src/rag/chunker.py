"""
Token-aware text chunker.

Splits document text into overlapping chunks that respect sentence boundaries
where possible. Token counts use tiktoken (cl100k_base) as a consistent
approximation — Cohere's tokenizer isn't public, but cl100k_base counts are
within a few percent for English technical text.

Each output chunk carries the full parent document metadata plus:
  chunk_index  — position within the document (0-based)
  chunk_count  — total chunks for this document
  text         — the chunk text (replaces the full document text)
"""

from __future__ import annotations

import re
from typing import Any

import tiktoken

# cl100k_base is GPT-4/Cohere-adjacent; accurate enough for sizing chunks
_ENC = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks of at most chunk_size tokens with chunk_overlap
        token overlap between consecutive chunks.

        Strategy: accumulate sentences until the chunk would exceed chunk_size,
        then start a new chunk that begins with enough trailing sentences from
        the previous chunk to cover chunk_overlap tokens.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = _token_len(sentence)

            # A single sentence longer than chunk_size — include it alone
            if s_tokens > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                chunks.append(sentence)
                current = []
                current_tokens = 0
                continue

            if current_tokens + s_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Roll back enough sentences to cover the overlap window
                overlap: list[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    pt = _token_len(prev)
                    if overlap_tokens + pt > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += pt
                current = overlap
                current_tokens = overlap_tokens

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def chunk_document(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Chunk a single document dict.

        Returns a list of chunk dicts. Each has all parent metadata fields
        (except 'text') plus chunk_index, chunk_count, and the chunk text.
        """
        text = doc.get("text", "")
        chunks = self.chunk_text(text)
        if not chunks:
            return []

        base = {k: v for k, v in doc.items() if k != "text"}
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                **base,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": chunk_text,
            })
        return result
