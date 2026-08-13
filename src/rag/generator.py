"""
Claude generation step via OpenRouter.

Takes a query and the top-k reranked chunks, formats them as context,
and calls Claude through OpenRouter to generate a grounded answer.
Answers are sourced only from the provided context — the prompt explicitly
instructs Claude not to use outside knowledge, keeping evaluation honest.
"""

from __future__ import annotations

from typing import Any

import requests

from rag.config import get_api_key, load_config

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_REFERER = "https://github.com/junaidsyed11/mfg-rag-benchmark"

_SYSTEM_PROMPT = """\
You are a quality engineering assistant for an aerospace manufacturing company.
Answer the user's question using ONLY the information in the provided context documents.
If the answer cannot be found in the context, say so clearly — do not guess.
When referencing specific information, cite the document ID (e.g. "per NCR-2024-00042").
Be concise and precise. Use technical language appropriate for a quality engineer.\
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("doc_id", "unknown")
        doc_type = chunk.get("doc_type", "")
        supplier = chunk.get("supplier", "")
        header = f"[{i}] {doc_id}"
        if doc_type:
            header += f" ({doc_type}"
            if supplier:
                header += f" · {supplier}"
            header += ")"
        parts.append(f"{header}:\n{chunk['document']}")
    return "\n\n".join(parts)


class ClaudeGenerator:
    def __init__(self, model: str, max_tokens: int, api_key: str) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """Generate an answer grounded in the provided chunks."""
        if not chunks:
            return "No relevant documents found for this query."

        context = _format_context(chunks)
        user_message = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        response = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": _REFERER,
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": self.max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    @property
    def model_name(self) -> str:
        return self.model


def get_generator(cfg: dict | None = None, api_key: str | None = None) -> ClaudeGenerator:
    """Build a ClaudeGenerator from config."""
    if cfg is None:
        cfg = load_config()
    if api_key is None:
        api_key = get_api_key("OPENROUTER_API_KEY")

    return ClaudeGenerator(
        model=cfg["generation"]["model"],
        max_tokens=cfg["generation"]["max_tokens"],
        api_key=api_key,
    )
