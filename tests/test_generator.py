from unittest.mock import MagicMock, patch

from rag.generator import ClaudeGenerator, _format_context, get_generator


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def test_format_context_includes_doc_id():
    chunks = [{"doc_id": "NCR-001", "doc_type": "NCR",
               "document": "Porosity detected.", "supplier": "Apex"}]
    ctx = _format_context(chunks)
    assert "NCR-001" in ctx
    assert "Porosity detected." in ctx


def test_format_context_numbers_chunks():
    chunks = [
        {"doc_id": "A", "doc_type": "NCR", "document": "text a", "supplier": ""},
        {"doc_id": "B", "doc_type": "ECR", "document": "text b", "supplier": ""},
    ]
    ctx = _format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_includes_supplier():
    chunks = [{"doc_id": "X", "doc_type": "NCR",
               "document": "text", "supplier": "Apex Precision"}]
    ctx = _format_context(chunks)
    assert "Apex Precision" in ctx


def test_format_context_handles_missing_supplier():
    chunks = [{"doc_id": "X", "doc_type": "NCR", "document": "text"}]
    ctx = _format_context(chunks)
    assert "X" in ctx


# ---------------------------------------------------------------------------
# ClaudeGenerator
# ---------------------------------------------------------------------------

def _generator() -> ClaudeGenerator:
    with patch("rag.generator.anthropic.Anthropic"):
        return ClaudeGenerator(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            api_key="test-key",
        )


def _mock_response(text: str):
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.content = [content]
    return resp


def test_generate_returns_string():
    g = _generator()
    g.client.messages.create.return_value = _mock_response("The answer is X.")
    chunks = [{"doc_id": "NCR-001", "doc_type": "NCR",
               "document": "Porosity found.", "supplier": "Apex"}]
    result = g.generate("What defects were found?", chunks)
    assert isinstance(result, str)
    assert result == "The answer is X."


def test_generate_empty_chunks_returns_no_docs_message():
    g = _generator()
    result = g.generate("any query", [])
    assert "No relevant documents" in result
    g.client.messages.create.assert_not_called()


def test_generate_passes_query_in_message():
    g = _generator()
    g.client.messages.create.return_value = _mock_response("answer")
    g.generate("find all porosity NCRs", [
        {"doc_id": "X", "doc_type": "NCR", "document": "text", "supplier": ""}
    ])
    call_kwargs = g.client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "find all porosity NCRs" in user_content


def test_generate_uses_system_prompt():
    g = _generator()
    g.client.messages.create.return_value = _mock_response("answer")
    g.generate("query", [{"doc_id": "X", "doc_type": "NCR",
                          "document": "text", "supplier": ""}])
    call_kwargs = g.client.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert len(call_kwargs["system"]) > 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_generator_reads_config():
    cfg = {"generation": {"model": "claude-sonnet-4-6", "max_tokens": 512}}
    with patch("rag.generator.anthropic.Anthropic"):
        g = get_generator(cfg=cfg, api_key="test")
    assert g.model_name == "claude-sonnet-4-6"
    assert g.max_tokens == 512
