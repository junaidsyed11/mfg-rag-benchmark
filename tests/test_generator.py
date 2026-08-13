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

def _mock_response(text: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return resp


def _generator() -> ClaudeGenerator:
    return ClaudeGenerator(
        model="anthropic/claude-sonnet-4-6",
        max_tokens=1024,
        api_key="test-key",
    )


def test_generate_returns_string():
    g = _generator()
    with patch("rag.generator.requests.post", return_value=_mock_response("The answer is X.")):
        chunks = [{"doc_id": "NCR-001", "doc_type": "NCR",
                   "document": "Porosity found.", "supplier": "Apex"}]
        result = g.generate("What defects were found?", chunks)
    assert result == "The answer is X."


def test_generate_empty_chunks_returns_no_docs_message():
    g = _generator()
    with patch("rag.generator.requests.post") as mock_post:
        result = g.generate("any query", [])
    assert "No relevant documents" in result
    mock_post.assert_not_called()


def test_generate_passes_query_in_request():
    g = _generator()
    with patch("rag.generator.requests.post", return_value=_mock_response("ans")) as mock_post:
        g.generate("find all porosity NCRs", [
            {"doc_id": "X", "doc_type": "NCR", "document": "text", "supplier": ""}
        ])
    payload = mock_post.call_args.kwargs["json"]
    user_content = payload["messages"][1]["content"]
    assert "find all porosity NCRs" in user_content


def test_generate_includes_system_prompt():
    g = _generator()
    with patch("rag.generator.requests.post", return_value=_mock_response("ans")) as mock_post:
        g.generate("query", [{"doc_id": "X", "doc_type": "NCR",
                               "document": "text", "supplier": ""}])
    payload = mock_post.call_args.kwargs["json"]
    assert payload["messages"][0]["role"] == "system"
    assert len(payload["messages"][0]["content"]) > 0


def test_generate_uses_openrouter_url():
    g = _generator()
    with patch("rag.generator.requests.post", return_value=_mock_response("ans")) as mock_post:
        g.generate("query", [{"doc_id": "X", "doc_type": "NCR",
                               "document": "text", "supplier": ""}])
    url = mock_post.call_args.args[0]
    assert "openrouter.ai" in url


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_generator_reads_config():
    cfg = {"generation": {"model": "anthropic/claude-sonnet-4-6", "max_tokens": 512}}
    g = get_generator(cfg=cfg, api_key="test")
    assert g.model_name == "anthropic/claude-sonnet-4-6"
    assert g.max_tokens == 512
