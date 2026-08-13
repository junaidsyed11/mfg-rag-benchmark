from unittest.mock import MagicMock, patch

from rag.embedder import CohereEmbedder, get_embedder


def _make_mock_response(vectors: list[list[float]]):
    """Build a mock Cohere embed response with the v2 SDK shape."""
    response = MagicMock()
    response.embeddings.float_ = vectors
    return response


def _embedder(model="embed-english-v4.0") -> CohereEmbedder:
    with patch("rag.embedder.cohere.ClientV2"):
        return CohereEmbedder(
            model=model,
            input_type_doc="search_document",
            input_type_query="search_query",
            api_key="test-key",
        )


# ---------------------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------------------

def test_embed_documents_returns_one_vector_per_text():
    emb = _embedder()
    fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    emb.co.embed.return_value = _make_mock_response(fake_vectors)

    result = emb.embed_documents(["chunk one", "chunk two"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]


def test_embed_documents_uses_search_document_input_type():
    emb = _embedder()
    emb.co.embed.return_value = _make_mock_response([[0.0, 0.1]])

    emb.embed_documents(["some text"])

    call_kwargs = emb.co.embed.call_args.kwargs
    assert call_kwargs["input_type"] == "search_document"


def test_embed_documents_batches_large_inputs():
    emb = _embedder()
    # 200 texts — should trigger 3 batches (96 + 96 + 8)
    texts = [f"chunk {i}" for i in range(200)]
    single_vector = [0.1] * 10

    def side_effect(**kwargs):
        n = len(kwargs["texts"])
        return _make_mock_response([single_vector] * n)

    emb.co.embed.side_effect = side_effect

    result = emb.embed_documents(texts)

    assert emb.co.embed.call_count == 3
    assert len(result) == 200


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

def test_embed_query_returns_single_vector():
    emb = _embedder()
    emb.co.embed.return_value = _make_mock_response([[0.7, 0.8, 0.9]])

    result = emb.embed_query("find all NCRs with porosity defects")

    assert isinstance(result, list)
    assert result == [0.7, 0.8, 0.9]


def test_embed_query_uses_search_query_input_type():
    emb = _embedder()
    emb.co.embed.return_value = _make_mock_response([[0.1, 0.2]])

    emb.embed_query("supplier audit findings")

    call_kwargs = emb.co.embed.call_args.kwargs
    assert call_kwargs["input_type"] == "search_query"


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------

def test_get_embedder_v4():
    cfg = {
        "embedding": {
            "backend": "cohere-v4",
            "models": {
                "cohere-v4": "embed-english-v4.0",
                "cohere-v3": "embed-english-v3.0",
            },
            "input_type": {
                "document": "search_document",
                "query": "search_query",
            },
        }
    }
    with patch("rag.embedder.cohere.ClientV2"):
        emb = get_embedder(cfg=cfg, api_key="test-key")

    assert emb.model_name == "embed-english-v4.0"


def test_get_embedder_v3():
    cfg = {
        "embedding": {
            "backend": "cohere-v3",
            "models": {
                "cohere-v4": "embed-english-v4.0",
                "cohere-v3": "embed-english-v3.0",
            },
            "input_type": {
                "document": "search_document",
                "query": "search_query",
            },
        }
    }
    with patch("rag.embedder.cohere.ClientV2"):
        emb = get_embedder(cfg=cfg, api_key="test-key")

    assert emb.model_name == "embed-english-v3.0"
