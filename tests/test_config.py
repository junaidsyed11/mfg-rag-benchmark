from rag.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["embedding"]["backend"] in ("cohere-v4", "cohere-v3")
    assert cfg["chunking"]["chunk_size"] > 0
    assert cfg["retrieval"]["rrf_k"] > 0


def test_config_has_required_keys():
    cfg = load_config()
    for key in ("embedding", "chunking", "retrieval", "reranking", "generation", "paths"):
        assert key in cfg, f"Missing top-level config key: {key}"
