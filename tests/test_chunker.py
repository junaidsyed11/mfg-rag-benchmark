from rag.chunker import Chunker


def test_chunk_short_text_is_single_chunk():
    c = Chunker(chunk_size=512, chunk_overlap=64)
    chunks = c.chunk_text("This is a short sentence. It fits in one chunk.")
    assert len(chunks) == 1


def test_chunk_long_text_splits():
    c = Chunker(chunk_size=20, chunk_overlap=5)
    # Build text that clearly exceeds 20 tokens
    long_text = " ".join(["The turbine blade failed dimensional inspection."] * 10)
    chunks = c.chunk_text(long_text)
    assert len(chunks) > 1


def test_chunk_overlap_present():
    c = Chunker(chunk_size=30, chunk_overlap=10)
    sentences = [
        "Inspection revealed a surface finish non-conformance on the hydraulic actuator.",
        "The part was measured using a calibrated CMM at incoming inspection.",
        "Results showed deviations of 0.15mm beyond the tolerance band specified.",
        "A disposition of Return to Supplier was issued pending corrective action.",
        "The supplier was notified and a formal corrective action request was raised.",
    ]
    text = " ".join(sentences)
    chunks = c.chunk_text(text)
    # With overlap, content from the end of chunk N should appear at start of chunk N+1
    if len(chunks) > 1:
        # Last sentence of chunk 0 should appear somewhere in chunk 1
        last_words_chunk0 = chunks[0].split()[-5:]
        combined = " ".join(last_words_chunk0)
        assert any(word in chunks[1] for word in last_words_chunk0), (
            f"No overlap found between chunks.\nChunk 0 tail: {combined}\nChunk 1: {chunks[1][:200]}"
        )


def test_chunk_document_preserves_metadata():
    c = Chunker(chunk_size=512, chunk_overlap=64)
    doc = {
        "doc_id": "NCR-2024-00001",
        "doc_type": "NCR",
        "supplier": "Apex Precision",
        "text": "Inspection revealed a non-conformance. The part was scrapped.",
    }
    chunks = c.chunk_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["doc_id"] == "NCR-2024-00001"
        assert chunk["doc_type"] == "NCR"
        assert chunk["supplier"] == "Apex Precision"
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "chunk_count" in chunk


def test_chunk_document_indices_correct():
    c = Chunker(chunk_size=20, chunk_overlap=5)
    long_text = " ".join(["Non-conformance detected on part surface finish."] * 15)
    doc = {"doc_id": "X", "text": long_text}
    chunks = c.chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["chunk_count"] == len(chunks)


def test_empty_text_returns_no_chunks():
    c = Chunker()
    assert c.chunk_text("") == []
    assert c.chunk_document({"doc_id": "X", "text": ""}) == []
