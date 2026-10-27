
from app.ingestion.chunker import DocumentChunk
from app.rag.vector_store import VectorStore


class FakeEmbeddingService:
    """Provide deterministic vectors for vector-store tests."""

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

#Test that the vector store correctly stores chunks and their metadata.
def test_add_chunks_stores_text_and_metadata(tmp_path):
    store = VectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path / "chroma",
    )

    chunks = [
        DocumentChunk("Authentication details.", "guide.pdf", 2, 0),
        DocumentChunk("Authorization details.", "guide.pdf", 3, 1),
    ]

    stored = store.add_chunks(chunks)

    assert stored == 2
    assert store.collection.count() == 2

    result = store.collection.get(
        include=["documents", "metadatas"]
    )

    assert "Authentication details." in result["documents"]
    assert all(
        metadata["source"] == "guide.pdf"
        for metadata in result["metadatas"]
    )
    assert {metadata["page_number"] for metadata in result["metadatas"]} == {2, 3}

#Test that the vector store raises a ValueError for empty text in chunks.
def test_reindexing_source_replaces_old_chunks(tmp_path):
    store = VectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path / "chroma",
    )

    store.add_chunks([
        DocumentChunk("Old chunk one.", "guide.txt", None, 0),
        DocumentChunk("Old chunk two.", "guide.txt", None, 1),
    ])

    store.add_chunks([
        DocumentChunk("Updated content.", "guide.txt", None, 0),
    ])

    assert store.collection.count() == 1

    result = store.collection.get(include=["documents"])

    assert result["documents"] == ["Updated content."]

#Test that the vector store correctly handles chunks with None as page_number.
def test_add_chunks_handles_empty_input(tmp_path):
    store = VectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path / "chroma",
    )

    assert store.add_chunks([]) == 0
    assert store.collection.count() == 0

#Test that the vector store correctly handles chunks with None as page_number.
def test_add_chunks_omits_unavailable_page_number(tmp_path):
    store = VectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path / "chroma",
    )

    store.add_chunks([
        DocumentChunk("Plain text content.", "notes.txt", None, 0),
    ])

    result = store.collection.get(include=["metadatas"])

    assert "page_number" not in result["metadatas"][0]
