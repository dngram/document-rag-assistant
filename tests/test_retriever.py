
import pytest

from app.ingestion.chunker import DocumentChunk
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class FakeEmbeddingService:
    """Create predictable vectors for retrieval tests."""

    def _vector(self, text):
        text = text.lower()

        authentication = any(
            word in text
            for word in ("authentication", "identity", "verify")
        )
        authorization = any(
            word in text
            for word in ("authorization", "permission")
        )
        database = "database" in text

        return [
            float(authentication),
            float(authorization),
            float(database),
            0.1,
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)

#Test that the retriever returns the most relevant chunk for a given query.
@pytest.fixture
def indexed_store(tmp_path):
    embeddings = FakeEmbeddingService()

    store = VectorStore(
        embedding_service=embeddings,
        persist_directory=tmp_path / "chroma",
    )

    store.add_chunks([
        DocumentChunk(
            "Authentication verifies user identity.",
            "security.pdf",
            4,
            0,
        ),
        DocumentChunk(
            "Authorization checks user permissions.",
            "security.pdf",
            5,
            1,
        ),
        DocumentChunk(
            "A database stores information in tables.",
            "database.txt",
            None,
            2,
        ),
    ])

    return embeddings, store

#Test that the retriever returns the most relevant chunk for a given query.
def test_retriever_returns_most_relevant_chunk(indexed_store):
    embeddings, store = indexed_store
    retriever = Retriever(embeddings, store)

    results = retriever.retrieve(
        "How do I verify a user's identity?",
        top_k=1,
    )

    assert len(results) == 1
    assert "Authentication" in results[0].text
    assert results[0].source == "security.pdf"
    assert results[0].page_number == 4

#Test that the retriever respects the top_k parameter and returns the correct number of results.
def test_retriever_respects_top_k(indexed_store):
    embeddings, store = indexed_store
    retriever = Retriever(embeddings, store)

    results = retriever.retrieve(
        "How do permissions work?",
        top_k=2,
    )

    assert len(results) == 2

#Test that the retriever returns an empty list when the vector store has no indexed chunks.
def test_retriever_returns_empty_for_empty_collection(tmp_path):
    embeddings = FakeEmbeddingService()

    store = VectorStore(
        embedding_service=embeddings,
        persist_directory=tmp_path / "empty_chroma",
    )

    retriever = Retriever(embeddings, store)

    assert retriever.retrieve("Explain authentication") == []

#Test that the retriever raises a ValueError for empty queries or non-positive top_k values.
@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("   ", 3),
        ("Explain authentication", 0),
        ("Explain authentication", -1),
    ],
)
def test_retriever_rejects_invalid_input(
    indexed_store, query, top_k
):
    embeddings, store = indexed_store
    retriever = Retriever(embeddings, store)

    with pytest.raises(ValueError):
        retriever.retrieve(query, top_k=top_k)
