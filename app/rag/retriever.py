
from dataclasses import dataclass

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved text chunk with source and relevance information."""
    text: str
    source: str
    page_number: int | None
    chunk_index: int
    similarity_score: float


class Retriever:
    """Find document chunks relevant to a user's question."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks for a query."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection = self.vector_store.collection
        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        query_embedding = self.embedding_service.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    source=metadata["source"],
                    page_number=metadata.get("page_number"),
                    chunk_index=metadata["chunk_index"],
                    similarity_score=1.0 - distance,
                )
            )

        return retrieved
