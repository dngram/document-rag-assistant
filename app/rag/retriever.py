
import math
import re
from collections import Counter
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
    """Retrieve relevant chunks using hybrid search and RRF fusion."""

    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into lowercase searchable terms."""
        return re.findall(r"[\w.-]+", text.lower())

    @staticmethod
    def _chunk_key(
        text: str,
        metadata: dict,
    ) -> tuple[str, int | None, int, str]:
        """Create a stable identity for a chunk across retrieval methods."""
        return (
            metadata["source"],
            metadata.get("page_number"),
            metadata["chunk_index"],
            text,
        )

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rank stored chunks using BM25 keyword relevance."""
        stored_chunks = self.vector_store.get_all_chunks()

        if not stored_chunks:
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        tokenized_documents = [
            self._tokenize(chunk["text"])
            for chunk in stored_chunks
        ]

        document_count = len(tokenized_documents)

        document_lengths = [
            len(tokens) for tokens in tokenized_documents
        ]

        average_length = (
            sum(document_lengths) / document_count
            if document_count
            else 0.0
        )

        if average_length == 0:
            return []

        term_frequencies = [
            Counter(tokens) for tokens in tokenized_documents
        ]

        document_frequencies = Counter()

        for frequencies in term_frequencies:
            document_frequencies.update(frequencies.keys())

        scored_chunks = []

        for chunk, frequencies, document_length in zip(
            stored_chunks,
            term_frequencies,
            document_lengths,
        ):
            score = 0.0

            for term in query_terms:
                frequency = frequencies.get(term, 0)

                if frequency == 0:
                    continue

                document_frequency = document_frequencies[term]

                inverse_document_frequency = math.log(
                    1
                    + (
                        document_count - document_frequency + 0.5
                    )
                    / (document_frequency + 0.5)
                )

                length_normalization = self.BM25_K1 * (
                    1
                    - self.BM25_B
                    + self.BM25_B
                    * document_length
                    / average_length
                )

                term_score = (
                    inverse_document_frequency
                    * frequency
                    * (self.BM25_K1 + 1)
                    / (frequency + length_normalization)
                )

                score += term_score

            if score > 0:
                metadata = chunk["metadata"]
                scored_chunks.append(
                    (
                        score,
                        RetrievedChunk(
                            text=chunk["text"],
                            source=metadata["source"],
                            page_number=metadata.get("page_number"),
                            chunk_index=metadata["chunk_index"],
                            similarity_score=score,
                        ),
                    )
                )

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            chunk
            for _, chunk in scored_chunks[:top_k]
        ]

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        total_chunks: int,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using embedding similarity in ChromaDB."""
        query_embedding = self.embedding_service.embed_query(query)

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        if not documents or not documents[0]:
            return []

        for text, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0],
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

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Combine semantic and keyword results using Reciprocal Rank Fusion."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        total_chunks = self.vector_store.collection.count()

        if total_chunks == 0:
            return []

        # Retrieve a wider candidate set before selecting the final top-k.
        candidate_count = min(max(top_k * 3, top_k), total_chunks)

        semantic_results = self._semantic_search(
            query=query,
            top_k=candidate_count,
            total_chunks=total_chunks,
        )

        keyword_results = self._keyword_search(
            query=query,
            top_k=candidate_count,
        )

        # Store each unique chunk and its combined reciprocal-rank score.
        fused_scores = {}
        chunk_by_key = {}

        for ranked_results in (semantic_results, keyword_results):
            for rank, chunk in enumerate(ranked_results, start=1):
                metadata = {
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }

                key = self._chunk_key(chunk.text, metadata)

                chunk_by_key[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    1.0 / (self.RRF_K + rank)
                )

        ranked_keys = sorted(
            fused_scores,
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        # similarity_score contains the RRF score in hybrid mode.
        return [
            RetrievedChunk(
                text=chunk_by_key[key].text,
                source=chunk_by_key[key].source,
                page_number=chunk_by_key[key].page_number,
                chunk_index=chunk_by_key[key].chunk_index,
                similarity_score=fused_scores[key],
            )
            for key in ranked_keys[:top_k]
        ]
