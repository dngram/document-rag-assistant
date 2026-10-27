
import hashlib
from pathlib import Path

import chromadb

from app.ingestion.chunker import DocumentChunk
from app.rag.embeddings import EmbeddingService


class VectorStore:
    """Persist document chunks and their embeddings in Chroma."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "document_chunks",
    ):
        self.embedding_service = embedding_service

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Store chunks, replacing previously indexed versions by source."""
        if not chunks:
            return 0

        grouped_chunks = {}

        for chunk in chunks:
            grouped_chunks.setdefault(chunk.source, []).append(chunk)

        stored_count = 0

        for source, source_chunks in grouped_chunks.items():
            existing = self.collection.get(
                where={"source": source},
                include=["metadatas"],
            )

            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])

            texts = [chunk.text for chunk in source_chunks]
            embeddings = self.embedding_service.embed_documents(texts)

            ids = []
            metadatas = []

            for chunk in source_chunks:
                identity = (
                    f"{chunk.source}|{chunk.page_number}|"
                    f"{chunk.chunk_index}|{chunk.text}"
                )

                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()

                ids.append(chunk_id)

                metadata = {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }

                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number

                metadatas.append(metadata)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            stored_count += len(source_chunks)

        return stored_count

    def get_all_chunks(self) -> list[dict]:
        """Return all stored text chunks and their metadata for keyword search."""
        stored = self.collection.get(
            include=["documents", "metadatas"],
        )

        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        return [
            {
                "text": text,
                "metadata": metadata,
            }
            for text, metadata in zip(documents, metadatas)
            if text is not None and metadata is not None
        ]
