
from pathlib import Path

from app.ingestion.chunker import chunk_sections
from app.ingestion.document_loader import load_document
from app.rag.answer_generator import AnswerGenerator, AnswerResult
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class RAGPipeline:
    """Coordinate document ingestion, indexing and question answering."""

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        llm_model: str = "qwen2.5:3b",
        chunk_size: int = 1000,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        self.chunk_size = chunk_size
        self.overlap = round(chunk_size * 0.15)

        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            embedding_service=self.embedding_service,
            persist_directory=persist_directory,
        )

        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

        self.answer_generator = AnswerGenerator(
            model_name=llm_model
        )

    def ingest_document(self, file_path: str | Path) -> int:
        """Extract, chunk and index a document."""
        sections = load_document(file_path)

        chunks = chunk_sections(
            sections,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        return self.vector_store.add_chunks(chunks)

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> AnswerResult:
        """Retrieve relevant chunks and generate an answer."""
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        return self.answer_generator.generate_answer(
            query=question,
            retrieved_chunks=retrieved_chunks,
        )
