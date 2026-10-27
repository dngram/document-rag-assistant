
from unittest.mock import MagicMock, patch

import pytest

from app.rag.answer_generator import AnswerResult
from app.rag.rag_pipeline import RAGPipeline

#Test that the RAGPipeline can ingest a document and return the number of chunks added.
@patch("app.rag.rag_pipeline.AnswerGenerator")
@patch("app.rag.rag_pipeline.Retriever")
@patch("app.rag.rag_pipeline.VectorStore")
@patch("app.rag.rag_pipeline.EmbeddingService")
def test_pipeline_ingests_document(
    mock_embedding,
    mock_store,
    mock_retriever,
    mock_generator,
    tmp_path,
):
    pipeline = RAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    pipeline.vector_store.add_chunks.return_value = 3

    with (
        patch(
            "app.rag.rag_pipeline.load_document",
            return_value=[
                MagicMock(
                    text="RAG retrieves relevant information.",
                    source="guide.txt",
                    page_number=None,
                )
            ],
        ),
        patch(
            "app.rag.rag_pipeline.chunk_sections",
            return_value=[MagicMock()],
        ) as mock_chunk,
    ):
        result = pipeline.ingest_document("guide.txt")

    assert result == 3
    mock_chunk.assert_called_once()
    pipeline.vector_store.add_chunks.assert_called_once()

#Test that the RAGPipeline can ask a question and return an answer with citations.
@patch("app.rag.rag_pipeline.AnswerGenerator")
@patch("app.rag.rag_pipeline.Retriever")
@patch("app.rag.rag_pipeline.VectorStore")
@patch("app.rag.rag_pipeline.EmbeddingService")
def test_pipeline_asks_question(
    mock_embedding,
    mock_store,
    mock_retriever,
    mock_generator,
    tmp_path,
):
    pipeline = RAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    pipeline.retriever.retrieve.return_value = [
        MagicMock(text="Authentication verifies identity.")
    ]

    expected = AnswerResult(
        answer="Authentication verifies identity. [1]",
        citations=[],
    )
    pipeline.answer_generator.generate_answer.return_value = expected

    result = pipeline.ask("How is identity verified?", top_k=2)

    assert result == expected
    pipeline.retriever.retrieve.assert_called_once_with(
        query="How is identity verified?",
        top_k=2,
    )
    pipeline.answer_generator.generate_answer.assert_called_once()

#Test that the RAGPipeline raises a ValueError when initialized with an invalid chunk size.
def test_pipeline_rejects_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="chunk_size"):
        RAGPipeline(
            persist_directory=tmp_path / "chroma",
            chunk_size=0,
        )
