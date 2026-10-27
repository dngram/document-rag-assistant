
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from langchain_app.rag_pipeline import LangChainRAGPipeline


# Test that the LangChain pipeline rejects an invalid chunk size during initialization.
def test_langchain_pipeline_rejects_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="chunk_size"):
        LangChainRAGPipeline(
            persist_directory=tmp_path / "chroma",
            chunk_size=0,
        )


# Test that the LangChain pipeline initializes its components with the expected configuration.
@patch("langchain_app.rag_pipeline.ChatPromptTemplate")
@patch("langchain_app.rag_pipeline.ChatOllama")
@patch("langchain_app.rag_pipeline.Chroma")
@patch("langchain_app.rag_pipeline.HuggingFaceEmbeddings")
def test_langchain_pipeline_initializes_components(
    mock_embeddings,
    mock_chroma,
    mock_ollama,
    mock_prompt,
    tmp_path,
):
    pipeline = LangChainRAGPipeline(
        persist_directory=tmp_path / "chroma",
        chunk_size=1000,
    )

    assert pipeline.chunk_size == 1000
    assert pipeline.overlap == 150

    mock_embeddings.assert_called_once()
    mock_chroma.assert_called_once()
    mock_ollama.assert_called_once_with(
        model="qwen2.5:3b",
        temperature=0,
    )
    mock_prompt.from_messages.assert_called_once()


# Test that the LangChain pipeline loads, splits and indexes a document.
@patch("langchain_app.rag_pipeline.ChatPromptTemplate")
@patch("langchain_app.rag_pipeline.ChatOllama")
@patch("langchain_app.rag_pipeline.Chroma")
@patch("langchain_app.rag_pipeline.HuggingFaceEmbeddings")
@patch("langchain_app.rag_pipeline.split_documents")
@patch("langchain_app.rag_pipeline.load_documents")
def test_langchain_pipeline_ingests_document(
    mock_load_documents,
    mock_split_documents,
    mock_embeddings,
    mock_chroma,
    mock_ollama,
    mock_prompt,
    tmp_path,
):
    mock_load_documents.return_value = [
        Document(
            page_content="Authentication verifies identity.",
            metadata={
                "source": "guide.txt",
                "page_number": None,
            },
        )
    ]

    mock_split_documents.return_value = [
        Document(
            page_content="Authentication verifies identity.",
            metadata={
                "source": "guide.txt",
                "page_number": None,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Authorization controls access.",
            metadata={
                "source": "guide.txt",
                "page_number": None,
                "chunk_index": 1,
            },
        ),
    ]

    mock_chroma.return_value.get.return_value = {
        "ids": [],
        "metadatas": [],
    }
    mock_chroma.return_value.add_documents.return_value = [
        "chunk-1",
        "chunk-2",
    ]

    pipeline = LangChainRAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    result = pipeline.ingest_document("guide.txt", chunk_size=500)

    assert result == 2

    mock_load_documents.assert_called_once()
    mock_split_documents.assert_called_once_with(
        mock_load_documents.return_value,
        chunk_size=500,
    )
    mock_chroma.return_value.add_documents.assert_called_once_with(
        mock_split_documents.return_value
    )


# Test that the LangChain pipeline rejects an invalid custom chunk size before loading a document.
@patch("langchain_app.rag_pipeline.ChatPromptTemplate")
@patch("langchain_app.rag_pipeline.ChatOllama")
@patch("langchain_app.rag_pipeline.Chroma")
@patch("langchain_app.rag_pipeline.HuggingFaceEmbeddings")
@patch("langchain_app.rag_pipeline.load_documents")
def test_langchain_pipeline_rejects_invalid_custom_chunk_size(
    mock_load_documents,
    mock_embeddings,
    mock_chroma,
    mock_ollama,
    mock_prompt,
    tmp_path,
):
    pipeline = LangChainRAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    with pytest.raises(ValueError, match="chunk_size"):
        pipeline.ingest_document(
            "guide.txt",
            chunk_size=0,
        )

    mock_load_documents.assert_not_called()


# Test that the LangChain pipeline returns an answer and citations for a question.
@patch("langchain_app.rag_pipeline.ChatPromptTemplate")
@patch("langchain_app.rag_pipeline.ChatOllama")
@patch("langchain_app.rag_pipeline.Chroma")
@patch("langchain_app.rag_pipeline.HuggingFaceEmbeddings")
def test_langchain_pipeline_answers_question(
    mock_embeddings,
    mock_chroma,
    mock_ollama,
    mock_prompt,
    tmp_path,
):
    mock_document = Document(
        page_content="Authentication verifies a user's identity.",
        metadata={
            "source": "security.pdf",
            "page_number": 4,
            "chunk_index": 0,
        },
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_document]
    mock_chroma.return_value.as_retriever.return_value = mock_retriever

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(
        content="Authentication verifies identity. [1]"
    )
    mock_prompt.from_messages.return_value.__or__.return_value = mock_chain

    pipeline = LangChainRAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    result = pipeline.ask(
        "How is identity verified?",
        top_k=3,
    )

    assert result["answer"] == "Authentication verifies identity. [1]"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "security.pdf"
    assert result["citations"][0]["page_number"] == 4

    mock_chroma.return_value.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    mock_retriever.invoke.assert_called_once_with(
        "How is identity verified?"
    )


# Test that the LangChain pipeline rejects empty questions and invalid top_k values.
@patch("langchain_app.rag_pipeline.ChatPromptTemplate")
@patch("langchain_app.rag_pipeline.ChatOllama")
@patch("langchain_app.rag_pipeline.Chroma")
@patch("langchain_app.rag_pipeline.HuggingFaceEmbeddings")
def test_langchain_pipeline_rejects_invalid_questions(
    mock_embeddings,
    mock_chroma,
    mock_ollama,
    mock_prompt,
    tmp_path,
):
    pipeline = LangChainRAGPipeline(
        persist_directory=tmp_path / "chroma",
    )

    with pytest.raises(ValueError, match="Question"):
        pipeline.ask("   ")

    with pytest.raises(ValueError, match="top_k"):
        pipeline.ask("Explain authentication", top_k=0)
