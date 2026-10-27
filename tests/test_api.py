
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.rag.answer_generator import AnswerResult, SourceCitation


client = TestClient(app)


# Test that the health endpoint returns a 200 status code and the expected JSON response.
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test that the /documents endpoint can successfully upload a document and return the expected response.
@patch("app.main.get_pipeline")
def test_upload_document(mock_get_pipeline):
    mock_get_pipeline.return_value.ingest_document.return_value = 2

    response = client.post(
        "/documents",
        files={
            "file": (
                "guide.txt",
                b"Authentication verifies identity.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "guide.txt"
    assert data["chunks_indexed"] == 2
    assert data["chunk_size"] == 1000
    assert data["overlap"] == 150

    mock_get_pipeline.return_value.ingest_document.assert_called_once()
    assert (
        mock_get_pipeline.return_value.ingest_document.call_args.kwargs[
            "chunk_size"
        ]
        == 1000
    )


# Test that the /documents endpoint passes a custom chunk size to the RAGPipeline.
@patch("app.main.get_pipeline")
def test_upload_document_with_custom_chunk_size(mock_get_pipeline):
    mock_get_pipeline.return_value.ingest_document.return_value = 3

    response = client.post(
        "/documents",
        data={"chunk_size": 500},
        files={
            "file": (
                "guide.txt",
                b"Authentication verifies identity.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["chunk_size"] == 500
    assert data["overlap"] == 75
    assert data["chunks_indexed"] == 3

    mock_get_pipeline.return_value.ingest_document.assert_called_once()
    assert (
        mock_get_pipeline.return_value.ingest_document.call_args.kwargs[
            "chunk_size"
        ]
        == 500
    )


# Test that the /documents endpoint rejects a chunk size below the allowed range.
def test_upload_rejects_chunk_size_below_minimum():
    response = client.post(
        "/documents",
        data={"chunk_size": 0},
        files={
            "file": (
                "guide.txt",
                b"Authentication verifies identity.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


# Test that the /documents endpoint rejects a chunk size above the allowed range.
def test_upload_rejects_chunk_size_above_maximum():
    response = client.post(
        "/documents",
        data={"chunk_size": 5001},
        files={
            "file": (
                "guide.txt",
                b"Authentication verifies identity.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


# Test that the /documents endpoint rejects unsupported file extensions and returns a 400 status code.
def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/documents",
        files={
            "file": (
                "guide.csv",
                b"a,b",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400


# Test that the /documents endpoint rejects empty files and returns a 400 status code.
def test_upload_rejects_empty_file():
    response = client.post(
        "/documents",
        files={
            "file": ("empty.txt", b"", "text/plain")
        },
    )

    assert response.status_code == 400


# Test that the /ask endpoint returns an answer and its source citations.
@patch("app.main.get_pipeline")
def test_ask_returns_answer_and_citations(mock_get_pipeline):
    mock_get_pipeline.return_value.ask.return_value = AnswerResult(
        answer="Authentication verifies identity. [1]",
        citations=[
            SourceCitation(
                citation_id=1,
                source="security.pdf",
                page_number=4,
                chunk_index=0,
                excerpt="Authentication verifies a user's identity.",
            )
        ],
    )

    response = client.post(
        "/ask",
        json={
            "question": "How is identity verified?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "Authentication verifies identity. [1]"
    assert data["citations"][0]["source"] == "security.pdf"
    assert data["citations"][0]["page_number"] == 4


# Test that the /ask endpoint rejects invalid top_k values and returns a 422 status code.
def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "Explain authentication",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
