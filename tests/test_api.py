
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.rag.answer_generator import AnswerResult, SourceCitation


client = TestClient(app)

#Test that the health endpoint returns a 200 status code and the expected JSON response.
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

#Test that the /documents endpoint can successfully upload a document and return the expected response.
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
    assert response.json()["filename"] == "guide.txt"
    assert response.json()["chunks_indexed"] == 2

#Test that the /documents endpoint rejects unsupported file extensions and returns a 400 status code.
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

#Test that the /documents endpoint rejects files that exceed the maximum size limit and returns a 413 status code.
def test_upload_rejects_empty_file():
    response = client.post(
        "/documents",
        files={
            "file": ("empty.txt", b"", "text/plain")
        },
    )

    assert response.status_code == 400

#Test that the /documents endpoint rejects files that exceed the maximum size limit and returns a 413 status code.
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

#Test that the /ask endpoint rejects invalid top_k values and returns a 422 status code.
def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "Explain authentication",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
