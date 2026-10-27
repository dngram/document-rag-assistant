
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from langchain_app.document_loader import (
    load_documents,
    split_documents,
)


# Test that a supported document is converted into a LangChain Document.
@patch("langchain_app.document_loader.load_document")
def test_load_documents(mock_load_document, tmp_path):
    mock_load_document.return_value = [
        type(
            "DocumentSection",
            (),
            {
                "text": "Authentication verifies identity.",
                "source": "guide.txt",
                "page_number": None,
            },
        )()
    ]

    documents = load_documents(tmp_path / "guide.txt")

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Authentication verifies identity."
    assert documents[0].metadata["source"] == "guide.txt"


# Test that unsupported file extensions are rejected.
def test_load_documents_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Supported formats"):
        load_documents(tmp_path / "guide.csv")


# Test that LangChain splits documents using the selected chunk size and 15% overlap.
def test_split_documents():
    document = Document(
        page_content=" ".join(["authentication"] * 200),
        metadata={"source": "guide.txt", "page_number": None},
    )

    chunks = split_documents([document], chunk_size=100)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


# Test that an invalid chunk size is rejected.
def test_split_documents_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        split_documents([], chunk_size=0)
