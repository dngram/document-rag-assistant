
import pytest

from app.ingestion.document_loader import (
    DocumentSection,
    load_document,
)

#Test that the document loader correctly handles a simple TXT file.
def test_load_document_handles_txt(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("RAG uses retrieval.", encoding="utf-8")

    result = load_document(file_path)

    assert result == [
        DocumentSection(
            text="RAG uses retrieval.",
            source="notes.txt",
            page_number=None,
        )
    ]

#Test that the document loader correctly handles a simple DOCX file.
def test_load_document_preserves_pdf_page_numbers(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "guide.pdf"
    file_path.write_bytes(b"test placeholder")

    from app.ingestion import document_loader

    monkeypatch.setattr(
        document_loader,
        "load_pdf",
        lambda path: [
            type("Page", (), {
                "text": "Page one",
                "page_number": 1,
            })(),
            type("Page", (), {
                "text": "Page two",
                "page_number": 2,
            })(),
        ],
    )

    result = load_document(file_path)

    assert result[0].page_number == 1
    assert result[1].page_number == 2
    assert all(section.source == "guide.pdf" for section in result)

#Test that the document loader correctly handles a simple DOCX file.
def test_load_document_rejects_unsupported_format(tmp_path):
    file_path = tmp_path / "notes.csv"
    file_path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(file_path)

#Test that the document loader raises a ValueError for empty text in a TXT file.
def test_load_document_rejects_empty_text(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        load_document(file_path)

#Test that the document loader raises a ValueError for empty text in a PDF file.
def test_load_document_raises_for_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(file_path)
