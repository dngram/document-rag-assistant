
import pytest
import pymupdf

from app.ingestion.pdf_loader import load_pdf

#Create a small PDF for testing.
def create_test_pdf(file_path, page_texts):
    """Create a small PDF for testing."""
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)

        document.save(file_path)

#Test that the PDF loader extracts text and page numbers correctly.
def test_load_pdf_extracts_text_and_page_numbers(tmp_path):
    file_path = tmp_path / "sample.pdf"
    create_test_pdf(
        file_path,
        ["Authentication uses tokens.", "Authorization checks permissions."],
    )

    pages = load_pdf(file_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Authentication uses tokens." in pages[0].text
    assert pages[1].page_number == 2
    assert "Authorization checks permissions." in pages[1].text

#Test that the PDF loader raises a ValueError for unsupported file extensions.
def test_load_pdf_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(file_path)

#Test that the PDF loader raises a FileNotFoundError for missing files.
def test_load_pdf_raises_for_missing_file(tmp_path):
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(file_path)
