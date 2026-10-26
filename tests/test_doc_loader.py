
from unittest.mock import patch

import pytest

from app.ingestion.doc_loader import load_doc

#Test that the DOC loader raises a ValueError for unsupported file extensions.
def test_load_doc_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a DOC file", encoding="utf-8")

    with pytest.raises(ValueError):
        load_doc(file_path)

#Test that the DOC loader raises a FileNotFoundError for missing files.
def test_load_doc_raises_for_missing_file(tmp_path):
    file_path = tmp_path / "missing.doc"

    with pytest.raises(FileNotFoundError):
        load_doc(file_path)

#Test that the DOC loader converts a legacy DOC file to DOCX and extracts text correctly.
@patch("app.ingestion.doc_loader.load_docx")
@patch("app.ingestion.doc_loader.win32com.client.DispatchEx")
def test_load_doc_converts_and_extracts_text(
    mock_dispatch,
    mock_load_docx,
    tmp_path,
):
    file_path = tmp_path / "sample.doc"
    file_path.write_bytes(b"fake legacy DOC content")

    mock_word = mock_dispatch.return_value
    mock_document = mock_word.Documents.Open.return_value

    def fake_load_docx(converted_path):
        assert converted_path.name == "converted.docx"
        converted_path.write_text("temporary test file", encoding="utf-8")
        return "Extracted legacy document text"

    mock_load_docx.side_effect = fake_load_docx

    result = load_doc(file_path)

    assert result == "Extracted legacy document text"
    mock_dispatch.assert_called_once_with("Word.Application")
    mock_document.SaveAs2.assert_called_once()
    mock_document.Close.assert_called_once_with(False)
    mock_word.Quit.assert_called_once()
