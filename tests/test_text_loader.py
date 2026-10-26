
import pytest

from app.ingestion.text_loader import load_txt

#Normal text is extracted correctly from a .txt file.
def test_load_txt_returns_file_contents(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Retrieval-Augmented Generation",
        encoding="utf-8",
    )

    result = load_txt(file_path)

    assert result == "Retrieval-Augmented Generation"

#Both TXT and txt extensions are accepted, and the function is case-insensitive
def test_load_txt_handles_uppercase_extension(tmp_path):
    file_path = tmp_path / "sample.TXT"
    file_path.write_text("Hello RAG", encoding="utf-8")

    assert load_txt(file_path) == "Hello RAG"

#A PDF cannot be passed to this loader.
def test_load_txt_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_text("Not a text document", encoding="utf-8")

    with pytest.raises(ValueError):
        load_txt(file_path)

#A non-existent file raises a FileNotFoundError when attempting to load it.
def test_load_txt_raises_for_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_txt(file_path)

