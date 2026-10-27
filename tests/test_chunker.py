
import pytest

from app.ingestion.chunker import chunk_sections
from app.ingestion.document_loader import DocumentSection

#Test that the chunker splits long text into multiple chunks.
def test_chunker_splits_long_text():
    section = DocumentSection(
        text="alpha beta gamma delta epsilon zeta eta theta",
        source="notes.txt",
        page_number=None,
    )

    chunks = chunk_sections(
        [section],
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)

#Test that the chunker preserves source metadata in the resulting chunks.
def test_chunker_preserves_source_metadata():
    section = DocumentSection(
        text="Authentication checks the user's identity.",
        source="security.pdf",
        page_number=4,
    )

    chunks = chunk_sections([section], chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].source == "security.pdf"
    assert chunks[0].page_number == 4
    assert chunks[0].chunk_index == 0

#Test that the chunker correctly handles text shorter than the specified chunk size.
def test_chunker_assigns_sequential_indexes():
    sections = [
        DocumentSection("First section.", "guide.pdf", 1),
        DocumentSection("Second section.", "guide.pdf", 2),
    ]

    chunks = chunk_sections(sections)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]

#Test that the chunker correctly handles text shorter than the specified chunk size.
def test_chunker_skips_empty_sections():
    sections = [
        DocumentSection("   ", "empty.txt", None),
        DocumentSection("Useful text.", "notes.txt", None),
    ]

    chunks = chunk_sections(sections)

    assert len(chunks) == 1
    assert chunks[0].text == "Useful text."

#Test that the chunker raises a ValueError for invalid chunk size or overlap values.
@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunker_rejects_invalid_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_sections(
            [DocumentSection("Some text.", "notes.txt", None)],
            chunk_size=chunk_size,
            overlap=overlap,
        )

#Test that the chunker preserves overlap between consecutive chunks.
def test_chunker_preserves_overlap_between_chunks():
    section = DocumentSection(
        text="alpha beta gamma delta epsilon zeta eta theta",
        source="notes.txt",
        page_number=None,
    )

    chunks = chunk_sections(
        [section],
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0].text.endswith("gamma")
    assert chunks[1].text.startswith("gamma")
