
from dataclasses import dataclass

from app.ingestion.document_loader import DocumentSection


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with metadata linking it to its source."""
    text: str
    source: str
    page_number: int | None
    chunk_index: int


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind(" ", start, end)

            if boundary >= start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Move forward while retaining the requested overlap.
        start = max(start + 1, end - overlap)

    return chunks


def chunk_sections(
    sections: list[DocumentSection],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Chunk document sections and preserve source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    result = []

    for section in sections:
        section_chunks = _split_text(
            section.text,
            chunk_size,
            overlap,
        )

        for chunk in section_chunks:
            result.append(
                DocumentChunk(
                    text=chunk,
                    source=section.source,
                    page_number=section.page_number,
                    chunk_index=len(result),
                )
            )

    return result
