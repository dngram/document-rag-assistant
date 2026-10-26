
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


def load_pdf(file_path: str | Path) -> list[PDFPage]:
    """Extract text from a PDF, preserving each page number."""
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got: {path.name}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append(
                PDFPage(
                    page_number=page_number,
                    text=page.get_text(),
                )
            )

    return pages
