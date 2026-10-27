
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.doc_loader import load_doc
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_loader import load_txt


@dataclass(frozen=True)
class DocumentSection:
    """Extracted text with its source information."""
    text: str
    source: str
    page_number: int | None = None


def load_document(file_path: str | Path) -> list[DocumentSection]:
    """Load a supported document into standardized sections."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        sections = [
            DocumentSection(text=load_txt(path), source=path.name)
        ]

    elif extension == ".pdf":
        sections = [
            DocumentSection(
                text=page.text,
                source=path.name,
                page_number=page.page_number,
            )
            for page in load_pdf(path)
            if page.text.strip()
        ]

    elif extension == ".docx":
        sections = [
            DocumentSection(text=load_docx(path), source=path.name)
        ]

    elif extension == ".doc":
        sections = [
            DocumentSection(text=load_doc(path), source=path.name)
        ]

    else:
        raise ValueError(
            f"Unsupported file format: {extension or '(no extension)'}"
        )

    if not any(section.text.strip() for section in sections):
        raise ValueError(
            f"No extractable text found in: {path.name}"
        )

    return sections
