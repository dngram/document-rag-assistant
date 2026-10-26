
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def load_docx(file_path: str | Path) -> str:
    """Extract paragraphs and table text from a .docx file."""
    path = Path(file_path)

    if path.suffix.lower() != ".docx":
        raise ValueError(
            f"Expected a .docx file, got: {path.name}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"DOCX file not found: {path}"
        )

    document = Document(path)
    extracted_parts = []

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                extracted_parts.append(block.text)

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]
                extracted_parts.append(" | ".join(cells))

    return "\n".join(extracted_parts)
