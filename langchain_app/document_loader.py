
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document_loader import load_document


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}


def load_documents(file_path: str | Path) -> list[Document]:
    """Load a supported document and convert its sections to LangChain Documents."""
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Supported formats: .txt, .pdf, .docx, .doc"
        )

    sections = load_document(path)
    documents = []

    for section in sections:
        documents.append(
            Document(
                page_content=section.text,
                metadata={
                    "source": section.source,
                    "page_number": section.page_number,
                },
            )
        )

    if not documents:
        raise ValueError("No extractable text found in the document.")

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
) -> list[Document]:
    """Split LangChain Documents into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    overlap = round(chunk_size * 0.15)

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

    return chunks
