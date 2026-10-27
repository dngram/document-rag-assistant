
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.rag_pipeline import RAGPipeline


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="Document RAG Assistant",
    description="Ask questions about your uploaded documents.",
    version="1.1.0",
)

_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Create the RAG pipeline once and reuse it."""
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class CitationResponse(BaseModel):
    citation_id: int
    source: str
    page_number: int | None
    chunk_index: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@app.get("/health")
def health_check():
    """Check whether the API process is responding."""
    return {"status": "ok"}


@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000, ge=1, le=5000),
):
    """Upload and index a document using the selected chunk size."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: .txt, .pdf, .docx, .doc",
        )

    try:
        content = file.file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 20 MB size limit.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / filename
            path.write_bytes(content)

            try:
                count = get_pipeline().ingest_document(
                    path,
                    chunk_size=chunk_size,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

        overlap = round(chunk_size * 0.15)

        return {
            "filename": filename,
            "chunks_indexed": count,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "message": "Document indexed successfully.",
        }

    finally:
        file.file.close()


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Answer a question using indexed documents."""
    try:
        result = get_pipeline().ask(
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                source=citation.source,
                page_number=citation.page_number,
                chunk_index=citation.chunk_index,
                excerpt=citation.excerpt,
            )
            for citation in result.citations
        ],
    )
