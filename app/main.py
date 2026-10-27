from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.rag_pipeline import RAGPipeline
from langchain_app.rag_pipeline import LangChainRAGPipeline


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="Document RAG Assistant",
    description="Ask questions about your uploaded documents.",
    version="1.2.0",
)

# Initialise each pipeline only when its endpoints are first used.
_pipeline = None
_langchain_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Create the custom RAG pipeline once and reuse it."""
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


def get_langchain_pipeline() -> LangChainRAGPipeline:
    """Create the LangChain RAG pipeline once and reuse it."""
    global _langchain_pipeline

    if _langchain_pipeline is None:
        _langchain_pipeline = LangChainRAGPipeline(
            persist_directory="chroma_db_langchain",
        )

    return _langchain_pipeline


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


def save_and_index_document(
    file: UploadFile,
    chunk_size: int,
    implementation: str,
) -> dict:
    """Validate an uploaded file and index it with the chosen pipeline."""
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
                if implementation == "custom":
                    count = get_pipeline().ingest_document(
                        path,
                        chunk_size=chunk_size,
                    )
                else:
                    count = get_langchain_pipeline().ingest_document(
                        path,
                        chunk_size=chunk_size,
                    )
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

        return {
            "filename": filename,
            "chunks_indexed": count,
            "chunk_size": chunk_size,
            "overlap": round(chunk_size * 0.15),
            "implementation": implementation,
            "message": "Document indexed successfully.",
        }

    finally:
        file.file.close()


def format_citations(citations: list) -> list[CitationResponse]:
    """Convert citation dictionaries or objects into API response models."""
    formatted = []

    for citation in citations:
        if isinstance(citation, dict):
            citation_data = citation.get
        else:
            citation_data = lambda key, default=None: getattr(
                citation, key, default
            )

        formatted.append(
            CitationResponse(
                citation_id=citation_data("citation_id", 0),
                source=citation_data("source", "Unknown source"),
                page_number=citation_data("page_number"),
                chunk_index=citation_data("chunk_index", 0),
                excerpt=citation_data("excerpt", ""),
            )
        )

    return formatted


@app.get("/health")
def health_check():
    """Check whether the API process is responding."""
    return {"status": "ok"}


@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000, ge=1, le=5000),
):
    """Upload and index a document using the custom RAG pipeline."""
    return save_and_index_document(file, chunk_size, "custom")


@app.post("/langchain/documents")
def upload_langchain_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000, ge=1, le=5000),
):
    """Upload and index a document using the LangChain pipeline."""
    return save_and_index_document(file, chunk_size, "langchain")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Answer a question using the custom RAG pipeline."""
    try:
        result = get_pipeline().ask(
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        citations=format_citations(result.citations),
    )


@app.post("/langchain/ask", response_model=AskResponse)
def ask_langchain_question(request: AskRequest):
    """Answer a question using the LangChain RAG pipeline."""
    try:
        result = get_langchain_pipeline().ask(
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        citations=format_citations(result["citations"]),
    )
