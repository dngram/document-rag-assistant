
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

CHUNK_SIZE_OPTIONS = [500, 750, 1000, 1500]
DEFAULT_CHUNK_SIZE = 1000

st.set_page_config(
    page_title="DocuMind | RAG Assistant",
    layout="wide",
)

# Minimal styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []


def check_backend():
    """Check whether the FastAPI backend is reachable."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )
        return response.ok
    except requests.RequestException:
        return False


def get_error_message(response):
    """Extract a useful error message from an API response."""
    try:
        data = response.json()
        return data.get("detail", response.text)
    except ValueError:
        return response.text or "An unexpected error occurred."


def display_citations(citations):
    """Display source information for a generated answer."""
    if not citations:
        return

    with st.expander("View source citations"):
        for citation in citations:
            source = citation.get("source", "Unknown source")
            page = citation.get("page_number")
            chunk_index = citation.get("chunk_index")
            excerpt = citation.get("excerpt", "")

            st.markdown(f"**{source}**")

            if page is not None:
                st.caption(f"Page {page}")

            if chunk_index is not None:
                st.caption(f"Chunk {chunk_index}")

            if excerpt:
                st.write(excerpt)

            st.divider()


# Sidebar
with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Your documents, searchable with AI")

    st.divider()
    st.subheader("Backend status")

    backend_available = check_backend()

    if backend_available:
        st.success("API connected")
    else:
        st.error("API unavailable")
        st.caption(
            "Start FastAPI in another terminal before using the assistant."
        )

    st.divider()

    # Chunking settings
    st.subheader("Chunking settings")

    chunk_size = st.selectbox(
        "Chunk size (characters)",
        options=CHUNK_SIZE_OPTIONS,
        index=CHUNK_SIZE_OPTIONS.index(DEFAULT_CHUNK_SIZE),
        help=(
            "Smaller chunks provide more focused passages. "
            "Larger chunks retain more surrounding context."
        ),
    )

    overlap = round(chunk_size * 0.15)

    st.caption(f"Overlap: {overlap} characters (15%)")

    st.divider()
    st.subheader("Upload documents")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["txt", "pdf", "docx", "doc"],
        help="Supported formats: TXT, PDF, DOCX and DOC.",
    )

    if st.button(
        "Index document",
        use_container_width=True,
        disabled=uploaded_file is None or not backend_available,
    ):
        try:
            with st.spinner(
                f"Indexing document with {chunk_size}-character chunks..."
            ):
                response = requests.post(
                    f"{API_BASE_URL}/documents",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                            or "application/octet-stream",
                        )
                    },
                    data={"chunk_size": chunk_size},
                    timeout=300,
                )

            if response.ok:
                result = response.json()
                filename = result.get("filename", uploaded_file.name)
                chunks_indexed = result.get("chunks_indexed", 0)
                actual_chunk_size = result.get("chunk_size", chunk_size)
                actual_overlap = result.get("overlap", overlap)

                # Update the displayed entry for this filename.
                st.session_state.indexed_documents = [
                    document
                    for document in st.session_state.indexed_documents
                    if document["filename"] != filename
                ]

                st.session_state.indexed_documents.append(
                    {
                        "filename": filename,
                        "chunks": chunks_indexed,
                        "chunk_size": actual_chunk_size,
                        "overlap": actual_overlap,
                    }
                )

                st.success(
                    f"Indexed {filename}: {chunks_indexed} chunks."
                )
                st.caption(
                    f"Chunk size: {actual_chunk_size} characters | "
                    f"Overlap: {actual_overlap} characters"
                )
            else:
                st.error(get_error_message(response))

        except requests.Timeout:
            st.error(
                "Indexing took too long. Check the backend and try again."
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

    if st.session_state.indexed_documents:
        st.divider()
        st.subheader("Indexed this session")

        for document in st.session_state.indexed_documents:
            st.markdown(f"**{document['filename']}**")
            st.caption(
                f"{document['chunks']} chunks | "
                f"Size: {document['chunk_size']} | "
                f"Overlap: {document['overlap']}"
            )


# Main area
st.title("Ask your documents")
st.write(
    "Upload documents in the sidebar, then ask questions about their contents."
)

st.info(
    "Answers are generated from retrieved document passages. "
    "Check the source excerpts to verify important details."
)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_citations(message.get("citations", []))


# Accept a new question
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question, "top_k": 3},
                    timeout=300,
                )

            if response.ok:
                result = response.json()
                answer = result.get("answer", "No answer returned.")
                citations = result.get("citations", [])

                st.markdown(answer)
                display_citations(citations)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )
            else:
                error_message = get_error_message(response)
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"**Request failed:** {error_message}",
                        "citations": [],
                    }
                )

        except requests.Timeout:
            error_message = (
                "The request timed out. "
                "Try again or check the backend."
            )
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "citations": [],
                }
            )

        except requests.RequestException as exc:
            error_message = f"Could not reach the API: {exc}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "citations": [],
                }
            )
