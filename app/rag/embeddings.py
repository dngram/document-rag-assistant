from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate embeddings for document chunks and user queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple document chunks into embeddings."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a user's question into an embedding."""
        embeddings = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()
