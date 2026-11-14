"""Local sentence-transformers embedding provider."""
import os
from typing import Protocol
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Sentence Transformers adapter (default: multilingual MiniLM, 384 dims)."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_MODEL_NAME
        ).strip()
        if not self.model_name:
            self.model_name = DEFAULT_MODEL_NAME
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=cache_folder,
            )
        except Exception as exc:  # Do not hide a broken model/provider.
            raise RuntimeError(
                f"Could not load embedding model {self.model_name!r} on {device}: {exc}"
            ) from exc

        self._model.eval()
        # sentence-transformers >= 5 renamed this method; support both.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_embedding_dimension
        dimensions = get_dimension()
        if not dimensions or dimensions <= 0:
            raise RuntimeError(
                f"Embedding model {self.model_name!r} did not report a valid dimension"
            )

        self.dimensions = int(dimensions)
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} texts"
            )

        result: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Embedding model returned dimension {len(vector)}, "
                    f"expected {self.dimensions}"
                )
            result.append([float(value) for value in vector])
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts, preserving input order.

        Empty texts are rejected because a zero/meaningless vector would be
        silently unusable for retrieval. Chunk empty/whitespace-only pages out
        before calling this method.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("texts must contain only strings")
            if not text.strip():
                raise ValueError("texts must not contain empty or whitespace-only strings")
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        """Embed one query with the exact preprocessing used for documents."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be empty")
        return self._encode([question])[0]
