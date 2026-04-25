from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_output: bool = False


class SentenceTransformerEmbedder:
    """
    Sentence-transformers embedding wrapper.
    """

    _MODEL_CACHE = {}

    def __init__(self, *, config: EmbeddingConfig = EmbeddingConfig()):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import sentence-transformers. Install `sentence-transformers` "
                "to enable embeddings."
            ) from e

        model_id = config.model_name
        # sentence-transformers accepts both "all-MiniLM-L6-v2" and the full repo id.
        # We keep this flexible in case the environment expects the full id.
        if "/" not in model_id:
            model_id = f"sentence-transformers/{model_id}"

        cached = self._MODEL_CACHE.get(model_id)
        if cached is None:
            cached = SentenceTransformer(model_id)
            self._MODEL_CACHE[model_id] = cached
        self._model = cached
        self._config = config

    @property
    def embedding_dim(self) -> Optional[int]:
        # Not always available reliably without encoding a sample.
        return None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns: float32 numpy array of shape (n_texts, embedding_dim)
        """

        cleaned = [t for t in texts if t and t.strip()]
        if not cleaned:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = self._model.encode(
            cleaned,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize_output,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])
