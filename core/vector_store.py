from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: Dict[str, Any]
    score: float


class FAISSVectorStore:
    """
    Persistent FAISS vector store using cosine similarity.

    - We normalize embeddings to unit length.
    - Use FAISS `IndexFlatIP` to approximate cosine similarity via inner product.
    - Store chunk texts and metadata in a sidecar JSON file.
    """

    def __init__(
        self,
        *,
        embedder: Any,
        persist_dir: str,
        embed_model_name: Optional[str] = None,
        index_filename: str = "faiss.index",
        metadata_filename: str = "metadata.json",
    ):
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import faiss. Install `faiss-cpu` to enable vector storage."
            ) from e

        self._faiss = faiss
        self._embedder = embedder
        self._embed_model_name = embed_model_name
        self._persist_dir = persist_dir
        self._index_path = os.path.join(persist_dir, index_filename)
        self._metadata_path = os.path.join(persist_dir, metadata_filename)

        self._index: Optional[Any] = None
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._embedding_dim: Optional[int] = None
        self._stored_embed_model_name: Optional[str] = None

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not os.path.isdir(self._persist_dir):
            return
        if not (os.path.exists(self._index_path) and os.path.exists(self._metadata_path)):
            return

        self._index = self._faiss.read_index(self._index_path)
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._texts = payload.get("texts", [])
        self._metadatas = payload.get("metadatas", [])
        self._embedding_dim = payload.get("embedding_dim")
        self._stored_embed_model_name = payload.get("embed_model_name")
        # Defensive repair for corrupted/mismatched metadata payloads.
        if len(self._metadatas) != len(self._texts):
            if len(self._metadatas) < len(self._texts):
                self._metadatas.extend({} for _ in range(len(self._texts) - len(self._metadatas)))
            else:
                self._metadatas = self._metadatas[: len(self._texts)]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _create_index(self, embedding_dim: int) -> None:
        self._embedding_dim = int(embedding_dim)
        self._index = self._faiss.IndexFlatIP(self._embedding_dim)

    def add_texts(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> int:
        """
        Embed and add texts into the FAISS index.
        """

        filtered_texts: list[str] = []
        filtered_metadatas: list[Dict[str, Any]] = []

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("`metadatas` must have the same length as `texts`.")

        if metadatas is None:
            for t in texts:
                if t and t.strip():
                    filtered_texts.append(t.strip())
        else:
            for t, md in zip(texts, metadatas):
                if t and str(t).strip():
                    filtered_texts.append(str(t).strip())
                    filtered_metadatas.append(md)

        if not filtered_texts:
            return 0
        if not filtered_metadatas:
            filtered_metadatas = [{} for _ in filtered_texts]

        vectors = self._embedder.embed_texts(filtered_texts)  # (n, d)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = self._normalize(vectors)

        if self._index is None:
            self._create_index(vectors.shape[1])

        # Add to index
        self._index.add(vectors)
        self._texts.extend(filtered_texts)
        self._metadatas.extend(filtered_metadatas)

        self.persist()
        return len(filtered_texts)

    def persist(self) -> None:
        os.makedirs(self._persist_dir, exist_ok=True)
        if self._index is None:
            return

        self._faiss.write_index(self._index, self._index_path)
        payload = {
            "embedding_dim": self._embedding_dim,
            "embed_model_name": self._embed_model_name,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def is_compatible_with_embedder(self) -> bool:
        """
        Check whether loaded index metadata matches the active embedder configuration.
        """
        if self._index is None:
            return True

        if self._embed_model_name and self._stored_embed_model_name:
            if self._embed_model_name != self._stored_embed_model_name:
                return False

        if self._embedding_dim is None:
            return True
        try:
            sample = self._embedder.embed_query("compatibility probe")
            sample = np.asarray(sample, dtype=np.float32)
            if sample.ndim != 2 or sample.shape[0] != 1:
                return False
            return int(sample.shape[1]) == int(self._embedding_dim)
        except Exception:
            return False

    def search(self, query: str, *, top_k: int = 2, min_score: Optional[float] = None) -> List[SearchResult]:
        """
        Retrieve top-k most similar stored chunks for the query.
        """

        if self._index is None or self._index.ntotal == 0:
            return []

        if not query or not query.strip():
            return []

        q_vec = self._embedder.embed_query(query)  # (1, d)
        q_vec = np.asarray(q_vec, dtype=np.float32)
        q_vec = self._normalize(q_vec)

        k = min(max(1, int(top_k)), int(self._index.ntotal))
        scores, indices = self._index.search(q_vec, k)

        results: list[SearchResult] = []
        # indices[0] shape (k,)
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            if min_score is not None and float(score) < float(min_score):
                continue
            text = self._texts[idx] if idx < len(self._texts) else ""
            if not text or not str(text).strip():
                continue
            metadata = self._metadatas[idx] if idx < len(self._metadatas) else {}
            results.append(
                SearchResult(
                    text=str(text),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    score=float(score),
                )
            )
        return results

    def rebuild_index_from_existing_texts(self) -> int:
        """
        Rebuild FAISS index from stored texts using current embedder.
        """
        texts = [t for t in self._texts if t and str(t).strip()]
        if not texts:
            self._index = None
            self._embedding_dim = None
            self.persist()
            return 0

        vectors = self._embedder.embed_texts(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = self._normalize(vectors)
        self._create_index(vectors.shape[1])
        self._index.add(vectors)
        self._texts = texts
        if len(self._metadatas) != len(self._texts):
            self._metadatas = (self._metadatas + [{} for _ in range(len(self._texts))])[: len(self._texts)]
        self.persist()
        return len(self._texts)

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)
