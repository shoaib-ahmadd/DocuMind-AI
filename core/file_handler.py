from __future__ import annotations

import os
import re
import shutil
import json
import time
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from core.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from core.ocr_engine import OCREngine, OCRConfig
from core.pdf_loader import load_pdf_text
from core.text_splitter import preprocess_and_chunk
from core.vector_store import FAISSVectorStore, SearchResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}
INGEST_METADATA_FILENAME = "ingest_metadata.json"

_MIN_CHUNK_LEN = 80    # characters — drop chunks shorter than this
_MAX_LINE_LEN = 500    # characters — drop lines that look like data dumps

_NOISY_MARKERS = (
    "format_hint",
    "sample_questions",
    "hybrid_eval",
    "sample_questions_hybrid_eval.jsonl",
    "return {",
)

_JSON_LINE_RE = re.compile(r'^\s*[\[{\]}\"]')
_ID_LINE_RE = re.compile(r'(?i)^\s*(?:"?id"?\s*[:=]|chunk_id\s*[:=]|doc_id\s*[:=])')
_KEY_VALUE_RE = re.compile(r'^\s*"[^"]+"\s*:\s*')
_SQL_RE = re.compile(
    r'(?i)\b(?:select|insert|update|delete|create|drop|alter)\b.+\b(?:from|into|table|view)\b'
)
_CODE_RE = re.compile(r'(?i)(?:\bdef\b|\bclass\b|\bimport\b|\breturn\b\s+\{|=>|::|```)')
_PATH_RE = re.compile(
    r'(?:[A-Za-z]:\\|(?:docs|data|uploads|tmp|src)[/\\]|[/\\]\w+[/\\])',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessedFileSummary:
    file_path: str
    num_chunks: int


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def _metadata_path(vector_store_dir: Union[str, Path]) -> Path:
    return Path(vector_store_dir) / INGEST_METADATA_FILENAME


def _compute_file_signature(file_path: Union[str, Path]) -> str:
    p = Path(file_path)
    stat = p.stat()
    token = f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _load_ingest_metadata(vector_store_dir: Union[str, Path]) -> Dict[str, Any]:
    mpath = _metadata_path(vector_store_dir)
    if not mpath.exists():
        return {}
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ingest_metadata(vector_store_dir: Union[str, Path], payload: Dict[str, Any]) -> None:
    persist_dir = Path(vector_store_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    with open(_metadata_path(vector_store_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_vector_store_ready(vector_store_dir: Union[str, Path]) -> bool:
    persist_dir = Path(vector_store_dir)
    return (
        (persist_dir / "faiss.index").exists()
        and (persist_dir / "metadata.json").exists()
    )


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _is_noisy_line(line: str) -> bool:
    """Return True if the line should be dropped before chunking."""
    stripped = line.strip()
    if not stripped:
        return True

    low = stripped.lower()
    if any(m in low for m in _NOISY_MARKERS):
        return True
    if _ID_LINE_RE.match(stripped):
        return True
    if _KEY_VALUE_RE.match(stripped):
        return True
    if _JSON_LINE_RE.match(stripped):
        return True
    if stripped.startswith("{") and stripped.endswith("}") and ":" in stripped:
        return True
    if _SQL_RE.search(stripped):
        return True
    if _CODE_RE.search(stripped):
        return True
    if _PATH_RE.search(stripped) and len(stripped) < 200:
        return True
    if len(stripped) > _MAX_LINE_LEN:
        return True

    words = re.findall(r"[A-Za-z0-9]+", stripped)
    if len(words) < 3:
        return True

    return False


def _clean_extracted_text_for_rag(text: str) -> str:
    """
    Produce clean, readable plain text suitable for chunking and embedding.

    Removes JSON/structured data lines, ID/metadata key-value lines, SQL fragments,
    code-like constructs, path-like noise, data-dump lines, and lines with fewer
    than 3 meaningful words.
    """
    if not text:
        return ""

    cleaned_lines = [line for line in text.splitlines() if not _is_noisy_line(line)]
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# File type helpers
# ---------------------------------------------------------------------------

def _is_supported_file(file_path: Union[str, Path]) -> bool:
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_PDF_EXTS or ext in SUPPORTED_IMAGE_EXTS


def _extract_text_from_file(
    file_path: str,
    *,
    ocr_engine: OCREngine,
    ocr_fallback_for_pdfs: bool,
    max_ocr_pages: Optional[int] = None,
) -> str:
    """
    Extract plain text from a PDF or image file.

    PDF flow:
    1. Try PyPDF text extraction.
    2. If empty and ocr_fallback_for_pdfs is True, fall back to OCR.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix in SUPPORTED_PDF_EXTS:
        pdf_result = load_pdf_text(file_path)
        text = (pdf_result.text or "").strip()
        if text:
            return text
        if not ocr_fallback_for_pdfs:
            return ""
        return ocr_engine.extract_text_from_pdf(file_path, max_pages=max_ocr_pages)

    if suffix in SUPPORTED_IMAGE_EXTS:
        return ocr_engine.extract_text_from_image(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")


def _make_plain_chunks(
    raw_text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Clean, chunk, and validate text.
    Returns only non-empty plain-text strings of at least _MIN_CHUNK_LEN characters.
    Never returns dicts or JSON objects.
    """
    clean_text = _clean_extracted_text_for_rag(raw_text)
    chunks = preprocess_and_chunk(clean_text, chunk_size=chunk_size, overlap=overlap)
    return [
        c for c in chunks
        if isinstance(c, str) and len(c.strip()) >= _MIN_CHUNK_LEN
    ]


# ---------------------------------------------------------------------------
# Public: chunk extraction (no vector storage)
# ---------------------------------------------------------------------------

def get_chunks_from_file(
    file_path: Union[str, Path],
    *,
    chunk_size: int = 900,
    overlap: int = 150,
    ocr_lang: str = "en",
    ocr_fallback_for_pdfs: bool = True,
    max_ocr_pages: Optional[int] = None,
) -> List[str]:
    """
    Extract and chunk text from a single uploaded file without storing vectors.
    Returns a list of clean plain-text strings.
    """
    file_path_str = str(file_path)
    if not _is_supported_file(file_path):
        raise ValueError(f"Unsupported file type: {file_path_str}")

    ocr_engine = OCREngine(config=OCRConfig(lang=ocr_lang))
    raw_text = _extract_text_from_file(
        file_path_str,
        ocr_engine=ocr_engine,
        ocr_fallback_for_pdfs=ocr_fallback_for_pdfs,
        max_ocr_pages=max_ocr_pages,
    )
    return _make_plain_chunks(raw_text, chunk_size=chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# Public: vector store helpers
# ---------------------------------------------------------------------------

def store_vectors(
    chunks: Sequence[str],
    *,
    vector_store_dir: Union[str, Path],
    embed_model_name: str = "all-MiniLM-L6-v2",
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    recreate_store: bool = False,
) -> int:
    """Embed and store plain-text chunk vectors into FAISS."""
    persist_dir = Path(vector_store_dir)
    if recreate_store and persist_dir.exists():
        shutil.rmtree(persist_dir)
        _get_cached_store.cache_clear()

    # Only store clean string chunks — never dicts or JSON
    safe_chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not safe_chunks:
        return 0

    embedder = SentenceTransformerEmbedder(
        config=EmbeddingConfig(model_name=embed_model_name, batch_size=64)
    )
    store = FAISSVectorStore(
        embedder=embedder,
        persist_dir=str(persist_dir),
        embed_model_name=embed_model_name,
    )
    return store.add_texts(safe_chunks, metadatas=metadatas)


@lru_cache(maxsize=4)
def _get_cached_store(vector_store_dir: str, embed_model_name: str) -> FAISSVectorStore:
    """Reuse embedder + FAISS store instance across repeated retrieval calls."""
    embedder = SentenceTransformerEmbedder(
        config=EmbeddingConfig(model_name=embed_model_name, batch_size=32)
    )
    store = FAISSVectorStore(
        embedder=embedder,
        persist_dir=vector_store_dir,
        embed_model_name=embed_model_name,
    )
    if not store.is_compatible_with_embedder():
        store.rebuild_index_from_existing_texts()
    return store


def clear_vector_store(vector_store_dir: Union[str, Path]) -> None:
    """Delete all vector store artifacts so the next ingestion starts clean."""
    persist_dir = Path(vector_store_dir)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    _get_cached_store.cache_clear()


def rebuild_vector_store_from_texts(
    *,
    vector_store_dir: Union[str, Path],
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """Rebuild FAISS index from already-persisted chunk texts using the current embedder."""
    persist_dir = Path(vector_store_dir)
    if not persist_dir.exists():
        return 0
    embedder = SentenceTransformerEmbedder(
        config=EmbeddingConfig(model_name=embed_model_name, batch_size=64)
    )
    store = FAISSVectorStore(
        embedder=embedder,
        persist_dir=str(persist_dir),
        embed_model_name=embed_model_name,
    )
    rebuilt = store.rebuild_index_from_existing_texts()
    _get_cached_store.cache_clear()
    return rebuilt


def retrieve_relevant_chunks(
    query: str,
    *,
    vector_store_dir: Union[str, Path],
    embed_model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[SearchResult]:
    """Retrieve the most relevant stored chunks for a given query."""
    persist_dir = Path(vector_store_dir)
    if not persist_dir.exists():
        return []
    store = _get_cached_store(str(persist_dir), embed_model_name)
    return store.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Public: end-to-end pipeline
# ---------------------------------------------------------------------------

def process_uploaded_files(
    file_paths: Sequence[Union[str, Path]],
    *,
    vector_store_dir: Union[str, Path],
    chunk_size: int = 900,
    overlap: int = 150,
    ocr_lang: str = "en",
    ocr_fallback_for_pdfs: bool = True,
    max_ocr_pages: Optional[int] = None,
    embed_model_name: str = "all-MiniLM-L6-v2",
    recreate_store: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[ProcessedFileSummary]:
    """
    End-to-end ingestion pipeline:
    1. Extract text via PyPDF / OCR.
    2. Clean (strip JSON, metadata, structured noise).
    3. Chunk into plain-text strings only.
    4. Embed and store vectors in FAISS.

    Returns a lightweight ProcessedFileSummary per file.
    """
    start_total = time.perf_counter()
    persist_dir = Path(vector_store_dir)

    def _progress(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, frac)), msg)

    if recreate_store and persist_dir.exists():
        shutil.rmtree(persist_dir)
        _get_cached_store.cache_clear()

    _progress(0.02, "Checking existing vector store...")
    previous = _load_ingest_metadata(vector_store_dir)
    old_files: Dict[str, str] = previous.get("files", {})
    file_paths_list = [str(fp) for fp in file_paths]
    new_files: Dict[str, str] = {fp: _compute_file_signature(fp) for fp in file_paths_list}

    unchanged_all = (
        not recreate_store
        and is_vector_store_ready(vector_store_dir)
        and old_files
        and old_files == new_files
        and previous.get("embed_model_name") == embed_model_name
        and int(previous.get("chunk_size", chunk_size)) == int(chunk_size)
        and int(previous.get("overlap", overlap)) == int(overlap)
    )

    if unchanged_all:
        print("[RAG] Skipping processing: all documents unchanged, using existing FAISS index.")
        _progress(1.0, "All files unchanged. Loaded existing index instantly.")
        return [
            ProcessedFileSummary(
                file_path=fp,
                num_chunks=int(previous.get("chunk_counts", {}).get(fp, 0)),
            )
            for fp in file_paths_list
        ]

    # Something changed — rebuild from scratch to avoid stale or duplicate chunks.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        _get_cached_store.cache_clear()

    t_model = time.perf_counter()
    ocr_engine = OCREngine(config=OCRConfig(lang=ocr_lang))
    embedder = SentenceTransformerEmbedder(
        config=EmbeddingConfig(model_name=embed_model_name, batch_size=64)
    )
    print(f"[RAG][timing] loading models: {time.perf_counter() - t_model:.2f}s")

    store = FAISSVectorStore(
        embedder=embedder,
        persist_dir=str(persist_dir),
        embed_model_name=embed_model_name,
    )

    summaries: List[ProcessedFileSummary] = []
    chunk_counts: Dict[str, int] = {}
    total_files = max(1, len(file_paths_list))

    for idx, fp in enumerate(file_paths):
        file_path_str = str(fp)
        if not _is_supported_file(fp):
            raise ValueError(f"Unsupported file type: {file_path_str}")

        # 1. Extract
        t_extract = time.perf_counter()
        raw_text = _extract_text_from_file(
            file_path_str,
            ocr_engine=ocr_engine,
            ocr_fallback_for_pdfs=ocr_fallback_for_pdfs,
            max_ocr_pages=max_ocr_pages,
        )
        print(
            f"[RAG][timing] extract ({Path(file_path_str).name}): "
            f"{time.perf_counter() - t_extract:.2f}s"
        )

        # 2. Clean + chunk — only plain-text strings, no dicts or JSON
        t_chunk = time.perf_counter()
        chunks = _make_plain_chunks(raw_text, chunk_size=chunk_size, overlap=overlap)
        print(
            f"[RAG][timing] chunk ({Path(file_path_str).name}): "
            f"{time.perf_counter() - t_chunk:.2f}s  ->  {len(chunks)} chunks"
        )

        # 3. Build per-chunk metadata (separate from chunk text)
        metadatas = [
            {
                "source_file": os.path.basename(file_path_str),
                "source_path": file_path_str,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # 4. Embed and store
        t_embed = time.perf_counter()
        store.add_texts(chunks, metadatas=metadatas)
        print(
            f"[RAG][timing] embed+index ({Path(file_path_str).name}): "
            f"{time.perf_counter() - t_embed:.2f}s"
        )

        summaries.append(ProcessedFileSummary(file_path=file_path_str, num_chunks=len(chunks)))
        chunk_counts[file_path_str] = len(chunks)
        _progress(
            (idx + 1) / total_files,
            f"Processed {idx + 1}/{total_files}: {Path(file_path_str).name}", 
        )

    # Ensure subsequent retrieval uses the freshly built index.
    _get_cached_store.cache_clear()

    _save_ingest_metadata(
        vector_store_dir,
        {
            "embed_model_name": embed_model_name,
            "chunk_size": int(chunk_size),
            "overlap": int(overlap),
            "files": new_files,
            "chunk_counts": chunk_counts,
            "updated_at": int(time.time()),
        },
    )

    print(f"[RAG][timing] total: {time.perf_counter() - start_total:.2f}s")
    _progress(1.0, "Processing complete.")
    return summaries
