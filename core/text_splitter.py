from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_SPACES_TABS_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


def clean_text(text: str) -> str:
    """
    Basic production-grade text cleaning:
    - Normalize whitespace
    - Remove excessive newlines
    - Trim
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")
    # Preserve newlines so we can chunk by paragraphs/sentences more meaningfully.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize spaces/tabs but keep newlines.
    text = _SPACES_TABS_RE.sub(" ", text)
    # Collapse excessive blank lines.
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    # Trim lines and overall.
    text = "\n".join([ln.strip() for ln in text.split("\n")]).strip()
    return text


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Chunking parameters.

    `chunk_size` and `overlap` are interpreted as character counts.

    This better matches typical RAG needs and avoids mixing unrelated content
    by keeping chunks small and semantically tight.
    """

    chunk_size: int = 500
    overlap: int = 50


def split_text_into_chunks(text: str, *, config: ChunkingConfig = ChunkingConfig()) -> List[str]:
    """
    Split cleaned text into overlapping chunks.

    This splitter is paragraph/sentence-aware:
    - Break into paragraphs (blank lines)
    - Break paragraphs into sentences
    - Greedily pack sentences until `chunk_size` characters
    - Add character-overlap between chunks to preserve continuity
    """

    if not text:
        return []

    chunk_size = max(1, int(config.chunk_size))
    overlap = max(0, int(config.overlap))
    if overlap >= chunk_size:
        raise ValueError("`overlap` must be smaller than `chunk_size`.")

    def _char_len(s: str) -> int:
        return len(s or "")

    # Paragraphs separated by blank lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p and p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # Sentence list
    sentences: list[str] = []
    for p in paragraphs:
        # Keep short paragraphs as-is; otherwise sentence split.
        if _char_len(p) <= chunk_size:
            sentences.append(p.strip())
            continue
        parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(p) if s and s.strip()]
        sentences.extend(parts if parts else [p.strip()])

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)
        current = []
        current_len = 0

    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue

        # If a single sentence is too long, hard-split by characters at whitespace boundaries if possible.
        if _char_len(s) > chunk_size:
            start = 0
            while start < len(s):
                end = min(len(s), start + chunk_size)
                piece = s[start:end]
                # Try not to cut mid-word (only if we have room to adjust).
                if end < len(s):
                    cut = piece.rfind(" ")
                    if cut >= max(50, int(chunk_size * 0.6)):
                        end = start + cut
                        piece = s[start:end]
                piece = piece.strip()
                if piece:
                    chunks.append(piece)
                if end >= len(s):
                    break
                start = max(0, end - overlap)
            continue

        # +1 for joining space if needed
        extra = 1 if current else 0
        if current_len + extra + _char_len(s) <= chunk_size:
            current.append(s)
            current_len += extra + _char_len(s)
        else:
            _flush()
            current.append(s)
            current_len = _char_len(s)

    _flush()

    # Apply overlap between chunks by characters to preserve continuity.
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    prev_tail: str = ""
    for chunk in chunks:
        c = chunk.strip()
        if prev_tail:
            combined = (prev_tail + " " + c).strip()
        else:
            combined = c
        overlapped.append(combined[:chunk_size].strip() if len(combined) > chunk_size else combined)
        prev_tail = c[-overlap:] if len(c) >= overlap else c

    return overlapped


def preprocess_and_chunk(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Clean then chunk.
    """

    cleaned = clean_text(text)
    return split_text_into_chunks(cleaned, config=ChunkingConfig(chunk_size=chunk_size, overlap=overlap))
