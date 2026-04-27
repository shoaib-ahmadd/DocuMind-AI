"""
chat_engine.py — Production-ready RAG chat engine
Stack: FAISS + Groq + SentenceTransformer embeddings
"""

from __future__ import annotations

import logging
import os
from groq import Groq
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import requests

from core.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from core.vector_store import FAISSVectorStore, SearchResult
from utils.text_cleaner import clean_retrieved_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatResponse:
    answer: str
    source_text: str
    confidence_score: float


@dataclass(frozen=True)
class ChatConfig:
    vector_store_dir: Union[str, Path]
    embed_model_name: str = "all-MiniLM-L6-v2"

    # Retrieval
    candidate_k: int = 30          # Cast a wide FAISS net
    top_k: int = 5                # Final context chunks
    similarity_threshold: float = -1.0  # Practical lower bound for MiniLM IP scores

    # Context / prompt
    max_context_chars: int = 2500

    # Groq
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ("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")

    # Reranking weights
    semantic_weight: float = 0.70
    lexical_weight: float = 0.30


# ---------------------------------------------------------------------------
# Stopwords / constants
# ---------------------------------------------------------------------------

_NOT_FOUND = "Not found in document"

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "as", "at", "by", "it", "this",
    "that", "from", "but", "not", "you", "your", "we", "they", "their",
    "i", "me", "my", "has", "have", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "its", "also", "more",
})

_QUERY_STOP_PHRASES: tuple[str, ...] = (
    "what is", "what are", "tell me about", "can you explain",
    "please explain", "define", "how do you calculate", "how is",
    "explain", "describe",
)


# ---------------------------------------------------------------------------
# Compiled regexes — NOISE REMOVAL ONLY (preserve formulas & KPIs)
# ---------------------------------------------------------------------------

# Metadata / path noise
_ANSWER_NOISE_RE = re.compile(
    r"(?im)^\s*(?:\[Source[^\]]*\]|Source\s*:\s*|File\s*:\s*|docs[/\\][^\n]+)\s*$"
)
_PATH_LIKE_RE = re.compile(
    r"(?:^|[\s\"'(\[])(?:[A-Za-z]:\\|(?:docs|data|uploads|tmp|src)[/\\]"
    r"|[/\\][^\s]+[/\\]|[^\s]+[/\\][^\s]+\.(?:pdf|png|jpe?g|webp|bmp|tiff?))\b",
    re.IGNORECASE,
)

# SQL DML — remove only DML statements, not SELECT-based KPI definitions
_SQL_DML_RE = re.compile(
    r"(?i)\b(?:insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table"
    r"|alter\s+table|create\s+table)\b.+",
    re.DOTALL,
)

# JSON artefacts
_JSON_OBJECT_LINE_RE = re.compile(r'^\s*\{.*:\s*.*\}\s*$')
_ID_LINE_RE = re.compile(r"(?i)^\s*(?:id|chunk_id|doc_id)\s*[:=]\s*\S+\s*$")

# Markdown (strip decoration but keep text)
_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_MARKDOWN_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_MARKDOWN_RULE_RE = re.compile(r"^\s*[-=|*]{3,}\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Excel cell refs that are STANDALONE (not part of a formula)
_BARE_EXCEL_REF_RE = re.compile(r"(?<![A-Za-z(,])([A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?)(?![A-Za-z(,\w])")

# Sentence splitting
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Project-internal meta noise
_PROJECT_META_RE = re.compile(r"(?i)\b(?:project|overview|build)\b|agent/|rag/")


# ---------------------------------------------------------------------------
# Keyword utilities
# ---------------------------------------------------------------------------

def _keywords(text: str) -> list[str]:
    return [
        w.lower() for w in _WORD_RE.findall(text or "")
        if w.lower() not in _STOPWORDS and len(w) > 2
    ]


def _keyword_set(text: str) -> set[str]:
    return set(_keywords(text))


def _preprocess_query(query: str) -> tuple[str, set[str]]:
    """Normalise query for FAISS search; return (normalised_str, keyword_set)."""
    q = (query or "").strip().lower()
    for phrase in _QUERY_STOP_PHRASES:
        if q.startswith(phrase):
            q = q[len(phrase):].strip()
            break
    q = re.sub(r"[^a-z0-9\s%*+\-/=_]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q, _keyword_set(q)


# ---------------------------------------------------------------------------
# Context cleaning — conservative: preserve formulas, KPIs, metrics
# ---------------------------------------------------------------------------

def _clean_raw_context(context: str) -> str:
    """
    Remove only true noise (SQL DML, code fences, path strings, JSON artefacts,
    markdown decoration).  Intentionally PRESERVE:
      - spreadsheet formulas: SUM(…), VLOOKUP(…), AOV = …
      - KPI definitions: "Revenue = Price × Qty"
      - arithmetic expressions: (A – B) / C
      - Excel refs embedded inside formulas
    """
    if not context:
        return ""

    c = context

    # Remove code fences completely
    c = _CODE_FENCE_RE.sub(" ", c)
    c = _INLINE_CODE_RE.sub(" ", c)

    # Remove SQL DML
    c = _SQL_DML_RE.sub(" ", c)

    # Remove JSON line artefacts
    c = re.sub(r'"[A-Za-z_][\w]*"\s*:\s*(?:"[^"]*"|\d+\.?\d*),?\s*', " ", c)
    c = re.sub(r'\{[^}]{0,80}\}', " ", c)
    c = re.sub(r'[\[\]{}\\]', " ", c)

    # Strip markdown decoration (keep content)
    c = _MARKDOWN_HEADING_RE.sub("", c)
    c = _MARKDOWN_EMPHASIS_RE.sub(r"\2", c)
    c = _MARKDOWN_RULE_RE.sub("", c)
    c = re.sub(r'#+\s*', "", c)

    # Remove standalone bare Excel cell refs (B3, A1:C5) not inside formula context.
    # We keep them when they appear adjacent to '=', '(', ',', or letters (formula context).
    c = _BARE_EXCEL_REF_RE.sub(" ", c)

    # Collapse whitespace
    c = re.sub(r'[ \t]+', " ", c)
    c = re.sub(r'\n{3,}', "\n\n", c)

    return c.strip()


# ---------------------------------------------------------------------------
# Answer output cleaning
# ---------------------------------------------------------------------------

def _clean_answer_output(text: str) -> str:
    """Remove metadata / path noise lines from a raw LLM answer."""
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _ANSWER_NOISE_RE.match(stripped):
            continue
        if _PATH_LIKE_RE.search(line) and len(stripped) < 200:
            continue
        if stripped in ("---", "```"):
            continue
        if _JSON_OBJECT_LINE_RE.match(stripped):
            continue
        if _ID_LINE_RE.match(stripped):
            continue
        lines.append(line)
    out = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", out)


def _finalize_answer(text: Optional[str], *, fallback: str = _NOT_FOUND) -> str:
    """
    Final answer sanitation.  We are PERMISSIVE — we only strip lines that are
    clearly system / meta artefacts.  We do NOT truncate to 3 sentences by
    default so that formula-heavy answers survive.
    """
    if not text or not text.strip():
        return fallback

    cleaned = _clean_answer_output(text)
    if not cleaned:
        return fallback

    kept: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Drop system-prompt echo lines
        if line.lower().startswith(("you are a", "system prompt", "document context:",
                                    "user question:", "rules:", "answer:")):
            continue
        # Drop Python/import artefacts
        if any(kw in line for kw in ("def ", "class ", "import ", "print(")):
            continue
        # Drop project meta noise
        if _PROJECT_META_RE.search(line) and len(line) < 80:
            continue
        # Keep everything else (including formulas, KPIs, multi-sentence answers)
        kept.append(line)

    if not kept:
        return fallback

    result = "\n".join(kept).strip()

    # If it's very short and looks like noise, return fallback
    if len(result) < 8 or len(_WORD_RE.findall(result)) < 3:
        return fallback

    return result


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _similarity_to_confidence(
    best: float,
    second: Optional[float] = None,
    num_results: int = 0,
) -> float:
    """
    Map FAISS inner-product score → [0, 1] confidence.

    MiniLM IP scores for good matches typically fall in 0.25–0.85.
    We normalise from the [-1, 1] range then apply a margin bonus.
    """
    if num_results == 0 or best < -0.5:
        return 0.0

    # Normalise: IP in [-1, 1] → [0, 1]
    base = max(0.0, min(1.0, (float(best) + 1.0) / 2.0))

    # Margin bonus: clear winner → higher confidence
    margin_bonus = 0.0
    if second is not None:
        margin = max(0.0, float(best) - float(second))
        margin_bonus = max(0.0, min(0.15, margin / 0.4)) * 0.15

    # Volume bonus: more results → slightly higher confidence
    volume_bonus = min(0.05, num_results * 0.005)

    return max(0.0, min(1.0, base + margin_bonus + volume_bonus))


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _rerank_results(
    query: str,
    results: Sequence[SearchResult],
    *,
    final_top_k: int,
    semantic_weight: float = 0.70,
    lexical_weight: float = 0.30,
) -> List[SearchResult]:
    """
    Blended reranking: semantic (FAISS) score + lexical overlap.
    Deduplicates chunks by normalised text fingerprint.
    """
    if not results:
        return []

    _, q_kw = _preprocess_query(query)

    scored: list[tuple[float, SearchResult]] = []
    seen: set[str] = set()

    for r in results:
        # Use clean_retrieved_text if available, else raw text
        try:
            cleaned = clean_retrieved_text(r.text or "")
        except Exception:
            cleaned = (r.text or "").strip()

        if not cleaned or len(cleaned) < 10:
            continue

        # Dedup fingerprint (first 120 normalised chars)
        fp = re.sub(r"\s+", " ", cleaned).strip().lower()[:120]
        if fp in seen:
            continue
        seen.add(fp)

        # Lexical overlap
        c_kw = _keyword_set(cleaned)
        if c_kw:
            overlap = len(q_kw & c_kw)
            # Jaccard-style: overlap / union
            lex_score = overlap / max(1.0, len(q_kw | c_kw))
        else:
            lex_score = 0.0

        blended = (semantic_weight * float(r.score)) + (lexical_weight * lex_score)

        scored.append((
            blended,
            SearchResult(text=cleaned, metadata=r.metadata, score=float(r.score)),
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max(1, final_top_k)]]


def _build_context(results: Sequence[SearchResult], *, max_chars: int) -> str:
    """
    Assemble context string from ranked chunks.
    Each chunk is separated by a clear delimiter for the LLM.
    """
    if not results:
        return ""

    parts: list[str] = []
    total = 0
    separator = "\n\n---\n\n"
    sep_len = len(separator)

    for r in results:
        chunk = (r.text or "").strip()
        if not chunk:
            continue
        chunk_len = len(chunk) + sep_len
        if total + chunk_len > max_chars:
            remaining = max_chars - total - sep_len
            if remaining > 150:
                parts.append(chunk[:remaining].rstrip())
            break
        parts.append(chunk)
        total += chunk_len

    return separator.join(parts).strip()


def _synthesize_fallback(query: str, context: str, max_sentences: int = 4) -> str:
    """
    Keyword-scored sentence extraction when Groq is unavailable.
    Returns best matching sentences from context.
    """
    q_kw = _keyword_set(query)
    if not context.strip() or not q_kw:
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context) if len(s.strip()) > 20]
    if not sentences:
        return ""

    scored: list[tuple[float, str]] = []
    for s in sentences:
        s_kw = _keyword_set(s)
        if not s_kw:
            continue
        # TF-like: keyword hits weighted by inverse sentence length
        hits = len(q_kw & s_kw)
        if hits == 0:
            continue
        score = hits / max(1.0, len(s) / 200.0)
        scored.append((score, s))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    best_sentences = [s for _, s in scored[:max_sentences]]
    answer = " ".join(best_sentences).strip()

    if len(answer) > 700:
        answer = answer[:700].rstrip() + "…"
    return answer


# ---------------------------------------------------------------------------
# Groq caller
# ---------------------------------------------------------------------------

def call_groq(
    prompt: str,
    *,
    model: str = "llama-3.3-70b-versatile",
    api_key: str = "",
) -> str:

    client = Groq(
    api_key=api_key or os.environ.get("GROQ_API_KEY")
)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"[Groq Error] {e}")
        return "Error generating response from Groq."


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(query: str, context: str) -> str:
    """
    Build a document-grounded RAG prompt.
    If context is present, instruct the model to use it; allow own knowledge
    as fallback so formula/KPI queries still get useful answers.
    """
    if context.strip():
        return (
    "You are a document assistant.\n\n"
    "Answer ONLY from the provided document context.\n"
    "If the answer is not present in the context, say:\n"
    "'I could not find this in the uploaded documents.'\n\n"
    "Do not use outside knowledge.\n"
    "Do not hallucinate.\n"
    "Keep answers concise, accurate, and relevant.\n"
    "Preserve formulas and KPI names exactly.\n\n"
    f"QUESTION:\n{query}\n\n"
    f"CONTEXT:\n{context}\n\n"
    "ANSWER:"
)
    else:
        return (
            "You are a knowledgeable assistant. Answer the following question clearly and concisely.\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """
    RAG chatbot — FAISS + Groq.

    Lifecycle:
      - FAISS index is loaded from disk on __init__.
      - The engine is stored in st.session_state (NOT @st.cache_resource)
        so that re-uploading documents discards the old instance cleanly.
    """

    def __init__(self, *, config: ChatConfig) -> None:
        self._cfg = config

        embedder = SentenceTransformerEmbedder(
            config=EmbeddingConfig(model_name=self._cfg.embed_model_name)
        )
        self._store = FAISSVectorStore(
            embedder=embedder,
            persist_dir=str(self._cfg.vector_store_dir),
        )

        try:
            n_vectors = self._store.index.ntotal if hasattr(self._store, "index") else "?"
        except Exception:
            n_vectors = "?"

        logger.info(
            "[ChatEngine] Initialized | model=%s | embed=%s | vectors=%s",
            self._cfg.groq_model,
            self._cfg.embed_model_name,
            n_vectors,
        )
        print(f"[ChatEngine] ✅ Initialized — vectors in index: {n_vectors}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> List[SearchResult]:
        """
        Two-pass FAISS search:
          Pass 1 — normalised/stripped query (strips question words, punctuation).
          Pass 2 — raw query, if pass 1 returns nothing.
        Results from both passes are merged and de-duplicated by score.
        """
        normalised, _ = _preprocess_query(query)

        results: List[SearchResult] = []

        # Pass 1: normalised
        if normalised:
            results = self._store.search(normalised, top_k=self._cfg.candidate_k)
            logger.info("[Retrieve] Pass-1 '%s' → %d results", normalised[:60], len(results))
            print(f"[ChatEngine] Pass-1 search '{normalised[:60]}' → {len(results)} results")

        # Pass 2: raw query fallback
        if not results or len(results) < 3:
            raw_results = self._store.search(query, top_k=self._cfg.candidate_k)
            logger.info("[Retrieve] Pass-2 (raw) '%s' → %d results", query[:60], len(raw_results))
            print(f"[ChatEngine] Pass-2 search (raw) → {len(raw_results)} results")

            # Merge: keep higher-scoring result for each unique chunk
            seen: dict[str, SearchResult] = {r.text[:80]: r for r in results}
            for r in raw_results:
                key = (r.text or "")[:80]
                if key not in seen or r.score > seen[key].score:
                    seen[key] = r
            results = list(seen.values())

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("[Retrieve] Total after merge: %d results", len(results))
        print(f"[ChatEngine] Total merged results: {len(results)}")
        return results

    # ------------------------------------------------------------------
    # Local fallback
    # ------------------------------------------------------------------

    def _local_fallback(self, query: str, context_clean: str, context_raw: str) -> str:
        """
        When Groq is unavailable: extract best sentences from context.
        Prefers cleaned context; falls back to raw if needed.
        """
        print("[ChatEngine] ⚠️  Groq unavailable — using local keyword fallback.")
        logger.warning("[ChatEngine] Using local keyword fallback.")

        ctx = context_clean if context_clean.strip() else context_raw
        if not ctx.strip():
            return _NOT_FOUND

        synth = _synthesize_fallback(query, ctx)
        if synth and len(synth.strip()) > 20:
            return synth.strip()

        # Last resort: first 400 chars of cleaned context
        snippet = ctx.strip()[:400].rstrip()
        return snippet if len(snippet) > 20 else _NOT_FOUND

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def answer(self, query: str) -> ChatResponse:
        q = (query or "").strip()
        if not q:
            return ChatResponse(
                answer="Please enter a question.",
                source_text="",
                confidence_score=0.0,
            )

        # ── 1. Retrieve ───────────────────────────────────────────────────
        raw_results = self._retrieve(q)

        logger.debug("[Answer] Raw results count: %d", len(raw_results))
        if raw_results:
            logger.debug(
                "[Answer] Score range: %.4f – %.4f",
                raw_results[-1].score, raw_results[0].score,
            )
            print(
                f"[ChatEngine] Score range: {raw_results[-1].score:.3f} – {raw_results[0].score:.3f}"
            )

        # ── 2. Rerank ─────────────────────────────────────────────────────
        ranked = _rerank_results(
            q, raw_results,
            final_top_k=self._cfg.top_k,
            semantic_weight=self._cfg.semantic_weight,
            lexical_weight=self._cfg.lexical_weight,
        )

        # ── 3. Apply similarity threshold ─────────────────────────────────
        filtered = [r for r in ranked if r.score >= self._cfg.similarity_threshold]

        # If threshold cuts everything, relax to top-3 so we always try to answer
        if not filtered and ranked:
            logger.warning(
                "[Answer] Threshold %.2f filtered all %d results — using top-3 anyway.",
                self._cfg.similarity_threshold, len(ranked),
            )
            print(
                f"[ChatEngine] ⚠️  Threshold too strict — using top-3 of {len(ranked)} results."
            )
            filtered = ranked[:3]

        logger.info(
            "[Answer] After rerank+filter: %d / %d (threshold=%.2f)",
            len(filtered), len(ranked), self._cfg.similarity_threshold,
        )
        print(
            f"[ChatEngine] Ranked: {len(ranked)}  After threshold: {len(filtered)}"
        )

        # ── 4. Confidence score ───────────────────────────────────────────
        best   = filtered[0].score if filtered else -1.0
        second = filtered[1].score if len(filtered) > 1 else None
        confidence = _similarity_to_confidence(best, second, num_results=len(filtered))

        # ── 5. Build context ──────────────────────────────────────────────
        context_raw   = _build_context(filtered, max_chars=self._cfg.max_context_chars)
        context_clean = _clean_raw_context(context_raw)

        logger.info(
            "[Answer] Context: raw=%d chars  clean=%d chars",
            len(context_raw), len(context_clean),
        )
        print(
            f"[ChatEngine] Context: raw={len(context_raw)}c  clean={len(context_clean)}c"
        )

        # ── 6. Build prompt ───────────────────────────────────────────────
        # Use cleaned context for the prompt; raw is kept for source display
        prompt_context = context_clean if context_clean.strip() else context_raw
        prompt = _build_prompt(q, prompt_context)

        logger.debug("[Answer] Prompt length: %d chars", len(prompt))

        # ── 7. Call Groq ────────────────────────────────────────────────
        raw_answer = call_groq(
            prompt,
            model=self._cfg.groq_model,
            api_key=self._cfg.groq_api_key,
        )

        # ── 8. Process answer ─────────────────────────────────────────────
        if raw_answer and len(raw_answer.strip()) > 5:
            answer = _finalize_answer(raw_answer)
            print(f"[ChatEngine] ✅ Groq answer ({len(answer)} chars).")
        else:
            # Groq failed or returned empty
            answer = self._local_fallback(q, context_clean, context_raw)

        # ── 9. Last-resort guard ──────────────────────────────────────────
        if not answer or not answer.strip():
            answer = _NOT_FOUND
            confidence = 0.0

        return ChatResponse(
            answer=answer.strip(),
            source_text=context_raw,
            confidence_score=round(confidence, 4),
        )
