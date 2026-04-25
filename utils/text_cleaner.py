from __future__ import annotations

import re
from typing import Iterable

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]+\s+", re.MULTILINE)
_JSON_LINE_RE = re.compile(r'^\s*\{.*:\s*.*\}\s*$', re.MULTILINE)
_SOURCE_LINE_RE = re.compile(r"^\s*(?:\[source[^\]]*\]|source\s*:|file\s*:).*$", re.IGNORECASE)
_PATH_LIKE_RE = re.compile(
    r"(?:^|[\s\"'(\[])(?:[A-Za-z]:\\|(?:docs|data|uploads|tmp|src)[/\\]|[/\\][^\s]+[/\\]|[^\s]+[/\\][^\s]+\.(?:pdf|png|jpe?g|webp|bmp|tiff?))\b",
    re.IGNORECASE,
)
_SPACES_RE = re.compile(r"[ \t]{2,}")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def _drop_noise_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        if _HEADING_RE.match(ln):
            continue
        if _SOURCE_LINE_RE.match(ln):
            continue
        # Path-like lines are often metadata/source pointers rather than content.
        if _PATH_LIKE_RE.search(ln) and len(ln) < 180:
            continue
        # Drop obvious single-line JSON records, keep natural prose with braces.
        if _JSON_LINE_RE.match(ln):
            continue
        out.append(line)
    return out


def clean_retrieved_text(text: str) -> str:
    """
    Clean retrieved chunk text while preserving semantic content.

    This strips common noise (paths, source tags, markdown structure, code fences),
    but keeps sentence and paragraph boundaries for better LLM grounding.
    """
    if not text:
        return ""

    t = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    t = _CODE_FENCE_RE.sub("", t)
    t = _BULLET_RE.sub("", t)
    lines = _drop_noise_lines(t.splitlines())
    t = "\n".join([ln.strip() for ln in lines if ln.strip()])
    t = _SPACES_RE.sub(" ", t)
    t = _MANY_NEWLINES_RE.sub("\n\n", t)
    return t.strip()