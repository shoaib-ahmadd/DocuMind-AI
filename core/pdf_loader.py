from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PdfLoadResult:
    """Result of loading a PDF as text."""

    text: str
    num_pages: int


def load_pdf_text(
    pdf_path: str,
    *,
    max_pages: Optional[int] = None,
) -> PdfLoadResult:
    """
    Extract text from a PDF using PyPDF.

    Notes:
    - PyPDF works best for "digital" PDFs where the text layer exists.
    - If the PDF is scanned, extracted text may be empty; callers can
      fall back to OCR (see `core/ocr_engine.py`).
    """

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import PyPDF. Install `pypdf` to enable PDF loading."
        ) from e

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    limit = num_pages if max_pages is None else min(num_pages, max_pages)

    parts: list[str] = []
    for page_idx in range(limit):
        page = reader.pages[page_idx]
        extracted = page.extract_text() or ""
        extracted = extracted.strip()
        if extracted:
            parts.append(extracted)

    return PdfLoadResult(text="\n\n".join(parts).strip(), num_pages=num_pages)
