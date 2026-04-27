from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for PaddleOCR."""

    lang: str = "en"
    use_angle_cls: bool = True


def _try_import_paddleocr() -> Any:
    try:
        from paddleocr import PaddleOCR  # type: ignore
        return PaddleOCR

    except Exception:
        return None


class OCREngine:
    """
    OCR wrapper around PaddleOCR for images (and optionally PDF pages).
    """

    def __init__(self, *, config: OCRConfig = OCRConfig()):
        PaddleOCR = _try_import_paddleocr()

        # If PaddleOCR is not installed, disable OCR gracefully
        if PaddleOCR is None:
            self._ocr = None
        else:
            self._ocr = PaddleOCR(
                lang=config.lang,
                use_angle_cls=config.use_angle_cls
            )

    @staticmethod
    def _extract_text_from_ocr_result(ocr_result: Any) -> str:
        """
        Convert PaddleOCR output to plain string.
        """

        if not ocr_result:
            return ""

        lines: list[str] = []

        if isinstance(ocr_result, list):
            for page in ocr_result:
                if not page:
                    continue

                for item in page:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue

                    text_score = item[1]

                    if not isinstance(text_score, (list, tuple)) or len(text_score) < 1:
                        continue

                    text = text_score[0]

                    if isinstance(text, str) and text.strip():
                        lines.append(text.strip())

        return "\n".join(lines).strip()

    def extract_text_from_image(
        self,
        image_path: str,
        *,
        min_chars_per_line: int = 1,
    ) -> str:

        # OCR disabled fallback
        if self._ocr is None:
            return "OCR is disabled because PaddleOCR is not installed."

        if not image_path:
            return ""

        ocr_result = self._ocr.ocr(image_path, cls=True)

        text = self._extract_text_from_ocr_result(ocr_result)

        if min_chars_per_line > 1:
            text = "\n".join(
                [
                    ln
                    for ln in text.splitlines()
                    if len(ln.strip()) >= min_chars_per_line
                ]
            )
            text = text.strip()

        return text

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        *,
        max_pages: Optional[int] = None,
        dpi: int = 300,
    ) -> str:

        # OCR disabled fallback
        if self._ocr is None:
            return "PDF OCR is disabled because PaddleOCR is not installed."

        try:
            from pdf2image import convert_from_path  # type: ignore

        except Exception:
            return "pdf2image dependency missing."

        rendered_pages = convert_from_path(
            pdf_path,
            dpi=dpi,
        )

        limit = (
            len(rendered_pages)
            if max_pages is None
            else min(len(rendered_pages), max_pages)
        )

        parts: list[str] = []

        for page_idx in range(limit):
            img = rendered_pages[page_idx]

            ocr_result = self._ocr.ocr(img, cls=True)

            text = self._extract_text_from_ocr_result(ocr_result)

            if text:
                parts.append(text)

        return "\n\n".join(parts).strip()
