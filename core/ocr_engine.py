from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for PaddleOCR."""

    lang: str = "en"
    use_angle_cls: bool = True


def _try_import_paddleocr() -> Any:
    try:
        # PaddleOCR is heavy; import lazily to give clearer errors.
        from paddleocr import PaddleOCR  # type: ignore

        return PaddleOCR
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import PaddleOCR. Install dependencies: `paddleocr` "
            "(and PaddlePaddle) to enable OCR."
        ) from e


class OCREngine:
    """
    OCR wrapper around PaddleOCR for images (and optionally PDF pages).
    """

    def __init__(self, *, config: OCRConfig = OCRConfig()):
        PaddleOCR = _try_import_paddleocr()
        self._ocr = PaddleOCR(lang=config.lang, use_angle_cls=config.use_angle_cls)

    @staticmethod
    def _extract_text_from_ocr_result(ocr_result: Any) -> str:
        """
        Convert PaddleOCR output to a plain string.

        PaddleOCR returns:
        - For `ocr(image_path)`: List[ List[ [box, (text, conf)], ... ] ] per page.
        """

        if not ocr_result:
            return ""

        lines: list[str] = []
        # Typical shape: [ [ (box, (text, score)), ... ] ]
        if isinstance(ocr_result, list):
            for page in ocr_result:
                if not page:
                    continue
                for item in page:
                    # item: (box, (text, score))
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
        """
        Extract text from an image file.
        """

        if not image_path:
            return ""

        ocr_result = self._ocr.ocr(image_path, cls=True)
        text = self._extract_text_from_ocr_result(ocr_result)
        if min_chars_per_line > 1:
            # Optional post-filter to remove very small noise.
            text = "\n".join([ln for ln in text.splitlines() if len(ln.strip()) >= min_chars_per_line])
            text = text.strip()
        return text

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        *,
        max_pages: Optional[int] = None,
        dpi: int = 300,
    ) -> str:
        """
        Extract text from a scanned PDF by rendering pages to images and OCR-ing them.

        This requires `pdf2image` and Poppler installed on the system.
        """

        try:
            from pdf2image import convert_from_path  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PDF OCR requires `pdf2image` (and Poppler). Install `pdf2image` first, "
                "then ensure Poppler is available on PATH."
            ) from e

        # Rendering all pages can be expensive; allow limiting.
        rendered_pages = convert_from_path(
            pdf_path,
            dpi=dpi,
        )
        limit = len(rendered_pages) if max_pages is None else min(len(rendered_pages), max_pages)

        parts: list[str] = []
        for page_idx in range(limit):
            img = rendered_pages[page_idx]
            # PaddleOCR supports PIL images directly
            ocr_result = self._ocr.ocr(img, cls=True)
            text = self._extract_text_from_ocr_result(ocr_result)
            if text:
                parts.append(text)

        return "\n\n".join(parts).strip()
