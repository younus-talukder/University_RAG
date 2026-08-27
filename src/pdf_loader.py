from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

from .config import DOCUMENT_DIR, SUPPORTED_DOCUMENT_EXTENSIONS


def _clean_text(text: str) -> str:
    if text is None:
        return ""

    # pypdf can emit replacement characters for PDF bullets or smart dashes.
    text = text.replace("\ufffd", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_pdf_files(document_dir: str | Path = DOCUMENT_DIR) -> List[Path]:
    document_path = Path(document_dir)
    if not document_path.exists():
        raise FileNotFoundError(f"Document directory not found: {document_path}")

    pdf_files = sorted(
        [
            path
            for path in document_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ]
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {document_path}")

    return pdf_files


def load_pdf_documents(document_dir: str | Path = DOCUMENT_DIR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pages: List[Dict[str, Any]] = []
    extraction_issues: List[Dict[str, Any]] = []

    for pdf_path in get_pdf_files(document_dir):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive path
            extraction_issues.append({
                "source": pdf_path.name,
                "page": None,
                "issue_type": "open_error",
                "error": f"Failed to open PDF: {exc}",
            })
            continue

        if len(reader.pages) == 0:
            extraction_issues.append({
                "source": pdf_path.name,
                "page": None,
                "issue_type": "empty_document",
                "error": "PDF has no pages.",
            })
            continue

        extracted_pages_for_document = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
                cleaned_text = _clean_text(raw_text)
            except Exception as exc:
                extraction_issues.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "issue_type": "extraction_error",
                    "error": f"Extraction failed: {exc}",
                })
                continue

            if not cleaned_text:
                extraction_issues.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "issue_type": "empty_page",
                    "error": "No extractable text found on this page. OCR may be required if the page is scanned.",
                })
                continue

            extracted_pages_for_document += 1
            pages.append({
                "text": cleaned_text,
                "source": pdf_path.name,
                "page": page_number,
                "document_path": str(pdf_path),
            })

        if extracted_pages_for_document == 0:
            extraction_issues.append({
                "source": pdf_path.name,
                "page": None,
                "issue_type": "no_extractable_text",
                "error": "No extractable text found in this PDF. OCR may be required.",
            })

    return pages, extraction_issues


__all__ = ["get_pdf_files", "load_pdf_documents"]
