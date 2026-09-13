from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from pypdf import PdfReader

from .config import DOCUMENT_DIR
from .corpus import (
    DocumentDescriptor,
    DocumentStatus,
    IngestionResult,
    discover_documents,
    mark_exact_duplicates,
)


def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.replace("\ufffd", "-").replace("\r\n", "\n").replace("\r", "\n")
    # Join only obvious lowercase word-wrap hyphenation; retain intentional
    # punctuation and all other source wording.
    text = re.sub(r"(?<=[a-z])\s*-\s*\n\s*(?=[a-z])", "", text)
    lines: list[str] = []
    blank_pending = False
    for raw_line in text.split("\n"):
        line = re.sub(r"[\t \u00a0]+", " ", raw_line).strip()
        if not line:
            blank_pending = bool(lines)
            continue
        if blank_pending and lines and lines[-1] != "":
            lines.append("")
        lines.append(line)
        blank_pending = False
    return "\n".join(lines).strip()


def get_pdf_files(document_dir: str | Path = DOCUMENT_DIR, recursive: bool = True) -> List[Path]:
    return [document.path for document in discover_documents(document_dir, recursive=recursive)]


def _record_issue(
    document: DocumentDescriptor,
    issues: list[dict[str, Any]],
    issue_type: str,
    message: str,
    page: int | None = None,
) -> None:
    issue = {
        "document_id": document.document_id,
        "source": document.source,
        "relative_path": document.relative_path,
        "page": page,
        "issue_type": issue_type,
        "error_type": issue_type,
        "error": message,
    }
    issues.append(issue)
    document.ingestion_errors.append(issue)


def ingest_documents(
    document_dir: str | Path = DOCUMENT_DIR,
    recursive: bool = True,
    documents: Sequence[DocumentDescriptor] | None = None,
) -> IngestionResult:
    descriptors = list(documents) if documents is not None else discover_documents(document_dir, recursive=recursive)
    mark_exact_duplicates(descriptors)
    pages: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for document in descriptors:
        if document.ingestion_status == DocumentStatus.DUPLICATE.value:
            _record_issue(
                document,
                issues,
                "duplicate_document",
                f"Exact duplicate of document_id={document.duplicate_of}; not indexed.",
            )
            continue

        try:
            reader = PdfReader(str(document.path))
            if reader.is_encrypted:
                try:
                    unlocked = reader.decrypt("")
                except Exception:
                    unlocked = 0
                if not unlocked:
                    document.ingestion_status = DocumentStatus.FAILED.value
                    _record_issue(document, issues, "encrypted_document", "Encrypted PDF cannot be read without a password.")
                    continue
            document.page_count = len(reader.pages)
        except Exception as exc:
            document.ingestion_status = DocumentStatus.FAILED.value
            _record_issue(document, issues, "open_error", f"Failed to open PDF: {type(exc).__name__}: {exc}")
            continue

        if document.page_count == 0:
            document.ingestion_status = DocumentStatus.FAILED.value
            _record_issue(document, issues, "empty_document", "PDF has no pages.")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
                cleaned_text = _clean_text(raw_text)
            except Exception as exc:
                _record_issue(
                    document,
                    issues,
                    "extraction_error",
                    f"Extraction failed: {type(exc).__name__}: {exc}",
                    page=page_number,
                )
                continue

            if not cleaned_text:
                document.empty_page_count += 1
                _record_issue(
                    document,
                    issues,
                    "empty_page",
                    "No extractable text found on this physical PDF page; OCR may be required.",
                    page=page_number,
                )
                continue

            document.extractable_page_count += 1
            pages.append(
                {
                    "text": cleaned_text,
                    "raw_text": raw_text,
                    "clean_text": cleaned_text,
                    "structured_lines": [line for line in cleaned_text.splitlines() if line.strip()],
                    "document_id": document.document_id,
                    "source": document.source,
                    "relative_path": document.relative_path,
                    # One-based physical PDF page index. Printed labels are not inferred.
                    "page": page_number,
                    "document_path": str(document.path),
                }
            )

        if document.extractable_page_count == 0:
            document.ingestion_status = DocumentStatus.FAILED.value
            _record_issue(
                document,
                issues,
                "requires_ocr",
                "No extractable text found in this PDF; it may be image-only and require OCR.",
            )
        elif document.ingestion_errors:
            document.ingestion_status = DocumentStatus.PARTIAL.value
        else:
            document.ingestion_status = DocumentStatus.SUCCESS.value

    return IngestionResult(descriptors, pages, issues)


def load_pdf_documents(
    document_dir: str | Path = DOCUMENT_DIR,
    recursive: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compatibility wrapper returning the historical pages/issues tuple."""
    result = ingest_documents(document_dir, recursive=recursive)
    return result.pages, result.issues


__all__ = ["get_pdf_files", "ingest_documents", "load_pdf_documents"]
