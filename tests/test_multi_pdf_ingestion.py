from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.chunker import chunk_pages
from src.corpus import (
    DocumentStatus,
    corpus_fingerprint,
    descriptor_for_path,
    discover_documents,
)
from src.evidence import SupportStatus, assess_evidence
from src.pdf_loader import ingest_documents
from src.vector_store import (
    atomic_publish_index,
    build_manifest,
    create_index,
    index_is_fresh,
    save_manifest,
)


class FakePage:
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error

    def extract_text(self) -> str:
        if self.error:
            raise self.error
        return self.text


class FakeReader:
    def __init__(self, pages: list[FakePage], encrypted: bool = False):
        self.pages = pages
        self.is_encrypted = encrypted

    def decrypt(self, password: str) -> int:
        return 0 if self.is_encrypted else 1


def metadata_for(document, text: str = "safe text") -> dict:
    return {
        "text": text,
        "document_id": document.document_id,
        "source": document.source,
        "relative_path": document.relative_path,
        "page": 1,
        "chunk_id": f"{document.document_id}-p000001-c0001",
        "word_start": 0,
        "word_end": len(text.split()),
    }


class DiscoveryAndIdentityTests(unittest.TestCase):
    def test_recursive_three_pdf_discovery_and_global_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = [root / "A.pdf", root / "academic" / "B.pdf", root / "services" / "C.PDF"]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"pdf-{index}".encode())

            documents = discover_documents(root)
            self.assertEqual([item.relative_path for item in documents], ["A.pdf", "academic/B.pdf", "services/C.PDF"])
            self.assertEqual(len({item.document_id for item in documents}), 3)

            def reader(path: str) -> FakeReader:
                return FakeReader([FakePage(f"Text extracted from {Path(path).name}")])

            with patch("src.pdf_loader.PdfReader", side_effect=reader):
                result = ingest_documents(root, documents=documents)
            chunks = chunk_pages(result.pages, chunk_size=20, overlap=2)
            self.assertEqual(len(result.pages), 3)
            self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 3)
            self.assertEqual({chunk["page"] for chunk in chunks}, {1})
            self.assertEqual({chunk["relative_path"] for chunk in chunks}, {"A.pdf", "academic/B.pdf", "services/C.PDF"})
            self.assertTrue(all(chunk["document_id"] in {item.document_id for item in documents} for chunk in chunks))

    def test_seventy_lightweight_descriptors_have_dynamic_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index in range(70):
                (root / f"document-{index:02d}.pdf").write_bytes(f"content-{index}".encode())
            documents = discover_documents(root)
            self.assertEqual(len(documents), 70)
            self.assertEqual(len({item.document_id for item in documents}), 70)


class CorpusChangeTests(unittest.TestCase):
    def test_add_remove_and_modify_change_fingerprint_and_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            a = root / "A.pdf"
            b = root / "B.pdf"
            c = root / "C.pdf"
            a.write_bytes(b"a-v1")
            b.write_bytes(b"b-v1")
            initial = discover_documents(root)
            initial_fingerprint = corpus_fingerprint(initial)
            save_manifest(build_manifest(initial, chunk_count=2, embedding_dimension=3), manifest_path)
            self.assertTrue(index_is_fresh(discover_documents(root), manifest_path=manifest_path))

            c.write_bytes(b"c-v1")
            added = discover_documents(root)
            self.assertNotEqual(initial_fingerprint, corpus_fingerprint(added))
            self.assertFalse(index_is_fresh(added, manifest_path=manifest_path))

            b.unlink()
            removed = discover_documents(root)
            self.assertNotEqual(corpus_fingerprint(added), corpus_fingerprint(removed))
            self.assertFalse(index_is_fresh(removed, manifest_path=manifest_path))

            before_modify = corpus_fingerprint(removed)
            a.write_bytes(b"a-v2")
            modified = discover_documents(root)
            self.assertNotEqual(before_modify, corpus_fingerprint(modified))
            self.assertFalse(index_is_fresh(modified, manifest_path=manifest_path))


class FailureIsolationTests(unittest.TestCase):
    def test_duplicate_is_reported_and_only_canonical_copy_is_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "A.pdf").write_bytes(b"identical")
            (root / "A_copy.pdf").write_bytes(b"identical")
            with patch("src.pdf_loader.PdfReader", return_value=FakeReader([FakePage("canonical text")])) as reader:
                result = ingest_documents(root)
            self.assertEqual(result.status_count(DocumentStatus.DUPLICATE), 1)
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(reader.call_count, 1)
            self.assertTrue(any(issue["issue_type"] == "duplicate_document" for issue in result.issues))

    def test_bad_pdf_does_not_prevent_safe_pdf_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bad.pdf").write_bytes(b"not-a-pdf")
            (root / "good.pdf").write_bytes(b"safe-pdf")

            def reader(path: str) -> FakeReader:
                if Path(path).name == "bad.pdf":
                    raise ValueError("malformed PDF")
                return FakeReader([FakePage("safe university content")])

            with patch("src.pdf_loader.PdfReader", side_effect=reader):
                result = ingest_documents(root)
            self.assertEqual(result.status_count(DocumentStatus.FAILED), 1)
            self.assertEqual(result.status_count(DocumentStatus.SUCCESS), 1)
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0]["source"], "good.pdf")

    def test_image_only_pdf_is_marked_as_requiring_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scan.pdf").write_bytes(b"image-pdf")
            with patch("src.pdf_loader.PdfReader", return_value=FakeReader([FakePage("")])):
                result = ingest_documents(root)
            self.assertEqual(result.status_count(DocumentStatus.FAILED), 1)
            self.assertFalse(result.pages)
            self.assertTrue(any(issue["issue_type"] == "requires_ocr" for issue in result.issues))


class AtomicPublicationTests(unittest.TestCase):
    def test_staging_failure_keeps_previous_artifacts_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "vector_db"
            target.mkdir()
            originals = {
                "index.faiss": b"old-index",
                "metadata.pkl": b"old-metadata",
                "index_manifest.json": b"old-manifest",
            }
            for name, content in originals.items():
                (target / name).write_bytes(content)

            pdf = root / "A.pdf"
            pdf.write_bytes(b"pdf")
            document = descriptor_for_path(pdf, root)
            document.ingestion_status = DocumentStatus.SUCCESS.value
            metadata = [metadata_for(document)]
            index = create_index(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
            manifest = build_manifest([document], chunk_count=1, embedding_dimension=3)

            with patch("src.vector_store.save_manifest", side_effect=RuntimeError("simulated staging failure")):
                with self.assertRaises(RuntimeError):
                    atomic_publish_index(index, metadata, manifest, [document], target_dir=target)
            for name, content in originals.items():
                self.assertEqual((target / name).read_bytes(), content)


class CrossDocumentEvidenceTests(unittest.TestCase):
    def test_conflicting_evidence_retains_two_document_sources(self) -> None:
        rows = [
            {
                **metadata_for(type("D", (), {"document_id": "doc-a", "source": "A.pdf", "relative_path": "a/A.pdf"})()),
                "text": "Course Code: ABC 234 Credit Value: 3.0",
                "score": 2.0,
            },
            {
                **metadata_for(type("D", (), {"document_id": "doc-b", "source": "B.pdf", "relative_path": "b/B.pdf"})()),
                "text": "Course Code: ABC 234 Credit Value: 4.0",
                "score": 1.9,
            },
        ]
        assessment = assess_evidence("How many credits is ABC 234?", rows)
        self.assertEqual(assessment.status, SupportStatus.CONFLICTING)
        self.assertEqual({item.source for item in assessment.evidence}, {"A.pdf", "B.pdf"})
        self.assertEqual({item.document_id for item in assessment.evidence}, {"doc-a", "doc-b"})


if __name__ == "__main__":
    unittest.main()
