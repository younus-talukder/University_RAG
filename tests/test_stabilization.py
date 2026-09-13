from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.answer_bank import AnswerBankMatch
from src import pipeline
from src.vector_store import (
    build_manifest,
    index_is_fresh,
    save_manifest,
    stale_metadata_sources,
    validate_index_metadata_consistency,
)
from src.corpus import descriptor_for_path


class FakeIndex:
    def __init__(self, ntotal: int):
        self.ntotal = ntotal


class IndexFreshnessTests(unittest.TestCase):
    def test_document_fingerprint_detects_collection_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc1 = root / "document1.pdf"
            doc2 = root / "document2.pdf"
            manifest_path = root / "index_manifest.json"
            doc1.write_bytes(b"one")
            doc2.write_bytes(b"two")

            manifest = build_manifest([doc1, doc2], chunk_count=2, embedding_dimension=1024)
            save_manifest(manifest, manifest_path)

            self.assertTrue(index_is_fresh([doc1, doc2], manifest_path=manifest_path))

            doc1.write_bytes(b"one modified")
            self.assertFalse(index_is_fresh([doc1, doc2], manifest_path=manifest_path))

            doc1.write_bytes(b"one")
            doc3 = root / "document3.pdf"
            doc3.write_bytes(b"three")
            self.assertFalse(index_is_fresh([doc1, doc2, doc3], manifest_path=manifest_path))

            doc2.unlink()
            self.assertFalse(index_is_fresh([doc1], manifest_path=manifest_path))


class MetadataConsistencyTests(unittest.TestCase):
    def test_index_count_and_metadata_sources_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc1 = root / "document1.pdf"
            doc1.write_bytes(b"pdf")
            descriptor = descriptor_for_path(doc1, root)
            metadata = [
                {
                    "text": "Course text",
                    "document_id": descriptor.document_id,
                    "source": "document1.pdf",
                    "relative_path": "document1.pdf",
                    "page": 1,
                    "chunk_id": "document1-p1-c1",
                    "word_start": 0,
                    "word_end": 2,
                }
            ]

            result = validate_index_metadata_consistency(FakeIndex(1), metadata, [doc1])
            self.assertTrue(result["ok"])
            self.assertEqual(result["vector_count"], result["metadata_count"])

            stale_metadata = [{**metadata[0], "source": "deleted.pdf", "relative_path": "deleted.pdf"}]
            self.assertEqual(stale_metadata_sources(stale_metadata, [doc1]), ["deleted.pdf"])
            stale_result = validate_index_metadata_consistency(FakeIndex(1), stale_metadata, [doc1])
            self.assertFalse(stale_result["ok"])


class AnswerPathTests(unittest.TestCase):
    def test_answer_bank_can_be_enabled_and_reports_mode(self) -> None:
        match = AnswerBankMatch(
            answer="Course code holo CSE 101.",
            score=1.0,
            matched_question="What is CSE 101?",
        )
        with patch.object(pipeline, "find_answer_bank_match", return_value=match):
            result = pipeline.answer_question(
                "What is CSE 101?",
                use_generation=False,
                use_answer_bank=True,
            )

        self.assertEqual(result["answer_mode"], "answer_bank")
        self.assertTrue(result["answer_bank_enabled"])

    def test_answer_bank_can_be_disabled_and_fast_path_reports_mode(self) -> None:
        metadata = [
            {
                "text": "Course No. / Course Code: CSE 101 2. Course Title: Computer Fundamentals and Programming 3. Course Type: Core Course 4. Credit Value: 3.0",
                "source": "document2.pdf",
                "page": 1,
                "chunk_id": "document2-p1-c1",
                "word_start": 0,
                "word_end": 20,
            }
        ]
        with (
            patch.object(pipeline, "find_answer_bank_match", side_effect=AssertionError("answer bank should be skipped")),
            patch.object(pipeline, "load_index", return_value=(FakeIndex(1), metadata)),
            patch.object(pipeline, "retrieve_lexical", return_value=[{**metadata[0], "score": 10.0}]),
        ):
            result = pipeline.answer_question(
                "What is CSE 101?",
                use_generation=False,
                use_answer_bank=False,
            )

        self.assertEqual(result["answer_mode"], "fast_extractive")
        self.assertFalse(result["answer_bank_enabled"])
        self.assertIn("sources", result)


class LanguagePipelineSmokeTests(unittest.TestCase):
    def test_basic_language_detection_and_response_metadata(self) -> None:
        metadata = [
            {
                "text": "Course No. / Course Code: CSE 101 2. Course Title: Computer Fundamentals and Programming 3. Course Type: Core Course 4. Credit Value: 3.0",
                "source": "document2.pdf",
                "page": 1,
                "chunk_id": "document2-p1-c1",
                "word_start": 0,
                "word_end": 20,
            }
        ]
        cases = [
            ("What is CSE 101?", "english"),
            ("CSE 101 er credit koto?", "banglish"),
            ("CSE 101 এর ক্রেডিট কত?", "bangla"),
        ]

        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(1), metadata)),
            patch.object(pipeline, "retrieve_lexical", return_value=[{**metadata[0], "score": 10.0}]),
        ):
            for question, expected_language in cases:
                with self.subTest(question=question):
                    result = pipeline.answer_question(
                        question,
                        use_generation=False,
                        use_answer_bank=False,
                    )
                    self.assertEqual(result["detected_language"], expected_language)
                    self.assertIn("answer_mode", result)
                    self.assertIn("response_language", result)


if __name__ == "__main__":
    unittest.main()
