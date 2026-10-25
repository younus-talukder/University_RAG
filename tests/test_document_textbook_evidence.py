from __future__ import annotations

import unittest

from src.document_textbook_evidence import assess_document_textbook
from src.evidence import SupportStatus


def piece(text: str, *, page: int, chunk: str, document: str = "doc-a") -> dict:
    return {"text": text, "page": page, "chunk_id": chunk,
            "document_id": document, "source": document + ".pdf",
            "relative_path": document + ".pdf", "word_start": int(chunk[-1]) * 100}


class DocumentTextbookEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cover = piece("Course Code: ABC 234", page=1, chunk="a1")
        self.heading = piece("Part D. Text Book 1.", page=8, chunk="a2")
        self.value = piece("1. Systems and Data (3rd Edition), A. Scholar.", page=8, chunk="a3")
        self.reference = piece("2. Reference Books & Materials: Other Book.", page=8, chunk="a4")

    def test_multilingual_questions_bind_one_labeled_entry(self) -> None:
        for question in ("Which textbook is listed for ABC 234?",
                         "ABC 234-এর পাঠ্যবই হিসেবে কোন বই আছে?",
                         "ABC 234-er textbook kon-ta?"):
            with self.subTest(question=question):
                assessment, candidate, value = assess_document_textbook(
                    question, [self.cover], [self.cover, self.heading, self.value, self.reference])
                self.assertEqual(assessment.status, SupportStatus.SUPPORTED)
                self.assertEqual(value, "Systems and Data (3rd Edition), A. Scholar")
                self.assertEqual(candidate["page"], 8)
                self.assertNotIn("Other Book", assessment.evidence[0].excerpt)

    def test_unretrieved_owner_cannot_license_document_scan(self) -> None:
        assessment, _, _ = assess_document_textbook(
            "Which textbook is listed for ABC 234?", [self.heading],
            [self.cover, self.heading, self.value, self.reference])
        self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)

    def test_foreign_or_multiple_owners_cannot_license_entry(self) -> None:
        for extra in (piece("Course Code: XYZ 333", page=2, chunk="a5"),
                      piece("Course Code: XYZ 333", page=1, chunk="b1", document="doc-b")):
            with self.subTest(extra=extra["document_id"]):
                if extra["document_id"] == "doc-b":
                    # A different PDF is harmless; it cannot lend its entry.
                    assessment, _, _ = assess_document_textbook(
                        "Which textbook is listed for XYZ 333?", [self.cover],
                        [self.cover, self.heading, self.value, self.reference, extra])
                else:
                    assessment, _, _ = assess_document_textbook(
                        "Which textbook is listed for ABC 234?", [self.cover],
                        [self.cover, self.heading, self.value, self.reference, extra])
                self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)

    def test_multiple_textbooks_or_missing_reference_boundary_abstains(self) -> None:
        second = piece("2. Another Textbook, B. Writer.", page=8, chunk="a5")
        second["word_start"] = 350
        for chunks in ([self.cover, self.heading, self.value],
                       [self.cover, self.heading, self.value, second, self.reference]):
            with self.subTest(count=len(chunks)):
                assessment, _, _ = assess_document_textbook(
                    "Which textbook is listed for ABC 234?", [self.cover], chunks)
                self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)


if __name__ == "__main__":
    unittest.main()
