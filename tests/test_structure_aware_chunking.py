from __future__ import annotations

import unittest

from src.chunker import chunk_pages, segment_page, validate_chunk_quality
from src.evidence import SupportStatus, assess_evidence
from src.pdf_loader import _clean_text


def page(text: str, document: str = "doc-a", source: str = "A.pdf", number: int = 1) -> dict:
    return {
        "text": text,
        "clean_text": text,
        "structured_lines": [line for line in text.splitlines() if line.strip()],
        "document_id": document,
        "source": source,
        "relative_path": source,
        "page": number,
    }


class CourseStructureTests(unittest.TestCase):
    def test_two_labeled_courses_on_one_page_are_separate(self) -> None:
        value = """Course Code: ABC 101
Course Title: Foundations
Credits: 3.00
Prerequisite: None
Introductory foundations and examples.

Course Code: XYZ 202
Course Title: Advanced Systems
Credits: 4.00
Prerequisite: ABC 101
Advanced systems material."""
        chunks = chunk_pages([page(value)])
        course_chunks = [chunk for chunk in chunks if chunk["block_type"] == "course_record"]
        self.assertEqual([chunk["course_code"] for chunk in course_chunks], ["ABC 101", "XYZ 202"])
        self.assertNotIn("XYZ 202", course_chunks[0]["text"])
        self.assertNotIn("Course Title: Foundations", course_chunks[1]["text"])

    def test_flattened_table_rows_are_separate(self) -> None:
        value = """First Year First Semester
Course Code Course Title Credits Pre-Requisite
ABC 101 Foundations 3.00 Nil
XYZ 202 Advanced Systems 4.00 ABC 101
HSS 303 Society and Culture 2.00 Nil
Total 9.00"""
        chunks = chunk_pages([page(value)])
        rows = [chunk for chunk in chunks if chunk["block_type"] == "table_row"]
        self.assertEqual([row["course_code"] for row in rows], ["ABC 101", "XYZ 202", "HSS 303"])
        self.assertTrue(all(sum(code in row["text"] for code in ("ABC 101", "XYZ 202", "HSS 303")) <= 2 for row in rows))
        evidence = assess_evidence("How many credits is XYZ 202?", [{**rows[1], "score": 1.0}])
        self.assertEqual(evidence.status, SupportStatus.SUPPORTED)

    def test_table_noise_does_not_merge_following_entity_rows(self) -> None:
        value = "Course Code Course Title Credits Pre-Requisite ABC101 Intro 3.00 Nil --- XYZ202 Lab 1.50 ABC101 ### MTH303 Math 3.00 Nil"
        rows = [block for block in segment_page(page(value)) if block.block_type == "table_row"]
        self.assertEqual([block.course_code for block in rows], ["ABC 101", "XYZ 202", "MTH 303"])


class PolicyAndFallbackTests(unittest.TestCase):
    def test_conservative_cleaning_preserves_useful_line_boundaries(self) -> None:
        cleaned = _clean_text("Heading\nFirst para line\n\nSecond para line\n")
        self.assertIn("Heading\nFirst para line", cleaned)
        self.assertIn("\n\nSecond para line", cleaned)

    def test_multiple_policies_split_by_generic_headings(self) -> None:
        value = """Attendance Policy
Students must attend at least seventy percent of scheduled classes.
Additional attendance details apply to laboratories.
Academic Progress
Students must maintain satisfactory progress each semester.
Unsatisfactory progress triggers an academic review."""
        chunks = chunk_pages([page(value)])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Attendance Policy", chunks[0]["text"])
        self.assertTrue(any("Academic Progress" in chunk["text"] for chunk in chunks[1:]))
        self.assertFalse(any("Academic Progress" in chunk["text"] and "Attendance Policy" in chunk["text"] for chunk in chunks))

    def test_unknown_template_uses_safe_fallback(self) -> None:
        value = "lowercase prose without any recognizable template continues with useful university service information for students and staff"
        chunks = chunk_pages([page(value)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["block_type"], "fallback")
        self.assertIn("useful university service information", chunks[0]["text"])

    def test_low_information_divider_can_be_skipped_safely(self) -> None:
        self.assertEqual(chunk_pages([page("13th Edition")]), [])


class ContinuationAndMultiDocumentTests(unittest.TestCase):
    def test_cross_page_continuation_is_linked_but_not_merged_with_new_section(self) -> None:
        pages = [
            page("General Requirements\nStudents submit documents and complete verification before", number=1),
            page("continuing the remaining verification steps on the next working day", number=2),
            page("Disciplinary Rules\nSeparate conduct rules apply to all enrolled students.", number=3),
        ]
        chunks = chunk_pages(pages)
        page_two = next(chunk for chunk in chunks if chunk["page"] == 2)
        page_three = next(chunk for chunk in chunks if chunk["page"] == 3)
        self.assertIsNotNone(page_two["continuation_of"])
        self.assertEqual(page_two["parent_section_id"], chunks[0]["parent_section_id"])
        self.assertIsNone(page_three["continuation_of"])
        self.assertNotIn("Disciplinary Rules", page_two["text"])

    def test_three_unknown_document_structures_all_produce_valid_chunks(self) -> None:
        pages = [
            page("Course Code: ABC 101\nCourse Title: Intro\nCredits: 3.00\nPrerequisite: None", "doc-a", "catalog/A.pdf"),
            page("Attendance Policy\nStudents are required to attend scheduled classes regularly.", "doc-b", "rules/B.pdf"),
            page("lowercase unstructured service details remain available to every enrolled student", "doc-c", "services/C.pdf"),
        ]
        chunks = chunk_pages(pages)
        self.assertEqual({chunk["document_id"] for chunk in chunks}, {"doc-a", "doc-b", "doc-c"})
        self.assertTrue(validate_chunk_quality(chunks)["ok"])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))


class QualityDiagnosticTests(unittest.TestCase):
    def test_quality_validator_flags_mixed_entities_without_deleting_chunk(self) -> None:
        value = "ABC 101, XYZ 202, HSS 303, MTH 404 and EEE 505 are listed together in this unstructured notice."
        chunks = chunk_pages([page(value)])
        quality = validate_chunk_quality(chunks, mixed_entity_threshold=3)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(quality["mixed_entity_chunks"], [chunks[0]["chunk_id"]])


if __name__ == "__main__":
    unittest.main()
