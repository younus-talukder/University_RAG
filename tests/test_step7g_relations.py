from __future__ import annotations

import unittest

from src.chunker import reconstruct_split_table_rows
from src.document_metadata import extract_document_metadata
from src.evidence import Evidence, EvidenceAssessment, SupportStatus, analyze_query, assess_evidence
from src.evidence_spans import expand_assessment_evidence
from src.relations import extract_relation, realize_relation


def item(text: str, *, chunk_id: str = "synthetic-1", page: int = 2) -> dict:
    return {"text": text, "chunk_id": chunk_id, "page": page,
            "source": "synthetic.pdf", "score": 1.0}


def relation_for(question: str, text: str, language: str = "bangla"):
    rows = [item(text)]
    # Isolate relation extraction with an already verified synthetic passage.
    # Separate metadata tests exercise the actual evidence gate.
    assessment = EvidenceAssessment(analyze_query(question), SupportStatus.SUPPORTED,
        (Evidence(None, "synthetic.pdf", None, 2, "synthetic-1", text, text, 1.0,
                  None, None, "general", "general", "supported"),), "synthetic verified passage")
    return extract_relation(question, assessment, rows, language)


class DocumentMetadataTests(unittest.TestCase):
    def test_edition_and_date_metadata(self) -> None:
        text = "Published by Example Department\nEdition\n13th Edition, July 2017"
        values = extract_document_metadata(text)
        self.assertEqual(values["edition"], "13th Edition")
        self.assertEqual(values["publication_date"], "July 2017")
        assessment = assess_evidence("প্রসপেক্টাসে কোন সংস্করণ ও তারিখ দেওয়া আছে?", [item(text)])
        self.assertEqual(assessment.status, SupportStatus.SUPPORTED)
        relation = extract_relation("প্রসপেক্টাসে কোন সংস্করণ ও তারিখ দেওয়া আছে?", assessment, [item(text)], "bangla")
        self.assertEqual(relation.relation, "edition_publication_date")
        self.assertIn("13th Edition", realize_relation(relation))
        self.assertIn("July 2017", realize_relation(relation))

    def test_publisher_metadata_requires_label(self) -> None:
        text = "Published by Example Department, Example University\nEdition 2nd Edition, May 2020"
        relation = relation_for("Prospectus-ta ke publish koreche?", text, "banglish")
        self.assertEqual(relation.relation, "published_by")
        self.assertIn("Example Department", realize_relation(relation))
        misleading = assess_evidence("Prospectus-ta ke publish koreche?", [item("Faculty publish research papers.")])
        self.assertEqual(misleading.status, SupportStatus.INSUFFICIENT)
        narrative = assess_evidence("Prospectus-ta ke publish koreche?", [
            item("A faculty paper was published by Another House in 2017.")])
        self.assertEqual(narrative.status, SupportStatus.INSUFFICIENT)

    def test_disclaimer_relation(self) -> None:
        text = "Disclaimer The information in this prospectus may change at the department's discretion."
        relation = relation_for("প্রসপেক্টাসের ডিসক্লেইমারে কী বলা হয়েছে?", text)
        self.assertEqual(relation.relation, "publication_disclaimer")
        self.assertIn("পরিবর্তন", realize_relation(relation))

    def test_bangla_disclaimer_omits_english_article_only_in_template(self) -> None:
        text = "Disclaimer The prospectus information may change at the discretion of the Example Department."
        relation = relation_for("প্রসপেক্টাসের ডিসক্লেইমারে কী বলা হয়েছে?", text)
        self.assertIn("the Example Department", relation.values[0])
        self.assertIn("Example Department-এর", realize_relation(relation))
        self.assertNotIn("the Example Department-এর", realize_relation(relation))


class RelationExtractionTests(unittest.TestCase):
    def test_reconstructed_course_row_has_exact_ownership(self) -> None:
        row = "ABC 111(B) Sample History 2.00 Nil"
        credit = relation_for("ABC 111(B) course-er credit koto?", row, "banglish")
        self.assertEqual(credit.relation, "course_credits")
        self.assertIn("2.00", realize_relation(credit))
        prerequisite = relation_for("ABC 111(B) course-er prerequisite ki?", row, "banglish")
        self.assertEqual(prerequisite.relation, "course_prerequisite")
        self.assertIn("kono prerequisite deya nei", realize_relation(prerequisite))
        self.assertIsNone(relation_for("ABC 111(A) course-er credit koto?", row, "banglish"))

    def test_established_under(self) -> None:
        text = "Example University (EU) started its operation under the Private University Act 2001."
        relation = relation_for("EU কোন আইনের অধীনে কার্যক্রম শুরু করেছিল?", text)
        self.assertEqual(relation.relation, "started_under")
        self.assertIn("Private University Act 2001", realize_relation(relation))

    def test_initial_two_program_list(self) -> None:
        text = "EU started its operation by offering Bachelor Degree Programs in Physics and Business Administration only."
        relation = relation_for("EU শুরুতে কোন দুইটি ব্যাচেলর ডিগ্রি প্রোগ্রাম অফার করেছিল?", text)
        self.assertEqual(relation.relation, "initially_offered")
        self.assertEqual(relation.values, ("Physics", "Business Administration"))
        self.assertIn("Physics", realize_relation(relation))

    def test_category_definition_and_eligibility(self) -> None:
        text = ("Category 4: Students who have passed all the prescribed courses and have no backlog of courses. "
                "A student of Category 4 is eligible for registration in all courses prescribed for the next semester.")
        relation = relation_for("রেজিস্ট্রেশনে Category 4 কী?", text)
        self.assertEqual(relation.relation, "category_definition")
        answer = realize_relation(relation)
        self.assertIn("Category 4", answer)
        self.assertIn("কোনো কোর্স বাকি না থাকা", answer)

    def test_full_percentage_distribution(self) -> None:
        text = ("For theoretical courses the distribution of marks is as follows: "
                "Assessment 25% Mid Semester 25% Final Exam 50% Total 100%")
        relation = relation_for("থিওরেটিক্যাল কোর্সে মার্কস বণ্টন কী?", text)
        self.assertEqual(relation.relation, "mark_distribution")
        answer = realize_relation(relation)
        self.assertIn("25%", answer)
        self.assertIn("50%", answer)
        self.assertEqual(answer.count("25%"), 2)


class EvidenceSpanTests(unittest.TestCase):
    def test_expand_truncated_excerpt_from_same_chunk(self) -> None:
        full = "Course Code: ABC 123 Prerequisite: XYZ 101. " + ("Context sentence. " * 45) + "Last complete sentence."
        assessment = assess_evidence("ABC 123 prerequisite?", [item(full)])
        self.assertEqual(assessment.status, SupportStatus.SUPPORTED)
        self.assertLess(len(assessment.evidence[0].excerpt), len(full))
        expanded = expand_assessment_evidence(assessment, [item(full)])
        self.assertEqual(expanded.evidence[0].excerpt, " ".join(full.split()))
        self.assertEqual(expanded.evidence[0].chunk_id, "synthetic-1")


class TableContinuationTests(unittest.TestCase):
    def test_split_code_and_title_reconstructs_one_row(self) -> None:
        lines = ["Course Code Course Title Credits Pre-Requisite", "ABC", "First part", "2.00", "Nil",
                 "First Year First Semester", "Course", "Code", "Course Title Credits", "Pre-", "Requisite",
                 "111(B) Second part", "XYZ 102 Another Course 3.00 Nil"]
        result = reconstruct_split_table_rows(lines)
        self.assertIn("ABC 111(B) First part Second part 2.00 Nil", " ".join(result))

    def test_ambiguous_suffixes_do_not_merge(self) -> None:
        lines = ["Course Code Course Title Credits Pre-Requisite", "ABC", "First part", "2.00", "Nil",
                 "111(B) Second part", "112(C) Third part", "XYZ 102 Another Course 3.00 Nil"]
        self.assertEqual(reconstruct_split_table_rows(lines), lines)


if __name__ == "__main__":
    unittest.main()
