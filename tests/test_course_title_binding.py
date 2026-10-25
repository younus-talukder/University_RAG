from __future__ import annotations

import unittest
from unittest.mock import patch

from src import pipeline
from src.course_title_evidence import assess_course_title, bound_course_title_value
from src.evidence import SupportStatus, analyze_query
from src.generation_context import VerifiedEvidenceItem, VerifiedEvidencePackage
from src.semantic_contract import build_semantic_contract, expressed_relations, validate_semantic_contract


def item(text: str, *, document: str = "doc-a", source: str = "a.pdf",
         page: int = 1, chunk: str = "chunk-1") -> dict:
    return {"text": text, "document_id": document, "source": source,
            "relative_path": source, "page": page, "chunk_id": chunk,
            "score": 0.9, "field_types": ["course_title"] if "Course Title:" in text else ["course_code"]}


class CourseTitleBindingTests(unittest.TestCase):
    def test_split_labeled_fields_bind_on_same_document_page(self) -> None:
        code = item("1. Course No. / Course Code: ABC 234", chunk="code")
        title = item("2. Course Title: Applied Systems Lab 3. Course Type: Lab", chunk="title")
        for question in ("What course title is listed for ABC 234?",
                         "ABC 234 কোর্সের নাম কী দেওয়া আছে?",
                         "ABC 234-er naam ki deya ache?"):
            with self.subTest(question=question):
                assessment = assess_course_title(question, [code], [code, title])
                self.assertEqual(assessment.status, SupportStatus.SUPPORTED)
                self.assertEqual(bound_course_title_value(assessment), "Applied Systems Lab")
                self.assertEqual(assessment.evidence[0].continuation_chunk_ids, ("title",))
                self.assertIn("Course Code: ABC 234", assessment.evidence[0].excerpt)
                self.assertIn("Course Title: Applied Systems Lab", assessment.evidence[0].excerpt)

    def test_banglish_naam_is_title_only_for_a_course(self) -> None:
        self.assertEqual(analyze_query("ABC 234-er naam ki deya ache?").requested_field, "course_title")
        self.assertNotEqual(analyze_query("ABC 234-er teacher-er naam ki?").requested_field, "course_title")

    def test_different_document_or_page_cannot_supply_title(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        for wrong in (item("Course Title: Wrong", page=2, chunk="next-page"),
                      item("Course Title: Wrong", document="doc-b", chunk="other-doc")):
            with self.subTest(wrong=wrong["chunk_id"]):
                assessment = assess_course_title("What is the title of ABC 234?", [code], [code, wrong])
                self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)
                self.assertIsNone(bound_course_title_value(assessment))

    def test_neighboring_course_on_same_page_is_ambiguous(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        other = item("Course Code: XYZ 987 Course Title: Neighboring Course", chunk="other")
        assessment = assess_course_title("What is the title of ABC 234?", [code], [code, other])
        self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)

    def test_unlabeled_heading_never_becomes_title(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        heading = item("Course Outline - Introduction", chunk="heading")
        assessment = assess_course_title("What is the title of ABC 234?", [code], [code, heading])
        self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)

    def test_conflicting_labeled_titles_abstain(self) -> None:
        a_code = item("Course Code: ABC 234", chunk="a-code")
        a_title = item("Course Title: Applied Systems Lab", chunk="a-title")
        b_code = item("Course Code: ABC 234", document="doc-b", source="b.pdf", chunk="b-code")
        b_title = item("Course Title: Systems Design Lab", document="doc-b", source="b.pdf", chunk="b-title")
        assessment = assess_course_title("What is the title of ABC 234?",
                                         [a_code, b_code], [a_code, a_title, b_code, b_title])
        self.assertEqual(assessment.status, SupportStatus.CONFLICTING)
        self.assertEqual(set(assessment.conflicting_values), {"Applied Systems Lab", "Systems Design Lab"})

    def test_same_page_multiple_title_labels_abstain(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        titles = item("Course Title: Applied Systems Lab Course Title: Other Lab", chunk="titles")
        assessment = assess_course_title("What is the title of ABC 234?", [code], [code, titles])
        self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)

    def test_pipeline_uses_same_deterministic_fact_for_three_languages(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        title = item("Course Title: Applied Systems Lab", chunk="title")
        with (patch.object(pipeline, "load_index", return_value=(object(), [code, title])),
              patch.object(pipeline, "_get_embedding_model", return_value=object()),
              patch.object(pipeline, "Retriever") as retriever,
              patch.object(pipeline, "generate_canonical_answer", side_effect=AssertionError("Qwen must not answer a title"))):
            retriever.return_value.retrieve.return_value = [code]
            for question in ("What course title is listed for ABC 234?",
                             "ABC 234 কোর্সের নাম কী দেওয়া আছে?",
                             "ABC 234-er naam ki deya ache?"):
                with self.subTest(question=question):
                    answer = pipeline.answer_question(question, use_generation=True, use_answer_bank=False)
                    self.assertEqual(answer["answerability_status"], "SUPPORTED")
                    self.assertIn("Applied Systems Lab", answer["answer"])
                    self.assertEqual(answer["answer_strategy"], "structured_exact")
                    self.assertFalse(answer["generation_used"])

    def test_pipeline_abstains_when_title_is_only_on_another_page(self) -> None:
        code = item("Course Code: ABC 234", chunk="code")
        wrong_page = item("Course Title: Unrelated Heading", page=2, chunk="title")
        with (patch.object(pipeline, "load_index", return_value=(object(), [code, wrong_page])),
              patch.object(pipeline, "_get_embedding_model", return_value=object()),
              patch.object(pipeline, "Retriever") as retriever,
              patch.object(pipeline, "generate_canonical_answer", side_effect=AssertionError("must abstain"))):
            retriever.return_value.retrieve.return_value = [code]
            answer = pipeline.answer_question("What course title is listed for ABC 234?",
                                              use_generation=True, use_answer_bank=False)
        self.assertNotEqual(answer["answerability_status"], "SUPPORTED")
        self.assertFalse(answer["generation_used"])
        self.assertIsNone(answer["source"])

    def test_title_contract_rejects_wrong_value_in_every_language(self) -> None:
        excerpt = "Course Code: ABC 234 Course Title: Systems Analysis and Design"
        self.assertNotIn("general_claim", expressed_relations(excerpt))
        for language in ("english", "bangla", "banglish"):
            package = VerifiedEvidencePackage(
                question="What is the title of ABC 234?", original_language=language,
                requested_entity="ABC 234", requested_field="course_title",
                support_status="supported",
                evidence=(VerifiedEvidenceItem("a.pdf", 1, None, "ABC 234", excerpt, "code", 1.0),),
            )
            contract = build_semantic_contract(package)
            self.assertTrue(validate_semantic_contract("Title holo Systems Analysis and Design.", contract)["passed"])
            wrong = validate_semantic_contract("Title holo Introduction.", contract)
            self.assertFalse(wrong["passed"])
            self.assertEqual(wrong["reason"], "WRONG_FIELD_VALUE")


if __name__ == "__main__":
    unittest.main()
