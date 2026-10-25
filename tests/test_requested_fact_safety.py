from __future__ import annotations

import unittest
from unittest.mock import patch

from src import pipeline
from src.answer_safety import answer_safety_failures
from src.crosslingual import Rewrite, crosslingual_assessment
from src.evidence import SupportStatus, assess_evidence, detect_requested_field


def chunk(text: str, *, page: int = 1, name: str = "chunk") -> dict:
    return {
        "text": text, "document_id": "synthetic-document", "source": "synthetic.pdf",
        "relative_path": "synthetic.pdf", "page": page, "chunk_id": name,
        "score": 0.9, "field_types": [],
    }


class RequestedFactSafetyTests(unittest.TestCase):
    def test_paired_subject_must_be_on_the_cited_passage(self) -> None:
        question = "Feasibility study-r sathe kon teaching strategy ache?"
        header = "Class Schedule/Lesson Plan: Suggested Activities Teaching Strategy(s)"
        pairing = "Problem-Based Learning | Project Scheduling, Feasibility Study"
        wrong = answer_safety_failures(
            question, "Class schedules and lesson plans are teaching strategies.",
            [pairing, header], cited_excerpt=header,
        )
        self.assertIn("REQUESTED_RELATION_NOT_ON_CITED_PASSAGE", wrong)
        self.assertEqual(answer_safety_failures(
            question, "Problem-Based Learning.", [pairing, header], cited_excerpt=pairing,
        ), ())

    def test_heading_and_code_cannot_support_unrelated_focus_claim(self) -> None:
        question = "ABC 234 real-life kon kaj-e focus kore?"
        heading = "Part A – Introduction. Course Code: ABC 234"
        fact = "Course Code: ABC 234. The course focuses on development of a database in a real-life scenario."
        self.assertIn("REQUESTED_FACT_NOT_IN_CITED_EVIDENCE", answer_safety_failures(
            question, "ABC 234 focuses on computer organization and architecture introduction.",
            [heading, fact], cited_excerpt=heading,
        ))
        self.assertEqual(answer_safety_failures(
            question, "It focuses on development of a database in a real-life scenario.",
            [heading, fact], cited_excerpt=fact,
        ), ())

    def test_named_operations_are_not_a_course_title_request(self) -> None:
        question = "ABC 234-এর উদ্দেশ্যে কোন ডেটাবেস অপারেশনগুলোর নাম আছে?"
        evidence = "Course Code: ABC 234. Course Title: Systems Lab. Objectives: constraints, DDL and DML operations."
        self.assertEqual(detect_requested_field(question), "objective")
        self.assertIn("WRONG_FIELD_ANSWER", answer_safety_failures(
            question, "কোর্সটির title হলো Systems Lab।", [evidence], cited_excerpt=evidence,
        ))
        self.assertEqual(answer_safety_failures(
            question, "DDL এবং DML operations।", [evidence], cited_excerpt=evidence,
        ), ())

    def test_course_code_alone_cannot_support_other_fields(self) -> None:
        code = chunk("Part A – Introduction 1. Course No. / Course Code: ABC 234")
        questions = (
            "Which textbook is listed for ABC 234?",
            "ABC 234-এর পাঠ্যবই হিসেবে কোন বই আছে?",
            "ABC 234-er textbook kon-ta?",
            "What computer-related area is ABC 234 closely relevant to?",
            "ABC 234 computer-er kon area-r sathe closely relevant?",
            "ABC 234 কম্পিউটার-সংশ্লিষ্ট কোন ক্ষেত্রের সঙ্গে প্রাসঙ্গিক?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(assess_evidence(question, [code]).status, SupportStatus.INSUFFICIENT)

    def test_crosslingual_rewrite_cannot_drop_requested_field(self) -> None:
        code = chunk("Part A – Introduction 1. Course No. / Course Code: ABC 234")
        assessment, reason = crosslingual_assessment(
            "ABC 234-er textbook kon-ta?", Rewrite("ABC 234", "DETERMINISTIC", "VALIDATED"), [code]
        )
        self.assertEqual(assessment.status, SupportStatus.INSUFFICIENT)
        self.assertEqual(assessment.request.requested_field, "textbook")
        self.assertEqual(reason, "REQUESTED_FIELD_CHANGED")

    def test_neighboring_fact_does_not_license_wrong_general_answer(self) -> None:
        question = "What computer-related area is ABC 234 closely relevant to?"
        evidence = "Course Code: ABC 234. It is closely relevant to computer hardware."
        self.assertIn("REQUESTED_FACT_NOT_IN_CITED_EVIDENCE", answer_safety_failures(
            question, "ABC 234 is closely relevant to computer engineering.", [evidence]
        ))
        self.assertEqual(answer_safety_failures(question, "Computer hardware.", [evidence]), ())

    def test_absence_claim_is_not_a_document_fact(self) -> None:
        question = "Which textbook is listed for ABC 234?"
        evidence = "Course Code: ABC 234. Text Book: Distributed Systems by A. Scholar."
        for answer in (
            "The textbook is not specified in the provided evidence.",
            "The textbook is not listed in the provided evidence.",
            "ABC 234-er textbook ta provide koreche na.",
            "পাঠ্যবই উল্লেখিত হয়নি।",
        ):
            with self.subTest(answer=answer):
                self.assertIn("UNVERIFIED_ABSENCE_CLAIM", answer_safety_failures(question, answer, [evidence]))

    def test_which_clos_requires_outcome_identifiers_even_on_right_page(self) -> None:
        question = "Which CLOs are assessed by the week 7 midterm lab exam?"
        evidence = "SQL operations. Week 7 Conduct hands-on lab exam. Assessment CLO1, CLO2."
        self.assertEqual(detect_requested_field(question), "learning_outcome")
        self.assertIn("REQUESTED_FIELD_MISSING", answer_safety_failures(
            question, "Week 7: SQL operations.", [evidence]
        ))
        self.assertEqual(answer_safety_failures(question, "CLO1 and CLO2.", [evidence]), ())

    def test_pipeline_rejects_wrong_field_after_supported_retrieval(self) -> None:
        evidence = chunk("SQL operations. Week 7 Conduct hands-on lab exam covering Labs 1-6. "
                         "Assessment CLO1, CLO2 Advanced SQL", page=5)
        with (patch.object(pipeline, "load_index", return_value=(object(), [evidence])),
              patch.object(pipeline, "_get_embedding_model", return_value=object()),
              patch.object(pipeline, "Retriever") as retriever,
              patch.object(pipeline, "build_extractive_answer", return_value="Week 7: SQL operations.")):
            retriever.return_value.retrieve.return_value = [evidence]
            result = pipeline.answer_question(
                "Which CLOs are assessed by the week 7 midterm lab exam?",
                use_generation=True, use_answer_bank=False,
            )
        self.assertEqual(result["pre_generation_support_status"], "supported")
        self.assertEqual(result["answerability_status"], "GENERATION_REJECTED")
        self.assertNotEqual(result["answer"], "Week 7: SQL operations.")
        self.assertEqual(result["generation_rejection_reason"], "REQUESTED_FIELD_MISSING")

    def test_pipeline_accepts_requested_clo_fact(self) -> None:
        evidence = chunk("SQL operations. Week 7 Conduct hands-on lab exam covering Labs 1-6. "
                         "Assessment CLO1, CLO2 Advanced SQL", page=5)
        with (patch.object(pipeline, "load_index", return_value=(object(), [evidence])),
              patch.object(pipeline, "_get_embedding_model", return_value=object()),
              patch.object(pipeline, "Retriever") as retriever,
              patch.object(pipeline, "build_extractive_answer", return_value="Week 7 assesses CLO1 and CLO2.")):
            retriever.return_value.retrieve.return_value = [evidence]
            result = pipeline.answer_question(
                "Which CLOs are assessed by the week 7 midterm lab exam?",
                use_generation=True, use_answer_bank=False,
            )
        self.assertEqual(result["answerability_status"], "SUPPORTED")
        self.assertIn("CLO1 and CLO2", result["answer"])


if __name__ == "__main__":
    unittest.main()
