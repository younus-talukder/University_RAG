from __future__ import annotations

import unittest

from src.answer_safety import requested_relation
from src.evidence import SupportStatus, assess_evidence, detect_requested_field
from src.fast_answer import detect_runtime_intent
from src.query_normalization import is_mark_distribution_query
from src.relations import extract_relation


SOURCE = (
    "For theoretical courses the distribution of marks is as follows: "
    "Assessment 30% Mid Semester 20% Final Exam 50%"
)
PASSAGE = {"text": SOURCE, "source": "synthetic-curriculum.pdf", "page": 18,
           "chunk_id": "synthetic-c0001", "parent_id": "synthetic"}


class MarkDistributionIntentTests(unittest.TestCase):
    def test_three_language_semantic_parity(self):
        questions = (
            "What is the marks distribution for theoretical courses?",
            "থিওরেটিক্যাল কোর্সে মার্কস বণ্টন কী?",
            "Theoretical course-e marks distribution ki?",
        )
        for language, question in zip(("english", "bangla", "banglish"), questions):
            with self.subTest(language=language):
                self.assertEqual(detect_requested_field(question), "assessment")
                self.assertEqual(detect_runtime_intent(question), "mark_distribution")
                self.assertEqual(requested_relation(question), "mark_distribution")
                assessment = assess_evidence(question, [PASSAGE])
                self.assertEqual(assessment.status, SupportStatus.SUPPORTED)
                relation = extract_relation(question, assessment, [PASSAGE], language)
                self.assertIsNotNone(relation)
                self.assertEqual(relation.relation, "mark_distribution")
                self.assertEqual(relation.values, ("30%", "20%", "50%"))

    def test_bangla_variants_and_context_guard(self):
        variants = (
            "থিওরেটিক্যাল কোর্সে নম্বর বণ্টন কী?",
            "থিওরিটিক্যাল কোর্সে মার্ক বণ্টন কী?",
            "থিওরেটিক্যাল কোর্সের নম্বর কীভাবে বণ্টন করা হয়?",
            "থিওরেটিক্যাল কোর্সে নম্বরের বণ্টন কী?",
            "থিওরেটিক্যাল কোর্সে মার্কের বণ্টন কী?",
        )
        for question in variants:
            with self.subTest(question=question):
                self.assertTrue(is_mark_distribution_query(question))
                self.assertEqual(detect_requested_field(question), "assessment")
                self.assertEqual(detect_runtime_intent(question), "mark_distribution")
                self.assertEqual(requested_relation(question), "mark_distribution")
        for question in ("নম্বর কত?", "মার্ক কত?", "What are the marks?"):
            with self.subTest(non_distribution=question):
                self.assertFalse(is_mark_distribution_query(question))
                self.assertNotEqual(detect_runtime_intent(question), "mark_distribution")


if __name__ == "__main__":
    unittest.main()
