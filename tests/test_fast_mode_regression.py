from __future__ import annotations

import csv
import unittest
from pathlib import Path

from src.config import VECTOR_DB_DIR
from src.fast_answer import detect_course_code, detect_runtime_intent, extract_week_number
from src.pipeline import answer_question
from src.vector_store import load_index


REGRESSION_PATH = Path(__file__).resolve().parent / "data" / "fast_mode_regression.csv"


def _load_cases() -> list[dict[str, str]]:
    with REGRESSION_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FastModeIntentTests(unittest.TestCase):
    def test_multilingual_intent_detection(self) -> None:
        cases = [
            ("What is CLO 2 of CSE 101?", "clo"),
            ("CSE 101-এর CLO 3 কী?", "clo"),
            ("CSE 101-er CLO 4 ki?", "clo"),
            ("What are the objectives of CSE 101?", "course_objective"),
            ("CSE 101 course-er objective ki?", "course_objective"),
            ("CSE 101 কোর্সের উদ্দেশ্য কী?", "course_objective"),
            ("What topics are covered in CSE 101?", "topics"),
            ("CSE 101 e ki ki topic porano hoy?", "topics"),
            ("CSE 101-এ কী কী বিষয় পড়ানো হয়?", "topics"),
            ("What is taught in week five?", "weekly_content"),
            ("Week 5 e ki porano hoy?", "weekly_content"),
            ("৫ম সপ্তাহে কী পড়ানো হয়?", "weekly_content"),
            ("Final exam e koto percent marks?", "final_exam_marks"),
            ("ফাইনাল পরীক্ষায় কত শতাংশ নম্বর?", "final_exam_marks"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(detect_runtime_intent(question), expected)

    def test_course_identification_variants(self) -> None:
        cases = [
            ("CSE101 credit?", "CSE 101"),
            ("cse 101 title?", "CSE 101"),
            ("MTH101 CLO 1?", "MTH 101"),
            ("ENG CSE 101 CLO 2?", "ENG (CSE) 101"),
            ("ENG(CSE)101 credit?", "ENG (CSE) 101"),
            ("HSS 111(B) credit?", "HSS 111(B)"),
            ("EEE221 title?", "EEE 221"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(detect_course_code(question), expected)

    def test_week_number_variants(self) -> None:
        cases = [
            ("week 5", 5),
            ("5th week", 5),
            ("week five", 5),
            ("৫ম সপ্তাহ", 5),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(extract_week_number(question), expected)


class FastModePipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index, cls.metadata = load_index(
            index_path=VECTOR_DB_DIR / "index.faiss",
            metadata_path=VECTOR_DB_DIR / "metadata.pkl",
        )

    def test_index_state_is_preserved(self) -> None:
        self.assertEqual(getattr(self.index, "ntotal", None), len(self.metadata))
        self.assertGreater(len(self.metadata), 0)

    def test_regression_cases_return_requested_field(self) -> None:
        cases = _load_cases()
        self.assertGreaterEqual(len(cases), 5)

        for row in cases:
            question = row["question"]
            with self.subTest(question=question):
                result = answer_question(
                    question,
                    use_generation=False,
                    use_answer_bank=True,
                    top_k=5,
                )
                answer = result["answer"]
                self.assertEqual(result["detected_language"], row["expected_language"])
                self.assertEqual(result["intent"], row["expected_intent"])
                self.assertEqual(result["sources"][0]["source"], row["expected_source"])
                self.assertEqual(result["answer_mode"], "answer_bank")

                folded_answer = answer.casefold()
                for expected in row["must_contain"].split("|"):
                    self.assertIn(expected.casefold(), folded_answer)
                for forbidden in filter(None, row["must_not_contain"].split("|")):
                    self.assertNotIn(forbidden.casefold(), folded_answer)


if __name__ == "__main__":
    unittest.main()
