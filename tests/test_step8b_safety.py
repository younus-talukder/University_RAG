from __future__ import annotations

import unittest

from src.answer_safety import answer_safety_failures, explicit_email_values, requested_relation
from src.evidence import SupportStatus, analyze_query, assess_evidence


def evidence(text: str, *, page: int = 1) -> dict:
    return {"text": text, "source": "synthetic.pdf", "document_id": "synthetic.pdf",
            "page": page, "chunk_id": f"synthetic-{page}", "parent_id": f"parent-{page}"}


class Step8BSafetyTests(unittest.TestCase):
    def test_semester_relations_are_distinct(self):
        self.assertEqual(requested_relation("একটি নিয়মিত সেমিস্টারের মোট মেয়াদ কত?"),
                         "regular_semester_duration")
        self.assertEqual(requested_relation("How many regular semesters are there?"),
                         "regular_semester_count")
        self.assertEqual(requested_relation("How many class weeks are in a semester?"),
                         "class_duration_weeks")
        self.assertEqual(requested_relation("What are the semester names?"), "semester_names")
        self.assertIn("RELATION_MISMATCH", answer_safety_failures(
            "একটি নিয়মিত সেমিস্টারের মোট মেয়াদ কত?",
            "এক শিক্ষাবর্ষে 2টি নিয়মিত সেমিস্টার আছে: Fall ও Spring।", (),
            extracted_relation="regular_semester_count"))

    def test_email_list_is_complete_and_scoped(self):
        source = ("Admission Office Email: a@example.edu, b@example.edu "
                  "Department of Architecture Email: c@example.edu")
        question = "What email addresses are listed for the Admission Office?"
        self.assertEqual(explicit_email_values(question, [source]),
                         ("a@example.edu", "b@example.edu"))
        self.assertIn("REQUIRED_VALUE_LOSS", answer_safety_failures(
            question, "a@example.edu", [source]))
        self.assertEqual(answer_safety_failures(
            question, "a@example.edu and b@example.edu", [source]), ())
        self.assertEqual(answer_safety_failures(
            "Give any one Admission Office email", "a@example.edu", [source]), ())

    def test_no_prerequisite_synonyms_do_not_conflict(self):
        rows = [evidence("Course Code: ABC 234 Course Title: Systems Credits: 3.0 Prerequisite: None"),
                evidence("ABC 234 Systems 3.0 Nil", page=2)]
        for row in rows:
            row["entity_type"] = "course"
            row["entity_id"] = "ABC 234"
            row["field_types"] = ["prerequisite"]
        self.assertEqual(assess_evidence("What is the prerequisite of ABC 234?", rows).status,
                         SupportStatus.SUPPORTED)

    def test_true_credit_conflict_remains(self):
        rows = [evidence("Course Code: ABC 234 Course Title: Systems Credit Value: 3.0"),
                evidence("Course Code: ABC 234 Course Title: Systems Credit Value: 4.0", page=2)]
        for row in rows:
            row["entity_type"] = "course"
            row["entity_id"] = "ABC 234"
            row["field_types"] = ["credits"]
        self.assertEqual(assess_evidence("How many credits is ABC 234?", rows).status,
                         SupportStatus.CONFLICTING)

    def test_policy_scope_and_true_ambiguity(self):
        for question in (
            "How is one credit assigned for theoretical courses?",
            "থিওরেটিক্যাল কোর্সে এক ক্রেডিট কীভাবে নির্ধারিত হয়?",
            "What percentage of total seats is reserved for children of Freedom Fighters?",
            "Freedom Fighters-এর সন্তানদের জন্য মোট আসনের কত শতাংশ সংরক্ষিত?",
        ):
            self.assertFalse(analyze_query(question).ambiguous, question)
        for question in ("What is the credit?", "What is its prerequisite?", "Who published it?"):
            self.assertTrue(analyze_query(question).ambiguous, question)

    def test_qualifier_and_entity_audit(self):
        self.assertIn("ESSENTIAL_QUALIFIER_LOSS", answer_safety_failures(
            "How many courses may be repeated?", "Students may repeat 2 courses.",
            ["Students may repeat at most 2 courses."], extracted_relation="repeat_course_maximum"))
        self.assertIn("ENTITY_MISMATCH", answer_safety_failures(
            "What is the credit of ABC 234?", "XYZ 987 carries 3 credits.", []))

    def test_bounded_relation_lists_and_prerequisite_components(self):
        self.assertIn("REQUIRED_VALUE_LOSS", answer_safety_failures(
            "Which bachelor programs were initially offered?", "Civil Engineering was offered.",
            ["Bachelor Degree Programs in Civil Engineering and Business Administration only"],
            extracted_relation="initially_offered",
            relation_values=("Civil Engineering", "Business Administration")))
        self.assertEqual(answer_safety_failures(
            "Which bachelor programs were initially offered?",
            "Civil Engineering and Business Administration were offered.",
            [], extracted_relation="initially_offered",
            relation_values=("Civil Engineering", "Business Administration")), ())
        self.assertIn("REQUIRED_VALUE_LOSS", answer_safety_failures(
            "What is the mark distribution?", "Assessment 30%.", [],
            extracted_relation="mark_distribution", relation_values=("30%", "70%")))
        self.assertIn("REQUIRED_VALUE_LOSS", answer_safety_failures(
            "What prerequisites are listed for ABC 234?", "ABC 234 requires DEF 101.",
            ["Course Code: ABC 234 Prerequisite: DEF 101, GHI 202"] ))


if __name__ == "__main__":
    unittest.main()
