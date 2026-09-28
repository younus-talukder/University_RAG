"""Generic same-relation conflict tests; no development question IDs."""

from __future__ import annotations

import unittest

from src.evidence import SupportStatus, _fact_values, assess_evidence


def passage(text: str, page: int, *, entity: str = "", field: str = "credits",
            edition: str = "") -> dict:
    item = {"text": text, "source": f"synthetic-{page}.pdf", "page": page,
            "document_id": f"doc-{page}", "chunk_id": f"doc-{page}-b1-c1",
            "field_types": [field], "edition": edition}
    if entity:
        item.update(entity_type="course", entity_id=entity)
    return item


class ConflictScopeTests(unittest.TestCase):
    def test_theoretical_rule_does_not_conflict_with_lab_or_course_value(self):
        rows = [
            passage("Assignment of Credits: Theoretical Courses: One lecture per week per semester will be equivalent to one credit. Laboratory work differs.", 1),
            passage("Assignment of Credits: Laboratory/Field Work: One credit is based on two hours per week.", 2),
            passage("Course Code: ABC 234 Course Title: Systems Credits: 3.00 Prerequisite: None", 3, entity="ABC 234"),
        ]
        for question in (
            "How is one credit assigned for theoretical courses?",
            "থিওরেটিক্যাল কোর্সে এক ক্রেডিট কীভাবে নির্ধারিত হয়?",
            "Theoretical course-e one credit kivabe assigned hoy?",
        ):
            with self.subTest(question=question):
                result = assess_evidence(question, rows)
                self.assertEqual(result.status, SupportStatus.SUPPORTED)
                self.assertEqual([item.page for item in result.evidence], [1])

    def test_distinct_theoretical_rule_values_still_conflict(self):
        rows = [
            passage("Theoretical Courses: 1 credit = 1 lecture per week.", 1),
            passage("Theoretical Courses: 1 credit = 2 lectures per week.", 2),
        ]
        result = assess_evidence("How is one credit assigned for theoretical courses?", rows)
        self.assertEqual(result.status, SupportStatus.CONFLICTING)
        self.assertEqual(len(result.evidence), 2)

    def test_course_value_is_not_semester_total_but_true_course_conflict_remains(self):
        question = "How many credits is ABC 234?"
        course = passage("Course Code: ABC 234 Course Title: Systems Credits: 3.00", 1, entity="ABC 234")
        total = passage("Course Code: ABC 234 Semester Total Credits: 19.00", 2, entity="ABC 234")
        self.assertEqual(assess_evidence(question, [course, total]).status, SupportStatus.SUPPORTED)
        different = passage("Course Code: ABC 234 Course Title: Systems Credits: 4.00", 3,
                            entity="ABC 234", edition="2025")
        course["edition"] = "2024"
        self.assertEqual(assess_evidence(question, [course, different]).status, SupportStatus.CONFLICTING)

    def test_compact_course_rows_detect_true_conflict_not_decimal_formatting(self):
        question = "How many credits is ABC 234?"
        first = passage("Course Code Course Title Credits Pre-Requisite ABC 234 Systems 3.00 Nil", 1,
                        entity="ABC 234")
        equivalent = passage("Course Code: ABC 234 Course Title: Systems Credits: 3.0 Prerequisite: None", 2,
                             entity="ABC 234")
        different = passage("Course Code Course Title Credits Pre-Requisite ABC 234 Systems 4.00 Nil", 3,
                            entity="ABC 234")
        self.assertEqual(assess_evidence(question, [first, equivalent]).status, SupportStatus.SUPPORTED)
        self.assertEqual(assess_evidence(question, [first, different]).status, SupportStatus.CONFLICTING)

    def test_same_policy_scope_70_and_75_conflict_but_categories_do_not_mix(self):
        question = "What attendance percentage is required by the university policy?"
        first = passage("University attendance policy Category 1: attendance of at least 70% in every course is required.", 1, field="attendance")
        second = passage("University attendance policy Category 1: attendance of at least 75% in every course is required.", 2, field="attendance")
        third = passage("University attendance policy Category 2: attendance of at least 75% in every course is required.", 3, field="attendance")
        self.assertEqual(assess_evidence(question, [first, second]).status, SupportStatus.CONFLICTING)
        self.assertEqual(assess_evidence(question, [first, third]).status, SupportStatus.SUPPORTED)

    def test_no_prerequisite_values_are_equivalent_only_in_verified_field(self):
        question = "What is the prerequisite of ABC 234?"
        rows = [
            passage("Course Code: ABC 234 Course Title: Systems Credits: 3.00 Prerequisite: Nil", 1,
                    entity="ABC 234", field="prerequisite"),
            passage("Course Code: ABC 234 Course Title: Systems Credits: 3.00 Prerequisite: None", 2,
                    entity="ABC 234", field="prerequisite"),
            passage("Course Code: ABC 234 Course Title: Systems Credits: 3.00 Prerequisite: N/A", 3,
                    entity="ABC 234", field="prerequisite"),
        ]
        self.assertEqual(assess_evidence(question, rows).status, SupportStatus.SUPPORTED)
        self.assertEqual(_fact_values("prerequisite", "Credits: N/A Prerequisite: DEF 101"), {"def 101"})
        self.assertEqual(_fact_values("credits", "Credits: N/A"), set())


if __name__ == "__main__":
    unittest.main()
