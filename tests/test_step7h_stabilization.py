from __future__ import annotations

import unittest

from src.answer_policy import effective_answer_field, format_exact_fact
from src.course_rows import course_field_value
from src.evidence import Evidence, EvidenceAssessment, SupportStatus, analyze_query
from src.generation_context import VerifiedEvidenceItem, VerifiedEvidencePackage
from src.grounding_validator import validate_grounding
from src.relations import extract_relation, realize_relation, validate_relation_grounding
from src.semantic_contract import build_semantic_contract


def package(question: str, field: str, text: str) -> VerifiedEvidencePackage:
    return VerifiedEvidencePackage(question, "english", None, field, "supported",
        (VerifiedEvidenceItem("synthetic.pdf", 1, None, None, text, "s-1", 1.0),))


def relation(question: str, text: str, language: str = "english"):
    evidence = Evidence(None, "synthetic.pdf", None, 1, "s-1", text, text, 1.0,
                        None, None, "general", "general", "supported")
    assessment = EvidenceAssessment(analyze_query(question), SupportStatus.SUPPORTED,
                                    (evidence,), "verified")
    item = extract_relation(question, assessment, [], language)
    return item, assessment


class CourseFieldTests(unittest.TestCase):
    def test_credit_decimal_is_atomic_and_owned_by_requested_row(self) -> None:
        text = "ABC 112 Sample Course 1.50 Nil Total 19.50 XYZ 113 Another Course 3.00 Nil"
        question = "How many credits is ABC 112?"
        self.assertEqual(course_field_value(question, "credits", [text]), "1.50")
        contract = build_semantic_contract(package(question, "credits", text))
        self.assertEqual(contract.required_cardinalities, ("1.50",))
        self.assertTrue(validate_grounding("ABC 112 carries 1.50 credits.",
                                           package(question, "credits", text).evidence, contract)["grounding_validation_passed"])
        self.assertEqual(validate_grounding("ABC 112 carries 3.00 credits.",
                                            package(question, "credits", text).evidence, contract)["grounding_validation_reason"],
                         "WRONG_FIELD_VALUE")

    def test_prerequisite_label_variants_and_nil(self) -> None:
        question = "What prerequisite is listed for CHEM 112 Chemistry Lab?"
        for label in ("Prerequisite", "Pre-requisite", "Pre Requisite", "Pre-Requisite", "Pre- Requisite"):
            text = f"Course Code: CHEM 112 Course Title: Chemistry Lab Credits: 1.50 {label}: Nil"
            with self.subTest(label=label):
                self.assertEqual(course_field_value(question, "prerequisite", [text]), "Nil")
                evidence = package(question, "prerequisite", text)
                answer = format_exact_fact("CHEM 112", "prerequisite", "Nil", "english")
                self.assertTrue(validate_grounding(answer, evidence.evidence,
                                                   build_semantic_contract(evidence))["grounding_validation_passed"])

    def test_equivalent_course_rows_coalesce_but_real_conflicts_do_not(self) -> None:
        q = "What prerequisite is listed for CSE 102?"
        self.assertEqual(course_field_value(q, "prerequisite", [
            "CSE 102 Example Lab 1.50 Nil",
            "Course Code: CSE 102 Course Title: Example Lab Credits: 1.50 Pre- Requisite: None",
        ]), "Nil")
        self.assertIsNone(course_field_value(q, "prerequisite", [
            "CSE 102 Example Lab 1.50 Nil",
            "Course Code: CSE 102 Course Title: Example Lab Credits: 1.50 Prerequisite: ABC 101",
        ]))
        self.assertEqual(course_field_value("CSE 205 prerequisite?", "prerequisite", [
            "CSE 205 Data Structures 3.00 CSE 101, CSE103, CSE 105",
            "Course Code: CSE 205 Course Title: Data Structures Credits: 3.00 Prerequisite: CSE 101, CSE 103, CSE 105 Introduction: topics",
        ]), "CSE 101, CSE 103, CSE 105")

    def test_banglish_sentence_final_decimal_is_valid(self) -> None:
        question = "CSE 102 Example Lab course-er credit koto?"
        text = "CSE 102 Example Lab 1.50 Nil"
        evidence = VerifiedEvidencePackage(question, "banglish", None, "credits", "supported",
            (VerifiedEvidenceItem("synthetic.pdf", 1, None, None, text, "s-1", 1.0),))
        self.assertTrue(validate_grounding("CSE 102 course-er credit 1.50.", evidence.evidence,
                                           build_semantic_contract(evidence))["grounding_validation_passed"])

    def test_verified_course_row_uses_same_semantic_standard_in_three_languages(self) -> None:
        question = "How many credits is ABC 112?"
        text = "ABC 112 Sample Course 1.50 Nil"
        for language, answer in (
            ("english", "ABC 112 carries 1.50 credits."),
            ("bangla", "ABC 112 কোর্সের ক্রেডিট 1.50।"),
            ("banglish", "ABC 112 course-er credit 1.50."),
        ):
            evidence = VerifiedEvidencePackage(question, language, None, "credits", "supported",
                (VerifiedEvidenceItem("synthetic.pdf", 1, None, None, text, "s-1", 1.0),))
            with self.subTest(language=language):
                self.assertTrue(validate_grounding(answer, evidence.evidence,
                                                   build_semantic_contract(evidence))["grounding_validation_passed"])


class FocusAndRelationTests(unittest.TestCase):
    def test_year_before_sentence_period_remains_atomic(self) -> None:
        question = "When was ABC University established?"
        evidence = package(question, "date", "ABC University was established in 1996.")
        contract = build_semantic_contract(evidence)
        self.assertEqual(contract.required_cardinalities, ("1996",))
        self.assertTrue(validate_grounding("ABC University was established in 1996.",
                                           evidence.evidence, contract)["grounding_validation_passed"])

    def test_repeat_course_count_not_grade(self) -> None:
        question = "How many courses can a student repeat for grade improvement?"
        text = "A student with grade C may repeat a maximum of four courses for grade improvement."
        self.assertEqual(effective_answer_field("grade", question), "course_count")
        item, assessment = relation(question, text)
        self.assertEqual(item.relation, "repeat_course_maximum")
        self.assertIn("at most 4 courses", realize_relation(item))
        self.assertFalse(validate_relation_grounding(question, item, assessment, "The grade is C.")["grounding_validation_passed"])

    def test_unrelated_establishment_claim_is_rejected(self) -> None:
        question = "UAP kobe established hoy?"
        text = ("University of Asia Pacific (UAP) was established in 1996. "
                "The University of Asia Pacific Foundation was established in 1995.")
        evidence = package(question, "general", text)
        contract = build_semantic_contract(evidence)
        bad = ("University of Asia Pacific (UAP) 1996. The University of Asia Pacific Foundation "
               "was established in 1995-e establish hoyechilo.")
        self.assertEqual(validate_grounding(bad, evidence.evidence, contract)["grounding_validation_reason"],
                         "ENTITY_FOCUS_MISMATCH")

    def test_attendance_every_course_qualifier_is_preserved(self) -> None:
        question = "What attendance is required to sit for the final examination?"
        text = ("A student is required to attend at least 70% of all the classes held in every course "
                "in order to sit for the final examination.")
        item, assessment = relation(question, text, "banglish")
        self.assertIn("every_course", item.labels)
        self.assertIn("protiti course-e", realize_relation(item))
        missing = "Final exam-e boshte kompakhe 70% class-e attend korte hobe."
        self.assertEqual(validate_relation_grounding(question, item, assessment, missing)["grounding_validation_reason"],
                         "MISSING_REQUIRED_QUALIFIER")
        self.assertTrue(validate_relation_grounding(question, item, assessment, realize_relation(item))["grounding_validation_passed"])


if __name__ == "__main__":
    unittest.main()
