from __future__ import annotations

import unittest

from src.generation_context import VerifiedEvidenceItem, VerifiedEvidencePackage
from src.realization_validator import validate_realization
from src.semistructured_realizer import realize_semistructured
from src.semantic_contract import build_semantic_contract


def contract():
    package = VerifiedEvidencePackage(
        question="What is required for CSE 205?",
        original_language="english",
        requested_entity="CSE 205",
        requested_field="requirement",
        support_status="supported",
        evidence=(VerifiedEvidenceItem(
            "synthetic.pdf", 1, None, "CSE 205",
            "CSE 205 requires 70% attendance.", "s-1", 1.0,
        ),),
    )
    return build_semantic_contract(package)


class RealizationValidationTests(unittest.TestCase):
    def test_semistructured_established_banglish_is_fact_preserving(self) -> None:
        canonical = "The University of Asia Pacific (UAP) was established in 1996."
        self.assertEqual(
            realize_semistructured(canonical, "banglish"),
            "University of Asia Pacific (UAP) 1996-e establish hoyechilo.",
        )

    def test_semistructured_fee_bangla_preserves_value(self) -> None:
        canonical = "The fee for re-examination of answer scripts is Taka 200 per script."
        self.assertEqual(
            realize_semistructured(canonical, "bangla"),
            "re-examination of answer scripts-এর ফি হলো Taka 200 per script।",
        )

    def test_semistructured_named_role_banglish_is_natural(self) -> None:
        canonical = "The Vice-Chancellor of the university is named Prof. Dr. Jamilur Reza Choudhury."
        self.assertEqual(
            realize_semistructured(canonical, "banglish"),
            "University-er Vice-Chancellor hisebe Prof. Dr. Jamilur Reza Choudhury-er naam deya ache.",
        )

    def test_fact_heavy_bangla_address_with_bengali_grammar_is_valid(self) -> None:
        canonical = "UAP Administration is located at 74/A, Green Road, Dhaka-1215."
        realized = "UAP Administration-এর ঠিকানাটি হলো 74/A, Green Road, Dhaka-1215।"
        result = validate_realization(canonical, realized, "bangla", contract())
        self.assertTrue(result["language_validation"]["validation_passed"])

    def setUp(self) -> None:
        self.canonical = "CSE 205 requires 70% attendance."
        self.contract = contract()

    def test_bangla_realization_preserves_facts(self) -> None:
        result = validate_realization(
            self.canonical,
            "CSE 205-এর জন্য 70% attendance প্রয়োজন।",
            "bangla",
            self.contract,
        )
        self.assertTrue(result["passed"])

    def test_positive_canonical_to_negative_bangla_is_rejected(self) -> None:
        result = validate_realization(
            self.canonical,
            "CSE 205-এর জন্য 70% attendance প্রয়োজন নয়।",
            "bangla",
            self.contract,
        )
        self.assertEqual(result["reason"], "WRONG_POLARITY")

    def test_removed_course_code_is_rejected(self) -> None:
        result = validate_realization(
            self.canonical,
            "কোর্সটির জন্য 70% attendance প্রয়োজন।",
            "bangla",
            self.contract,
        )
        self.assertEqual(result["reason"], "MISSING_CANONICAL_FACTS")

    def test_changed_percentage_is_rejected(self) -> None:
        result = validate_realization(
            self.canonical,
            "CSE 205-এর জন্য 75% attendance প্রয়োজন।",
            "bangla",
            self.contract,
        )
        self.assertEqual(result["reason"], "MISSING_CANONICAL_FACTS")

    def test_valid_banglish_realization_is_accepted(self) -> None:
        result = validate_realization(
            self.canonical,
            "CSE 205-er jonno 70% attendance lagbe.",
            "banglish",
            self.contract,
        )
        self.assertTrue(result["passed"])

    def test_pure_english_banglish_is_rejected(self) -> None:
        result = validate_realization(
            self.canonical,
            self.canonical,
            "banglish",
            self.contract,
        )
        self.assertEqual(result["reason"], "BANGLISH_BODY_PURE_ENGLISH")

    def test_bengali_script_banglish_is_rejected(self) -> None:
        result = validate_realization(
            self.canonical,
            "CSE 205-এর jonno 70% attendance lagbe.",
            "banglish",
            self.contract,
        )
        self.assertEqual(result["reason"], "BANGLISH_CONTAINS_BENGALI_SCRIPT")


if __name__ == "__main__":
    unittest.main()
