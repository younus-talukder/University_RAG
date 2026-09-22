from __future__ import annotations

import unittest

from src.generation_context import VerifiedEvidenceItem, VerifiedEvidencePackage
from src.grounding_validator import validate_grounding
from src.language_validator import validate_language
from src.semantic_contract import build_semantic_contract


def package(question: str, evidence: str, field: str = "accreditation") -> VerifiedEvidencePackage:
    return VerifiedEvidencePackage(
        question=question,
        original_language="english",
        requested_entity="ABC University",
        requested_field=field,
        support_status="supported",
        evidence=(VerifiedEvidenceItem("synthetic.pdf", 1, None, None, evidence, "s-1", 1.0),),
    )


class SemanticContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package = package("Who accredited ABC University?", "ABC University is accredited by XYZ.")
        self.contract = build_semantic_contract(self.package)

    def test_wrong_bangla_negation_is_rejected(self) -> None:
        result = validate_grounding("ABC University XYZ দ্বারা স্বীকৃত নয়।", self.package.evidence, self.contract)
        self.assertFalse(result["grounding_validation_passed"])
        self.assertEqual(result["grounding_validation_reason"], "WRONG_POLARITY")

    def test_correct_bangla_polarity_is_accepted(self) -> None:
        result = validate_grounding("ABC University XYZ দ্বারা স্বীকৃত।", self.package.evidence, self.contract)
        self.assertTrue(result["grounding_validation_passed"])

    def test_wrong_banglish_negation_is_rejected(self) -> None:
        result = validate_grounding("ABC University XYZ dara accredited noy.", self.package.evidence, self.contract)
        self.assertEqual(result["grounding_validation_reason"], "WRONG_POLARITY")

    def test_negative_evidence_requires_negative_answer(self) -> None:
        negative = package("Does Course ABC 123 have a prerequisite?", "Course ABC 123 has no prerequisite.", "prerequisite")
        contract = build_semantic_contract(negative)
        self.assertTrue(validate_grounding("Course ABC 123 has no prerequisite.", negative.evidence, contract)["grounding_validation_passed"])
        self.assertEqual(
            validate_grounding("Course ABC 123 has a prerequisite.", negative.evidence, contract)["grounding_validation_reason"],
            "WRONG_POLARITY",
        )

    def test_wrong_relation_value_is_rejected(self) -> None:
        published = package("Who published the prospectus?", "Published by ABC University.", "publisher")
        result = validate_grounding(
            "Published by XYZ University.", published.evidence, build_semantic_contract(published)
        )
        self.assertFalse(result["grounding_validation_passed"])
        self.assertEqual(result["grounding_validation_reason"], "WRONG_RELATION_VALUE")

    def test_fact_plan_is_derived_from_evidence_not_reference_answer(self) -> None:
        self.assertEqual(self.contract.fact_plan[0].relation, "accredited_by")
        self.assertEqual(self.contract.fact_plan[0].value, "XYZ")
        self.assertEqual(self.contract.required_polarity, "affirmative")
        self.assertNotIn("reference", self.contract.prompt_block().casefold())

    def test_implicit_publisher_citation_is_uncertain_for_multilingual_generation(self) -> None:
        citation = VerifiedEvidencePackage(
            question="Prospectus-ta ke publish koreche?", original_language="banglish",
            requested_entity=None, requested_field="publisher", support_status="supported",
            evidence=(VerifiedEvidenceItem("x", 1, None, None, "5. L. L. Lapin, PWS Publisher.", "x", 1.0),),
        )
        contract = build_semantic_contract(citation)
        result = validate_grounding("Prospectus-ta L. L. Lapin publish koreche.", citation.evidence, contract)
        self.assertEqual(result["grounding_validation_reason"], "SEMANTIC_CONTRACT_UNCERTAIN")

    def test_how_many_requires_every_evidence_derived_count(self) -> None:
        counts = package(
            "How many undergraduate and postgraduate disciplines does UAP offer?",
            "UAP offers undergraduate programs in nine disciplines and postgraduate programs in eight disciplines.",
            "general",
        )
        contract = build_semantic_contract(counts)
        self.assertEqual(contract.required_cardinalities, ("9", "8"))
        incomplete = validate_grounding(
            "UAP offers undergraduate programs in nine disciplines and postgraduate programs in multiple disciplines.",
            counts.evidence,
            contract,
        )
        self.assertEqual(incomplete["grounding_validation_reason"], "MISSING_REQUIRED_ENTITY")
        complete = validate_grounding(
            "UAP offers undergraduate programs in nine disciplines and postgraduate programs in eight disciplines.",
            counts.evidence,
            contract,
        )
        self.assertTrue(complete["grounding_validation_passed"])

    def test_supported_but_wrong_strict_relation_modifier_is_rejected(self) -> None:
        operation = VerifiedEvidencePackage(
            question="UAP kon act-er odhine operation start korechilo?",
            original_language="banglish",
            requested_entity="UAP",
            requested_field="general",
            support_status="supported",
            evidence=(VerifiedEvidenceItem(
                "x", 1, None, None,
                "The University of Asia Pacific (UAP), under the Private University Act 1992, started its operation in 1996.",
                "x", 1.0,
            ),),
        )
        contract = build_semantic_contract(operation)
        result = validate_grounding(
            "UAP operation start korechilo 1996-er Private University Act 1992 odhine.",
            operation.evidence,
            contract,
        )
        self.assertEqual(result["grounding_validation_reason"], "WRONG_RELATION_VALUE")
        correct = validate_grounding(
            "UAP Private University Act 1992-er odhine operation start korechilo.",
            operation.evidence,
            contract,
        )
        self.assertTrue(correct["grounding_validation_passed"])

    def test_conditional_answer_requires_every_numeric_constraint(self) -> None:
        conditions = VerifiedEvidencePackage(
            question="Kokhon ekjon student repeat examination dite pare?",
            original_language="banglish",
            requested_entity="student",
            requested_field="general",
            support_status="supported",
            evidence=(VerifiedEvidenceItem(
                "x", 1, None, None,
                "A student would be allowed to appear at the Repeat Examination if they fail in three theory courses or less but do not exceed 10 credit hours.",
                "x", 1.0,
            ),),
        )
        contract = build_semantic_contract(conditions)
        self.assertEqual(contract.required_cardinalities, ("3", "10"))
        incomplete = validate_grounding(
            "Student tin-ti theory course-e fail korle ebong credit limit exceed na korle repeat examination dite pare.",
            conditions.evidence,
            contract,
        )
        self.assertEqual(incomplete["grounding_validation_reason"], "MISSING_REQUIRED_ENTITY")

    def test_count_question_without_evidence_count_is_uncertain(self) -> None:
        missing_count = VerifiedEvidencePackage(
            question="Prospectus onujayi kotojon graduate degree peyechilo?",
            original_language="banglish",
            requested_entity=None,
            requested_field="general",
            support_status="supported",
            evidence=(VerifiedEvidenceItem("x", 1, None, None, "Graduate degree information.", "x", 1.0),),
        )
        contract = build_semantic_contract(missing_count)
        self.assertTrue(contract.exact_cardinality_required)
        result = validate_grounding("Graduate-ra degree peyechilo.", missing_count.evidence, contract)
        self.assertEqual(result["grounding_validation_reason"], "SEMANTIC_CONTRACT_UNCERTAIN")


class Step7DLanguageTests(unittest.TestCase):
    def test_pure_english_banglish_is_rejected(self) -> None:
        result = validate_language("The university is accredited by XYZ.", "banglish")
        self.assertEqual(result["validation_reason"], "BANGLISH_BODY_PURE_ENGLISH")

    def test_valid_banglish_structure_passes(self) -> None:
        result = validate_language("University-ti XYZ dara accredited.", "banglish")
        self.assertTrue(result["validation_passed"])

    def test_banglish_bengali_script_leakage_is_rejected(self) -> None:
        result = validate_language("University-ti XYZ দ্বারা accredited.", "banglish")
        self.assertFalse(result["validation_passed"])

    def test_foreign_script_corruption_is_rejected(self) -> None:
        result = validate_language("উত্তরটি হলো। 翻译如下", "bangla")
        self.assertEqual(result["validation_reason"], "UNEXPECTED_FOREIGN_SCRIPT")

    def test_mixed_script_token_is_rejected(self) -> None:
        result = validate_language("CSE ইngineers থেকে স্বীকৃতি পেয়েছে।", "bangla")
        self.assertEqual(result["validation_reason"], "MIXED_SCRIPT_TOKEN")


if __name__ == "__main__":
    unittest.main()
