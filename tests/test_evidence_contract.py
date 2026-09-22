from __future__ import annotations

import unittest
from unittest.mock import patch

from src import pipeline
from src.evidence import SupportStatus, analyze_query, assess_evidence, identify_entities


def chunk(text: str, source: str = "document-a.pdf", page: int = 1, chunk_id: str = "a-1") -> dict:
    return {"text": text, "source": source, "page": page, "chunk_id": chunk_id, "score": 7.5}


class FakeIndex:
    ntotal = 1


class GenericEntityAndFieldTests(unittest.TestCase):
    def test_generic_course_patterns_preserve_original(self) -> None:
        cases = ["CSE 101", "CSE101", "EEE 221", "ENG (CSE) 101", "HSS 111(B)"]
        for value in cases:
            with self.subTest(value=value):
                entities = identify_entities(f"What is the credit of {value}?")
                self.assertEqual(entities[0].kind, "course_code")
                self.assertEqual(entities[0].value, value)

    def test_unknown_field_remains_general(self) -> None:
        self.assertEqual(analyze_query("Explain the university library services").requested_field, "general")


class EvidenceAssessmentTests(unittest.TestCase):
    def test_valid_entity_and_field_is_supported_with_excerpt(self) -> None:
        result = assess_evidence(
            "How many credits is ABC 234?",
            [chunk("Course Code: ABC 234 Course Title: Systems Credit Value: 3.0")],
        )
        self.assertEqual(result.status, SupportStatus.SUPPORTED)
        self.assertTrue(result.evidence[0].excerpt)
        self.assertEqual(result.evidence[0].source, "document-a.pdf")

    def test_wrong_entity_is_insufficient(self) -> None:
        result = assess_evidence(
            "What is the prerequisite of ABC 234?",
            [chunk("Course Code: XYZ 987 Prerequisite: XYZ 100")],
        )
        self.assertEqual(result.status, SupportStatus.INSUFFICIENT)

    def test_missing_field_is_insufficient(self) -> None:
        result = assess_evidence(
            "What is the prerequisite of ABC 234?",
            [chunk("Course Code: ABC 234 Course Title: Systems")],
        )
        self.assertEqual(result.status, SupportStatus.INSUFFICIENT)

    def test_conflicting_values_are_not_silently_selected(self) -> None:
        result = assess_evidence(
            "How many credits is ABC 234?",
            [
                chunk("Course Code: ABC 234 Credit Value: 3.0", page=1, chunk_id="a-1"),
                chunk("Course Code: ABC 234 Credit Value: 4.0", page=7, chunk_id="a-7"),
            ],
        )
        self.assertEqual(result.status, SupportStatus.CONFLICTING)
        self.assertEqual(set(result.conflicting_values), {"3.0", "4.0"})
        self.assertEqual(len(result.evidence), 2)

    def test_ambiguous_credit_request_asks_for_context(self) -> None:
        result = assess_evidence("What is the credit?", [chunk("ABC 234 Credit Value: 3.0")])
        self.assertEqual(result.status, SupportStatus.AMBIGUOUS)

    def test_invented_entity_does_not_return_unrelated_evidence(self) -> None:
        result = assess_evidence(
            "What is the email for ZZZ 999?",
            [chunk("Contact for ABC 234: registrar@example.edu")],
        )
        self.assertEqual(result.status, SupportStatus.INSUFFICIENT)


class PipelineEvidenceContractTests(unittest.TestCase):
    def _answer(self, question: str, rows: list[dict]) -> dict:
        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "retrieve_lexical", return_value=rows),
        ):
            return pipeline.answer_question(question, use_generation=False)

    def test_answer_bank_is_off_by_default(self) -> None:
        rows = [chunk("Course Code: ABC 234 Credit Value: 3.0")]
        with (
            patch.object(pipeline, "find_answer_bank_match", side_effect=AssertionError("must be opt-in")),
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "retrieve_lexical", return_value=rows),
        ):
            result = pipeline.answer_question("How many credits is ABC 234?", use_generation=False)
        self.assertFalse(result["answer_bank_enabled"])
        self.assertEqual(result["support_status"], "supported")

    def test_supported_pipeline_answer_has_bound_citation(self) -> None:
        result = self._answer(
            "How many credits is ABC 234?",
            [chunk("Course Code: ABC 234 Course Title: Systems Credit Value: 3.0")],
        )
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(result["source"], "document-a.pdf")
        self.assertEqual(result["chunk_id"], "a-1")
        self.assertTrue(result["supporting_excerpt"])

    def test_multilingual_wrong_entity_uses_same_abstention_logic(self) -> None:
        rows = [chunk("Course Code: XYZ 987 Credit Value: 4.0")]
        cases = [
            ("How many credits is ABC 234?", "english"),
            ("ABC 234 এর ক্রেডিট কত?", "bangla"),
            ("ABC 234 er credit koto?", "banglish"),
        ]
        for question, language in cases:
            with self.subTest(question=question):
                result = self._answer(question, rows)
                self.assertEqual(result["support_status"], "insufficient")
                self.assertEqual(result["detected_language"], language)
                self.assertFalse(result["sources"])

    def test_conflict_pipeline_exposes_both_passages(self) -> None:
        result = self._answer(
            "How many credits is ABC 234?",
            [
                chunk("Course Code: ABC 234 Credit Value: 3.0", page=1, chunk_id="a-1"),
                chunk("Course Code: ABC 234 Credit Value: 4.0", page=7, chunk_id="a-7"),
            ],
        )
        self.assertEqual(result["support_status"], "conflicting")
        self.assertEqual(len(result["sources"]), 2)
        self.assertNotIn("Credit value:", result["answer"])

    def test_conflict_and_ambiguity_messages_cover_all_languages(self) -> None:
        conflicts = [
            chunk("Course Code: ABC 234 Credit Value: 3.0", page=1, chunk_id="a-1"),
            chunk("Course Code: ABC 234 Credit Value: 4.0", page=7, chunk_id="a-7"),
        ]
        cases = [
            ("How many credits is ABC 234?", "What is the credit?", "english"),
            ("ABC 234 এর ক্রেডিট কত?", "ক্রেডিট কত?", "bangla"),
            ("ABC 234 er credit koto?", "Credit koto?", "banglish"),
        ]
        for conflict_question, ambiguous_question, language in cases:
            with self.subTest(language=language):
                conflict = self._answer(conflict_question, conflicts)
                ambiguous = self._answer(ambiguous_question, conflicts)
                self.assertEqual(conflict["support_status"], "conflicting")
                self.assertEqual(ambiguous["support_status"], "ambiguous")
                self.assertEqual(conflict["detected_language"], language)
                self.assertEqual(ambiguous["detected_language"], language)

    def test_generation_receives_only_verified_excerpt(self) -> None:
        rows = [
            chunk("Course Code: XYZ 987 Course overview: unrelated material.", source="wrong.pdf", chunk_id="wrong"),
            chunk("Course Code: ABC 234 Course overview: systems principles.", source="right.pdf", chunk_id="right"),
        ]
        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "_get_embedding_model", return_value=object()),
            patch.object(pipeline, "Retriever") as retriever,
            patch.object(pipeline, "generate_canonical_answer", return_value="ABC 234 explains systems principles.") as generate,
        ):
            retriever.return_value.retrieve.return_value = rows
            result = pipeline.answer_question("Explain ABC 234.", use_generation=True)
        package = generate.call_args.args[0]
        context = [item.excerpt for item in package.evidence]
        self.assertEqual(result["support_status"], "supported")
        self.assertIn("ABC 234", context[0])
        self.assertNotIn("XYZ 987", " ".join(context))

    def test_exact_fact_bypasses_generation(self) -> None:
        rows = [chunk("Course Code: ABC 234 Credit Value: 3.0")]
        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "_get_embedding_model", return_value=object()),
            patch.object(pipeline, "Retriever") as retriever,
            patch.object(pipeline, "generate_canonical_answer") as generate,
        ):
            retriever.return_value.retrieve.return_value = rows
            result = pipeline.answer_question("How many credits is ABC 234?", use_generation=True)
        generate.assert_not_called()
        self.assertEqual(result["answer_strategy"], "structured_exact")
        self.assertIn("3.0", result["answer"])

    def test_unbound_generated_fact_abstains_without_unsafe_fallback(self) -> None:
        rows = [chunk("Course Code: ABC 234 Course overview: systems principles.")]
        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "_get_embedding_model", return_value=object()),
            patch.object(pipeline, "Retriever") as retriever,
            patch.object(pipeline, "generate_canonical_answer", return_value="ABC 234 requires 9.0 years."),
        ):
            retriever.return_value.retrieve.return_value = rows
            result = pipeline.answer_question("Explain ABC 234.", use_generation=True)
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(result["final_status"], "GENERATION_REJECTED")
        self.assertNotIn("9.0", result["answer"])

    def test_unbound_generated_text_claim_also_abstains(self) -> None:
        rows = [chunk("Course Code: ABC 234 Course overview: programming fundamentals.")]
        with (
            patch.object(pipeline, "load_index", return_value=(FakeIndex(), rows)),
            patch.object(pipeline, "_get_embedding_model", return_value=object()),
            patch.object(pipeline, "Retriever") as retriever,
            patch.object(pipeline, "generate_canonical_answer", return_value="ABC 234 focuses on Calculus."),
        ):
            retriever.return_value.retrieve.return_value = rows
            result = pipeline.answer_question("Explain ABC 234.", use_generation=True)
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(result["final_status"], "GENERATION_REJECTED")
        self.assertNotIn("Calculus", result["answer"])


if __name__ == "__main__":
    unittest.main()
