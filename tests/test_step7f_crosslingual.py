from __future__ import annotations

import unittest
from unittest.mock import patch

from src import pipeline
from src.crosslingual import (
    Rewrite, canonical_rewrite, crosslingual_assessment, detect_chunk_language,
    deterministic_query, fuse_queries, validate_rewrite,
)
from src.evidence import SupportStatus, assess_evidence


def chunk(text: str, chunk_id: str) -> dict:
    return {"text": text, "chunk_id": chunk_id, "source": "synthetic.pdf", "page": 1,
            "score": 1.0, "entity_type": "course", "entity_id": "ABC 123"}


class QuerySafetyTests(unittest.TestCase):
    def test_deterministic_entity_and_field(self) -> None:
        rewrite = deterministic_query("ABC 123 কোর্সের পূর্বশর্ত কী?")
        self.assertIsNotNone(rewrite)
        self.assertIn("ABC 123", rewrite.query)
        self.assertIn("prerequisite", rewrite.query)

    def test_course_entity_change_is_rejected(self) -> None:
        self.assertFalse(validate_rewrite("ABC 123 prerequisite?", "ABC 124 prerequisite")[0])

    def test_percentage_change_is_rejected(self) -> None:
        self.assertFalse(validate_rewrite("70% attendance?", "75% attendance")[0])

    def test_field_change_is_rejected(self) -> None:
        self.assertFalse(validate_rewrite("ABC 123 prerequisite?", "ABC 123 credits")[0])

    def test_rewrite_cache_is_bounded_and_validated(self) -> None:
        question = "XYZ 984 er prerequisite ki?"
        first = canonical_rewrite(question)
        second = canonical_rewrite(question)
        self.assertEqual(first.query, second.query)
        self.assertEqual(second.validation, "VALIDATED_CACHED")

    def test_language_signal_is_conservative(self) -> None:
        self.assertEqual(detect_chunk_language("x"), "unknown")
        self.assertEqual(detect_chunk_language("This university course has prerequisites and credits."), "english_dominant")

    def test_equal_rank_fusion_preserves_provenance(self) -> None:
        original = [chunk("first", "a"), chunk("shared", "b")]
        canonical = [chunk("shared", "b"), chunk("other", "c")]
        merged = fuse_queries(original, canonical)
        self.assertEqual(merged[0]["chunk_id"], "b")
        self.assertEqual(merged[0]["original_query_rank"], 2)
        self.assertEqual(merged[0]["canonical_query_rank"], 1)


class PipelineFallbackTests(unittest.TestCase):
    def _run(self, question: str, original: list[dict], canonical: list[dict] | None = None,
             generation_failure: bool = False):
        with (
            patch.object(pipeline, "load_index", return_value=(object(), original)),
            patch.object(pipeline, "_get_embedding_model", return_value=object()),
            patch.object(pipeline, "Retriever") as retriever,
            patch.object(pipeline, "find_answer_bank_match", side_effect=AssertionError("dataset accessed")),
            patch.object(pipeline, "rewrite_query_for_retrieval", side_effect=AssertionError("Qwen unnecessary")),
            patch.object(pipeline, "select_answer_strategy", return_value="gguf_generation") if generation_failure else patch.object(pipeline, "select_answer_strategy", wraps=pipeline.select_answer_strategy),
            patch.object(pipeline, "generate_canonical_answer", side_effect=RuntimeError("synthetic generation failure")) if generation_failure else patch.object(pipeline, "generate_canonical_answer", wraps=pipeline.generate_canonical_answer),
        ):
            retriever.return_value.retrieve.side_effect = [original] + ([canonical] if canonical is not None else [])
            result = pipeline.answer_question(question, use_generation=True)
            calls = retriever.return_value.retrieve.call_count
        return result, calls

    def test_bangla_fallback_recovers_evidence(self) -> None:
        original = [chunk("Course Code: ABC 123 Course Title: Example", "a")]
        canonical = [chunk("Course Code: ABC 123 Prerequisite: XYZ 101", "b")]
        result, calls = self._run("ABC 123 কোর্সের পূর্বশর্ত কী?", original, canonical)
        self.assertEqual(result["support_status"], "supported")
        self.assertTrue(result["fallback_triggered"])
        self.assertEqual(calls, 2)

    def test_banglish_fallback_recovers_evidence(self) -> None:
        original = [chunk("Course Code: ABC 123 Course Title: Example", "a")]
        canonical = [chunk("Course Code: ABC 123 Prerequisite: XYZ 101", "b")]
        result, calls = self._run("ABC 123 er prerequisite ki?", original, canonical)
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(calls, 2)

    def test_english_does_not_rewrite(self) -> None:
        original = [chunk("Course Code: ABC 123 Course Title: Example", "a")]
        result, calls = self._run("What is the prerequisite of ABC 123?", original)
        self.assertFalse(result["fallback_triggered"])
        self.assertEqual(calls, 1)

    def test_supported_generation_failure_does_not_retrieve_again(self) -> None:
        original = [chunk("Course Code: ABC 123 Prerequisite: XYZ 101", "b")]
        result, calls = self._run("ABC 123 কোর্সের পূর্বশর্ত কী?", original, generation_failure=True)
        self.assertEqual(result["final_status"], "GENERATION_REJECTED")
        self.assertFalse(result["fallback_triggered"])
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
