from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evidence import analyze_query, identify_entities
from src.language_detector import detect_language, detect_language_details
from src.query_normalization import (
    build_query_representations,
    dense_query_for_strategy,
    extract_course_entities,
    normalize_retrieval_text,
    normalize_unicode,
)
from src.reranker import (
    REQUIRED_FILES,
    RerankerConfiguration,
    RerankerUnavailableError,
    rerank_candidate_pool,
    resolve_reranker_snapshot,
    validate_reranker_snapshot,
)
from src.retriever import Retriever
from src.sparse_index import sparse_tokenize


class QueryNormalizationTests(unittest.TestCase):
    def test_unicode_whitespace_hyphen_and_full_width_are_normalized(self) -> None:
        self.assertEqual(normalize_unicode("ＣＳＥ\u00a0–\u00a0২০৫"), "CSE - 205")

    def test_original_query_is_preserved_separately(self) -> None:
        original = "CSE–২০৫ er credit koy?"
        representations = build_query_representations(original)
        self.assertEqual(representations.original_query, original)
        self.assertIn("cse 205", representations.normalized_query)
        self.assertIn("koto", representations.retrieval_query)

    def test_generic_course_forms_share_compact_identity(self) -> None:
        forms = ("ABC123", "ABC 123", "ABC-123")
        self.assertEqual({extract_course_entities(value)[0].compact for value in forms}, {"ABC123"})
        self.assertEqual(extract_course_entities("XYZ 201(A)")[0].compact, "XYZ201A")
        self.assertEqual(extract_course_entities("ENG(CSE)101")[0].compact, "ENGCSE101")

    def test_banglish_variants_are_conservatively_canonicalized(self) -> None:
        normalized = normalize_retrieval_text("kii hbe ase jnno krte koy credit lage")
        self.assertEqual(normalized, "ki hobe ache jonno korte koto credit lagbe")

    def test_bangla_week_and_mixed_prerequisite_are_normalized(self) -> None:
        normalized = normalize_retrieval_text("৫ম সপ্তাহে প্রিরিকুইজিট কী?")
        self.assertIn("সপ্তাহ 5", normalized)
        self.assertIn("prerequisite", normalized)

    def test_email_decimal_percentage_and_date_survive(self) -> None:
        normalized = normalize_retrieval_text("INFO@Example.edu, 3.00, 70%, 2024-01-31")
        for value in ("info@example.edu", "3.00", "70%", "2024-01-31"):
            self.assertIn(value, normalized)

    def test_empty_input_is_safe(self) -> None:
        self.assertEqual(normalize_retrieval_text(""), "")

    def test_dense_strategies_are_bounded_and_explicit(self) -> None:
        query = "CSE-২০৫ credit koy?"
        self.assertEqual(dense_query_for_strategy(query, "original"), query)
        self.assertIn("cse 205", dense_query_for_strategy(query, "normalized"))
        self.assertLessEqual(dense_query_for_strategy(query, "original_plus_normalized").count("cse 205"), 1)
        with self.assertRaises(ValueError):
            dense_query_for_strategy(query, "invalid")


class SharedUnderstandingTests(unittest.TestCase):
    def test_unseen_entity_and_field_use_shared_normalization(self) -> None:
        request = analyze_query("ABC-১২৩ er prereq kii?")
        self.assertEqual(request.entities[0].normalized, "ABC123")
        self.assertEqual(request.requested_field, "prerequisite")

    def test_entity_extraction_handles_unparenthesized_suffix(self) -> None:
        self.assertEqual(identify_entities("XYZ201A er credit koto?")[0].normalized, "XYZ201A")

    def test_sparse_tokens_retain_normalized_technical_values(self) -> None:
        tokens = sparse_tokenize("ABC-১২৩, 3.00, 70%, info@example.edu, 2024-01-31")
        for value in ("abc123", "3.00", "70%", "info@example.edu", "2024-01-31"):
            self.assertIn(value, tokens)


class LanguageDetectionTests(unittest.TestCase):
    def test_representative_languages(self) -> None:
        cases = (
            ("What are the course requirements?", "english"),
            ("কয় ক্রেডিট লাগবে?", "bangla"),
            ("koy credit lage?", "banglish"),
            ("kii requirement?", "banglish"),
            ("exam er rule ki?", "banglish"),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(detect_language(query), expected)

    def test_detection_reason_uses_debug_signals_not_probability(self) -> None:
        details = detect_language_details("koy credit lage?")
        self.assertEqual(details.language, "banglish")
        self.assertTrue(details.reason)
        self.assertIn("koto", details.banglish_markers)


class RerankerScaffoldingTests(unittest.TestCase):
    class MockScorer:
        def score_pairs(self, query, passages):
            return [0.1, 0.9, 0.5][:len(passages)]

    def test_mock_reranker_scores_only_bounded_pool_deterministically(self) -> None:
        candidates = [
            {"chunk_id": "a", "text": "A", "final_rank": 1},
            {"chunk_id": "b", "text": "B", "final_rank": 2},
            {"chunk_id": "c", "text": "C", "final_rank": 3},
        ]
        first = rerank_candidate_pool("query", candidates, self.MockScorer(), candidate_k=2)
        second = rerank_candidate_pool("query", candidates, self.MockScorer(), candidate_k=2)
        self.assertEqual([item["chunk_id"] for item in first], ["b", "a"])
        self.assertEqual(first, second)

    def test_disabled_reranker_keeps_step5_retriever_available(self) -> None:
        Retriever(embedding_model=object(), reranker_enabled=False)
        scorer = self.MockScorer()
        with patch("src.retriever.get_reranker_model", return_value=scorer) as loader:
            enabled = Retriever(embedding_model=object(), reranker_enabled=True)
        self.assertIs(enabled.reranker, scorer)
        loader.assert_called_once_with()

    def test_pinned_model_identity_rejects_mutable_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved pinned"):
            RerankerConfiguration(revision="main")
        with self.assertRaisesRegex(ValueError, "approved pinned"):
            RerankerConfiguration(model_name="another/model")
        with self.assertRaisesRegex(ValueError, "approved pinned"):
            RerankerConfiguration(revision="0" * 40)

    def test_missing_snapshot_has_clear_error(self) -> None:
        configuration = RerankerConfiguration()
        with self.assertRaisesRegex(RerankerUnavailableError, "pinned revision"):
            validate_reranker_snapshot(Path("missing") / configuration.revision, configuration)

    def test_snapshot_validation_checks_required_files_and_identity(self) -> None:
        configuration = RerankerConfiguration()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / configuration.revision
            snapshot.mkdir()
            for name in REQUIRED_FILES:
                content = b"x"
                if name == "config.json":
                    content = json.dumps({"architectures": ["XLMRobertaForSequenceClassification"]}).encode()
                (snapshot / name).write_bytes(content)
            validation = validate_reranker_snapshot(snapshot, configuration)
        self.assertEqual(validation["reranker_model"], configuration.model_name)
        self.assertEqual(validation["reranker_revision"], configuration.revision)
        self.assertNotIn("snapshot_path", validation)

    def test_resolver_enforces_local_only(self) -> None:
        configuration = RerankerConfiguration(local_only=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / configuration.revision
            snapshot.mkdir()
            for name in REQUIRED_FILES:
                content = b"x" if name != "config.json" else json.dumps(
                    {"architectures": ["XLMRobertaForSequenceClassification"]}
                ).encode()
                (snapshot / name).write_bytes(content)
            with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                resolved, _ = resolve_reranker_snapshot(configuration)
        self.assertEqual(resolved.name, configuration.revision)
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertEqual(download.call_args.kwargs["revision"], configuration.revision)

    def test_score_count_and_finite_validation(self) -> None:
        candidates = [{"chunk_id": "a", "text": "A", "final_rank": 1}]

        class WrongCount:
            def score_pairs(self, query, passages):
                return []

        class NonFinite:
            def score_pairs(self, query, passages):
                return [math.nan]

        with self.assertRaisesRegex(ValueError, "different number"):
            rerank_candidate_pool("query", candidates, WrongCount(), 1)
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            rerank_candidate_pool("query", candidates, NonFinite(), 1)

    def test_stable_ties_preserve_rrf_order_and_provenance(self) -> None:
        class TieScorer:
            def score_pairs(self, query, passages):
                return [1.0] * len(passages)

        candidates = [
            {"chunk_id": "b", "text": "B", "source": "one.pdf", "page": 7, "final_rank": 1},
            {"chunk_id": "a", "text": "A", "source": "two.pdf", "page": 9, "final_rank": 2},
        ]
        ranked = rerank_candidate_pool("query", candidates, TieScorer(), 2)
        self.assertEqual([item["chunk_id"] for item in ranked], ["b", "a"])
        self.assertEqual(ranked[0]["source"], "one.pdf")
        self.assertEqual(ranked[0]["page"], 7)

    def test_only_query_and_passage_text_reach_scorer(self) -> None:
        class RecordingScorer:
            def score_pairs(self, query, passages):
                self.query = query
                self.passages = list(passages)
                return [0.0]

        scorer = RecordingScorer()
        candidate = {
            "chunk_id": "a", "text": "document evidence", "final_rank": 1,
            "question_id": "SHOULD_NOT_PASS", "expected_page": 99, "ground_truth": "SECRET",
        }
        rerank_candidate_pool("original question", [candidate], scorer, 1)
        self.assertEqual(scorer.query, "original question")
        self.assertEqual(scorer.passages, ["document evidence"])


if __name__ == "__main__":
    unittest.main()
