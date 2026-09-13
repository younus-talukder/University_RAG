from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.corpus import DocumentStatus, descriptor_for_path
from src.retriever import (
    HybridRetrievalConfiguration,
    reciprocal_rank_fusion,
    select_retrieval_results,
)
from src.sparse_index import (
    SparseIndex,
    SparseIndexError,
    load_sparse_index,
    save_sparse_index,
    sparse_configuration_fingerprint,
    sparse_tokenize,
)
from src.vector_store import atomic_publish_index, build_manifest, create_index


def row(chunk_id: str, text: str, **extra):
    return {
        "text": text,
        "document_id": extra.pop("document_id", "doc-a"),
        "source": extra.pop("source", "A.pdf"),
        "relative_path": extra.pop("relative_path", "A.pdf"),
        "page": extra.pop("page", 1),
        "chunk_id": chunk_id,
        "block_id": extra.pop("block_id", f"block-{chunk_id}"),
        "word_start": 0,
        "word_end": len(text.split()),
        "vector_row": extra.pop("vector_row", 0),
        "embedding_content_sha256": extra.pop("embedding_content_sha256", ""),
        **extra,
    }


def bound_rows(*rows):
    from src.vector_store import order_metadata_for_embedding
    return order_metadata_for_embedding(list(rows))


class SparseTokenizerTests(unittest.TestCase):
    def test_course_code_aliases_are_generic(self) -> None:
        tokens = sparse_tokenize("CSE 205, CSE205, and HSS 111(B)")
        for expected in ("cse", "205", "cse205", "hss", "111", "hss111", "hss111b"):
            self.assertIn(expected, tokens)

    def test_numbers_email_percentage_and_date_are_preserved(self) -> None:
        tokens = sparse_tokenize("Credit 3.00; attendance 70%; email info@example.edu; date 2024-01-31")
        for expected in ("3.00", "70%", "info@example.edu", "2024-01-31"):
            self.assertIn(expected, tokens)

    def test_bangla_and_banglish_tokens_are_retained_without_translation(self) -> None:
        tokens = sparse_tokenize("বিশ্ববিদ্যালয়ের CSE 205 prerequisite ki")
        self.assertIn("বিশ্ববিদ্যালয়ের", tokens)
        self.assertIn("prerequisite", tokens)
        self.assertIn("cse205", tokens)


class SparseIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = bound_rows(
            row("a", "general semantic discussion"),
            row("b", "Course Code: ABC 123 Prerequisite: XYZ 100 Credit Value: 3.00", course_code="ABC 123", entity_id="ABC 123", field_types=["prerequisite", "credits"]),
            row("c", "Course Code: ABC 124 Credit Value: 4.00", course_code="ABC 124", entity_id="ABC 124", field_types=["credits"]),
        )
        self.index = SparseIndex.build(self.metadata)

    def test_exact_identifier_and_number_retrieve_positive_matches(self) -> None:
        results = self.index.search("ABC123 3.00", self.metadata, top_k=5)
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertTrue(all(item["sparse_score"] > 0 for item in results))

    def test_empty_or_unmatched_query_returns_no_fillers(self) -> None:
        self.assertEqual(self.index.search("", self.metadata, top_k=3), [])
        self.assertEqual(self.index.search("শুধু অমিল বাংলা শব্দ", self.metadata, top_k=3), [])

    def test_exact_metadata_entity_prefers_requested_field(self) -> None:
        candidates = self.index.metadata_candidates("ABC123", "prerequisite", self.metadata, top_k=5)
        self.assertEqual([item["chunk_id"] for item in candidates], ["b"])

    def test_serialization_round_trip_validates_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse.pkl"
            save_sparse_index(self.index, path)
            loaded = load_sparse_index(path, self.metadata)
        self.assertEqual(loaded.vocabulary_size, self.index.vocabulary_size)

    def test_changed_chunk_text_invalidates_sparse_binding(self) -> None:
        changed = [dict(item) for item in self.metadata]
        changed[0]["text"] = "modified text"
        with self.assertRaisesRegex(SparseIndexError, "different chunk text"):
            self.index.validate(changed)

    def test_tokenizer_configuration_changes_fingerprint(self) -> None:
        baseline = sparse_configuration_fingerprint()
        with patch("src.sparse_index.SPARSE_TOKENIZER_SCHEMA", "different-tokenizer"):
            self.assertNotEqual(baseline, sparse_configuration_fingerprint())


class ReciprocalRankFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = HybridRetrievalConfiguration(
            dense_candidate_k=3,
            sparse_candidate_k=3,
            metadata_candidate_k=3,
            rrf_k=10,
            dense_weight=1.0,
            sparse_weight=1.0,
            metadata_weight=0.5,
        )

    def test_fusion_uses_ranks_not_raw_score_scales(self) -> None:
        dense = [{**row("a", "A"), "dense_score": 0.99}, {**row("b", "B"), "dense_score": 0.10}]
        sparse = [{**row("b", "B"), "sparse_score": 5000.0}]
        fused = reciprocal_rank_fusion(dense, sparse, [], self.configuration)
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertAlmostEqual(fused[0]["fusion_score"], 1 / 12 + 1 / 11)

    def test_duplicate_from_all_channels_is_merged_once(self) -> None:
        item = row("same", "same evidence")
        fused = reciprocal_rank_fusion(
            [{**item, "dense_score": 0.5}],
            [{**item, "sparse_score": 3.0}],
            [item],
            self.configuration,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertEqual(fused[0]["sparse_rank"], 1)
        self.assertEqual(fused[0]["metadata_rank"], 1)

    def test_same_inputs_and_ties_have_deterministic_chunk_id_order(self) -> None:
        dense = [row("b", "B"), row("a", "A")]
        first = reciprocal_rank_fusion(dense, [], [], self.configuration)
        second = reciprocal_rank_fusion(dense, [], [], self.configuration)
        self.assertEqual([item["chunk_id"] for item in first], [item["chunk_id"] for item in second])

    def test_empty_sparse_channel_preserves_dense_ordering(self) -> None:
        dense = [row("a", "A"), row("b", "B"), row("c", "C")]
        fused = reciprocal_rank_fusion(dense, [], [], self.configuration)
        self.assertEqual([item["chunk_id"] for item in fused], ["a", "b", "c"])

    def test_final_selection_is_exact_prefix_of_final_ranking(self) -> None:
        ranked = [row("a", "A"), row("b", "B"), row("c", "C")]
        self.assertEqual(select_retrieval_results(ranked, 2), ranked[:2])


class SparseAtomicPublicationTests(unittest.TestCase):
    def test_sparse_staging_failure_preserves_all_published_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "vector_db"
            target.mkdir()
            originals = {
                "index.faiss": b"old-index",
                "metadata.pkl": b"old-metadata",
                "index_manifest.json": b"old-manifest",
                "sparse_index.pkl": b"old-sparse",
            }
            for name, content in originals.items():
                (target / name).write_bytes(content)
            pdf = root / "A.pdf"
            pdf.write_bytes(b"pdf")
            document = descriptor_for_path(pdf, root)
            document.ingestion_status = DocumentStatus.SUCCESS.value
            metadata = bound_rows(row(
                "a",
                "safe evidence",
                document_id=document.document_id,
                source=document.source,
                relative_path=document.relative_path,
            ))
            sparse = SparseIndex.build(metadata)
            index = create_index(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
            manifest = build_manifest([document], 1, 3, sparse_index=sparse)
            with patch("src.vector_store.save_sparse_index", side_effect=RuntimeError("sparse staging failed")):
                with self.assertRaisesRegex(RuntimeError, "sparse staging failed"):
                    atomic_publish_index(
                        index,
                        metadata,
                        manifest,
                        [document],
                        target_dir=target,
                        sparse_index=sparse,
                    )
            for name, content in originals.items():
                self.assertEqual((target / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
