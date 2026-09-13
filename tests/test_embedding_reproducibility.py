from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import build_index as build_script
from src.corpus import DocumentStatus, IngestionResult, descriptor_for_path
from src.embeddings import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfiguration,
    EmbeddingModel,
    EmbeddingSnapshotError,
    EmbeddingValidationError,
    validate_embeddings,
    validate_model_snapshot,
    resolve_model_snapshot,
)
from src.vector_store import (
    IndexCompatibilityError,
    build_manifest,
    create_index,
    index_configuration_fingerprint,
    index_is_fresh,
    order_metadata_for_embedding,
    validate_runtime_index_compatibility,
)


def make_snapshot(root: Path, safetensors: bool = False) -> Path:
    snapshot = root / "snapshot"
    files = [
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors" if safetensors else "pytorch_model.bin",
    ]
    for relative in files:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return snapshot


class FakeEncoder:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            base = float(sum(ord(char) for char in text) % 7 + 1)
            vector = np.asarray([base, base + 1.0, base + 2.0], dtype=np.float32)
            if kwargs.get("normalize_embeddings"):
                vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


def fake_embedding_model(batch_size: int = 2) -> EmbeddingModel:
    model = EmbeddingModel.__new__(EmbeddingModel)
    model.configuration = EmbeddingConfiguration(
        model_name="fixture",
        revision="fixture-revision",
        dimension=3,
        batch_size=batch_size,
        normalize_embeddings=True,
        use_safetensors=False,
    )
    model.model = FakeEncoder()
    return model


class SnapshotValidationTests(unittest.TestCase):
    def test_complete_pytorch_snapshot_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_model_snapshot(make_snapshot(Path(tmpdir)))
        self.assertTrue(result["snapshot_complete"])
        self.assertEqual(result["weight_format"], "pytorch_bin")

    def test_incomplete_snapshot_is_rejected_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingSnapshotError, "snapshot incomplete"):
                validate_model_snapshot(snapshot)

    def test_safetensors_policy_requires_matching_weight_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = make_snapshot(Path(tmpdir), safetensors=False)
            config = replace(DEFAULT_EMBEDDING_CONFIG, use_safetensors=True)
            with self.assertRaisesRegex(EmbeddingSnapshotError, "model.safetensors"):
                validate_model_snapshot(snapshot, config)

    def test_mutable_remote_revision_is_rejected(self) -> None:
        config = replace(DEFAULT_EMBEDDING_CONFIG, revision="main")
        with self.assertRaisesRegex(EmbeddingSnapshotError, "immutable 40-character commit hash"):
            resolve_model_snapshot(config)


class EmbeddingOutputContractTests(unittest.TestCase):
    def test_valid_normalized_vectors_are_accepted(self) -> None:
        values = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        config = replace(DEFAULT_EMBEDDING_CONFIG, dimension=3)
        self.assertEqual(validate_embeddings(values, 1, config).shape, (1, 3))

    def test_wrong_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(EmbeddingValidationError, "dimension mismatch"):
            validate_embeddings(np.ones((1, 3), dtype=np.float32), 1)

    def test_nan_and_inf_are_rejected(self) -> None:
        config = replace(DEFAULT_EMBEDDING_CONFIG, dimension=3, normalize_embeddings=False)
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                values = np.asarray([[invalid, 0.0, 1.0]], dtype=np.float32)
                with self.assertRaisesRegex(EmbeddingValidationError, "NaN or infinite"):
                    validate_embeddings(values, 1, config)

    def test_non_unit_vectors_are_rejected_when_normalization_is_required(self) -> None:
        config = replace(DEFAULT_EMBEDDING_CONFIG, dimension=3)
        with self.assertRaisesRegex(EmbeddingValidationError, "Normalized embedding contract"):
            validate_embeddings(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32), 1, config)

    def test_same_text_and_different_batch_sizes_are_equivalent(self) -> None:
        texts = ["alpha", "beta", "alpha"]
        model = fake_embedding_model()
        first = model.embed_many(texts, batch_size=1)
        second = model.embed_many(texts, batch_size=3)
        np.testing.assert_allclose(first, second, atol=1e-7)
        np.testing.assert_allclose(first[0], first[2], atol=1e-7)


class OrderingAndCompatibilityTests(unittest.TestCase):
    def test_document_and_chunk_order_is_deterministic(self) -> None:
        rows = [
            {"relative_path": "b/B.pdf", "page": 1, "block_id": "b2", "chunk_id": "z", "text": "z"},
            {"relative_path": "a/A.pdf", "page": 2, "block_id": "b1", "chunk_id": "y", "text": "y"},
            {"relative_path": "a/A.pdf", "page": 1, "block_id": "b1", "chunk_id": "x", "text": "x"},
        ]
        forward = order_metadata_for_embedding(rows)
        reverse = order_metadata_for_embedding(list(reversed(rows)))
        self.assertEqual([row["chunk_id"] for row in forward], ["x", "y", "z"])
        self.assertEqual(forward, reverse)
        self.assertEqual([row["vector_row"] for row in forward], [0, 1, 2])

    def test_revision_normalization_and_dimension_change_fingerprint(self) -> None:
        base = DEFAULT_EMBEDDING_CONFIG
        values = {
            "corpus_fingerprint": "corpus",
            "chunker_schema_version": "chunker",
            "chunking_configuration_id": "chunks",
            "faiss_index_type": "IndexFlatIP",
            **base.compatibility_dict(),
        }
        baseline = index_configuration_fingerprint(values)
        for key, value in (
            ("embedding_revision", "different"),
            ("embedding_normalized", not base.normalize_embeddings),
            ("embedding_dimension", base.dimension + 1),
        ):
            self.assertNotEqual(baseline, index_configuration_fingerprint({**values, key: value}))

    def test_manifest_revision_change_marks_index_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document = root / "A.pdf"
            document.write_bytes(b"pdf")
            manifest = build_manifest([document], 1, DEFAULT_EMBEDDING_CONFIG.dimension)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(index_is_fresh([document], manifest_path=path))
            self.assertFalse(index_is_fresh(
                [document],
                manifest_path=path,
                expected_embedding_revision="different-revision",
            ))

    def test_query_runtime_refuses_revision_and_normalization_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document = root / "A.pdf"
            document.write_bytes(b"pdf")
            manifest = build_manifest([document], 1, DEFAULT_EMBEDDING_CONFIG.dimension)
            index = create_index(np.eye(1, DEFAULT_EMBEDDING_CONFIG.dimension, dtype=np.float32))
            for config in (
                replace(DEFAULT_EMBEDDING_CONFIG, revision="different"),
                replace(DEFAULT_EMBEDDING_CONFIG, normalize_embeddings=False),
            ):
                with self.subTest(config=config):
                    with self.assertRaisesRegex(IndexCompatibilityError, "must be rebuilt"):
                        validate_runtime_index_compatibility(index, manifest, config)

    def test_create_index_rejects_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            create_index(np.asarray([[np.nan, 0.0]], dtype=np.float32), normalized=False)


class BuildFailureSafetyTests(unittest.TestCase):
    def test_mid_embedding_failure_preserves_published_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / "documents"
            target = root / "vector_db"
            results = root / "results"
            document_dir.mkdir()
            target.mkdir()
            results.mkdir()
            pdf = document_dir / "A.pdf"
            pdf.write_bytes(b"pdf")
            descriptor = descriptor_for_path(pdf, document_dir)
            descriptor.ingestion_status = DocumentStatus.SUCCESS.value
            ingestion = IngestionResult([descriptor], [{"text": "safe text"}], [])
            originals = {
                "index.faiss": b"old-index",
                "metadata.pkl": b"old-metadata",
                "index_manifest.json": b"old-manifest",
            }
            for name, content in originals.items():
                (target / name).write_bytes(content)

            class FailingModel:
                def embed_many(self, texts, progress_callback=None):
                    raise RuntimeError("simulated failure at batch 2")

            chunk = {
                "text": "safe text",
                "document_id": descriptor.document_id,
                "source": descriptor.source,
                "relative_path": descriptor.relative_path,
                "page": 1,
                "chunk_id": "chunk-1",
                "block_id": "block-1",
                "word_start": 0,
                "word_end": 2,
            }
            with (
                patch.object(build_script, "DOCUMENT_DIR", document_dir),
                patch.object(build_script, "VECTOR_DB_DIR", target),
                patch.object(build_script, "INGESTION_REPORT_PATH", results / "ingestion.json"),
                patch.object(build_script, "BUILD_REPORT_PATH", results / "build.json"),
                patch.object(build_script, "ingest_documents", return_value=ingestion),
                patch.object(build_script, "chunk_pages", return_value=[chunk]),
                patch.object(build_script, "validate_chunk_quality", return_value={"ok": True}),
                patch.object(build_script, "get_embedding_model", return_value=FailingModel()),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    build_script.build_vector_index(force_rebuild=True)

            for name, content in originals.items():
                self.assertEqual((target / name).read_bytes(), content)
            report = json.loads((results / "build.json").read_text(encoding="utf-8"))
            self.assertFalse(report["build_success"])


if __name__ == "__main__":
    unittest.main()
