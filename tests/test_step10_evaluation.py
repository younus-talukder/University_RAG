"""Step 10 benchmark provenance, generation-mode, and recovery guards."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src import config
from src.evaluation.manifest import model_identity
from src.evaluation.runner import checkpoint_records, prepare_run, run_benchmark
from src.evaluation.schema import expand, load_and_validate
from scripts.freeze_development_dataset import freeze


class Step10EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "questions.csv"
        with self.dataset.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", "english_question",
                                                      "ground_truth_answer_en", "ground_truth_status"])
            writer.writeheader()
            for number in range(3):
                writer.writerow({"question_id": f"A{number}",
                                 "english_question": f"Question {number}?",
                                 "ground_truth_answer_en": "A grounded answer.",
                                 "ground_truth_status": "verified"})

    def test_full_generation_configuration_and_pilot_classification(self):
        folder, _, manifest = prepare_run(self.dataset, self.root / "runs",
                                          languages=("english",), run_classification="PILOT")
        effective = json.loads((folder / "run_config.json").read_text())
        self.assertTrue(effective["use_generation"])
        self.assertEqual(effective["run_mode"], "full")
        self.assertEqual(manifest["run_classification"], "PILOT")
        self.assertIn("model_hashes", manifest["model_runtime"])
        self.assertIn("environment", manifest)
        self.assertIn("resource_preflight", manifest)

    def test_skip_generation_cannot_masquerade_as_full(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_run(self.dataset, self.root / "runs", languages=("english",),
                        use_generation=False, run_mode="full")
        folder, _, _ = prepare_run(self.dataset, self.root / "runs", languages=("english",),
                                   use_generation=False, run_mode="skip_generation")
        self.assertFalse(json.loads((folder / "run_config.json").read_text())["use_generation"])

    def test_final_run_lock_rejects_diagnostic_profile(self):
        for options in ({"use_generation": False, "run_mode": "skip_generation"},
                        {"strict": False}, {"limit": 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "FINAL_BENCHMARK"):
                prepare_run(self.dataset, self.root / "runs", languages=("english",),
                            run_classification="FINAL_BENCHMARK", **options)

    def test_final_run_invalidates_changed_dataset(self):
        def changing_inference(question, *, top_k, use_generation, use_answer_bank):
            if question == "Question 0?":
                with self.dataset.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
            return {"answerability_status": "INSUFFICIENT_EVIDENCE",
                    "answer": "Insufficient evidence."}

        with self.assertRaisesRegex(RuntimeError, "run lock changed"):
            run_benchmark(self.dataset, self.root / "runs", languages=("english",),
                          run_classification="FINAL_BENCHMARK", infer=changing_inference)
        folders = list((self.root / "runs").iterdir())
        self.assertEqual(len(folders), 1)
        manifest = json.loads((folders[0] / "run_manifest.json").read_text())
        self.assertFalse(manifest["run_complete"])
        self.assertTrue(manifest["run_invalidated"])

    def test_resume_after_generated_rows_without_duplicates(self):
        calls = []

        def generated(question, *, top_k, use_generation, use_answer_bank):
            self.assertTrue(use_generation)
            self.assertFalse(use_answer_bank)
            calls.append(question)
            return {"answerability_status": "SUPPORTED", "final_answer_allowed": True,
                    "answer": "A grounded answer.", "generation_used": True,
                    "generation_attempts": 2, "retry_used": True,
                    "requested_answer_strategy": "gguf_generation",
                    "answer_strategy": "gguf_generation",
                    "grounding_validation_passed": True,
                    "semantic_validation_passed": True,
                    "language_validation_passed": True}

        folder = run_benchmark(self.dataset, self.root / "runs", languages=("english",),
                               run_classification="PILOT", checkpoint_every=1,
                               stop_after=2, infer=generated)
        self.assertFalse(json.loads((folder / "run_manifest.json").read_text())["run_complete"])
        run_benchmark(self.dataset, self.root / "runs", languages=("english",),
                      run_classification="PILOT", checkpoint_every=1,
                      resume=folder, infer=generated)
        manifest = json.loads((folder / "run_manifest.json").read_text())
        rows = list(checkpoint_records(folder, manifest["compatibility_sha256"]))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["evaluation_id"] for row in rows}), 3)
        self.assertEqual(len(calls), 3)
        summary = json.loads((folder / "metrics_overall.json").read_text())
        self.assertTrue(summary["run_complete"])
        self.assertEqual(summary["overall"]["generation_required_cases"], 3)
        self.assertEqual(summary["overall"]["generation_attempts"], 6)
        self.assertEqual(summary["overall"]["successful_retries"], 3)

    def test_model_shard_sha256_provenance(self):
        one = self.root / "qwen-test-00001-of-00002.gguf"
        two = self.root / "qwen-test-00002-of-00002.gguf"
        one.write_bytes(b"first-shard")
        two.write_bytes(b"second-shard")
        settings = replace(config.load_generator_settings(), model_path=str(one))
        with patch("src.evaluation.manifest.config.load_generator_settings", return_value=settings):
            identity = model_identity()
        self.assertEqual(identity["model_hashes"][one.name], hashlib.sha256(one.read_bytes()).hexdigest())
        self.assertEqual(identity["model_hashes"][two.name], hashlib.sha256(two.read_bytes()).hexdigest())
        self.assertEqual(identity["model_shard_sizes_bytes"][one.name], len(one.read_bytes()))

    def test_development_freeze_is_immutable_and_auditable(self):
        dataset = self.root / "q007.csv"
        with dataset.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", "english_question",
                                                      "bangla_question", "banglish_question",
                                                      "ground_truth_answer_en", "expected_page",
                                                      "ground_truth_status"])
            writer.writeheader()
            writer.writerow({"question_id": "Q007", "english_question": "Which two programs?",
                             "bangla_question": "কোন দুইটি প্রোগ্রাম?",
                             "banglish_question": "Kon duita program?",
                             "ground_truth_answer_en": "Computer Science & Engineering",
                             "expected_page": "4", "ground_truth_status": "verified"})
        output = self.root / "freeze.json"
        manifest = freeze(dataset, output, annotation_version="v1-final",
                          q007_adjudication="page4-direct-approval")
        self.assertEqual(manifest["language_variant_count"], 3)
        self.assertEqual(manifest["ground_truth_status_distribution"], {"verified": 1})
        self.assertEqual(freeze(dataset, output, annotation_version="v1-final",
                                q007_adjudication="page4-direct-approval"), manifest)
        with self.assertRaisesRegex(ValueError, "already exists and differs"):
            freeze(dataset, output, annotation_version="v2", q007_adjudication="changed")

    def test_q007_confirmed_development_annotation_is_evaluation_only(self):
        rows, validation = load_and_validate(Path("data/questions/questions.csv"))
        self.assertTrue(validation["valid"])
        q007 = [row for row in expand(rows) if row["base_question_id"] == "Q007"]
        self.assertEqual({row["language"] for row in q007}, {"english", "bangla", "banglish"})
        self.assertEqual({tuple(row["expected_pages"]) for row in q007}, {("4",)})
        self.assertTrue(all("Computer Science & Engineering" in row["reference_answer"]
                            for row in q007))
        self.assertTrue(all("Computer Science & Technology" not in row["reference_answer"]
                            for row in q007))


if __name__ == "__main__":
    unittest.main()
