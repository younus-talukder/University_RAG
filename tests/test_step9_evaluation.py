from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch
from pathlib import Path

from openpyxl import Workbook

from src.evaluation.human_review import (agreement_statistics, create_sample,
                                         human_metrics, import_scores, validate_scores)
from src.evaluation.manifest import atomic_json, corpus_identity
from src.evaluation.metrics import (answer_proxies, group_summary, retrieval_scores,
                                    safe_return, wilson_interval, classify_error)
from src.evaluation.runner import checkpoint_records, infer_one, prepare_run, run_benchmark
from src.evaluation.schema import evidence_labels, expand, load_and_validate


def fake_response(question: str, *, top_k: int, use_generation: bool,
                  use_answer_bank: bool) -> dict:
    assert top_k == 3
    assert use_answer_bank is False
    return {
        "answerability_status": "SUPPORTED", "final_answer_allowed": True,
        "answer": "CSE 101 has 3 credits.", "detected_language": "english",
        "answer_strategy": "structured_exact", "grounding_validation_passed": True,
        "language_validation_passed": True, "semantic_validation_passed": True,
        "source": "curriculum.pdf", "page": 4, "document_id": "doc-1",
        "supporting_excerpt": "CSE 101 | 3 credits",
        "retrieved_context": [{"source": "curriculum.pdf", "page": 4,
                               "chunk_id": "c1", "text": "CSE 101 | 3 credits", "score": 0.8}],
        "latency_seconds": {"retrieval": 0.1, "generation": 0.0},
    }


class DatasetSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def csv_file(self, rows, fields=None):
        path = self.root / "questions.csv"
        fields = fields or list(rows[0])
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_csv_dynamic_optional_fields_and_hash(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "What is CSE 101?"}])
        rows, validation = load_and_validate(path)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["row_count"], 1)
        self.assertEqual(validation["language_variant_count"], 1)
        self.assertEqual(len(validation["dataset_sha256"]), 64)
        self.assertEqual(list(expand(rows))[0]["evaluation_id"], "A1:english")

    def test_xlsx_input(self):
        path = self.root / "questions.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["question_id", "english_question", "bangla_question", "expected_page"])
        sheet.append(["A1", "English?", "বাংলা?", 4])
        workbook.save(path)
        rows, validation = load_and_validate(path)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["language_variant_count"], 2)
        self.assertEqual([r["language"] for r in expand(rows)], ["english", "bangla"])

    def test_missing_question_text(self):
        path = self.csv_file([{"question_id": "A1", "english_question": ""}])
        _, validation = load_and_validate(path)
        self.assertFalse(validation["valid"])

    def test_duplicate_question_ids(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"},
                              {"question_id": "A1", "english_question": "Two?"}])
        _, validation = load_and_validate(path)
        self.assertFalse(validation["valid"])

    def test_reference_requirement(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"}])
        _, validation = load_and_validate(path, require_references=True)
        self.assertFalse(validation["valid"])

    def test_multiple_sources_pages_and_pairs(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?",
                               "expected_source": "a.pdf;b.pdf", "expected_page": "4;8",
                               "expected_evidence": '[{"source":"a.pdf","page":4}]'}])
        rows, validation = load_and_validate(path)
        self.assertTrue(validation["valid"])
        descriptor = list(expand(rows))[0]
        self.assertEqual(descriptor["expected_sources"], ["a.pdf", "b.pdf"])
        self.assertEqual(descriptor["expected_pages"], ["4", "8"])
        self.assertEqual(descriptor["expected_evidence"], [{"source": "a.pdf", "page": "4"}])

    def test_invalid_page(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?", "expected_page": "4;x"}])
        _, validation = load_and_validate(path)
        self.assertFalse(validation["valid"])

    def test_6000_descriptor_simulation(self):
        rows = [{"_base_question_id": f"S{number:05d}", "_annotation_status": "unlabeled",
                 "_expected_sources": [], "_expected_pages": [], "_expected_evidence": [],
                 "english_question": f"Question {number}?"} for number in range(6000)]
        descriptors = list(expand(rows))
        self.assertEqual(len(descriptors), 6000)
        self.assertEqual(len({row["evaluation_id"] for row in descriptors}), 6000)


class BenchmarkRunnerTests(DatasetSchemaTests):
    def test_partial_resume_and_artifacts(self):
        path = self.csv_file([
            {"question_id": f"A{number}", "english_question": f"CSE 101 question {number}?",
             "ground_truth_answer_en": "CSE 101 has 3 credits.",
             "expected_source": "curriculum.pdf", "expected_page": "4", "ground_truth_status": "verified"}
            for number in range(3)])
        captured = []
        def infer(question, **kwargs):
            captured.append((question, kwargs))
            return fake_response(question, **kwargs)
        run_dir = run_benchmark(path, self.root / "runs", languages=("english",),
                                checkpoint_every=1, stop_after=2, infer=infer)
        partial = json.loads((run_dir / "metrics_overall.json").read_text(encoding="utf-8"))
        self.assertFalse(partial["run_complete"])
        self.assertEqual(partial["completed_rows"], 2)
        self.assertEqual(len(list(checkpoint_records(run_dir,
            json.loads((run_dir / "run_manifest.json").read_text())["compatibility_sha256"]))), 2)
        resumed = run_benchmark(path, self.root / "runs", languages=("english",),
                                checkpoint_every=1, resume=run_dir, infer=infer)
        self.assertEqual(run_dir, resumed)
        self.assertEqual(len(captured), 3)
        self.assertEqual(len({item[0] for item in captured}), 3)
        self.assertTrue(json.loads((run_dir / "run_manifest.json").read_text())["run_complete"])
        uninterrupted = run_benchmark(path, self.root / "runs", languages=("english",),
                                      checkpoint_every=1, infer=fake_response)
        resumed_metrics = json.loads((run_dir / "metrics_overall.json").read_text())
        direct_metrics = json.loads((uninterrupted / "metrics_overall.json").read_text())
        self.assertEqual(resumed_metrics["overall"]["safe_answer_coverage"],
                         direct_metrics["overall"]["safe_answer_coverage"])
        self.assertEqual(resumed_metrics["overall"]["answerability_status_distribution"],
                         direct_metrics["overall"]["answerability_status_distribution"])
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        for name in ("run_config.json", "dataset_validation.json", "retrieval_trace.csv",
                     "metrics_by_language.csv", "multilingual_parity.csv", "benchmark_report.md"):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertTrue(all(set(kwargs) == {"top_k", "use_generation", "use_answer_bank"}
                            for _, kwargs in captured))

    def test_resume_rejects_dataset_change(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"}])
        run_dir = run_benchmark(path, self.root / "runs", languages=("english",), infer=fake_response)
        path.write_text(path.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe resume"):
            run_benchmark(path, self.root / "runs", languages=("english",),
                          resume=run_dir, infer=fake_response)

    def test_resume_rejects_configuration_change(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"}])
        run_dir = run_benchmark(path, self.root / "runs", languages=("english",),
                                checkpoint_every=1, infer=fake_response)
        with self.assertRaisesRegex(ValueError, "unsafe resume"):
            run_benchmark(path, self.root / "runs", languages=("english",),
                          checkpoint_every=2, resume=run_dir, infer=fake_response)

    def test_strict_profile_rejects_reranker(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"}])
        with patch("src.config.RERANKER_ENABLED", True), self.assertRaisesRegex(ValueError, "reranker OFF"):
            prepare_run(path, self.root / "runs", languages=("english",))

    def test_duplicate_checkpoint_id_is_rejected(self):
        path = self.csv_file([{"question_id": "A1", "english_question": "One?"},
                              {"question_id": "A2", "english_question": "Two?"}])
        run_dir = run_benchmark(path, self.root / "runs", languages=("english",),
                                checkpoint_every=1, infer=fake_response)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        second = run_dir / "checkpoint" / "batch-000002.json"
        payload = json.loads(second.read_text(encoding="utf-8"))
        payload["records"][0]["evaluation_id"] = "A1:english"
        atomic_json(second, payload)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            list(checkpoint_records(run_dir, manifest["compatibility_sha256"]))

    def test_error_isolation(self):
        descriptor = {"evaluation_id": "A1:english", "base_question_id": "A1",
                      "language": "english", "question": "One?", "reference_answer": "Answer",
                      "expected_sources": [], "expected_pages": [], "expected_evidence": [],
                      "annotation_status": "verified", "intent": "", "difficulty": "", "course_code": ""}
        def broken(*args, **kwargs):
            raise RuntimeError("one-row failure")
        result = infer_one(descriptor, "run", infer=broken)
        self.assertEqual(result["answerability_status"], "SYSTEM_ERROR")
        self.assertEqual(result["error_type"], "RuntimeError")


class MetricsAndReviewTests(unittest.TestCase):
    def test_retrieval_multiple_labels_and_not_available(self):
        row = {"question": "What is CSE 101 credit?", "expected_sources": ["a.pdf", "b.pdf"],
               "expected_pages": ["4", "8"], "expected_evidence": [],
               "retrieval_top3": [{"source": "b.pdf", "page": 8,
                                   "text": "CSE 101 credit 3"}]}
        scored = retrieval_scores(row)
        self.assertTrue(scored["hit_1"])
        self.assertTrue(scored["page_accuracy"])
        row["expected_sources"] = row["expected_pages"] = []
        self.assertEqual(retrieval_scores(row)["hit_1"], "NOT_AVAILABLE")

    def test_paired_evidence_prevents_cross_pair_false_hit(self):
        row = {"question": "Which page?", "expected_sources": ["a.pdf", "b.pdf"],
               "expected_pages": ["4", "8"],
               "expected_evidence": [{"source": "a.pdf", "page": "4"},
                                     {"source": "b.pdf", "page": "8"}],
               "retrieval_top3": [{"source": "a.pdf", "page": 8, "text": "unrelated"}]}
        self.assertFalse(retrieval_scores(row)["hit_1"])

    def test_abstention_not_scored_as_answer(self):
        row = {"answer_returned": False, "reference_answer": "3 credits",
               "annotation_status": "verified", "final_answer": "Insufficient evidence"}
        self.assertEqual(answer_proxies(row)["token_f1"], "NOT_AVAILABLE")
        self.assertFalse(safe_return(row))

    def test_unsafe_return_is_not_safe_coverage(self):
        row = {"answer_returned": True, "grounding_status": False,
               "semantic_status": True, "language_status": True,
               "answer_safety_failures": []}
        self.assertFalse(safe_return(row))

    def test_generation_disabled_is_not_a_grounding_failure(self):
        row = {"answerability_status": "GENERATION_REJECTED", "generation_attempts": 0,
               "grounding_status": False, "annotation_status": "verified"}
        self.assertEqual(classify_error(row, {}), "UNKNOWN")

    def test_wilson_interval(self):
        interval = wilson_interval(8, 10)
        self.assertEqual(interval["n"], 10)
        self.assertLess(interval["lower"], 0.8)
        self.assertGreater(interval["upper"], 0.8)

    def test_sample_and_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sample.csv"
            rows = [{"evaluation_id": f"A{number}:{lang}", "language": lang,
                     "question": "Question?", "final_answer": "Answer",
                     "answerability_status": "SUPPORTED" if number % 2 else "INSUFFICIENT_EVIDENCE",
                     "answer_strategy": "structured_exact" if number % 2 else "unsupported",
                     "intent": "test", "difficulty": "easy"}
                    for lang in ("english", "bangla", "banglish") for number in range(10)]
            sample = create_sample(rows, output, sample_size=30, seed=5)
            self.assertEqual(len(sample), 30)
            self.assertEqual(Counter(row["language"] for row in sample),
                             Counter({"english": 10, "bangla": 10, "banglish": 10}))
            self.assertTrue(all(row["correctness"] == "" for row in sample))
            scored = [{"evaluation_id": "A1:english", "reviewer_id": "r1", "correctness": "5",
                       "semantic_consistency": "PASS", "overall_acceptability": "ACCEPT"}]
            valid = validate_scores(scored, {"A1:english"})
            self.assertEqual(human_metrics(valid)["mean_correctness"], 5)
            self.assertFalse(agreement_statistics(valid)["available"])
            with self.assertRaises(ValueError):
                validate_scores([{**scored[0], "correctness": "6"}], {"A1:english"})

    def test_multi_reviewer_agreement(self):
        rows = [{"evaluation_id": "A1", "reviewer_id": "r1", "correctness": "5",
                 "overall_acceptability": "ACCEPT"},
                {"evaluation_id": "A1", "reviewer_id": "r2", "correctness": "5",
                 "overall_acceptability": "ACCEPT"}]
        result = agreement_statistics(rows)
        self.assertTrue(result["available"])
        self.assertEqual(result["fields"]["overall_acceptability"]["percent_agreement"], 1)

    def test_blinded_review_key_and_import(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            results = folder / "results.csv"
            with results.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["evaluation_id", "language", "answer_strategy"])
                writer.writeheader()
                writer.writerow({"evaluation_id": "A1:english", "language": "english",
                                 "answer_strategy": "structured_exact"})
            review = folder / "review.csv"
            key = folder / "key.csv"
            create_sample([{"evaluation_id": "A1:english", "language": "english",
                            "question": "Question?", "final_answer": "Answer"}],
                          review, sample_size=1, blinded=True, blinding_key=key)
            with review.open(encoding="utf-8-sig", newline="") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertNotIn("evaluation_id", row)
            self.assertNotIn("answer_strategy", row)
            row["correctness"] = "5"
            row["overall_acceptability"] = "ACCEPT"
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            report = import_scores(results, review, folder, blinding_key_path=key)
            self.assertEqual(report["overall"]["mean_correctness"], 5)
            self.assertFalse(report["inter_rater_agreement"]["available"])

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            atomic_json(path, {"one": 1})
            atomic_json(path, {"two": 2})
            self.assertEqual(json.loads(path.read_text()), {"two": 2})
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_corpus_manifest(self):
        corpus = corpus_identity()
        self.assertEqual(corpus["chunk_count"], 491)
        self.assertEqual(corpus["document_count"], 1)


if __name__ == "__main__":
    unittest.main()
