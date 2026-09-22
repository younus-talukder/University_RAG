from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_step7c.py"
SPEC = importlib.util.spec_from_file_location("evaluate_step7c", SCRIPT)
step7c = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(step7c)


def dataset(count: int) -> list[dict[str, str]]:
    return [
        {
            "question_id": f"Q{index:03d}",
            "expected_language": "english",
            "question": f"Question {index}?",
            "reference_answer": "Reference",
        }
        for index in range(1, count + 1)
    ]


def fake_answer(*_args, **_kwargs):
    return {
        "answer": "Reference",
        "answer_strategy": "structured_exact",
        "generation_used": False,
        "generation_attempts": 0,
        "retry_used": False,
        "support_status": "supported",
        "target_language": "english",
        "language_validation_passed": True,
        "grounding_validation_passed": True,
        "grounding_validation_reason": "GENERATION_NOT_USED",
        "source": "doc.pdf",
        "page": 1,
        "supporting_excerpt": "Reference",
        "latency_seconds": {"retrieval": 0.1, "answering": 0.1, "generation": 0.0, "retry": 0.0, "total": 0.2},
    }


class Step7CEvaluatorTests(unittest.TestCase):
    def test_checkpoint_resume_has_no_duplicates_or_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(step7c, "answer_question", side_effect=fake_answer):
            path = Path(directory) / "checkpoint.csv"
            rows = dataset(10)
            first = step7c.evaluate(rows[:5], path)
            resumed = step7c.evaluate(step7c.rows_to_resume(rows, first), path, initial_rows=first)
            keys = {(row["question_id"], row["language"]) for row in resumed}
            self.assertEqual(len(resumed), 10)
            self.assertEqual(len(keys), 10)
            self.assertFalse(path.with_suffix(".csv.tmp").exists())

    def test_atomic_checkpoint_round_trip_preserves_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.csv"
            row = {
                "question_id": "Q001", "language": "english", "generation_used": False,
                "retry_used": False, "language_validation_passed": True,
                "runtime_language_validation_passed": True, "grounding_validation_passed": True,
                "fact_tokens_preserved": False, "fact_preservation_applicable": False,
                "answer_bank_enabled": False, "reranker_enabled": False, "generation_attempts": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "rss_bytes": 1,
                "available_ram_bytes": 2, "swap_used_bytes": 3, "automatic_precision": 0,
                "automatic_recall": 0, "automatic_f1": 0, "retrieval_seconds": 0,
                "answering_seconds": 0, "generation_seconds": 0, "retry_seconds": 0,
                "total_seconds": 0, "tokens_per_second": 0, "page": "", "fact_tokens_required": "[]",
            }
            step7c.write_csv(path, [row])
            loaded = step7c.read_checkpoint(path)
            self.assertEqual(list(row), list(loaded[0]))


if __name__ == "__main__":
    unittest.main()
