from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluator import evaluate_dataset, retrieval_framework_note


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_stdout()
    print(retrieval_framework_note())
    results = evaluate_dataset()
    print(f"Saved {len(results)} evaluation rows to results/evaluation_results.csv")
