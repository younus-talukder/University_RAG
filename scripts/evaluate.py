from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate the chatbot against data/questions/questions.csv.")
    parser.add_argument(
        "--mode",
        choices=("fast", "gguf"),
        default="fast",
        help="Evaluation mode: fast extraction or local GGUF generation. Default: fast.",
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Deprecated alias for --mode gguf.",
    )
    parser.add_argument(
        "--answer-bank",
        action="store_true",
        help="Allow curated answer-bank lookup. Disabled by default for controlled thesis evaluation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N question variants.")
    args = parser.parse_args()

    use_generation = args.generation or args.mode == "gguf"
    print(retrieval_framework_note())
    print(f"Evaluation mode: {'gguf' if use_generation else 'fast'}")
    print(f"Answer bank enabled: {args.answer_bank}")
    results = evaluate_dataset(
        use_generation=use_generation,
        use_answer_bank=args.answer_bank,
        limit=args.limit,
    )
    print(f"Saved {len(results)} evaluation rows to results/evaluation_results.csv")
