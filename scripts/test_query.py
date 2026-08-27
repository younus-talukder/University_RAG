from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import TOP_K
from src.pipeline import answer_question


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_test_query(question: str, top_k: int = TOP_K, use_generation: bool = True) -> None:
    _configure_stdout()
    print(f"Question: {question}\n")

    result = answer_question(question, top_k=top_k, use_generation=use_generation)
    print(f"Detected Language: {result.get('detected_language')}\n")
    print(f"Mode: {result.get('generation_mode')}\n")
    print("Answer:")
    print(result.get("answer", ""))

    print("\nSources:")
    for i, item in enumerate(result.get("sources", []), start=1):
        print(f"{i}. {item.get('source', 'unknown')} - Page {item.get('page', 'unknown')}")

    print("\nRetrieved Chunks:")
    for i, item in enumerate(result.get("retrieved_context", []), start=1):
        score = item.get("score", 0.0)
        source = item.get("source", "unknown")
        page = item.get("page", "unknown")
        text = item.get("text", "")
        print(f"\n[{i}] score={score:.4f} | source={source} | page={page}")
        print(text[:500])

    print("\nEND")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one question through the RAG pipeline.")
    parser.add_argument("question", nargs="*", help="Question text to ask.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of chunks to retrieve. Default: {TOP_K}.")
    parser.add_argument("--fast", action="store_true", help="Use fast extractive mode instead of local LLM generation.")
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    elif not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    else:
        try:
            question = input("Question: ").strip()
        except EOFError:
            question = ""

    if not question:
        parser.error("question is required")

    run_test_query(question, top_k=args.top_k, use_generation=not args.fast)


if __name__ == "__main__":
    main()
