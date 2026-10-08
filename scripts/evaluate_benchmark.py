"""Dataset-agnostic, sequential benchmark entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.runner import run_benchmark
from src.evaluation.schema import LANGUAGES


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", type=Path)
    command.add_argument("--output-dir", type=Path, default=ROOT / "results" / "benchmarks")
    command.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    command.add_argument("--limit", type=int)
    command.add_argument("--resume", type=Path, help="Existing run directory; mismatched provenance is rejected.")
    command.add_argument("--run-name", default="benchmark")
    command.add_argument("--checkpoint-every", type=int, default=10)
    command.add_argument("--config", type=Path, help="Optional JSON defaults; explicit CLI flags override them.")
    command.add_argument("--human-sample-size", type=int, default=30)
    command.add_argument("--skip-generation", action="store_true")
    command.add_argument("--retrieval-only", action="store_true")
    command.add_argument("--classification", choices=("DEVELOPMENT", "PILOT", "FINAL_BENCHMARK"),
                         default="DEVELOPMENT")
    command.add_argument("--non-strict", action="store_true", help="Label a non-strict diagnostic run.")
    command.add_argument("--require-references", action="store_true")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--stop-after", type=int, help="Stop after N new rows for a resumability test.")
    return command


def main(argv: list[str] | None = None) -> int:
    command = parser()
    inputs = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    preliminary, _ = bootstrap.parse_known_args(inputs)
    if preliminary.config:
        values = json.loads(preliminary.config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("--config must contain a JSON object")
        allowed = {action.dest for action in command._actions}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        command.set_defaults(**values)
    args = command.parse_args(inputs)
    if args.dataset is None:
        command.error("--dataset is required (or provide it in --config)")
    if args.skip_generation and args.retrieval_only:
        command.error("choose only one of --skip-generation and --retrieval-only")
    if args.classification == "FINAL_BENCHMARK" and args.non_strict:
        command.error("FINAL_BENCHMARK requires the strict profile")
    run_dir = run_benchmark(
        Path(args.dataset), Path(args.output_dir), languages=tuple(dict.fromkeys(args.languages)),
        limit=args.limit, run_name=args.run_name, checkpoint_every=args.checkpoint_every,
        use_generation=not (args.skip_generation or args.retrieval_only),
        run_classification=args.classification, strict=not args.non_strict,
        require_references=args.require_references,
        resume=Path(args.resume) if args.resume else None,
        seed=args.seed, stop_after=args.stop_after,
        human_sample_size=args.human_sample_size,
        run_mode="retrieval_only" if args.retrieval_only else
                 ("skip_generation" if args.skip_generation else "full"),
    )
    print(f"Benchmark artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
