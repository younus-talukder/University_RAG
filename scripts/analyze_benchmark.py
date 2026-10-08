"""Rebuild metrics and report from complete or partial checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import analyze_run
from src.evaluation.runner import export_records
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--human-sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    export_records(args.run_dir, manifest["compatibility_sha256"])
    summary = analyze_run(args.run_dir, human_sample_size=args.human_sample_size, seed=args.seed)
    print(f"{summary['completed_rows']}/{summary['expected_rows']} run_complete={summary['run_complete']}")


if __name__ == "__main__":
    main()
