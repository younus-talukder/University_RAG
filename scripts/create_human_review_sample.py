"""Create a stratified human-review CSV; ratings remain blank."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.human_review import create_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blinded", action="store_true")
    parser.add_argument("--blinding-key", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.results:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            for row in csv.DictReader(handle):
                row["system_id"] = path.stem if len(args.results) > 1 else ""
                rows.append(row)
    if len(args.results) > 1 and not args.blinded:
        parser.error("multi-system review must use --blinded")
    sample = create_sample(rows, args.output, sample_size=args.sample_size,
                           seed=args.seed, blinded=args.blinded,
                           blinding_key=args.blinding_key)
    print(f"Created {len(sample)} unrated review rows at {args.output}")


if __name__ == "__main__":
    main()
