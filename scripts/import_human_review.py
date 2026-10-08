"""Validate actual human scores and report sample-only human metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.human_review import import_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blinding-key", type=Path)
    args = parser.parse_args()
    report = import_scores(args.results, args.reviews, args.output_dir,
                           blinding_key_path=args.blinding_key)
    print(f"Imported {report['overall']['n']} genuine review rows; "
          f"inter-rater agreement available={report['inter_rater_agreement']['available']}")


if __name__ == "__main__":
    main()
