"""Exercise 6,000 descriptor/checkpoint bookkeeping without inference results."""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.manifest import atomic_json, corpus_identity, model_identity
from src.evaluation.schema import expand


def simulate(count: int = 6000, checkpoint_every: int = 10) -> dict:
    if count < 1 or checkpoint_every < 1:
        raise ValueError("count and checkpoint interval must be positive")
    tracemalloc.start()
    try:
        rows = ({"_base_question_id": f"S{number:06d}", "_annotation_status": "unlabeled",
                 "_expected_sources": [], "_expected_pages": [], "_expected_evidence": [],
                 "english_question": f"Synthetic descriptor {number}?"}
                for number in range(count))
        # Bookkeeping only: no runtime model call, answer, or fabricated result.
        ids = []
        batches = 0
        for row in rows:
            for descriptor in expand([row], ("english",)):
                ids.append(descriptor["evaluation_id"])
                if len(ids) % checkpoint_every == 0:
                    batches += 1
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if len(ids) != len(set(ids)):
        raise ValueError("descriptor IDs are not unique")
    return {
        "simulation_only": True, "inference_results_fabricated": False,
        "descriptors": len(ids), "unique_evaluation_ids": len(set(ids)),
        "planned_full_checkpoint_batches": batches + bool(len(ids) % checkpoint_every),
        "peak_python_allocation_bytes": peak,
        "corpus_schema_has_document_count": "document_count" in corpus_identity(),
        "model_config_recorded": bool(model_identity().get("generator_model_name")),
        "note": "Schema and bookkeeping simulation only; no 6,000-question or 70-PDF benchmark was run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "step9_scale_simulation.json")
    args = parser.parse_args()
    report = simulate(args.count, args.checkpoint_every)
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
