"""Gate-only evaluation harness (development tool, never used at serving time).

Runs the production entry point ``src.pipeline.answer_question`` for every
variant and stops at the first generator call. All retrieval, passage
selection, evidence and answerability gates are the pipeline's own functions;
this script only observes them:

* ``generate_canonical_answer`` is replaced by a stub that raises
  ``GenerationReached`` (a BaseException, so the pipeline's error handling
  cannot turn it into an answer). Deterministic (template / structured)
  answers never reach the stub and complete normally.
* The Qwen retrieval-rewrite used by the Bangla/Banglish fallback is replayed
  from a cache built once with ``--build-rewrite-cache`` (the real model,
  temperature 0), so a gate-only run never loads the 7B generator.
* ``Retriever.retrieve``, ``assess_answerability`` and
  ``build_verified_evidence_package`` are wrapped to record their results;
  ``_repair_trace_hook`` records which repairs acted.

Output: one JSON file per variant in --out-dir (existing files are skipped, so
the run is resumable). Repair flags are passed as RAG_REPAIR_<NAME>=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANG_COLUMNS = {"english": "english_question", "bangla": "bangla_question", "banglish": "banglish_question"}


class GenerationReached(BaseException):
    def __init__(self, package):
        super().__init__("generation reached")
        self.package = package


def _slim(item: dict, rank: int) -> dict:
    return {"rank": rank, "chunk_id": item.get("chunk_id"), "source": item.get("source"),
            "page": item.get("page"), "score": round(float(item.get("score") or 0.0), 6)}


def _variants(dataset: Path, ids: set[str] | None):
    with dataset.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            qid = row["question_id"].strip()
            for language, column in LANG_COLUMNS.items():
                eid = f"{qid}:{language}"
                if ids and eid not in ids:
                    continue
                yield eid, language, row[column], row


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def build_rewrite_cache(args) -> None:
    from src.crosslingual import deterministic_query
    from src.generator import rewrite_query_for_retrieval
    cache_path = Path(args.rewrite_cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for eid, language, question, _ in _variants(Path(args.dataset), None):
        if language == "english" or question in cache or deterministic_query(question):
            continue
        started = time.perf_counter()
        cache[question] = rewrite_query_for_retrieval(question)
        _atomic_json(cache_path, cache)
        print(json.dumps({"eid": eid, "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    print("REWRITE_CACHE_DONE", len(cache), flush=True)


def run_gate_eval(args) -> None:
    import src.pipeline as pipeline
    from src.repair_flags import enabled_repairs
    from src.retriever import Retriever

    cache = json.loads(Path(args.rewrite_cache).read_text(encoding="utf-8")) if args.rewrite_cache else {}
    capture: dict = {}

    def replay_rewrite(question: str) -> str:
        capture["rewrite_cache_hit"] = question in cache
        return cache.get(question, "")

    def stop_generation(package, *unused, **unused_kw):
        raise GenerationReached(package)

    original_assess = pipeline.assess_answerability
    original_package = pipeline.build_verified_evidence_package
    original_retrieve = Retriever.retrieve

    def record_assess(*a, **k):
        decision = original_assess(*a, **k)
        capture["decision"] = decision.to_dict()
        return decision

    def record_package(question, language, assessment, retrieved=()):
        capture["selected"] = [_slim(item, rank) for rank, item in enumerate(retrieved, 1)]
        capture["verified"] = [{"chunk_id": e.chunk_id, "source": e.source, "page": e.page,
                                "continuation_chunk_ids": list(getattr(e, "continuation_chunk_ids", ()) or ()),
                                "excerpt": e.excerpt} for e in assessment.evidence]
        return original_package(question, language, assessment, retrieved)

    def record_retrieve(self, query, *a, **k):
        pool = original_retrieve(self, query, *a, **k)
        capture.setdefault("pools", []).append({"query": query, "items": [_slim(i, r) for r, i in enumerate(pool, 1)]})
        return pool

    pipeline._repair_trace_hook = lambda trace: capture.__setitem__("repair_trace", dict(trace))
    pipeline.rewrite_query_for_retrieval = replay_rewrite
    pipeline.generate_canonical_answer = stop_generation
    pipeline.assess_answerability = record_assess
    pipeline.build_verified_evidence_package = record_package
    Retriever.retrieve = record_retrieve

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = list(enabled_repairs())
    ids = set(args.ids.split(",")) if args.ids else None
    done = 0
    for eid, language, question, row in _variants(Path(args.dataset), ids):
        target = out_dir / (eid.replace(" ", "_").replace(":", "__") + ".json")
        if target.exists():
            continue
        capture.clear()
        started = time.perf_counter()
        record = {"evaluation_id": eid, "language": language, "question": question, "flags": flags,
                  "expected_source": row.get("expected_source"), "expected_page": row.get("expected_page")}
        try:
            response = pipeline.answer_question(question, top_k=3, use_generation=True, use_answer_bank=False)
            record.update(outcome="FINAL", final_status=response.get("answerability_status") or response.get("final_status"),
                          answer=response.get("answer"), answer_strategy=response.get("answer_strategy"),
                          generation_used=bool(response.get("generation_used")),
                          deterministic_answer=bool(response.get("answerability_status") == "SUPPORTED" and not response.get("generation_used")),
                          cited={"source": response.get("source"), "page": response.get("page"), "chunk_id": response.get("chunk_id")},
                          trace={k: response.get(k) for k in ("pre_generation_support_status", "pre_generation_support_reason",
                                 "answerability_reason", "crosslingual_gate_reason", "bounded_evidence_widened",
                                 "metadata_evidence_widened", "same_page_owner_widened", "repair_trace",
                                 "final_candidate_chunk_ids", "runtime_error")})
            if not capture.get("selected"):
                capture["selected"] = [_slim(i, r) for r, i in enumerate(response.get("retrieved_context") or [], 1)]
                capture["verified"] = [{k: e.get(k) for k in ("chunk_id", "source", "page", "continuation_chunk_ids", "excerpt")}
                                       for e in response.get("evidence") or []]
        except GenerationReached as reached:
            record.update(outcome="GENERATION_REQUIRED", final_status="GENERATION_REQUIRED", deterministic_answer=False,
                          package=reached.package.to_dict())
        record.update(decision=capture.get("decision"), pools=capture.get("pools"), selected=capture.get("selected"),
                      verified=capture.get("verified"), repair_trace=capture.get("repair_trace"), rewrite_cache_hit=capture.get("rewrite_cache_hit"),
                      seconds=round(time.perf_counter() - started, 3))
        _atomic_json(target, record)
        done += 1
        print(json.dumps({"eid": eid, "outcome": record["outcome"], "status": record["final_status"],
                          "gate": (record["decision"] or {}).get("answerability_status"), "s": record["seconds"]}, ensure_ascii=False), flush=True)
    print("GATE_EVAL_DONE", done, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--rewrite-cache", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--build-rewrite-cache", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.build_rewrite_cache:
        build_rewrite_cache(args)
    else:
        run_gate_eval(args)


if __name__ == "__main__":
    main()
