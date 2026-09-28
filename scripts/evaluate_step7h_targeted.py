"""Checkpointed Step 7H gate: frozen English flags, human rejects, 7G cases."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_step7g_full as base
from src.course_rows import course_field_value

OUTPUT = ROOT / "results/step7h_targeted_results.csv"
SUMMARY = ROOT / "results/step7h_targeted_summary.json"


def fingerprint() -> str:
    digest = hashlib.sha256(base.run_fingerprint().encode())
    for relative in ("src/answer_policy.py", "src/course_rows.py", "src/grounding_validator.py",
                     "src/semantic_contract.py", "scripts/evaluate_step7h_targeted.py"):
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def run() -> None:
    old = base.read_csv(ROOT / "results/step7g_full_300_results.csv")
    old_by_key = {(row["question_id"], row["language"]): row for row in old}
    flagged = [(row["question_id"], "english") for row in old if row["language"] == "english"
               and row["final_status"] == "ANSWER_RETURNED" and not base.truth(row["grounding_validation_passed"])]
    if len(flagged) != 15:
        raise RuntimeError(f"Expected exactly 15 frozen English flags, got {len(flagged)}")
    human = [("Q030", "english"), ("Q006", "banglish"), ("Q023", "banglish")]
    nine = [(row["question_id"], row["language"]) for row in base.read_csv(ROOT / "results/step7g_targeted_results.csv")]
    if len(nine) != 9:
        raise RuntimeError(f"Expected nine Step-7G cases, got {len(nine)}")
    keys = list(dict.fromkeys(flagged + human + nine))
    dataset = {(row["question_id"], row["expected_language"]): row for row in base.load_dataset()}
    current = fingerprint()
    existing = base.read_csv(OUTPUT)
    if existing and any(row["step7h_fingerprint"] != current for row in existing):
        raise RuntimeError("Step-7H checkpoint differs from current code/index; preserve it and start a new checkpoint explicitly")
    completed = {(row["question_id"], row["language"]): row for row in existing}
    for key in keys:
        if key in completed:
            continue
        previous = old_by_key[key]
        row = base.evaluate(dataset[key], previous["expected_page_offline"], current)
        row.update({
            "step7h_fingerprint": current,
            "gate_groups": ";".join(group for group, members in (("english_flag", flagged), ("human_reject", human), ("step7g_nine", nine)) if key in members),
            "previous_final_status": previous["final_status"],
            "previous_answer": previous["final_answer"],
            "previous_grounding_passed": previous["grounding_validation_passed"],
            "previous_grounding_reason": previous["grounding_validation_reason"],
            "answer_changed": row["final_answer"] != previous["final_answer"],
        })
        completed[key] = row
        base.write_csv(OUTPUT, [completed[item] for item in keys if item in completed])
        print(f"{len(completed)}/{len(keys)} {key[1]} {key[0]} {row['final_status']} {row['answer_strategy']} {row['total_seconds']:.1f}s", flush=True)

    failures: list[str] = []
    for key in keys:
        row = completed[key]
        returned = row["final_status"] == "ANSWER_RETURNED"
        if returned and (not base.truth(row["grounding_validation_passed"]) or not base.truth(row["language_validation_passed"])):
            failures.append(f"{key}: returned despite failed validation")
        if row["runtime_error"]:
            failures.append(f"{key}: runtime error {row['runtime_error']}")
    for key in flagged:
        row = completed[key]
        old_row = old_by_key[key]
        if key[0] == "Q030":
            if row["final_status"] == "ANSWER_RETURNED" and not (row["relation_type"] == "repeat_course_maximum" and "4" in row["final_answer"]):
                failures.append("Q030: wrong repeat-course count")
        elif key[0] == "Q096":
            if row["final_status"] != "ANSWER_RETURNED" or "CHEM 112" not in row["final_answer"] or not base.truth(row["grounding_validation_passed"]):
                failures.append("Q096: prerequisite label/ownership unresolved")
        else:
            value = course_field_value(old_row["question"], "credits", [old_row["supporting_excerpt"]])
            if not value or row["final_status"] != "ANSWER_RETURNED" or value not in row["final_answer"] or not base.truth(row["grounding_validation_passed"]):
                failures.append(f"{key[0]}: correct credit false flag unresolved")
    for key in human:
        row = completed[key]
        if row["final_status"] != "ANSWER_RETURNED":
            continue
        answer = row["final_answer"].casefold()
        if key[0] == "Q006" and ("foundation" in answer or "1995" in answer):
            failures.append("Q006 Banglish: unrelated entity claim remains")
        if key[0] == "Q023" and not ("protiti course" in answer or "proti course" in answer):
            failures.append("Q023 Banglish: every-course qualifier missing")
    for key in nine:
        row = completed[key]
        if row["final_status"] != "ANSWER_RETURNED" or row["answer_strategy"] != "semi_structured_relation":
            failures.append(f"{key}: Step-7G relation regression")
    report = {"completed": len(completed), "expected": len(keys), "english_flags": len(flagged),
              "human_rejects": human, "step7g_cases": nine, "failures": failures,
              "gate_passed": not failures and len(completed) == len(keys), "fingerprint": current}
    SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
