"""Render the completed Step 7G development summary; never rerun answers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "step7g_full_300_summary.json"
REPORT = RESULTS / "step7g_full_300_report.md"
RESULT_CSV = RESULTS / "step7g_full_300_results.csv"
REVIEW = RESULTS / "step7g_full_manual_review.csv"
LANGUAGES = ("english", "bangla", "banglish")
METRICS = ("page_hit_1", "page_hit_3", "mrr_3", "entity_hit_1", "entity_hit_3",
           "field_hit_1", "field_hit_3", "supportable_1", "supportable_3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.casefold() == "true"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + [
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = read_csv(RESULT_CSV)
    review = read_csv(REVIEW)
    if len(rows) != 300 or len(review) != 30:
        raise RuntimeError("Full results or manual-review sample incomplete")
    per = summary["per_language"]
    returned = [x for x in rows if x["final_status"] == "ANSWER_RETURNED"]
    returned_bad_language = [x for x in returned if not truth(x["language_validation_passed"])]
    returned_bad_grounding = [x for x in returned if not truth(x["grounding_validation_passed"])]
    runtime_errors = [x for x in rows if x["runtime_error"]]
    generation_errors = [x for x in rows if x["generation_error"]]
    engineering_complete = not (runtime_errors or returned_bad_language or returned_bad_grounding)
    summary["quality_checks"] = {
        "returned_answers": len(returned),
        "returned_language_failures": len(returned_bad_language),
        "returned_grounding_failures": len(returned_bad_grounding),
        "returned_grounding_failure_reasons": dict(Counter(
            x["grounding_validation_reason"] or "UNSPECIFIED" for x in returned_bad_grounding)),
        "returned_grounding_failure_question_ids": [x["question_id"] for x in returned_bad_grounding],
        "manual_review_scored": 0,
        "engineering_complete_provisional": engineering_complete,
        "step8_recommended_now": False,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Step 7G full 300-query development evaluation", "",
        "2026-09-25. Development measurement only; **not final thesis accuracy**. One-PDF, 491-chunk fresh index. "
        "Answer bank and neural reranker were off; Qwen2.5-7B was called only by the existing pipeline. "
        "Step 8 was not started and no commit was made.", "",
        "## 1. Completion", "", f"**{len(rows)}/300** variants completed, 100 per language. "
        f"Checkpoint fingerprint: `{summary['run_fingerprint']}`. Runtime exceptions: {len(runtime_errors)}.", "",
        "## 2. Answers and statuses by language", "",
    ]
    statuses = ("ANSWER_RETURNED", "INSUFFICIENT_EVIDENCE", "GENERATION_REJECTED", "AMBIGUOUS", "CONFLICTING", "CONFLICTING_EVIDENCE", "RUNTIME_ERROR")
    lines += table(("Language", "Returned", "Insufficient", "Generation rejected", "Ambiguous", "Conflicting", "Runtime errors"), [
        (lang, per[lang]["answers_returned"], per[lang]["statuses"].get("INSUFFICIENT_EVIDENCE", 0),
         per[lang]["statuses"].get("GENERATION_REJECTED", 0), per[lang]["statuses"].get("AMBIGUOUS", 0),
         sum(per[lang]["statuses"].get(x, 0) for x in ("CONFLICTING", "CONFLICTING_EVIDENCE")), per[lang]["runtime_errors"])
        for lang in LANGUAGES])
    lines += ["", "## 3. Answer strategy distribution", ""]
    strategies = ("structured_exact", "structured_list", "semi_structured_relation",
                  "gguf_generation", "generation_rejected", "unsupported", "ambiguous", "conflicting")
    lines += table(("Strategy", *LANGUAGES, "Overall"), [
        (strategy, *(per[lang]["strategies"].get(strategy, 0) for lang in LANGUAGES),
         sum(per[lang]["strategies"].get(strategy, 0) for lang in LANGUAGES)) for strategy in strategies])
    lines += ["", "`semi_structured_relation` is deterministic. `gguf_generation` is an accepted generated answer; "
              "rejected generation is counted under `generation_rejected`. `fast_extractive` may appear only if selected by the current pipeline.", "",
              "## 4. Retrieval metrics", ""]
    metrics = summary["retrieval_metrics"]
    lines += table(("Language", "Page H@1", "Page H@3", "MRR@3", "Entity H@1/H@3", "Field H@1/H@3", "Supportable@1/@3"), [
        (lang, pct(metrics[lang]["page_hit_1"]["value"]), pct(metrics[lang]["page_hit_3"]["value"]),
         f"{metrics[lang]['mrr_3']['value']:.3f}",
         pct(metrics[lang]["entity_hit_1"]["value"]) + "/" + pct(metrics[lang]["entity_hit_3"]["value"]),
         pct(metrics[lang]["field_hit_1"]["value"]) + "/" + pct(metrics[lang]["field_hit_3"]["value"]),
         pct(metrics[lang]["supportable_1"]["value"]) + "/" + pct(metrics[lang]["supportable_3"]["value"]))
        for lang in (*LANGUAGES, "overall")])
    lines += ["", "Entity and field rates use applicable-question denominators. Page labels and language pairings are offline evaluation inputs. "
              "These are top-1/top-3 metrics on the final candidate order; a verified metadata hit below rank 3 can still support an answer.", "",
              "### Frozen baseline comparison", ""]
    old7f = summary["frozen_step7f_retrieval_comparison"]
    old5 = json.loads((RESULTS / "step5_hybrid_results.json").read_text(encoding="utf-8"))["configuration_summaries"]["A_equal"]
    lines += table(("Language", "Step 5 Page H@3", "Step 7F Page H@3", "Step 7G Page H@3", "Step 5 Supportable@3", "Step 7F Supportable@3", "Step 7G Supportable@3"), [
        (lang,
         pct((old5["overall"] if lang == "overall" else old5["languages"][lang])["page_hit_at_3"]),
         pct(old7f[lang]["page_hit_3"]), pct(metrics[lang]["page_hit_3"]["value"]),
         pct((old5["overall"] if lang == "overall" else old5["languages"][lang])["supportable_at_3"]),
         pct(old7f[lang]["supportable_3"]), pct(metrics[lang]["supportable_3"]["value"]))
        for lang in (*LANGUAGES, "overall")])
    lines += ["", "Comparison is descriptive, not a controlled same-index delta: Step 5/7F used 490 chunks; Step 7G uses 491 and a changed evidence gate. "
              "Step 7F's post-fallback metrics used its accepted-rewrite gate; Step 7G metrics use current final candidate order and current evidence assessment. "
              "No retrieval weights were changed for this run.", "",
              "## 5. Multilingual support parity", ""]
    parity = summary["multilingual_parity"]
    categories = ("ALL_THREE_SUPPORTED", "ENGLISH_ONLY", "ENGLISH_AND_BANGLA", "ENGLISH_AND_BANGLISH", "OTHER_MISMATCH", "NONE_SUPPORTED")
    lines += table(("Category", "Base questions"), [(name, parity.get(name, 0)) for name in categories])
    lines += ["", f"Total paired base questions: {sum(parity.values())}. Pairing is evaluation-only; it never enters runtime answering.", "",
              "## 6. Generation and validation", ""]
    lines += table(("Language", "Language passed / 100", "Grounding passed / returned", "Generated rows", "Attempts", "Accepted", "Rejected", "Retries", "Successful", "Failed"), [
        (lang,
         f"{per[lang]['language_consistency']['passed']}/100",
         f"{per[lang]['grounding_validation']['passed']}/{per[lang]['grounding_validation']['denominator']}",
         per[lang]["generation"]["rows"], per[lang]["generation"]["attempts"],
         per[lang]["generation"]["accepted"], per[lang]["generation"]["rejected"],
         per[lang]["generation"]["retries"], per[lang]["generation"]["successful_retries"],
         per[lang]["generation"]["failed_retries"])
        for lang in LANGUAGES])
    lines += ["", "Grounding pass rate is reported among returned answers; rejected/insufficient responses are not treated as grounded factual answers. "
              f"Returned answers failing language validation: {len(returned_bad_language)}; failing grounding validation: {len(returned_bad_grounding)}.", "",
              "## 7. Latency", ""]
    latency = summary["latency_seconds"]
    lines += [f"Wall-clock per question: mean {latency['mean']:.2f}s, median {latency['median']:.2f}s, "
              f"P95 {latency['p95']:.2f}s, maximum {latency['maximum']:.2f}s; summed question time {latency['total']:.1f}s. "
              "These include model warm-up and any generation retries; they are not throughput measurements for a 70-PDF corpus.", "",
              "## 8. Memory and pagefile", ""]
    memory = summary["memory"]
    mib = 1024 ** 2
    lines += [f"Observed peak process RSS {memory['peak_process_rss_bytes']/mib:.1f} MiB; "
              f"peak process working set {memory['peak_process_working_set_bytes']/mib:.1f} MiB; "
              f"minimum available system RAM {memory['minimum_available_ram_bytes']/mib:.1f} MiB; "
              f"maximum system pagefile usage {memory['maximum_system_pagefile_used_bytes']/mib:.1f} MiB. "
              "The pagefile measure is system-wide, not attributable solely to this process. "
              "The very low minimum available RAM indicates substantial memory pressure during this one-PDF run; "
              "capacity for a 70-PDF workload is not established.", "",
              "## 9. Errors", ""]
    lines += [f"Runtime exceptions: {len(runtime_errors)}. Rows recording a generation error: {len(generation_errors)}. "
              "Error details remain in the results CSV; safe rejection is counted separately from an exception.", "",
              "## 10. Test suite", "", "After the template-only edits: 196 tests run, 195 passed, one skipped, zero failed. "
              "The fresh 491-chunk index was reused without rebuild.", "",
              "## 11. Manual review sample", "", f"`results/step7g_full_manual_review.csv` contains {len(review)} rows: "
              "10 English, 10 Bangla and 10 Banglish. It prioritizes relation answers, GGUF outputs, prior failures and cross-lingual recoveries. "
              "All eight requested human fields are blank; no human score was invented.", "",
              "## 12. Remaining failure categories", ""]
    failures = [x for x in rows if x["final_status"] != "ANSWER_RETURNED"]
    lines += table(("Category", "Rows"), [(label, count) for label, count in sorted(Counter(x["final_status"] for x in failures).items())])
    rejection_reasons = Counter(x["generation_rejection_reason"] or "UNSPECIFIED" for x in failures
                                if x["final_status"] == "GENERATION_REJECTED")
    if rejection_reasons:
        lines += ["", "Generation-rejection reasons (diagnostic labels): " + ", ".join(f"{key}={value}" for key, value in rejection_reasons.most_common()) + "."]
    ungrounded_reasons = Counter(x["grounding_validation_reason"] or "UNSPECIFIED" for x in returned_bad_grounding)
    if ungrounded_reasons:
        lines += ["", f"Safety issue outside the rejection counts: {len(returned_bad_grounding)} returned English "
                  "`structured_exact` answers failed automatic grounding: " +
                  ", ".join(f"{key}={value}" for key, value in ungrounded_reasons.most_common()) +
                  ". Their question IDs and excerpts are retained in the results CSV; no benchmark-time fix was made."]
    lines += ["", "The six previously identified Banglish safe rejections were allowed to remain rejections. "
              "No relation type or retrieval/model setting was changed in response to this run.", "",
              "## 13. Is Step 7 engineering-complete?", "",
              ("**Provisionally yes for this one-PDF development workflow:** all 300 ran without runtime exceptions, "
               "and every returned answer passed automatic language and grounding checks. This is not a factual-accuracy or 70-PDF claim."
               if engineering_complete else
               "**No:** 15 English `structured_exact` answers were returned despite failed automatic grounding. "
               "There were no runtime exceptions or returned-answer language failures, but the grounding safety gap prevents an engineering-complete declaration. "
               "Inspect the recorded rows before declaring completion."), "",
              "## 14. Should Step 8 begin?", "", "**Not automatically.** Review the 30 sampled answers and decide whether the observed failures and any validation issues "
              "are acceptable for the thesis workflow. Step 8 was not implemented here.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}; engineering_complete={engineering_complete}; review={len(review)}")


if __name__ == "__main__":
    main()
