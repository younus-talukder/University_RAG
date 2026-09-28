from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict

from .answer_policy import effective_answer_field, evidence_value, format_exact_fact, format_structured_text, select_answer_strategy
from .answer_safety import answer_safety_failures, explicit_email_values, relation_compatible, requested_relation
from .answerability import (
    AnswerabilityDecision, AnswerabilityStatus, EvidenceLevel, assess_answerability,
    clearly_out_of_domain, finalize_answerability, remove_untrusted_directives, safe_response,
)
from .course_rows import course_field_value, requested_course_code
from .answer_bank import find_answer_bank_match
from .config import TOP_K, VECTOR_DB_DIR
from .embeddings import EmbeddingModel, get_embedding_model
from .evidence import (
    SupportStatus,
    answer_claim_is_bound,
    assess_evidence,
    response_for_status,
    select_primary_evidence,
)
from .evidence_spans import expand_assessment_evidence
from .document_metadata import metadata_fields_for_query
from .fast_answer import build_extractive_answer, build_topic_answer, detect_runtime_intent, retrieve_lexical
from .generator import generate_canonical_answer, realize_canonical_answer
from .generator import rewrite_query_for_retrieval
from .crosslingual import canonical_rewrite, crosslingual_assessment, fuse_queries
from .generation_context import build_verified_evidence_package, evidence_excerpts
from .grounding_validator import validate_grounding
from .language_detector import detect_language_details
from .language_validator import generation_rejected_answer, unsupported_answer, validate_language
from .realization_validator import validate_realization
from .relations import extract_relation, realize_relation, validate_relation_grounding
from .retriever import Retriever
from .query_normalization import build_query_representations
from .semantic_contract import build_semantic_contract, repair_instruction
from .semistructured_realizer import realize_semistructured
from .vector_store import load_index


StatusCallback = Callable[[str], None]
DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "generation_debug_log.jsonl"


@lru_cache(maxsize=1)
def _get_embedding_model() -> EmbeddingModel:
    return get_embedding_model()


def _log_generation_debug(
    question: str,
    first_answer: str,
    first_validation: Dict[str, Any],
    second_answer: str | None = None,
    second_validation: Dict[str, Any] | None = None,
) -> None:
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "question": question,
        "first_answer": first_answer,
        "first_validation": first_validation,
        "second_answer": second_answer,
        "second_validation": second_validation,
    }
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _truncate_chunk_for_generation(question: str, chunk_text: str, max_words: int = 300) -> str:
    text = " ".join(str(chunk_text).split()).strip()
    if not text:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    normalized_question = re.sub(r"[^\w\s]", " ", question.lower())
    keywords = {
        token for token in normalized_question.split() if len(token) >= 3 and token not in {"what", "when", "where", "which", "that", "this", "with", "from", "into", "they", "them"}
    }

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if keywords:
        scored_segments = []
        for segment in sentences:
            lower_segment = segment.lower()
            overlap = sum(1 for keyword in keywords if keyword in lower_segment)
            scored_segments.append((overlap, segment))
        scored_segments.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        used_words = 0
        for _, segment in scored_segments:
            segment_words = segment.split()
            if used_words + len(segment_words) <= max_words:
                selected.append(segment)
                used_words += len(segment_words)
            else:
                remaining = max_words - used_words
                if remaining > 0:
                    selected.append(" ".join(segment_words[:remaining]))
                break

        if selected:
            return " ".join(selected).strip()

    return " ".join(words[:max_words]).strip()


def _answer_question_impl(
    question: str,
    top_k: int = TOP_K,
    status_callback: StatusCallback | None = None,
    use_generation: bool = True,
    use_answer_bank: bool = False,
) -> Dict[str, Any]:
    pipeline_started = time.perf_counter()
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    report("Detecting question language...")
    representations = build_query_representations(question)
    language_detection = detect_language_details(question)
    detected_language = language_detection.language
    representations = replace(representations, target_language=detected_language)
    runtime_intent = detect_runtime_intent(question)

    answer_bank_match = None
    if use_answer_bank:
        report("Checking curated question dataset...")
        answer_bank_match = find_answer_bank_match(question, detected_language)

    if answer_bank_match:
        answer = answer_bank_match.answer
        validation = validate_language(answer, detected_language)
        return {
            "question": question,
            "normalized_query": representations.normalized_query,
            "retrieval_query": representations.retrieval_query,
            "detected_language": detected_language,
            "language_detection_reason": language_detection.reason,
            "answer": answer,
            "answer_mode": "answer_bank",
            "intent": runtime_intent,
            "generation_mode": "answer_bank",
            "answer_bank_enabled": use_answer_bank,
            "support_status": SupportStatus.SUPPORTED.value,
            "final_status": "ANSWER_RETURNED",
            "support_reason": "Explicit answer-bank/debug mode matched the evaluation dataset.",
            "requested_entity": None,
            "requested_field": None,
            "document_id": None,
            "source": "data/questions/questions.csv",
            "relative_path": None,
            "page": None,
            "chunk_id": None,
            "supporting_excerpt": f"Matched curated question: {answer_bank_match.matched_question}",
            "response_language": validation["response_language"],
            "language_consistency": validation["language_consistency"],
            "language_validation_failed": bool(validation.get("language_validation_failed", False)),
            "language_validation_passed": bool(validation.get("validation_passed", validation["language_consistency"])),
            "language_validation_reason": validation.get("validation_reason"),
            "grounding_validation_passed": True,
            "grounding_validation_reason": "ANSWER_BANK_MODE",
            "generation_used": False,
            "generation_attempts": 0,
            "retry_used": False,
            "target_language": detected_language,
            "answer_strategy": "answer_bank",
            "answerability_status": None,
            "answerability_reason": "ANSWER_BANK_MODE_NOT_DOCUMENT_ASSESSED",
            "evidence_level": "NONE",
            "final_answer_allowed": True,
            "answerability_seconds": 0.0,
            "latency_seconds": {"retrieval": 0.0, "answering": time.perf_counter() - pipeline_started, "generation": 0.0, "retry": 0.0, "total": time.perf_counter() - pipeline_started},
            "sources": [
                {
                    "source": "data/questions/questions.csv",
                    "page": None,
                    "score": answer_bank_match.score,
                }
            ],
            "retrieved_context": [
                {
                    "text": f"Matched curated question: {answer_bank_match.matched_question}",
                    "score": answer_bank_match.score,
                    "source": "data/questions/questions.csv",
                    "page": None,
                    "chunk_id": None,
                }
            ],
        }

    index_path = str(VECTOR_DB_DIR / "index.faiss")
    metadata_path = str(VECTOR_DB_DIR / "metadata.pkl")

    try:
        report("Checking FAISS vector index...")
        index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "FAISS index is missing. Build the index first using the ingestion/build scripts."
        ) from exc

    if use_generation:
        report("Loading embedding model...")
        embedding_model = _get_embedding_model()
        report("Retrieving relevant context...")
        retriever = Retriever(embedding_model=embedding_model, top_k=max(top_k, 30))
        original_pool = retriever.retrieve(question, index_path=index_path, metadata_path=metadata_path)
        retrieved = original_pool[:top_k]
    else:
        report("Retrieving relevant context with fast lexical search...")
        retrieved = retrieve_lexical(question, metadata=metadata, top_k=top_k)
    retrieval_seconds = time.perf_counter() - pipeline_started

    report("Validating retrieved evidence...")
    assessment = assess_evidence(question, retrieved)
    initial_support_status = assessment.status.value
    metadata_evidence_widened = False
    if (use_generation and assessment.status is SupportStatus.INSUFFICIENT
            and metadata_fields_for_query(question)):
        # The dense retriever already supplied a bounded pool. A labeled
        # front-matter field can sit below top-k without licensing a corpus
        # scan, an unlabeled publisher guess, or cross-document value mixing.
        widened = assess_evidence(question, original_pool)
        if widened.status is not SupportStatus.INSUFFICIENT:
            assessment, retrieved = widened, original_pool
            metadata_evidence_widened = True
    representations = replace(representations,
                              requested_entity=assessment.request.primary_entity or "",
                              requested_field=assessment.request.requested_field)
    initial_top = list(original_pool[:top_k] if use_generation else retrieved)
    fallback_trace: dict[str, Any] = {
        "initial_support_status": initial_support_status,
        "metadata_evidence_widened": metadata_evidence_widened,
        "initial_top1": initial_top[:1], "initial_top3": initial_top[:3],
        "fallback_triggered": False, "rewrite_method": "NONE",
        "canonical_retrieval_query": "", "rewrite_validation": "NOT_ATTEMPTED",
        "fallback_top1": [], "fallback_top3": [],
        "merged_support_status": assessment.status.value,
        "crosslingual_gate_reason": "NOT_ELIGIBLE",
        "initial_retrieval_seconds": retrieval_seconds,
        "rewrite_seconds": 0.0, "canonical_retrieval_seconds": 0.0,
        "multi_query_fusion_seconds": 0.0, "fallback_seconds": 0.0,
    }
    if (use_generation and detected_language in {"bangla", "banglish"}
            and assessment.status is SupportStatus.INSUFFICIENT):
        fallback_trace["fallback_triggered"] = True
        fallback_started = time.perf_counter()
        try:
            rewrite_started = time.perf_counter()
            rewrite = canonical_rewrite(question, rewrite_query_for_retrieval)
            fallback_trace["rewrite_seconds"] = time.perf_counter() - rewrite_started
            representations = replace(representations,
                                      canonical_retrieval_query=rewrite.query,
                                      rewrite_method=rewrite.method,
                                      retrieval_terms=tuple(rewrite.query.split()))
            fallback_trace.update({"rewrite_method": rewrite.method,
                                   "canonical_retrieval_query": rewrite.query,
                                   "rewrite_validation": rewrite.validation})
            if rewrite.query:
                canonical_started = time.perf_counter()
                canonical_pool = retriever.retrieve(rewrite.query, index_path=index_path,
                                                    metadata_path=metadata_path)
                fallback_trace["canonical_retrieval_seconds"] = time.perf_counter() - canonical_started
                fallback_trace["fallback_top1"] = canonical_pool[:1]
                fallback_trace["fallback_top3"] = canonical_pool[:3]
                fusion_started = time.perf_counter()
                merged = fuse_queries(original_pool, canonical_pool, top_k)
                reassessed, gate_reason = crosslingual_assessment(question, rewrite, merged)
                fallback_trace["multi_query_fusion_seconds"] = time.perf_counter() - fusion_started
                fallback_trace["merged_support_status"] = reassessed.status.value
                fallback_trace["crosslingual_gate_reason"] = gate_reason
                if reassessed.status in {SupportStatus.SUPPORTED, SupportStatus.CONFLICTING, SupportStatus.AMBIGUOUS}:
                    retrieved, assessment = merged, reassessed
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            fallback_trace["crosslingual_gate_reason"] = f"REWRITE_UNAVAILABLE:{type(exc).__name__}"
        fallback_trace["fallback_seconds"] = time.perf_counter() - fallback_started
    retrieval_seconds = time.perf_counter() - pipeline_started
    if assessment.status is SupportStatus.SUPPORTED:
        assessment = expand_assessment_evidence(assessment, metadata)
    trust_started = time.perf_counter()
    trust_relation = extract_relation(question, assessment, retrieved, detected_language) if assessment.status is SupportStatus.SUPPORTED else None
    intended_relation = requested_relation(question)
    extracted_relation = trust_relation.relation if trust_relation else None
    trust_strategy = (select_answer_strategy(assessment.status, assessment.request.requested_field, runtime_intent, question)
                      if assessment.status is SupportStatus.SUPPORTED else None)
    trust = assess_answerability(
        question, assessment, retrieved,
        cross_lingual_fallback_used=bool(fallback_trace["fallback_triggered"]),
        requested_relation=intended_relation or extracted_relation,
        relation_supported=(relation_compatible(intended_relation, extracted_relation)
                            if trust_relation else None),
        generation_required=trust_strategy == "gguf_generation" if trust_strategy else None,
    )
    answerability_seconds = time.perf_counter() - trust_started
    pre_generation_support_status = assessment.status.value
    pre_generation_support_reason = assessment.reason
    verified = [item.to_dict() for item in assessment.evidence]
    verified_retrieved = [
        {
            # Extractors and the generator operate on the verified local passage,
            # never on the rest of a broad retrieved chunk.
            "text": item["excerpt"],
            "score": item["retrieval_score"],
            "document_id": item["document_id"],
            "source": item["source"],
            "relative_path": item["relative_path"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "supporting_excerpt": item["excerpt"],
        }
        for item in verified
    ]

    if trust.answerability_status is not AnswerabilityStatus.SUPPORTED:
        answer = safe_response(trust.answerability_status, detected_language)
        validation = validate_language(answer, detected_language)
        sources = [
            {
                "document_id": item.document_id,
                "source": item.source,
                "relative_path": item.relative_path,
                "page": item.page,
                "chunk_id": item.chunk_id,
                "score": item.retrieval_score,
                "supporting_excerpt": item.excerpt,
            }
            for item in assessment.evidence
        ]
        primary = assessment.evidence[0] if assessment.evidence else None
        return {
            "question": question,
            "normalized_query": representations.normalized_query,
            "retrieval_query": representations.retrieval_query,
            "detected_language": detected_language,
            "language_detection_reason": language_detection.reason,
            "answer": answer,
            "answer_mode": assessment.status.value,
            "intent": runtime_intent,
            "generation_mode": assessment.status.value,
            "answer_bank_enabled": use_answer_bank,
            "support_status": assessment.status.value,
            "final_status": trust.answerability_status.value,
            "support_reason": assessment.reason,
            "requested_entity": assessment.request.primary_entity,
            "requested_field": assessment.request.requested_field,
            "conflicting_values": list(assessment.conflicting_values),
            "source": primary.source if primary else None,
            "document_id": primary.document_id if primary else None,
            "relative_path": primary.relative_path if primary else None,
            "page": primary.page if primary else None,
            "chunk_id": primary.chunk_id if primary else None,
            "supporting_excerpt": primary.excerpt if primary else None,
            "response_language": validation["response_language"],
            "language_consistency": validation["language_consistency"],
            "language_validation_failed": bool(validation.get("language_validation_failed", False)),
            "language_validation_passed": bool(validation.get("validation_passed", validation["language_consistency"])),
            "language_validation_reason": validation.get("validation_reason"),
            "grounding_validation_passed": True,
            "grounding_validation_reason": "GENERATION_NOT_USED",
            "generation_used": False,
            "generation_attempts": 0,
            "retry_used": False,
            "target_language": detected_language,
            "answer_strategy": "unsupported" if assessment.status is SupportStatus.INSUFFICIENT else assessment.status.value,
            **trust.to_dict(),
            "answerability_seconds": answerability_seconds,
            "latency_seconds": {"retrieval": retrieval_seconds, "answering": time.perf_counter() - pipeline_started - retrieval_seconds, "generation": 0.0, "retry": 0.0, "total": time.perf_counter() - pipeline_started},
            "sources": sources,
            "evidence": verified,
            "retrieved_context": retrieved,
            "pre_generation_evidence": verified,
            "pre_generation_support_status": pre_generation_support_status,
            "pre_generation_support_reason": pre_generation_support_reason,
            "raw_first_generation": "",
            "first_language_validation": {},
            "first_semantic_validation": {},
            "first_grounding_validation": {},
            "retry_reason": "",
            "raw_retry_output": "",
            "retry_language_validation": {},
            "retry_semantic_validation": {},
            "retry_grounding_validation": {},
            "final_fallback_reason": assessment.status.value,
            **fallback_trace,
        }

    package = build_verified_evidence_package(question, detected_language, assessment, retrieved)
    semantic_contract = build_semantic_contract(package)
    context = list(evidence_excerpts(package))
    requested_strategy = trust_strategy
    relation = trust_relation
    answer_field = effective_answer_field(assessment.request.requested_field, question)
    answer_strategy = requested_strategy
    generation_used = False
    generation_attempts = 0
    retry_attempted = False
    generation_error: str | None = None
    generation_seconds = 0.0
    retry_seconds = 0.0
    canonical_seconds = 0.0
    realization_seconds = 0.0
    answering_started = time.perf_counter()
    raw_first_generation = ""
    first_language_validation: Dict[str, Any] = {}
    first_grounding_validation: Dict[str, Any] = {}
    retry_reason = ""
    raw_retry_output = ""
    retry_language_validation: Dict[str, Any] = {}
    retry_grounding_validation: Dict[str, Any] = {}
    canonical_answer = ""
    canonical_validation: Dict[str, Any] = {}
    canonical_grounding: Dict[str, Any] = {}
    realization_output = ""
    realization_validation: Dict[str, Any] = {}
    final_status = "ANSWER_RETURNED"

    answer = ""
    if relation:
        answer = realize_relation(relation)
        answer_mode = "semi_structured_relation"
        answer_strategy = "semi_structured_relation"
    elif requested_strategy == "structured_exact":
        if runtime_intent == "final_exam_marks":
            answer = build_extractive_answer(question, verified_retrieved, detected_language, runtime_intent)
        course_field = bool(answer_field in {"credits", "prerequisite"} and requested_course_code(question))
        value = (course_field_value(question, answer_field, context) if course_field
                 else evidence_value(answer_field, context)) if not answer else None
        if answer_field == "email":
            listed = explicit_email_values(question, context)
            if listed:
                value = ", ".join(listed)
        if value:
            answer = format_exact_fact(assessment.request.primary_entity, answer_field, value, detected_language)
        if not answer and not course_field:
            extracted = build_extractive_answer(question, verified_retrieved, "english", runtime_intent)
            if extracted.startswith((
                "Course code:", "Course title:", "Course type:", "Credit value:", "Prerequisite:",
                "Final Exam:", "Term Examination:", "80% and above",
            )):
                answer = format_structured_text(extracted, detected_language)
        answer_mode = "structured_exact"
    elif requested_strategy == "structured_list":
        answer = build_extractive_answer(question, verified_retrieved, detected_language, runtime_intent)
        answer_mode = "structured_list"
        if answer == unsupported_answer(detected_language) and use_generation:
            answer = ""
            answer_strategy = "gguf_generation"
    elif not use_generation:
        report("Building fast answer from retrieved document text...")
        answer = build_extractive_answer(question, verified_retrieved, detected_language, runtime_intent)
        answer_mode = "fast_extractive"
    else:
        answer_mode = "gguf_generation"

    generation_rejection_reason: str | None = None
    final_fallback_reason = ""
    validation: Dict[str, Any] = {}
    grounding: Dict[str, Any] = {}

    if not answer and use_generation and answer_strategy == "gguf_generation":
        generation_used = True
        canonical_package = replace(package, original_language="english")
        canonical_contract = build_semantic_contract(canonical_package)
        report("Generating a canonical English answer from verified evidence...")
        generation_attempts = 1
        generation_started = time.perf_counter()
        try:
            canonical_answer = generate_canonical_answer(canonical_package, canonical_contract)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            generation_error = str(exc)
        canonical_seconds += time.perf_counter() - generation_started
        raw_first_generation = canonical_answer
        canonical_validation = validate_language(canonical_answer, "english")
        canonical_grounding = validate_grounding(canonical_answer, package.evidence, canonical_contract)
        first_language_validation = dict(canonical_validation)
        first_grounding_validation = dict(canonical_grounding)

        canonical_semantic = canonical_grounding.get("semantic_validation") or {}
        canonical_repairable = (
            not canonical_validation.get("validation_passed", False)
            or bool(canonical_semantic.get("repairable"))
        )
        if (
            (not canonical_validation.get("validation_passed", False)
             or not canonical_grounding.get("grounding_validation_passed", False))
            and canonical_repairable
            and generation_attempts < 3
        ):
            retry_attempted = True
            retry_reason = str(
                canonical_grounding.get("grounding_validation_reason")
                if not canonical_grounding.get("grounding_validation_passed")
                else canonical_validation.get("validation_reason")
            )
            report("Repairing the canonical answer once...")
            retry_started = time.perf_counter()
            generation_attempts += 1
            try:
                repaired = generate_canonical_answer(
                    canonical_package,
                    canonical_contract,
                    previous_answer=canonical_answer,
                    correction_reason=repair_instruction(retry_reason),
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                generation_error = str(exc)
                repaired = ""
            elapsed_retry = time.perf_counter() - retry_started
            retry_seconds += elapsed_retry
            canonical_seconds += elapsed_retry
            raw_retry_output = repaired
            retry_language_validation = validate_language(repaired, "english")
            retry_grounding_validation = validate_grounding(repaired, package.evidence, canonical_contract)
            canonical_answer = repaired
            canonical_validation = retry_language_validation
            canonical_grounding = retry_grounding_validation

        canonical_ok = bool(
            canonical_answer
            and canonical_validation.get("validation_passed")
            and canonical_grounding.get("grounding_validation_passed")
        )
        if canonical_ok and detected_language == "english":
            answer = canonical_answer
            validation = canonical_validation
            grounding = canonical_grounding
        elif canonical_ok:
            report(f"Realizing the validated canonical answer in {detected_language}...")
            realization_started = time.perf_counter()
            realization_output = realize_semistructured(canonical_answer, detected_language) or ""
            if not realization_output:
                generation_attempts += 1
                try:
                    realization_output = realize_canonical_answer(canonical_answer, detected_language)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    generation_error = str(exc)
            realization_seconds += time.perf_counter() - realization_started
            realization_validation = validate_realization(
                canonical_answer, realization_output, detected_language, semantic_contract
            )
            validation = dict(realization_validation.get("language_validation") or validate_language(realization_output, detected_language))
            grounding = validate_grounding(
                realization_output, package.evidence, replace(semantic_contract, target_language="english")
            )
            realization_ok = bool(realization_validation.get("passed") and grounding.get("grounding_validation_passed"))
            if not realization_ok and not retry_attempted and generation_attempts < 3:
                retry_attempted = True
                retry_reason = str(
                    realization_validation.get("reason")
                    if not realization_validation.get("passed")
                    else grounding.get("grounding_validation_reason")
                )
                report("Repairing the target-language realization once...")
                retry_started = time.perf_counter()
                generation_attempts += 1
                try:
                    repaired = realize_canonical_answer(
                        canonical_answer,
                        detected_language,
                        previous_answer=realization_output,
                        correction_reason=repair_instruction(retry_reason),
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    generation_error = str(exc)
                    repaired = ""
                elapsed_retry = time.perf_counter() - retry_started
                retry_seconds += elapsed_retry
                realization_seconds += elapsed_retry
                raw_retry_output = repaired
                realization_output = repaired
                realization_validation = validate_realization(
                    canonical_answer, realization_output, detected_language, semantic_contract
                )
                retry_language_validation = dict(realization_validation.get("language_validation") or {})
                grounding = validate_grounding(
                    realization_output, package.evidence, replace(semantic_contract, target_language="english")
                )
                retry_grounding_validation = dict(grounding)
                validation = retry_language_validation
                realization_ok = bool(realization_validation.get("passed") and grounding.get("grounding_validation_passed"))
            if realization_ok:
                answer = realization_output
        generation_seconds = canonical_seconds + realization_seconds

        if not answer:
            if not canonical_ok:
                generation_rejection_reason = str(
                    canonical_grounding.get("grounding_validation_reason")
                    if not canonical_grounding.get("grounding_validation_passed")
                    else canonical_validation.get("validation_reason")
                )
            else:
                generation_rejection_reason = str(
                    realization_validation.get("reason")
                    if not realization_validation.get("passed")
                    else grounding.get("grounding_validation_reason")
                )

    if answer and not validation:
        report("Validating response language and factual grounding...")
        validation = validate_language(answer, detected_language)
        grounding = (validate_relation_grounding(question, relation, assessment, answer) if relation
                     else validate_grounding(answer, package.evidence, semantic_contract))
        first_language_validation = dict(validation)
        first_grounding_validation = dict(grounding)

    # A recovered distribution answer must not silently drop components of a
    # multi-percentage source. This is a narrow fallback safety check, not a
    # change to the original Step-7E answer path.
    distribution_requested = bool(re.search(r"\bdistribution\b|\bbonton\b|বণ্টন", question, re.I))
    source_percentages = set(re.findall(r"\b(\d+(?:\.\d+)?)\s*%", " ".join(x.excerpt for x in assessment.evidence)))
    answer_percentages = set(re.findall(r"\b(\d+(?:\.\d+)?)\s*%", answer))
    incomplete_distribution = bool(
        fallback_trace["fallback_triggered"] and distribution_requested
        and len(source_percentages) > 1 and not source_percentages.issubset(answer_percentages)
    )
    if incomplete_distribution:
        generation_rejection_reason = "INCOMPLETE_DISTRIBUTION_FROM_RECOVERED_EVIDENCE"
    safety_failures = answer_safety_failures(
        question, answer, context, extracted_relation=extracted_relation,
        relation_values=relation.values if relation else (),
    ) if answer else ()
    if safety_failures:
        generation_rejection_reason = ",".join(safety_failures)
    generation_failed = generation_used and (
        not answer
        or not validation.get("validation_passed", False)
        or not grounding.get("grounding_validation_passed", False)
        or incomplete_distribution or bool(safety_failures)
    )
    relation_failed = bool(relation and (not answer or not validation.get("validation_passed", False)
                                          or not grounding.get("grounding_validation_passed", False)
                                          or bool(safety_failures)))
    non_generated_validation_failed = bool(
        answer and not generation_used
        and (not validation.get("validation_passed", False)
             or not grounding.get("grounding_validation_passed", False)
             or bool(safety_failures))
    )
    structured_extraction_failed = (
        (not answer and requested_strategy in {"structured_exact", "structured_list"})
        or incomplete_distribution or relation_failed or non_generated_validation_failed
    )
    if assessment.status is SupportStatus.SUPPORTED and (generation_failed or structured_extraction_failed):
        generation_rejection_reason = generation_rejection_reason or str(
            grounding.get("grounding_validation_reason")
            if grounding and not grounding.get("grounding_validation_passed")
            else validation.get("validation_reason") or "NO_SAFE_ANSWER_FROM_SUPPORTED_EVIDENCE"
        )
        final_fallback_reason = "GENERATION_REJECTED"
        final_status = "GENERATION_REJECTED"
        answer = generation_rejected_answer(detected_language)
        answer_mode = "generation_rejected"
        answer_strategy = "generation_rejected"
        validation = validate_language(answer, detected_language)
        grounding = {
            "grounding_validation_passed": False,
            "grounding_validation_reason": "GENERATION_REJECTED",
            "unsupported_facts": grounding.get("unsupported_facts", []) if grounding else [],
        }

    semantic_signal = grounding.get("semantic_validation") if grounding else None
    semantic_passed = semantic_signal is None or bool(semantic_signal.get("passed", False))
    if relation or requested_strategy in {"structured_exact", "structured_list"}:
        trust = replace(trust, structured_extractable=bool(answer and final_status == "ANSWER_RETURNED"))
    trust = finalize_answerability(
        trust, answer_present=bool(answer and final_status == "ANSWER_RETURNED"),
        language_passed=bool(validation.get("validation_passed", False)),
        grounding_passed=bool(grounding.get("grounding_validation_passed", False)),
        semantic_passed=semantic_passed, generation_used=generation_used,
        rejection_reason=generation_rejection_reason or "",
    )
    if not trust.final_answer_allowed:
        final_status = AnswerabilityStatus.GENERATION_REJECTED.value
        answer = safe_response(AnswerabilityStatus.GENERATION_REJECTED, detected_language)
        answer_mode = "generation_rejected"
        answer_strategy = "generation_rejected"

    primary = select_primary_evidence(answer, assessment.evidence) or (assessment.evidence[0] if assessment.evidence else None)

    sources = [
        {
            "document_id": item.get("document_id"),
            "source": item.get("source"),
            "relative_path": item.get("relative_path"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
            "supporting_excerpt": item.get("supporting_excerpt"),
        }
        for item in verified_retrieved
    ]

    return {
        "question": question,
        "normalized_query": representations.normalized_query,
        "retrieval_query": representations.retrieval_query,
        "detected_language": detected_language,
        "language_detection_reason": language_detection.reason,
        "answer": answer,
        "answer_mode": answer_mode,
        "intent": runtime_intent,
        "generation_mode": answer_mode,
        "answer_bank_enabled": use_answer_bank,
        "support_status": assessment.status.value,
        "final_status": final_status,
        "support_reason": assessment.reason,
        "requested_entity": assessment.request.primary_entity,
        "requested_field": assessment.request.requested_field,
        "source": primary.source if primary else None,
        "document_id": primary.document_id if primary else None,
        "relative_path": primary.relative_path if primary else None,
        "page": primary.page if primary else None,
        "chunk_id": primary.chunk_id if primary else None,
        "supporting_excerpt": primary.excerpt if primary else None,
        "response_language": validation["response_language"],
        "language_consistency": validation["language_consistency"],
        "language_validation_failed": bool(validation.get("language_validation_failed", False)),
        "language_validation_passed": bool(validation.get("validation_passed", validation["language_consistency"])),
        "language_validation_reason": validation.get("validation_reason"),
        "grounding_validation_passed": bool(grounding.get("grounding_validation_passed", False)),
        "grounding_validation_reason": grounding.get("grounding_validation_reason"),
        "semantic_validation_passed": semantic_passed,
        "generation_rejection_reason": generation_rejection_reason,
        "unsupported_generated_facts": grounding.get("unsupported_facts", []),
        "generation_used": generation_used,
        "relation_type": relation.relation if relation else None,
        "relation_extracted": ({"subject": relation.subject, "relation": relation.relation,
                                "values": list(relation.values), "source": relation.source,
                                "page": relation.page, "chunk_id": relation.chunk_id,
                                "evidence_span": relation.evidence_span,
                                "continuation_chunk_ids": list(relation.continuation_chunk_ids)} if relation else None),
        "generation_attempts": generation_attempts,
        "retry_used": retry_attempted,
        "target_language": detected_language,
        "answer_strategy": answer_strategy,
        **trust.to_dict(),
        "answerability_seconds": answerability_seconds,
        "requested_answer_strategy": requested_strategy,
        "generation_error": generation_error,
        "canonical_answer": canonical_answer,
        "canonical_validation_passed": bool(canonical_validation.get("validation_passed", False)),
        "canonical_grounding_passed": bool(canonical_grounding.get("grounding_validation_passed", False)),
        "canonical_validation_reason": canonical_grounding.get("grounding_validation_reason") or canonical_validation.get("validation_reason"),
        "target_language_realization": realization_output,
        "realization_validation_passed": bool(realization_validation.get("passed", False)) if detected_language != "english" and generation_used else bool(canonical_answer),
        "realization_validation_reason": realization_validation.get("reason") if realization_validation else ("NOT_REQUIRED" if detected_language == "english" else ""),
        "latency_seconds": {
            "retrieval": retrieval_seconds,
            "answering": time.perf_counter() - answering_started,
            "generation": generation_seconds,
            "retry": retry_seconds,
            "canonical_generation": canonical_seconds,
            "language_realization": realization_seconds,
            "total": time.perf_counter() - pipeline_started,
        },
        "sources": sources,
        "evidence": [item.to_dict() for item in assessment.evidence],
        "retrieved_context": retrieved,
        "pre_generation_evidence": verified,
        "pre_generation_support_status": pre_generation_support_status,
        "pre_generation_support_reason": pre_generation_support_reason,
        "raw_first_generation": raw_first_generation,
        "first_language_validation": first_language_validation,
        "first_semantic_validation": first_grounding_validation.get("semantic_validation", {}),
        "first_grounding_validation": first_grounding_validation,
        "retry_reason": retry_reason,
        "raw_retry_output": raw_retry_output,
        "retry_language_validation": retry_language_validation,
        "retry_semantic_validation": retry_grounding_validation.get("semantic_validation", {}),
        "retry_grounding_validation": retry_grounding_validation,
        "final_fallback_reason": final_fallback_reason,
        **fallback_trace,
    }


def _preflight_response(question: str, status: AnswerabilityStatus, reason: str,
                        started: float, error: str = "") -> Dict[str, Any]:
    language = detect_language_details(question).language
    answer = safe_response(status, language)
    validation = validate_language(answer, language)
    decision = AnswerabilityDecision(status, reason, EvidenceLevel.NONE)
    return {
        "question": question, "answer": answer, "detected_language": language,
        "answer_mode": status.value.casefold(), "answer_strategy": status.value.casefold(),
        "support_status": SupportStatus.INSUFFICIENT.value, "final_status": status.value,
        "source": None, "page": None, "supporting_excerpt": None,
        "sources": [], "evidence": [], "retrieved_context": [],
        "generation_used": False, "generation_attempts": 0, "retry_used": False,
        "generation_validation_status": "NOT_ATTEMPTED",
        "language_validation_passed": bool(validation["validation_passed"]),
        "grounding_validation_passed": False,
        "latency_seconds": {"retrieval": 0.0, "answering": 0.0, "generation": 0.0,
                            "retry": 0.0, "total": time.perf_counter() - started},
        "answerability_seconds": time.perf_counter() - started,
        "runtime_error": error,
        **decision.to_dict(),
    }


def answer_question(
    question: str,
    top_k: int = TOP_K,
    status_callback: StatusCallback | None = None,
    use_generation: bool = True,
    use_answer_bank: bool = False,
) -> Dict[str, Any]:
    """Public trust boundary: bypass instructions and failures cannot leak facts."""
    started = time.perf_counter()
    if not question or not question.strip():
        return _preflight_response(question or "", AnswerabilityStatus.AMBIGUOUS_QUERY,
                                   "EMPTY_QUERY", started)
    cleaned = remove_untrusted_directives(question)
    if clearly_out_of_domain(question):
        return _preflight_response(question, AnswerabilityStatus.OUT_OF_DOMAIN,
                                   "OUT_OF_DOMAIN", started)
    try:
        result = _answer_question_impl(cleaned, top_k=top_k, status_callback=status_callback,
                                       use_generation=use_generation, use_answer_bank=use_answer_bank)
    except Exception as exc:
        return _preflight_response(question, AnswerabilityStatus.SYSTEM_ERROR,
                                   "SYSTEM_EXCEPTION", started,
                                   f"{type(exc).__name__}: {exc}")
    result["question"] = question
    return result


__all__ = ["answer_question"]
