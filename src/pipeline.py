from __future__ import annotations

import json
import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict

from .answer_bank import find_answer_bank_match
from .config import TOP_K, VECTOR_DB_DIR
from .embeddings import EmbeddingModel
from .evidence import (
    SupportStatus,
    answer_claim_is_bound,
    assess_evidence,
    response_for_status,
    select_primary_evidence,
)
from .fast_answer import build_extractive_answer, build_topic_answer, detect_runtime_intent, retrieve_lexical
from .generator import generate_answer, regenerate_answer_for_language
from .language_detector import detect_language
from .language_validator import unsupported_answer, validate_language
from .retriever import Retriever
from .vector_store import load_index


StatusCallback = Callable[[str], None]
DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "generation_debug_log.jsonl"


@lru_cache(maxsize=1)
def _get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


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


def answer_question(
    question: str,
    top_k: int = TOP_K,
    status_callback: StatusCallback | None = None,
    use_generation: bool = True,
    use_answer_bank: bool = False,
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    report("Detecting question language...")
    detected_language = detect_language(question)
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
            "detected_language": detected_language,
            "answer": answer,
            "answer_mode": "answer_bank",
            "intent": runtime_intent,
            "generation_mode": "answer_bank",
            "answer_bank_enabled": use_answer_bank,
            "support_status": SupportStatus.SUPPORTED.value,
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
        retriever = Retriever(embedding_model=embedding_model, top_k=top_k)
        retrieved = retriever.retrieve(question, index_path=index_path, metadata_path=metadata_path)
    else:
        report("Retrieving relevant context with fast lexical search...")
        retrieved = retrieve_lexical(question, metadata=metadata, top_k=top_k)

    report("Validating retrieved evidence...")
    assessment = assess_evidence(question, retrieved)
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

    if assessment.status is not SupportStatus.SUPPORTED:
        answer = response_for_status(assessment.status, detected_language)
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
            "detected_language": detected_language,
            "answer": answer,
            "answer_mode": assessment.status.value,
            "intent": runtime_intent,
            "generation_mode": assessment.status.value,
            "answer_bank_enabled": use_answer_bank,
            "support_status": assessment.status.value,
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
            "sources": sources,
            "evidence": verified,
            "retrieved_context": retrieved,
        }

    if use_generation:
        context = [
            _truncate_chunk_for_generation(question, item["supporting_excerpt"], max_words=300)
            for item in verified_retrieved[:2]
            if item.get("supporting_excerpt")
        ]
    else:
        context = [item["text"] for item in verified_retrieved if item.get("text")]

    topic_answer = (
        build_topic_answer(question, retrieved=verified_retrieved, language=detected_language)
        if runtime_intent == "topics"
        else None
    )

    answer_mode = "gguf_generation" if use_generation else "fast_extractive"

    if not context:
        answer = unsupported_answer(detected_language)
        answer_mode = "unsupported"
    elif topic_answer:
        answer = topic_answer
        answer_mode = "structured_extractive"
    elif not use_generation:
        report("Building fast answer from retrieved document text...")
        answer = build_extractive_answer(
            question,
            retrieved=verified_retrieved,
            language=detected_language,
            intent=runtime_intent,
        )
    else:
        report("Loading/generating with the local Qwen model...")
        answer = generate_answer(
            question=question,
            context=context,
            language=detected_language,
            requested_entity=assessment.request.primary_entity,
            requested_field=assessment.request.requested_field,
        )

    if answer == unsupported_answer(detected_language) or not answer.strip():
        answer = response_for_status(SupportStatus.INSUFFICIENT, detected_language)
        answer_mode = "unsupported"
        assessment = type(assessment)(
            assessment.request,
            SupportStatus.INSUFFICIENT,
            (),
            "Verified passages did not yield the requested factual value.",
        )
        verified_retrieved = []
        context = []

    report("Validating response language...")
    validation = validate_language(answer, detected_language)
    best_answer = answer
    best_validation = validation
    retry_attempted = False

    if (
        assessment.status is SupportStatus.SUPPORTED
        and use_generation
        and not validation["language_consistency"]
        and context
        and not retry_attempted
    ):
        report("Retrying answer once in the detected language...")
        retry_attempted = True
        retry_answer = regenerate_answer_for_language(
            question=question,
            context=context,
            language=detected_language,
            previous_answer=answer,
        )
        retry_validation = validate_language(retry_answer, detected_language)
        _log_generation_debug(
            question=question,
            first_answer=answer,
            first_validation=validation,
            second_answer=retry_answer,
            second_validation=retry_validation,
        )

        if retry_validation["language_consistency"]:
            best_answer = retry_answer
            best_validation = retry_validation
        else:
            best_answer = retry_answer.strip() or answer
            best_validation = retry_validation
            best_validation["language_validation_failed"] = True

    elif use_generation and not validation["language_consistency"] and not context:
        best_validation["language_validation_failed"] = True
        _log_generation_debug(
            question=question,
            first_answer=answer,
            first_validation=best_validation,
        )

    if use_generation and not best_validation["language_consistency"]:
        best_validation["language_validation_failed"] = True

    answer = best_answer
    validation = best_validation

    if assessment.status is SupportStatus.SUPPORTED and use_generation and not answer_claim_is_bound(answer, assessment.evidence):
        assessment = type(assessment)(
            assessment.request,
            SupportStatus.INSUFFICIENT,
            (),
            "Generated factual tokens were not present in the verified excerpts.",
        )
        answer = response_for_status(SupportStatus.INSUFFICIENT, detected_language)
        answer_mode = "unsupported"
        validation = validate_language(answer, detected_language)
        verified_retrieved = []

    primary = select_primary_evidence(answer, assessment.evidence)

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
        "detected_language": detected_language,
        "answer": answer,
        "answer_mode": answer_mode,
        "intent": runtime_intent,
        "generation_mode": answer_mode,
        "answer_bank_enabled": use_answer_bank,
        "support_status": assessment.status.value,
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
        "sources": sources,
        "evidence": [item.to_dict() for item in assessment.evidence],
        "retrieved_context": retrieved,
    }


__all__ = ["answer_question"]
