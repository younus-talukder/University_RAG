from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import TOP_K
from .embeddings import EmbeddingModel
from .fast_answer import detect_course_code, detect_runtime_intent, extract_week_number
from .vector_store import load_index, search


BANGLA_DIGITS = str.maketrans(
    "\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef",
    "0123456789",
)

RETRIEVAL_RERANK_WEIGHTS = {
    "candidate_pool_min": 15,
    "candidate_pool_multiplier": 5,
    "course_match_boost": 0.8,
    "course_text_boost": 0.2,
    "course_mismatch_penalty": -1.0,
    "intent_match_boost": 0.8,
    "requested_field_boost": 3.2,
    "topic_subject_boost": 0.8,
    "topic_plan_boost": 2.2,
    "topic_assessment_penalty": -1.8,
    "lexical_overlap_weight": 0.12,
    "lexical_overlap_cap": 0.9,
    "section_heading_boost": 0.45,
    "metadata_mismatch_penalty": -1.6,
}

STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "course", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "er", "e", "ki", "kii",
    "koto", "koy", "kon", "kono", "hoy", "holo", "ache", "ebong",
    "\u0995\u09c0", "\u0995\u09bf", "\u098f\u09ac\u0982", "\u0995\u09cb\u09a8",
}

INTENT_SECTION_PATTERNS = {
    "clo": [
        r"Course Learning Outcomes",
        r"\bCLO\s*\d+\b",
    ],
    "course_objective": [
        r"Course Objectives",
        r"Course Summary",
        r"\bobjective",
    ],
    "topics": [
        r"Course Content",
        r"Alignment of topics",
        r"Topics?\s*/\s*Content",
        r"\b[A-Z][A-Za-z /&-]{2,45}:",
    ],
    "weekly_content": [
        r"Weekly plan",
        r"Class Schedule",
        r"\bWeek\s+\d+\b",
    ],
    "assessment": [
        r"Assessment Strategy",
        r"Assessment Techniques",
        r"Evaluation Policy",
        r"Class Tests?",
        r"Quizzes?",
        r"Assignment",
        r"Presentation",
    ],
    "prerequisite": [
        r"Pre[- ]?requisites?",
        r"Prereq",
    ],
    "final_exam_marks": [
        r"Final Exam",
        r"Term Examination",
        r"\d+(?:\.\d+)?\s*%",
    ],
    "mark_distribution": [
        r"Weighting",
        r"Assessment Type",
        r"Evaluation Policy",
        r"Class Tests?",
        r"Mid-Term Examination",
        r"Final Exam",
        r"\d+(?:\.\d+)?\s*%",
    ],
    "course_metadata": [
        r"Course No\.",
        r"Course Code",
        r"Course Title",
        r"Course Type",
        r"Credit Value",
    ],
    "course_code": [r"Course No\.", r"Course Code"],
    "course_title": [r"Course Title"],
    "course_type": [r"Course Type"],
    "course_credit": [r"Credit Value", r"credits?"],
}


@dataclass(frozen=True)
class QueryProfile:
    original: str
    normalized: str
    tokens: List[str]
    course_code: str
    intent: str
    clo_number: str
    week_number: int | None
    topic_subject: str


def normalize_query_for_retrieval(question: str) -> str:
    text = str(question).translate(BANGLA_DIGITS).casefold()
    text = re.sub(r"[\u2010-\u2015_/(),:;?!.]+", " ", text)
    replacements = [
        (r"\b([a-z]+)-er\b", r"\1 er"),
        (r"\b([a-z]+)-e\b", r"\1 e"),
        (r"\bkoy\b", "koto"),
        (r"\bkii\b", "ki"),
        (r"\bunder\s+e\b", "under"),
        (r"\bunder-e\b", "under"),
        (r"\bboraddo\b", "allocated"),
        (r"\bporano\s+hoy\b", "porano"),
        (r"\bfinal\s+exam\s+e\b", "final exam"),
        (r"\bweek\s+(\d+)\s+e\b", r"week \1"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z]+|\d+(?:\.\d+)?|[\u0980-\u09ff]+", text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _clo_number(question: str) -> str:
    match = re.search(r"\bclo\s*(\d+)\b", normalize_query_for_retrieval(question), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def detect_retrieval_course_code(question: str) -> str:
    return detect_course_code(question)


def _topic_subject(question: str) -> str:
    normalized = normalize_query_for_retrieval(question)
    patterns = [
        r"^(.+?)\s+er\s+under\b",
        r"^(.+?)\s+er\s+odhine\b",
        r"^(.+?)\s+এর\s+অধীনে\b",
        r"\bunder\s+(?:the\s+)?(.+?)(?:\s+topic|\s+topics|$)",
        r"\b(?:types?\s+of|ধরন(?:গুলো)?|type)\s+(.+?)(?:\s+mentioned|\s+in\b|$)",
        r"\bmention(?:ed)?\s+kora\s+(.+?)\s+er\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip(" -:.,")
            if subject and subject not in {"course", "topic", "topics"}:
                return subject
    return ""


def build_query_profile(question: str) -> QueryProfile:
    normalized = normalize_query_for_retrieval(question)
    course_code = detect_retrieval_course_code(question)
    return QueryProfile(
        original=question,
        normalized=normalized,
        tokens=_tokens(normalized),
        course_code=course_code,
        intent=detect_runtime_intent(question),
        clo_number=_clo_number(question),
        week_number=extract_week_number(question),
        topic_subject=_topic_subject(question),
    )


def _section_matches(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def section_tags(text: str) -> set[str]:
    tags = set()
    for intent, patterns in INTENT_SECTION_PATTERNS.items():
        if _section_matches(text, patterns):
            tags.add(intent)
    return tags


def _course_component(profile: QueryProfile, source: str, text: str) -> tuple[float, float]:
    if not profile.course_code:
        return 0.0, 0.0
    compact_requested = re.sub(r"[^A-Za-z0-9]", "", profile.course_code).casefold()
    compact_text = re.sub(r"[^A-Za-z0-9]", "", text).casefold()
    if compact_requested in compact_text:
        return (
            RETRIEVAL_RERANK_WEIGHTS["course_match_boost"]
            + RETRIEVAL_RERANK_WEIGHTS["course_text_boost"],
            0.0,
        )
    other_codes = re.findall(r"\b[A-Za-z]{2,12}\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?\b", text)
    if other_codes:
        return 0.0, RETRIEVAL_RERANK_WEIGHTS["course_mismatch_penalty"]
    return 0.0, 0.0


def _intent_component(profile: QueryProfile, text: str) -> tuple[float, float]:
    patterns = INTENT_SECTION_PATTERNS.get(profile.intent, [])
    matches = _section_matches(text, patterns)
    if matches == 0:
        return 0.0, 0.0
    if profile.intent == "weekly_content" and profile.week_number is None:
        return 0.0, 0.0
    return RETRIEVAL_RERANK_WEIGHTS["intent_match_boost"], RETRIEVAL_RERANK_WEIGHTS["section_heading_boost"] * min(matches, 2)


def _requested_field_component(profile: QueryProfile, text: str) -> float:
    if profile.intent == "clo" and profile.clo_number:
        return RETRIEVAL_RERANK_WEIGHTS["requested_field_boost"] if re.search(
            rf"\bCLO\s*{re.escape(profile.clo_number)}\b",
            text,
            flags=re.IGNORECASE,
        ) else 0.0

    if profile.intent == "weekly_content" and profile.week_number is not None:
        return RETRIEVAL_RERANK_WEIGHTS["requested_field_boost"] if re.search(
            rf"\bWeek\s+{profile.week_number}\b",
            text,
            flags=re.IGNORECASE,
        ) else 0.0

    if profile.intent == "topics" and profile.topic_subject:
        subject_pattern = r"\s+".join(re.escape(part) for part in profile.topic_subject.split())
        if re.search(rf"\b{subject_pattern}\s*:", text, flags=re.IGNORECASE):
            return RETRIEVAL_RERANK_WEIGHTS["topic_subject_boost"]
        if re.search(rf"\b{subject_pattern}\b", text, flags=re.IGNORECASE):
            return RETRIEVAL_RERANK_WEIGHTS["topic_subject_boost"] / 2

    if profile.intent == "final_exam_marks":
        return RETRIEVAL_RERANK_WEIGHTS["requested_field_boost"] if re.search(
            r"\b(?:Final Exam|Term Examination)\b[^\d%]{0,50}\d+(?:\.\d+)?\s*%",
            text,
            flags=re.IGNORECASE,
        ) else 0.0

    requested_fields = {
        "course_credit": [r"Credit Value"],
        "course_type": [r"Course Type"],
        "course_title": [r"Course Title"],
        "course_code": [r"Course No\.", r"Course Code"],
    }
    if profile.intent in requested_fields and _section_matches(text, requested_fields[profile.intent]):
        return RETRIEVAL_RERANK_WEIGHTS["requested_field_boost"]
    return 0.0


def _topic_plan_component(profile: QueryProfile, text: str) -> float:
    if profile.intent not in {"topics", "weekly_content"}:
        return 0.0

    score = 0.0
    if re.search(r"Alignment of topics|Topics\s*/\s*Content|Class Schedule/Lesson Plan/Weekly plan", text, flags=re.IGNORECASE):
        score += RETRIEVAL_RERANK_WEIGHTS["topic_plan_boost"]
    elif re.search(r"\bWeek\s+\d+\b.*\bCLO\d\b|\bCLO\d\b.*\bWeek\s+\d+\b", text, flags=re.IGNORECASE):
        score += RETRIEVAL_RERANK_WEIGHTS["topic_plan_boost"] / 2

    if profile.intent == "topics" and re.search(r"Assessment Techniques|Assessment Strategy|Evaluation Policy|Part C", text, flags=re.IGNORECASE):
        score += RETRIEVAL_RERANK_WEIGHTS["topic_assessment_penalty"]

    if profile.intent == "weekly_content":
        if profile.week_number is None and re.search(r"MID-TERM EXAMINATION|FINAL EXAMINATION", text, flags=re.IGNORECASE):
            score += RETRIEVAL_RERANK_WEIGHTS["requested_field_boost"]
        elif re.search(r"\bWeek\s+\d+\b", text, flags=re.IGNORECASE):
            score += RETRIEVAL_RERANK_WEIGHTS["topic_plan_boost"] / 2
    return score


def _metadata_mismatch_component(profile: QueryProfile, text: str) -> float:
    if profile.intent in {
        "course_metadata",
        "course_code",
        "course_title",
        "course_type",
        "course_credit",
        "course_objective",
        "unknown",
    }:
        return 0.0
    has_metadata = bool(re.search(r"Course No\.|Course Code|Course Title|Course Type|Credit Value", text, flags=re.IGNORECASE))
    if profile.intent == "topics":
        has_intent_section = bool(re.search(r"Course Content|Alignment of topics|Topics?\s*/\s*Content", text, flags=re.IGNORECASE))
    else:
        has_intent_section = bool(INTENT_SECTION_PATTERNS.get(profile.intent) and _section_matches(text, INTENT_SECTION_PATTERNS[profile.intent]))
    if has_metadata and not has_intent_section:
        return RETRIEVAL_RERANK_WEIGHTS["metadata_mismatch_penalty"]
    return 0.0


def _lexical_component(profile: QueryProfile, text: str) -> float:
    if not profile.tokens:
        return 0.0
    text_tokens = set(_tokens(text))
    overlap = len(set(profile.tokens) & text_tokens)
    return min(
        RETRIEVAL_RERANK_WEIGHTS["lexical_overlap_cap"],
        overlap * RETRIEVAL_RERANK_WEIGHTS["lexical_overlap_weight"],
    )


def score_candidate_components(profile: QueryProfile, item: Dict[str, Any]) -> Dict[str, float]:
    text = str(item.get("text", ""))
    source = str(item.get("source", ""))
    semantic_score = float(item.get("score", 0.0))
    course_boost, penalty = _course_component(profile, source, text)
    intent_boost, section_boost = _intent_component(profile, text)
    field_boost = _requested_field_component(profile, text)
    lexical_boost = _lexical_component(profile, text)
    topic_plan_boost = _topic_plan_component(profile, text)
    metadata_penalty = _metadata_mismatch_component(profile, text)
    final_score = (
        semantic_score
        + course_boost
        + intent_boost
        + field_boost
        + lexical_boost
        + section_boost
        + topic_plan_boost
        + penalty
        + metadata_penalty
    )
    return {
        "semantic_score": semantic_score,
        "course_boost": course_boost,
        "intent_boost": intent_boost,
        "field_boost": field_boost,
        "lexical_boost": lexical_boost,
        "section_boost": section_boost,
        "topic_plan_boost": topic_plan_boost,
        "penalty": penalty,
        "metadata_penalty": metadata_penalty,
        "final_score": final_score,
    }


def rerank_candidates(question: str, candidates: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    profile = build_query_profile(question)
    ranked = []
    for item in candidates:
        components = score_candidate_components(profile, item)
        reranked = {**item, "score": components["final_score"], "_semantic_score": components["semantic_score"]}
        if debug:
            reranked["rerank_debug"] = components
            reranked["section_tags"] = sorted(section_tags(str(item.get("text", ""))))
        ranked.append(reranked)

    ranked.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("_semantic_score", item.get("score", 0.0))),
            str(item.get("source", "")),
            int(item.get("page") or 0),
        ),
        reverse=True,
    )
    return ranked


def select_retrieval_results(ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    if len(ranked) <= top_k:
        return ranked

    selected = [ranked[0]]
    seen = {(str(ranked[0].get("source", "")), ranked[0].get("chunk_id"))}

    semantic_backfill = sorted(
        ranked[1:],
        key=lambda item: (
            float(item.get("_semantic_score", item.get("score", 0.0))),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    for item in semantic_backfill:
        item_key = (str(item.get("source", "")), item.get("chunk_id"))
        if item_key in seen:
            continue
        selected.append(item)
        seen.add(item_key)
        if len(selected) >= top_k:
            return selected

    for item in ranked[1:]:
        item_key = (str(item.get("source", "")), item.get("chunk_id"))
        if item_key not in seen:
            selected.append(item)
            if len(selected) >= top_k:
                break
    return selected


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel | None = None, top_k: int = TOP_K):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        index_path: str | None = None,
        metadata_path: str | None = None,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if index_path is None or metadata_path is None:
            from .config import VECTOR_DB_DIR
            index_path = str(VECTOR_DB_DIR / "index.faiss")
            metadata_path = str(VECTOR_DB_DIR / "metadata.pkl")

        index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
        query_vector = self.embedding_model.embed(question)
        profile = build_query_profile(question)
        candidate_pool = min(
            len(metadata),
            max(
                int(RETRIEVAL_RERANK_WEIGHTS["candidate_pool_min"]),
                self.top_k * int(RETRIEVAL_RERANK_WEIGHTS["candidate_pool_multiplier"]),
            ),
        )
        raw_results = search(index=index, query_vector=query_vector, metadata=metadata, top_k=candidate_pool)
        ranked = rerank_candidates(question, raw_results, debug=debug)

        normalized = []
        selected = select_retrieval_results(ranked, self.top_k)
        for item in selected:
            result = {
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
                "document_id": item.get("document_id"),
                "source": item.get("source", "unknown"),
                "relative_path": item.get("relative_path"),
                "page": item.get("page", None),
                "chunk_id": item.get("chunk_id", None),
            }
            if debug:
                result["rerank_debug"] = item.get("rerank_debug", {})
                result["section_tags"] = item.get("section_tags", [])
            normalized.append(result)

        return normalized


__all__ = [
    "RETRIEVAL_RERANK_WEIGHTS",
    "Retriever",
    "build_query_profile",
    "detect_retrieval_course_code",
    "normalize_query_for_retrieval",
    "rerank_candidates",
    "score_candidate_components",
    "select_retrieval_results",
    "section_tags",
]
