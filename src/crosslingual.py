"""Conservative auxiliary retrieval for multilingual, evidence-insufficient queries.

No evaluation data, document-specific names, answers, or second model live here.
"""

from __future__ import annotations

import re
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .evidence import analyze_query, assess_evidence, SupportStatus
from .query_normalization import extract_course_entities, normalize_retrieval_text, normalize_unicode

SCHEMA_VERSION = "step7f-v1"
_CACHE: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
_CACHE_LIMIT = 128

# Small, generic academic concepts. These are search terms, never answer facts.
CONCEPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "credits": ("credits", (r"\bcredits?\b", r"ক্রেডিট", r"\bkredit\b")),
    "prerequisite": ("prerequisite", (r"\bpre\s*[- ]?req(?:uisite)?s?\b", r"পূর্বশর্ত", r"প্রিরিকুইজিট")),
    "course_title": ("course title", (r"\btitle\b", r"শিরোনাম", r"কোর্সের নাম")),
    "course_code": ("course code", (r"\bcode\b", r"কোড")),
    "attendance": ("attendance", (r"\battendance\b", r"উপস্থিতি", r"হাজিরা")),
    "assessment": ("marks assessment", (r"\bmarks?\b", r"মার্কস", r"নম্বর", r"পরীক্ষা")),
    "semester": ("semester", (r"\bsemester\b", r"সেমিস্টার")),
    "program": ("degree program", (r"\bprograms?\b", r"প্রোগ্রাম")),
    "publication": ("publication", (r"\bpublish(?:ed|er)?\b", r"প্রকাশ")),
    "publisher": ("publisher", (r"\bpublisher\b", r"প্রকাশক")),
    "accreditation": ("accreditation", (r"\baccreditation\b", r"অ্যাক্রেডিটেশন")),
    "topic": ("course topics", (r"\btopics?\b", r"বিষয়বস্তু", r"বিষয়")),
    "objective": ("objective", (r"\bobjective\b", r"উদ্দেশ্য")),
    "learning_outcome": ("CLO learning outcome", (r"\bCLO\b", r"শিখনফল")),
    "weekly_content": ("weekly content", (r"\bweekly\b", r"সাপ্তাহিক", r"সপ্তাহ")),
    "requirement": ("requirement", (r"\brequirement\b", r"প্রয়োজন", r"যোগ্যতা")),
    "percentage": ("percentage", (r"\bpercent(?:age)?\b", r"শতাংশ", r"\d+%")),
    "date": ("date", (r"\bdate\b", r"\bwhen\b", r"তারিখ", r"কবে", r"কখন")),
    "exam": ("exam", (r"\bexam\b", r"পরীক্ষা")),
    "disclaimer": ("disclaimer", (r"\bdisclaimer\b", r"ডিসক্লেইমার")),
    "registration": ("registration", (r"\bregistration\b", r"রেজিস্ট্রেশন")),
    "prospectus": ("prospectus", (r"\bprospectus\b", r"প্রসপেক্টাস")),
    "establishment": ("established", (r"\bestablish(?:ed|ment)?\b", r"প্রতিষ্ঠিত")),
    "graduation": ("graduates degree", (r"\bgraduates?\b", r"গ্র্যাজুয়েট")),
    "law": ("act law", (r"\bact\b", r"\blaw\b", r"আইন")),
    "discipline": ("disciplines", (r"\bdisciplines?\b", r"ডিসিপ্লিন")),
    "category": ("category", (r"\bcategory\b", r"ক্যাটাগরি")),
    "edition": ("edition", (r"\bedition\b", r"সংস্করণ")),
}


@dataclass(frozen=True)
class Rewrite:
    query: str
    method: str
    validation: str


def detect_chunk_language(text: str) -> str:
    bengali = len(re.findall(r"[\u0980-\u09ff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if bengali + latin < 20:
        return "unknown"
    if bengali >= 20 and latin >= 20 and min(bengali, latin) / max(bengali, latin) >= 0.2:
        return "mixed"
    return "bangla_dominant" if bengali > latin else "english_dominant"


def detected_concepts(question: str) -> tuple[str, ...]:
    normal = normalize_unicode(question)
    return tuple(name for name, (_, patterns) in CONCEPTS.items()
                 if any(re.search(pattern, normal, re.I) for pattern in patterns))


def _protected(text: str) -> set[str]:
    normal = normalize_unicode(text)
    tokens = {entity.compact for entity in extract_course_entities(normal)}
    tokens.update(x.casefold() for x in re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normal))
    tokens.update(re.sub(r"\s+", "", x) for x in re.findall(r"\d+(?:\.\d+)?\s*%", normal))
    tokens.update(x.casefold() for x in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b", normal))
    tokens.update(re.findall(r"\b\d+(?:\.\d+)?\b", normal))
    tokens.update(x.upper() for x in re.findall(r"\b[A-Z]{2,}(?:\.[A-Z]+)*\b", normal))
    return tokens


def validate_rewrite(original: str, rewritten: str) -> tuple[bool, str]:
    query = rewritten.strip().strip('"\' ')
    if not query or len(query) > 180 or "\n" in query or re.search(r"[\u0980-\u09ff]", query):
        return False, "NOT_CONCISE_ENGLISH_QUERY"
    original_facts, rewritten_facts = _protected(original), _protected(query)
    if not original_facts.issubset(rewritten_facts):
        return False, "PROTECTED_FACT_CHANGED"
    if rewritten_facts - original_facts:
        return False, "NEW_PROTECTED_FACT"
    original_entities = analyze_query(original).entities
    rewritten_entities = analyze_query(query).entities
    for entity in original_entities:
        if not any(other.kind == entity.kind and other.normalized == entity.normalized
                   for other in rewritten_entities):
            return False, "ENTITY_CHANGED"
    original_field, rewritten_field = analyze_query(original).requested_field, analyze_query(query).requested_field
    if original_field != "general" and rewritten_field not in {original_field, "general"}:
        return False, "FIELD_CHANGED"
    concepts = set(detected_concepts(original))
    if concepts and not concepts.intersection(detected_concepts(query)):
        return False, "CONCEPT_CHANGED"
    return True, "VALIDATED"


def deterministic_query(question: str) -> Rewrite | None:
    request = analyze_query(question)
    concepts = detected_concepts(question)
    entities = [entity.raw for entity in extract_course_entities(question)]
    # Course field is reliable; context-free generic questions need at least two
    # independently detected concepts. Never infer an answer or document fact.
    if not entities and len(concepts) < 2:
        return None
    terms: list[str] = [*entities]
    for name in concepts[:5]:
        term = CONCEPTS[name][0]
        if term not in terms:
            terms.append(term)
    if request.requested_field != "general" and request.requested_field in CONCEPTS:
        term = CONCEPTS[request.requested_field][0]
        if term not in terms:
            terms.append(term)
    # Preserve non-course quantities, dates, email addresses and official acronyms.
    normal = normalize_unicode(question)
    for token in re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d+(?:\.\d+)?%?\b|\b[A-Z]{2,}(?:\.[A-Z]+)*\b", normal):
        if token not in terms and not any(token in entity for entity in entities):
            terms.append(token)
    query = " ".join(terms)
    valid, reason = validate_rewrite(question, query)
    return Rewrite(query, "DETERMINISTIC", reason) if valid else None


def canonical_rewrite(question: str, qwen_rewrite: Callable[[str], str] | None = None) -> Rewrite:
    # Explicit target-language switch for future multilingual document corpora.
    # English is only an auxiliary representation for the present configuration.
    if os.getenv("CROSSLINGUAL_RETRIEVAL_TARGET", "english").casefold() != "english":
        return Rewrite("", "NONE", "TARGET_LANGUAGE_NOT_CONFIGURED_FOR_ENGLISH_REWRITE")
    key = (normalize_retrieval_text(question), SCHEMA_VERSION)
    if key in _CACHE:
        query, method = _CACHE[key]
        _CACHE.move_to_end(key)
        return Rewrite(query, method, "VALIDATED_CACHED")
    deterministic = deterministic_query(question)
    if deterministic:
        result = deterministic
    elif qwen_rewrite:
        candidate = qwen_rewrite(question)
        valid, reason = validate_rewrite(question, candidate)
        result = Rewrite(candidate.strip(), "QWEN", reason) if valid else Rewrite("", "NONE", reason)
    else:
        result = Rewrite("", "NONE", "NO_RELIABLE_DETERMINISTIC_QUERY")
    if result.query:
        _CACHE[key] = (result.query, result.method)
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return result


def fuse_queries(original: Sequence[dict[str, Any]], canonical: Sequence[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Equal-weight rank fusion: 1/(60+original_rank)+1/(60+canonical_rank)."""
    merged: dict[str, dict[str, Any]] = {}
    for label, items in (("original", original), ("canonical", canonical)):
        for rank, item in enumerate(items, 1):
            key = str(item.get("chunk_id") or f"{item.get('source')}:{item.get('page')}:{rank}")
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["original_query_rank"] = None
                merged[key]["canonical_query_rank"] = None
                merged[key]["multi_query_score"] = 0.0
                merged[key]["retrieval_provenance"] = {}
            merged[key][f"{label}_query_rank"] = rank
            merged[key]["multi_query_score"] += 1.0 / (60 + rank)
            merged[key]["retrieval_provenance"][label] = {
                "dense_rank": item.get("dense_rank"), "sparse_rank": item.get("sparse_rank"),
                "metadata_rank": item.get("metadata_rank"), "rrf_rank": rank,
            }
    ranked = sorted(merged.values(), key=lambda x: (-x["multi_query_score"],
                    x["original_query_rank"] or 10**6, x["canonical_query_rank"] or 10**6,
                    str(x.get("chunk_id"))))
    for item in ranked:
        item["chunk_language"] = detect_chunk_language(str(item.get("text", "")))
    return ranked[:limit]


def crosslingual_assessment(original_question: str, rewrite: Rewrite, merged: Sequence[dict[str, Any]]):
    """Use the same evidence gate; do not promote a rewrite that loses anchors."""
    original = assess_evidence(original_question, merged)
    if original.status is not SupportStatus.INSUFFICIENT:
        return original, "ORIGINAL_QUERY_GATE"
    if not rewrite.query or rewrite.validation not in {"VALIDATED", "VALIDATED_CACHED"}:
        return original, "REWRITE_NOT_VALIDATED"
    candidate = assess_evidence(rewrite.query, merged)
    if candidate.status is SupportStatus.CONFLICTING:
        return candidate, "CANONICAL_QUERY_CONFLICT"
    if candidate.status is not SupportStatus.SUPPORTED:
        return original, "CANONICAL_QUERY_GATE_FAILED"
    required = [token for token in re.findall(r"[A-Za-z]{3,}", rewrite.query.casefold())
                if token not in {"course", "degree", "university"}]
    evidence_text = " ".join(x.excerpt.casefold() for x in candidate.evidence)
    if required and sum(token in evidence_text for token in required) < min(2, len(required)):
        return original, "CANONICAL_TERMS_NOT_IN_EVIDENCE"
    return candidate, "CANONICAL_QUERY_GATE"
