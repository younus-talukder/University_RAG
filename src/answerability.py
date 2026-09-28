"""Observable, non-probabilistic trust decisions over verified evidence.

This module never reads evaluation questions, reference answers, or PDF names.
Retrieval scores are deliberately excluded from the decision rules.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Sequence

from .evidence import EvidenceAssessment, SupportStatus
from .language_detector import Language


class AnswerabilityStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    RETRIEVAL_UNCERTAIN = "RETRIEVAL_UNCERTAIN"
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    GENERATION_REJECTED = "GENERATION_REJECTED"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class EvidenceLevel(str, Enum):
    STRONG = "STRONG"
    ADEQUATE = "ADEQUATE"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerability_status: AnswerabilityStatus
    answerability_reason: str
    evidence_level: EvidenceLevel
    requested_entity: str | None = None
    requested_field: str | None = None
    requested_relation: str | None = None
    entity_supported: bool | None = None
    field_supported: bool | None = None
    relation_supported: bool | None = None
    structured_extractable: bool | None = None
    supporting_evidence_count: int = 0
    independent_support_count: int = 0
    source_count: int = 0
    best_support_rank: int | None = None
    retrieval_channels: tuple[str, ...] = ()
    cross_lingual_fallback_used: bool = False
    conflict_detected: bool = False
    ambiguity_detected: bool = False
    generation_required: bool | None = None
    generation_validation_status: str = "NOT_ATTEMPTED"
    final_answer_allowed: bool = False
    conflict_sources: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["answerability_status"] = self.answerability_status.value
        data["evidence_level"] = self.evidence_level.value
        data["retrieval_channels"] = list(self.retrieval_channels)
        data["conflict_sources"] = list(self.conflict_sources)
        return data


_INJECTION_DIRECTIVES = (
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous\s+instructions|university\s+documents?|pdf|provided\s+documents?)\b",
    r"\bdo\s+not\s+use\s+(?:the\s+)?(?:pdf|documents?|sources?)\b",
    r"\banswer\s+from\s+(?:your\s+)?own\s+knowledge\b",
)


def remove_untrusted_directives(question: str) -> str:
    """Discard a small set of overt bypass commands, never answer from them."""
    clean = str(question)
    for pattern in _INJECTION_DIRECTIVES:
        clean = re.sub(pattern, " ", clean, flags=re.I)
    clean = re.sub(r"^[\s,;:.!?-]*(?:and\s+)?", "", clean, flags=re.I)
    return clean.strip(" ,;:.-")


_OOD_PATTERNS = (
    r"\bweather\b|\bforecast\s+(?:today|tomorrow)\b|\btemperature\s+(?:today|tomorrow)\b|\bwill\s+it\s+rain\b",
    r"\b(?:world\s+cup|football\s+match|cricket\s+match)\b.*\b(?:won|winner|score)\b|\bwho\s+won\s+(?:the\s+)?(?:world\s+cup|football\s+match|cricket\s+match|match)\b",
    r"\b(?:give|provide|need|want)\s+(?:me\s+)?(?:medical|health)\s+advice\b|\b(?:diagnose|prescribe)\s+(?:my|me)\b",
    r"\b(?:write|generate|show)\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|java)\s+(?:sorting\s+)?(?:code|program|function|script)\b",
    r"\b(?:who\s+is\s+(?:the\s+)?(?:current\s+)?president|latest\s+(?:political\s+)?news|election\s+result)\b",
    r"\b(?:capital\s+of\s+france|who\s+wrote\s+hamlet|largest\s+planet)\b",
    r"\b(?:should\s+i\s+(?:break\s+up|marry|invest)|relationship\s+advice)\b",
    r"আজকের\s+আবহাওয়া|বিশ্বকাপ\s+কে\s+জিতেছে|চিকিৎসা\s+পরামর্শ",
)


def clearly_out_of_domain(question: str) -> bool:
    clean = remove_untrusted_directives(question)
    if not clean:
        return True
    if not any(re.search(pattern, clean, re.I) for pattern in _OOD_PATTERNS):
        return False
    # Course and institutional context wins over a lexical domain cue. A course
    # about medicine, programming or sport is still a university question.
    if re.search(r"\b(?:university|campus|course|syllabus|curriculum|department|semester|faculty|admission|tuition|prospectus)\b|"
                 r"বিশ্ববিদ্যালয়|কোর্স|সিলেবাস|বিভাগ|ভর্তি", clean, re.I):
        return False
    return True


def safe_response(status: AnswerabilityStatus, language: Language) -> str:
    messages = {
        AnswerabilityStatus.INSUFFICIENT_EVIDENCE: (
            "I couldn't find enough information in the university documents to answer reliably.",
            "বিশ্ববিদ্যালয়ের নথিতে নির্ভরযোগ্যভাবে উত্তর দেওয়ার মতো পর্যাপ্ত তথ্য পাওয়া যায়নি।",
            "University document-e reliable answer deyar moto porjapto tothyo paoa jayni.",
        ),
        AnswerabilityStatus.RETRIEVAL_UNCERTAIN: (
            "I found related information, but couldn't verify that it answers this specific question.",
            "সম্পর্কিত তথ্য পাওয়া গেছে, কিন্তু তা এই নির্দিষ্ট প্রশ্নের উত্তর কি না নিশ্চিত করা যায়নি।",
            "Related tothyo paoa geche, kintu eta ei specific proshner uttor kina verify kora jayni.",
        ),
        AnswerabilityStatus.AMBIGUOUS_QUERY: (
            "Which course, document, program, or policy do you mean?",
            "আপনি কোন কোর্স, নথি, প্রোগ্রাম বা নীতির কথা বলছেন?",
            "Apni kon course, document, program, ba policy-r kotha bolchen?",
        ),
        AnswerabilityStatus.CONFLICTING_EVIDENCE: (
            "The university documents contain conflicting information about this.",
            "বিশ্ববিদ্যালয়ের নথিতে এই বিষয়ে পরস্পরবিরোধী তথ্য রয়েছে।",
            "University document-gulote ei bishoye conflicting tothyo ache.",
        ),
        AnswerabilityStatus.GENERATION_REJECTED: (
            "Relevant information was found, but I couldn't produce a reliable answer.",
            "প্রাসঙ্গিক তথ্য পাওয়া গেছে, কিন্তু নির্ভরযোগ্য উত্তর তৈরি করা যায়নি।",
            "Relevant tothyo paoa geche, kintu reliable uttor toiri kora jayni.",
        ),
        AnswerabilityStatus.OUT_OF_DOMAIN: (
            "I can answer questions supported by the provided university documents, but not this one.",
            "আমি দেওয়া বিশ্ববিদ্যালয়ের নথিভিত্তিক প্রশ্নের উত্তর দিতে পারি, কিন্তু এই প্রশ্নটির নয়।",
            "Ami deya university document-er tothyo diye proshner uttor dite pari, kintu ei proshner noy.",
        ),
        AnswerabilityStatus.SYSTEM_ERROR: (
            "I couldn't check the university documents right now. Please try again later.",
            "এখন বিশ্ববিদ্যালয়ের নথি যাচাই করা যাচ্ছে না। পরে আবার চেষ্টা করুন।",
            "Ekhon university document verify kora jacche na. Pore abar cheshta korun.",
        ),
    }
    index = {"english": 0, "bangla": 1, "banglish": 2}[language]
    return messages[status][index]


def _identity(item: dict[str, Any]) -> tuple[str, str | int | None]:
    return (str(item.get("document_id") or item.get("relative_path") or item.get("source") or ""), item.get("chunk_id"))


def _parent_identity(item: dict[str, Any], lookup: dict[tuple[str, str | int | None], dict[str, Any]]) -> tuple[str, str, str]:
    original = lookup.get(_identity(item), {})
    document = str(item.get("document_id") or item.get("relative_path") or item.get("source") or "")
    chunk = str(item.get("chunk_id") or "")
    parent = str(item.get("parent_id") or original.get("parent_id") or re.sub(r"-c\d{4}$", "", chunk))
    if not parent:
        parent = f"page:{item.get('page')}:{chunk}"
    return document, str(item.get("page") or ""), parent


def assess_answerability(
    question: str,
    assessment: EvidenceAssessment,
    retrieved: Sequence[dict[str, Any]],
    *,
    cross_lingual_fallback_used: bool = False,
    requested_relation: str | None = None,
    relation_supported: bool | None = None,
    generation_required: bool | None = None,
) -> AnswerabilityDecision:
    """Apply one language-independent rule set to existing evidence signals."""
    lookup = {_identity(item): item for item in retrieved}
    verified = [item.to_dict() for item in assessment.evidence]
    independent = {_parent_identity(item, lookup) for item in verified}
    sources = {str(item.get("document_id") or item.get("relative_path") or item.get("source") or "") for item in verified}
    matched = {_identity(item) for item in verified}
    ranks = [rank for rank, item in enumerate(retrieved, 1) if _identity(item) in matched]
    channels: set[str] = set()
    for item in retrieved:
        if _identity(item) not in matched:
            continue
        if item.get("dense_rank") is not None:
            channels.add("dense")
        if item.get("sparse_rank") is not None:
            channels.add("bm25")
        if item.get("metadata_rank") is not None:
            channels.add("metadata")
        for provenance in (item.get("retrieval_provenance") or {}).values():
            if provenance.get("dense_rank") is not None:
                channels.add("dense")
            if provenance.get("sparse_rank") is not None:
                channels.add("bm25")
            if provenance.get("metadata_rank") is not None:
                channels.add("metadata")
        if item.get("canonical_query_rank") is not None:
            channels.add("cross_lingual")
    request = assessment.request
    # assess_evidence has already checked *all* requested entities against the
    # same passage; Evidence.matched_entity stores only its primary display name.
    entity_supported = bool(verified)
    field_supported = bool(verified and all(item.get("matched_field") == request.requested_field for item in verified))
    conflict_sources = tuple({"source": item["source"], "page": item["page"], "excerpt": item["excerpt"]} for item in verified) if assessment.status is SupportStatus.CONFLICTING else ()
    common = dict(
        requested_entity=request.primary_entity, requested_field=request.requested_field,
        requested_relation=requested_relation, entity_supported=entity_supported,
        field_supported=field_supported, relation_supported=relation_supported,
        supporting_evidence_count=len(verified), independent_support_count=len(independent),
        source_count=len(sources), best_support_rank=min(ranks) if ranks else None,
        retrieval_channels=tuple(sorted(channels)),
        cross_lingual_fallback_used=cross_lingual_fallback_used,
        conflict_detected=assessment.status is SupportStatus.CONFLICTING,
        ambiguity_detected=assessment.status is SupportStatus.AMBIGUOUS,
        generation_required=generation_required, conflict_sources=conflict_sources,
    )
    if assessment.status is SupportStatus.AMBIGUOUS:
        return AnswerabilityDecision(AnswerabilityStatus.AMBIGUOUS_QUERY, "AMBIGUOUS_ENTITY", EvidenceLevel.NONE, **common)
    if assessment.status is SupportStatus.CONFLICTING:
        return AnswerabilityDecision(AnswerabilityStatus.CONFLICTING_EVIDENCE, "CONFLICTING_VALUES", EvidenceLevel.WEAK, **common)
    if assessment.status is SupportStatus.INSUFFICIENT or not verified:
        # A same-entity/same-field lexical hit rejected by the structural owner
        # gate is a retrieval ambiguity, not evidence of the requested relation.
        owner_uncertain = bool(request.entities and any(
            all(entity.normalized in re.sub(r"[^A-Za-z0-9]", "", str(item.get("text") or "")).upper()
                for entity in request.entities if entity.kind == "course_code")
            and request.requested_field in (item.get("field_types") or [])
            for item in retrieved
        ))
        status = AnswerabilityStatus.RETRIEVAL_UNCERTAIN if owner_uncertain else AnswerabilityStatus.INSUFFICIENT_EVIDENCE
        reason = "ENTITY_OWNERSHIP_UNCERTAIN" if owner_uncertain else ("ENTITY_NOT_SUPPORTED" if request.entities else "FIELD_NOT_SUPPORTED")
        level = EvidenceLevel.WEAK if owner_uncertain else EvidenceLevel.NONE
        return AnswerabilityDecision(status, reason, level, **common)
    if relation_supported is False:
        return AnswerabilityDecision(AnswerabilityStatus.RETRIEVAL_UNCERTAIN, "RELATION_NOT_SUPPORTED", EvidenceLevel.WEAK, **common)
    if not entity_supported or not field_supported:
        return AnswerabilityDecision(AnswerabilityStatus.RETRIEVAL_UNCERTAIN, "ENTITY_OR_FIELD_NOT_VERIFIED", EvidenceLevel.WEAK, **common)
    labeled = any(item.get("entity_type") == "course" or request.requested_field in (item.get("field_types") or [])
                  or request.requested_field == "document_metadata"
                  for item in retrieved if _identity(item) in matched)
    level = EvidenceLevel.STRONG if (labeled or relation_supported or len(independent) >= 2) else EvidenceLevel.ADEQUATE
    reason = ("DIRECT_RELATION_SUPPORT" if relation_supported else "DIRECT_STRUCTURED_SUPPORT" if labeled
              else "MULTIPLE_AGREEING_PASSAGES" if len(independent) >= 2 else "SINGLE_VERIFIED_PASSAGE")
    return AnswerabilityDecision(AnswerabilityStatus.SUPPORTED, reason, level, **common)


def finalize_answerability(
    decision: AnswerabilityDecision,
    *,
    answer_present: bool,
    language_passed: bool,
    grounding_passed: bool,
    semantic_passed: bool,
    generation_used: bool,
    rejection_reason: str = "",
) -> AnswerabilityDecision:
    if decision.answerability_status is not AnswerabilityStatus.SUPPORTED:
        return decision
    if answer_present and language_passed and grounding_passed and semantic_passed:
        return replace(decision, generation_validation_status="ACCEPTED" if generation_used else "NOT_REQUIRED",
                       final_answer_allowed=True)
    lowered = rejection_reason.casefold()
    reason = ("GENERATION_LANGUAGE_FAILURE" if not language_passed or "language" in lowered else
              "GENERATION_SEMANTIC_FAILURE" if not semantic_passed or any(token in lowered for token in ("semantic", "polarity", "relation")) else
              "GENERATION_GROUNDING_FAILURE" if not grounding_passed else "NO_SAFE_ANSWER_FROM_SUPPORTED_EVIDENCE")
    return replace(decision, answerability_status=AnswerabilityStatus.GENERATION_REJECTED,
                   answerability_reason=reason, generation_validation_status=f"REJECTED:{reason}",
                   final_answer_allowed=False)


__all__ = ["AnswerabilityDecision", "AnswerabilityStatus", "EvidenceLevel", "assess_answerability",
           "clearly_out_of_domain", "finalize_answerability", "remove_untrusted_directives", "safe_response"]
