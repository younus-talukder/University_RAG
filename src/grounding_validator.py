from __future__ import annotations

import re
from typing import Any, Sequence

from .semantic_contract import SemanticContract, validate_semantic_contract


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,12}(?:\s*\([A-Z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Z0-9]+\))?\b")
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:\.\d+)?%?(?![\w])")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
PROPER_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NON_FACTUAL_CAPITALIZED = {"The", "This", "That", "Course", "University", "Document", "Evidence", "Relevant", "According"}


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _facts(text: str) -> dict[str, set[str]]:
    emails = {_compact(value) for value in EMAIL_RE.findall(text)}
    codes = {_compact(value) for value in COURSE_CODE_RE.findall(text)}
    numbers = {_compact(value) for value in NUMBER_RE.findall(text)}
    acronyms = {_compact(value) for value in ACRONYM_RE.findall(text)}
    proper_words = {_compact(value) for value in PROPER_WORD_RE.findall(text) if value not in NON_FACTUAL_CAPITALIZED}
    return {"emails": emails, "course_codes": codes, "numbers": numbers, "named_identifiers": acronyms | proper_words}


def validate_grounding(
    answer: str,
    evidence: Sequence[Any],
    semantic_contract: SemanticContract | None = None,
) -> dict[str, Any]:
    excerpts = [
        str(item.get("excerpt") or item.get("text") or "") if isinstance(item, dict)
        else str(getattr(item, "excerpt", ""))
        for item in evidence
    ]
    context = "\n".join(excerpts)
    if not answer or not answer.strip():
        return {"grounding_validation_passed": False, "grounding_validation_reason": "EMPTY_ANSWER", "unsupported_facts": []}
    if not context.strip():
        return {"grounding_validation_passed": False, "grounding_validation_reason": "NO_VERIFIED_EVIDENCE", "unsupported_facts": []}
    if re.search(
        r"\b(?:does not provide|not explicitly (?:stated|provided)|could not be found|cannot be found|no reliable information|"
        r"paowa\s+jayni|paoa\s+jayni|deya\s+hoyni)\b|"
        r"(?:(?:তথ্য|ফি|নাম|ঠিকানা)?\s*(?:দেওয়া|প্রদান\s+করা)\s+হয়নি|পাওয়া\s+যায়নি|নির্ভরযোগ্য.*পাওয়া\s+যায়নি)",
        answer,
        re.I,
    ):
        return {
            "grounding_validation_passed": False,
            "grounding_validation_reason": "MODEL_ABSTAINED_DESPITE_VERIFIED_EVIDENCE",
            "unsupported_facts": [],
            "semantic_validation": None,
        }

    semantic = validate_semantic_contract(answer, semantic_contract) if semantic_contract else None
    if semantic and not semantic["passed"]:
        return {
            "grounding_validation_passed": False,
            "grounding_validation_reason": semantic["reason"],
            "unsupported_facts": list(semantic["details"]),
            "semantic_validation": semantic,
        }

    answer_facts = _facts(answer)
    evidence_facts = _facts(context)
    unsupported: list[str] = []
    for category in ("emails", "course_codes", "numbers", "named_identifiers"):
        for value in sorted(answer_facts[category] - evidence_facts[category]):
            # Ordinary language acronyms used by the answer layer are not factual claims.
            if category == "named_identifiers" and value in {"the", "email"}:
                continue
            unsupported.append(f"{category}:{value}")
    return {
        "grounding_validation_passed": not unsupported,
        "grounding_validation_reason": "FACTUAL_TOKENS_SUPPORTED" if not unsupported else "UNSUPPORTED_FACTUAL_TOKENS",
        "unsupported_facts": unsupported,
        "semantic_validation": semantic,
    }


__all__ = ["validate_grounding"]
