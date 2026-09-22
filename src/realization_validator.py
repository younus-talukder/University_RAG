from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .language_detector import Language
from .language_validator import validate_language
from .semantic_contract import SemanticContract, detect_polarity, expressed_relations, validate_semantic_contract


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,12}(?:\s*\([A-Z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Z0-9]+\))?\b")
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:\.\d+)?%?(?![\w])")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def protected_facts(text: str) -> dict[str, set[str]]:
    return {
        "emails": {_compact(value) for value in EMAIL_RE.findall(text)},
        "course_codes": {_compact(value) for value in COURSE_CODE_RE.findall(text)},
        "numbers": {_compact(value) for value in NUMBER_RE.findall(text)},
        "acronyms": {_compact(value) for value in ACRONYM_RE.findall(text)},
    }


def validate_realization(
    canonical_answer: str,
    realized_answer: str,
    target_language: Language,
    semantic_contract: SemanticContract,
) -> dict[str, Any]:
    language = validate_language(realized_answer, target_language)
    if not language["validation_passed"]:
        return {
            "passed": False,
            "reason": language["validation_reason"],
            "repairable": True,
            "details": [],
            "language_validation": language,
        }

    canonical_polarity = detect_polarity(canonical_answer)
    realized_polarity = detect_polarity(realized_answer)
    if canonical_polarity != "unknown" and realized_polarity != canonical_polarity:
        return {
            "passed": False,
            "reason": "WRONG_POLARITY",
            "repairable": True,
            "details": [f"canonical={canonical_polarity};realization={realized_polarity}"],
            "language_validation": language,
        }

    canonical_facts = protected_facts(canonical_answer)
    realized_facts = protected_facts(realized_answer)
    missing = [
        f"{category}:{value}"
        for category in canonical_facts
        for value in sorted(canonical_facts[category] - realized_facts[category])
    ]
    if missing:
        return {
            "passed": False,
            "reason": "MISSING_CANONICAL_FACTS",
            "repairable": True,
            "details": missing,
            "language_validation": language,
        }
    added = [
        f"{category}:{value}"
        for category in canonical_facts
        for value in sorted(realized_facts[category] - canonical_facts[category])
    ]
    if added:
        return {
            "passed": False,
            "reason": "UNSUPPORTED_REALIZATION_FACTS",
            "repairable": False,
            "details": added,
            "language_validation": language,
        }

    canonical_relations = set(expressed_relations(canonical_answer))
    realized_relations = set(expressed_relations(realized_answer))
    strict_relations = {"published_by", "accredited_by", "established", "operates_under", "located_at", "prerequisite"}
    required_strict = canonical_relations.intersection(strict_relations)
    if required_strict and not required_strict <= realized_relations:
        return {
            "passed": False,
            "reason": "WRONG_RELATION",
            "repairable": True,
            "details": [f"required={sorted(required_strict)};realized={sorted(realized_relations)}"],
            "language_validation": language,
        }

    # The canonical answer has already passed evidence validation. Reuse the same
    # relation/value contract without the old multilingual uncertainty shortcut.
    semantic = validate_semantic_contract(realized_answer, replace(semantic_contract, target_language="english"))
    if not semantic["passed"]:
        return {
            **semantic,
            "language_validation": language,
        }
    return {
        "passed": True,
        "reason": "CANONICAL_MEANING_PRESERVED",
        "repairable": False,
        "details": [],
        "language_validation": language,
    }


__all__ = ["protected_facts", "validate_realization"]
