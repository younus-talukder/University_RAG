from __future__ import annotations

from typing import Dict

from .language_detector import BANGLA_RE, Language, detect_language


def detect_response_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"

    stripped = text.strip()
    if BANGLA_RE.search(stripped):
        return "bangla"

    try:
        return detect_language(stripped)
    except ValueError:
        return "unknown"


def validate_language(answer: str, expected_language: Language) -> Dict[str, object]:
    response_language = detect_response_language(answer)
    language_consistency = response_language == expected_language
    return {
        "expected_language": expected_language,
        "response_language": response_language,
        "language_consistency": language_consistency,
        "language_validation_failed": not language_consistency,
    }


def unsupported_answer(language: Language) -> str:
    if language == "bangla":
        return "উপলব্ধ বিশ্ববিদ্যালয়ের নথিতে এই তথ্যটি পাওয়া যায়নি।"
    if language == "banglish":
        return "Available university documents-e ei information ta paoa jayni."
    return "The information could not be found in the available university documents."


__all__ = ["detect_response_language", "unsupported_answer", "validate_language"]
