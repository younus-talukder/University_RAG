from __future__ import annotations

import re
from typing import Dict

from .language_detector import BANGLISH_MARKERS, Language


LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
BANGLA_WORD_RE = re.compile(r"[\u0980-\u09FF]+")
BANGLISH_SUFFIX_RE = re.compile(r"\b[A-Za-z]+-(?:er|e|te|tir|ta|gulo|gulor)\b", re.I)
MIXED_SCRIPT_TOKEN_RE = re.compile(r"(?:[\u0980-\u09FF][A-Za-z]|[A-Za-z][\u0980-\u09FF])")
UNEXPECTED_SCRIPT_RE = re.compile(r"[\u0400-\u052F\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF]")
BANGLISH_STRUCTURE = {
    "holo", "ache", "jonno", "ebong", "korte", "kora", "hoy", "deya", "lagbe",
    "dara", "theke", "pare", "peyeche", "peyecchhe", "chilo", "niye", "ki", "koyta",
}


def _signals(text: str) -> dict[str, object]:
    bangla_words = BANGLA_WORD_RE.findall(text)
    latin_words = [value.casefold() for value in LATIN_WORD_RE.findall(text)]
    bangla_letters = sum(len(value) for value in bangla_words)
    latin_letters = sum(len(value) for value in latin_words)
    markers = sorted({value for value in latin_words if value in BANGLISH_MARKERS or value in BANGLISH_STRUCTURE} | {
        value for value in latin_words if value in {"korun", "dekhun", "pawa", "paoa", "nei"}
    })
    if BANGLISH_SUFFIX_RE.search(text):
        markers.append("suffix")
    return {
        "bangla_word_count": len(bangla_words),
        "latin_word_count": len(latin_words),
        "bangla_letter_count": bangla_letters,
        "latin_letter_count": latin_letters,
        "bangla_letter_share": bangla_letters / max(1, bangla_letters + latin_letters),
        "banglish_markers": sorted(set(markers)),
    }


def validate_language(answer: str, expected_language: Language) -> Dict[str, object]:
    stripped = str(answer or "").strip()
    signals = _signals(stripped)
    bangla_words = int(signals["bangla_word_count"])
    latin_words = int(signals["latin_word_count"])
    share = float(signals["bangla_letter_share"])
    markers = list(signals["banglish_markers"])

    if not stripped:
        passed, response_language, reason = False, "unknown", "EMPTY_ANSWER"
    elif UNEXPECTED_SCRIPT_RE.search(stripped):
        passed, response_language, reason = False, "other", "UNEXPECTED_FOREIGN_SCRIPT"
    elif MIXED_SCRIPT_TOKEN_RE.search(stripped):
        passed, response_language, reason = False, "mixed", "MIXED_SCRIPT_TOKEN"
    elif expected_language == "bangla":
        # Official names, addresses, course codes, and acronyms can legitimately
        # dominate a short Bengali sentence. Require Bengali words plus a clear
        # Bengali grammatical predicate/suffix rather than a raw script ratio alone.
        bangla_grammar = bool(re.search(
            r"(?:হলো|হয়|ছিল|করেছে|পেয়েছে|অবস্থিত|প্রকাশ|প্রতিষ্ঠিত|থেকে|জন্য|সালে|ঠিকানাটি|টি|টা)(?:[\s।,.]|$)",
            stripped,
        ))
        passed = bangla_words >= 2 and (share >= 0.25 or bangla_grammar)
        response_language = "bangla" if passed else ("english" if latin_words else "unknown")
        reason = "BANGLA_BODY_MEANINGFUL" if passed else "BANGLA_BODY_TOO_ENGLISH"
    elif expected_language == "banglish":
        if bangla_words:
            passed, response_language, reason = False, "bangla", "BANGLISH_CONTAINS_BENGALI_SCRIPT"
        elif not latin_words:
            passed, response_language, reason = False, "unknown", "EMPTY_ANSWER"
        elif markers:
            passed, response_language, reason = True, "banglish", "BANGLISH_MARKERS_PRESENT"
        else:
            passed, response_language, reason = False, "english", "BANGLISH_BODY_PURE_ENGLISH"
    else:
        if bangla_words >= 2 or share >= 0.15:
            passed, response_language, reason = False, "bangla", "ENGLISH_CONTAINS_BENGALI_SENTENCE"
        elif markers:
            passed, response_language, reason = False, "banglish", "ENGLISH_CONTAINS_BANGLISH_FRAMING"
        else:
            passed = bool(latin_words)
            response_language = "english" if latin_words else "unknown"
            reason = "ENGLISH_BODY_VALID" if latin_words else "EMPTY_ANSWER"
    return {
        "expected_language": expected_language,
        "response_language": response_language,
        "language_consistency": passed,
        "language_validation_failed": not passed,
        "validation_passed": passed,
        "validation_reason": reason,
        **signals,
    }


def detect_response_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    signals = _signals(text)
    if int(signals["bangla_word_count"]):
        return "bangla"
    if signals["banglish_markers"]:
        return "banglish"
    return "english" if LATIN_WORD_RE.search(text) else "unknown"


def unsupported_answer(language: Language) -> str:
    if language == "bangla":
        return "উপলব্ধ বিশ্ববিদ্যালয়ের নথিতে এই তথ্যটি নির্ভরযোগ্যভাবে পাওয়া যায়নি।"
    if language == "banglish":
        return "Available university document-e ei information-ta reliable vabe paoa jayni."
    return "The information could not be found reliably in the available university documents."


def generation_rejected_answer(language: Language) -> str:
    if language == "bangla":
        return "প্রাসঙ্গিক তথ্য বিশ্ববিদ্যালয়ের নথিতে পাওয়া গেছে, কিন্তু এই মুহূর্তে আমি সেটির একটি নির্ভরযোগ্য বাংলা উত্তর তৈরি করতে পারিনি।"
    if language == "banglish":
        return "Relevant information university document-e paoa geche, kintu ei muhurte ami eta reliable Banglish-e present korte parini."
    return "Relevant information was found in the university documents, but a reliable answer could not be produced at this time."


__all__ = ["detect_response_language", "generation_rejected_answer", "unsupported_answer", "validate_language"]
