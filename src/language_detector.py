from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .query_normalization import normalize_retrieval_text

Language = Literal["bangla", "english", "banglish"]

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

BANGLISH_MARKERS = {
    "abar",
    "ache",
    "amar",
    "ami",
    "apni",
    "bujha",
    "chara",
    "deya",
    "dite",
    "e",
    "ebong",
    "er",
    "gulo",
    "hobe",
    "hole",
    "holo",
    "ei",
    "jonno",
    "jay",
    "jayni",
    "jete",
    "ki",
    "kivabe",
    "kobe",
    "kokhon",
    "kon",
    "kora",
    "korbo",
    "korle",
    "korte",
    "koto",
    "kothay",
    "naki",
    "naam",
    "nibo",
    "neya",
    "onushthito",
    "o",
    "paoa",
    "parbo",
    "porano",
    "ta",
    "hoy",
}

ENGLISH_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

UNIVERSITY_TERMS = {
    "cgpa",
    "clo",
    "course",
    "credit",
    "exam",
    "grade",
    "plo",
    "probation",
    "retake",
    "semester",
    "requirement",
    "prerequisite",
}


@dataclass(frozen=True)
class LanguageDetection:
    language: Language
    reason: str
    banglish_markers: tuple[str, ...] = ()
    english_function_words: tuple[str, ...] = ()
    university_terms: tuple[str, ...] = ()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:\.\d+)?", normalize_retrieval_text(text))


def detect_language_details(question: str) -> LanguageDetection:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    text = question.strip()
    if BANGLA_RE.search(text):
        return LanguageDetection("bangla", "Bengali Unicode characters detected.")

    if not LATIN_RE.search(text):
        raise ValueError("Language detection failed: question has no Bangla or Latin text.")

    tokens = _tokens(text)
    if not tokens:
        raise ValueError("Language detection failed: question has no usable tokens.")

    banglish = tuple(sorted({token for token in tokens if token in BANGLISH_MARKERS}))
    english = tuple(sorted({token for token in tokens if token in ENGLISH_FUNCTION_WORDS}))
    university = tuple(sorted({token for token in tokens if token in UNIVERSITY_TERMS}))
    banglish_hits = len(banglish)
    english_hits = len(english)
    university_hits = len(university)

    # Latin script alone is not English. Banglish often mixes English university
    # terms with Bangla grammar markers, so style markers outweigh domain terms.
    if banglish_hits >= 2:
        return LanguageDetection("banglish", "Multiple Latin-script Bangla grammar markers detected.", banglish, english, university)
    if banglish_hits >= 1 and university_hits >= 1:
        return LanguageDetection("banglish", "Banglish grammar plus university terminology detected.", banglish, english, university)
    if re.search(r"\b[A-Za-z]+-(er|e|gulo)\b", text.lower()):
        return LanguageDetection("banglish", "Banglish suffix attachment detected.", banglish, english, university)
    if banglish_hits >= 1 and english_hits <= 2:
        return LanguageDetection("banglish", "Latin-script Bangla marker detected with little English grammar.", banglish, english, university)

    return LanguageDetection("english", "Latin text without sufficient Banglish grammar signals.", banglish, english, university)


def detect_language(question: str) -> Language:
    return detect_language_details(question).language


__all__ = ["Language", "LanguageDetection", "detect_language", "detect_language_details"]
