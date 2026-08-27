from __future__ import annotations

import re
from typing import Literal

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
    "kora",
    "korbo",
    "korle",
    "korte",
    "koto",
    "kothay",
    "naki",
    "nibo",
    "o",
    "paoa",
    "parbo",
    "ta",
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
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:\.\d+)?", text.lower())


def detect_language(question: str) -> Language:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    text = question.strip()
    if BANGLA_RE.search(text):
        return "bangla"

    if not LATIN_RE.search(text):
        raise ValueError("Language detection failed: question has no Bangla or Latin text.")

    tokens = _tokens(text)
    if not tokens:
        raise ValueError("Language detection failed: question has no usable tokens.")

    banglish_hits = sum(1 for token in tokens if token in BANGLISH_MARKERS)
    english_hits = sum(1 for token in tokens if token in ENGLISH_FUNCTION_WORDS)
    university_hits = sum(1 for token in tokens if token in UNIVERSITY_TERMS)

    # Latin script alone is not English. Banglish often mixes English university
    # terms with Bangla grammar markers, so style markers outweigh domain terms.
    if banglish_hits >= 2:
        return "banglish"
    if banglish_hits >= 1 and university_hits >= 1:
        return "banglish"
    if re.search(r"\b[A-Za-z]+-(er|e|gulo)\b", text.lower()):
        return "banglish"
    if banglish_hits >= 1 and english_hits <= 2:
        return "banglish"

    return "english"


__all__ = ["Language", "detect_language"]
