from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .language_detector import Language
from .language_validator import unsupported_answer


STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "course", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "why",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _course_code(text: str) -> str:
    match = re.search(r"\b([a-z]{2,4})(?:\s*\([a-z]{2,4}\))?\s*[- ]?\s*(\d{3})\b", text.lower())
    if not match:
        return ""
    prefix = match.group(1)
    number = match.group(2)
    return f"{prefix} {number}"


def retrieve_lexical(question: str, metadata: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    requested_course = _course_code(question)
    ranked = []
    for item in metadata:
        text = str(item.get("text", ""))
        searchable = f"{text} {item.get('source', '')}".lower()
        score = 0.0
        for token in query_tokens:
            if token in searchable:
                score += 1.0
            if re.search(rf"\b{re.escape(token)}\b", searchable):
                score += 1.0

        if requested_course:
            course_pattern = re.escape(requested_course).replace(r"\ ", r"\s*")
            exact_course_re = re.compile(
                rf"course\s+(?:no\.\s*/\s*)?(?:course\s+)?code:\s*{course_pattern}\b"
            )
            loose_course_re = re.compile(rf"\b{course_pattern}\b")
            if exact_course_re.search(searchable):
                score += 20.0
            elif loose_course_re.search(searchable):
                score += 4.0
            else:
                score -= 5.0

        if score > 0:
            ranked.append({**item, "score": score})

    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ranked[:top_k]


def _extract_field(text: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|$)", text)
    if not match:
        return ""
    return match.group(1).strip()


def build_extractive_answer(question: str, retrieved: Sequence[Dict[str, Any]], language: Language) -> str:
    if not retrieved:
        return unsupported_answer(language)

    top_text = re.sub(r"\s+", " ", str(retrieved[0].get("text", ""))).strip()
    title = _extract_field(top_text, "Course Title")
    course_type = _extract_field(top_text, "Course Type")
    credit = _extract_field(top_text, "Credit Value")
    if title:
        details = [f"{title}"]
        if course_type:
            details.append(f"Course type: {course_type}.")
        if credit:
            details.append(f"Credit value: {credit}.")
        return " ".join(details)

    query_tokens = set(_tokens(question))
    sentences: list[str] = []
    for item in retrieved:
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence_tokens = set(_tokens(sentence))
            if query_tokens & sentence_tokens:
                sentences.append(sentence.strip())
            if len(sentences) >= 3:
                break
        if len(sentences) >= 3:
            break

    if not sentences:
        text = re.sub(r"\s+", " ", str(retrieved[0].get("text", ""))).strip()
        sentences = [text[:700].rstrip()]

    prefix = {
        "bangla": "উপলব্ধ নথিতে পাওয়া প্রাসঙ্গিক তথ্য:",
        "banglish": "Available document-e paoa relevant information:",
        "english": "Relevant information from the available university documents:",
    }[language]
    return f"{prefix}\n\n" + " ".join(sentences)


__all__ = ["build_extractive_answer", "retrieve_lexical"]
