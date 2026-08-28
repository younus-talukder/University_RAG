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


def _extract_course_code(text: str) -> str:
    match = re.search(
        r"Course\s+No\.\s*/\s*Course\s+Code:\s*(.*?)(?=\s+2\.\s+Course\s+Title:)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_course_learning_outcome(text: str, outcome_number: str) -> str:
    match = re.search(
        rf"\bCLO\s*{re.escape(outcome_number)}\b\s*(.*?)(?=\s+CLO\s*\d+\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else ""


def _extract_objectives(text: str) -> str:
    match = re.search(
        r"Course Objectives and Course Summary:\s*(.*?)(?=\s+This is a core course\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else ""


def _extract_assessment_percentage(text: str, assessment_name: str) -> str:
    match = re.search(
        rf"\b{assessment_name}\b\s*\((\d+(?:\.\d+)?)\s*%\)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def build_extractive_answer(question: str, retrieved: Sequence[Dict[str, Any]], language: Language) -> str:
    if not retrieved:
        return unsupported_answer(language)

    top_text = re.sub(r"\s+", " ", str(retrieved[0].get("text", ""))).strip()
    clo_match = re.search(r"\bCLO\s*(\d+)\b", question, flags=re.IGNORECASE)
    if clo_match:
        outcome = _extract_course_learning_outcome(top_text, clo_match.group(1))
        if outcome:
            return f"CLO {clo_match.group(1)}: {outcome}."

    if re.search(r"\b(objective|objectives|purpose|aim|aims)\b", question, flags=re.IGNORECASE):
        objectives = _extract_objectives(top_text)
        if objectives:
            return objectives + "."

    if re.search(r"\b(final|term)\s+exam", question, flags=re.IGNORECASE):
        exam_name = "Final Exam" if re.search(r"\bfinal\b", question, flags=re.IGNORECASE) else "Mid Term"
        percentage = _extract_assessment_percentage(top_text, exam_name)
        if percentage:
            return f"{exam_name}: {percentage}%."

    title = _extract_field(top_text, "Course Title")
    course_code = _extract_course_code(top_text)
    course_type = _extract_field(top_text, "Course Type")
    credit = _extract_field(top_text, "Credit Value")
    if title:
        asks_code = re.search(r"\b(code|number)\b", question, flags=re.IGNORECASE)
        asks_title = re.search(r"\btitle\b", question, flags=re.IGNORECASE)
        asks_type = re.search(r"\b(type|kind)\b", question, flags=re.IGNORECASE)
        asks_credit = re.search(r"\b(credit|credits)\b", question, flags=re.IGNORECASE)

        if asks_code and asks_title:
            if course_code:
                return f"Course code: {course_code}. Course title: {title}."
            return f"Course title: {title}."

        if asks_type or asks_credit:
            details = []
            if asks_type and course_type:
                details.append(f"Course type: {course_type}.")
            if asks_credit and credit:
                details.append(f"Credit value: {credit}.")
            if details:
                return " ".join(details)

        if asks_code and course_code:
            return f"Course code: {course_code}."
        if asks_title:
            return f"Course title: {title}."

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
