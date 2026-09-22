from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .evidence import SupportStatus
from .language_detector import Language


AnswerStrategy = Literal[
    "structured_exact",
    "structured_list",
    "gguf_generation",
    "unsupported",
    "ambiguous",
    "conflicting",
]


@dataclass(frozen=True)
class LanguageStyle:
    language: Language
    generation_instruction: str
    retry_instruction: str


LANGUAGE_STYLES: Mapping[Language, LanguageStyle] = {
    "english": LanguageStyle(
        "english",
        "Write only natural, concise English. Preserve technical terms, identifiers, and numbers exactly.",
        "Rewrite in natural concise English. Keep every fact unchanged and add no information.",
    ),
    "bangla": LanguageStyle(
        "bangla",
        "Use one or two short natural Bengali sentences. Rephrase only the verified facts; preserve affirmative/negative meaning, official names, technical identifiers, and numbers exactly. Do not invent explanations or use synonyms that change meaning.",
        "Rewrite in short natural Bengali Unicode. Preserve every fact and its affirmative/negative meaning; keep official names and technical identifiers unchanged; add nothing.",
    ),
    "banglish": LanguageStyle(
        "banglish",
        "Use one or two short Latin-script Bangla sentences with Bangla grammar (for example holo, ache, jonno, ebong, korte, kora hoy, or deya hoy where natural). Rephrase only verified facts. Keep technical English terms, official names, identifiers, and numbers unchanged. Use no Bengali script and no pure-English prose.",
        "Rewrite in short Latin-script Bangla. Preserve every fact and its affirmative/negative meaning, use no Bengali script, keep technical identifiers unchanged, and add nothing.",
    ),
}


EXACT_FIELDS = frozenset({
    "credits", "prerequisite", "percentage", "course_code", "course_title", "course_type",
    "email", "date", "duration", "semester", "grade", "attendance",
})
LIST_FIELDS = frozenset({"topic", "weekly_content", "learning_outcome"})


def effective_answer_field(requested_field: str, question: str) -> str:
    if requested_field == "semester" and re.search(r"\b(?:how long|duration|koto)\b|সময়কাল|কতদিন", question, re.I):
        return "duration"
    return requested_field


def select_answer_strategy(
    support_status: SupportStatus | str,
    requested_field: str,
    runtime_intent: str,
    question: str = "",
) -> AnswerStrategy:
    status = support_status.value if isinstance(support_status, SupportStatus) else str(support_status)
    if status != SupportStatus.SUPPORTED.value:
        return status if status in {"unsupported", "ambiguous", "conflicting"} else "unsupported"  # type: ignore[return-value]
    explanatory_form = bool(re.search(r"\b(?:explain|describe|why|how\s+is|how\s+does|kivabe)\b|কীভাবে|কিভাবে|ব্যাখ্যা", question, re.I))
    if explanatory_form and requested_field in {"credits", "attendance", "requirement", "policy"}:
        return "gguf_generation"
    if requested_field in EXACT_FIELDS or runtime_intent in {
        "course_credit", "prerequisite", "course_code", "course_title", "course_type", "final_exam_marks",
    }:
        return "structured_exact"
    if requested_field in LIST_FIELDS or runtime_intent in {"topics", "weekly_content", "mark_distribution", "clo"}:
        return "structured_list"
    return "gguf_generation"


def _finish(text: str, ending: str = ".") -> str:
    cleaned = " ".join(str(text).split()).strip().rstrip(".।")
    return cleaned + ending if cleaned else ""


def _entity_label(entity: str | None, language: Language) -> str:
    value = " ".join(str(entity or "").split()).strip()
    if value:
        if re.fullmatch(r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?", value):
            value = value.upper()
        return value
    return {"english": "The requested item", "bangla": "অনুরোধ করা বিষয়টির", "banglish": "Requested item-er"}[language]


def format_exact_fact(entity: str | None, field: str, value: str, language: Language) -> str:
    entity_text = _entity_label(entity, language)
    clean_value = " ".join(str(value).split()).strip().rstrip(".।")
    if not clean_value:
        return ""

    if language == "english":
        templates = {
            "credits": f"{entity_text} carries {clean_value} credits",
            "prerequisite": f"The prerequisite for {entity_text} is {clean_value}",
            "percentage": f"The requested percentage for {entity_text} is {clean_value.rstrip('%')}%",
            "course_code": f"The course code is {clean_value}",
            "course_title": f"The course title is {clean_value}",
            "course_type": f"The course type is {clean_value}",
            "email": f"The email address is {clean_value}",
            "date": f"The relevant date is {clean_value}",
            "duration": f"The duration is {clean_value}",
            "semester": f"The semester is {clean_value}",
            "grade": f"The grade is {clean_value}",
            "attendance": f"The attendance requirement is {clean_value}",
        }
        return _finish(templates.get(field, f"{entity_text}: {clean_value}"), ".")
    if language == "bangla":
        subject = entity_text if entity else "অনুরোধ করা বিষয়টির"
        templates = {
            "credits": f"{subject} কোর্সটির ক্রেডিট {clean_value}",
            "prerequisite": f"{subject} কোর্সটির prerequisite হলো {clean_value}",
            "percentage": f"{subject} বিষয়টির প্রাসঙ্গিক percentage {clean_value.rstrip('%')}%",
            "course_code": f"কোর্সটির code হলো {clean_value}",
            "course_title": f"কোর্সটির title হলো {clean_value}",
            "course_type": f"কোর্সটির type হলো {clean_value}",
            "email": f"প্রাসঙ্গিক email address হলো {clean_value}",
            "date": f"প্রাসঙ্গিক তারিখ হলো {clean_value}",
            "duration": f"সময়কাল হলো {clean_value}",
            "semester": f"সেমিস্টারটি হলো {clean_value}",
            "grade": f"প্রাসঙ্গিক grade হলো {clean_value}",
            "attendance": f"Attendance-এর শর্ত হলো {clean_value}",
        }
        return _finish(templates.get(field, f"{subject}: {clean_value}"), "।")

    subject = entity_text if entity else "Requested item-er"
    templates = {
        "credits": f"{subject} course-er credit {clean_value}",
        "prerequisite": f"{subject} course-er prerequisite holo {clean_value}",
        "percentage": f"{subject} item-er relevant percentage {clean_value.rstrip('%')}%",
        "course_code": f"Course-er code holo {clean_value}",
        "course_title": f"Course-er title holo {clean_value}",
        "course_type": f"Course-er type holo {clean_value}",
        "email": f"Relevant email address holo {clean_value}",
        "date": f"Relevant date holo {clean_value}",
        "duration": f"Duration holo {clean_value}",
        "semester": f"Semester holo {clean_value}",
        "grade": f"Relevant grade holo {clean_value}",
        "attendance": f"Attendance-er requirement holo {clean_value}",
    }
    return _finish(templates.get(field, f"{subject} value holo {clean_value}"), ".")


def format_structured_text(answer: str, language: Language) -> str:
    """Format legacy extractor output without generic language wrappers."""
    cleaned = " ".join(str(answer).split()).strip()
    if not cleaned:
        return cleaned
    if cleaned.casefold().startswith("mark distribution:"):
        details = cleaned.split(":", 1)[1].strip()
        details = re.sub(r"\s+-\s+", "\n- ", details)
        if language == "bangla":
            return f"নম্বর বণ্টন:\n- {details.lstrip('- ')}"
        if language == "banglish":
            return f"Marks distribution-er details:\n- {details.lstrip('- ')}"
        return f"Mark distribution:\n- {details.lstrip('- ')}"
    if language == "english":
        return cleaned
    patterns = (
        ("course_code", r"^Course code:\s*(.+?)[.]?$"),
        ("course_title", r"^Course title:\s*(.+?)[.]?$"),
        ("course_type", r"^Course type:\s*(.+?)[.]?$"),
        ("credits", r"^Credit value:\s*(.+?)[.]?$"),
        ("prerequisite", r"^Prerequisite:\s*(.+?)[.]?$"),
        ("percentage", r"^(?:Final Exam|Term Examination):\s*(\d+(?:\.\d+)?%?)[.]?$"),
    )
    for field, pattern in patterns:
        match = re.match(pattern, cleaned, re.I)
        if match:
            return format_exact_fact(None, field, match.group(1), language)
    # Lists and explanations should normally use the generator. This fallback is
    # concise and language-bearing, but never pretends an English paragraph was translated.
    if language == "bangla":
        return "প্রাসঙ্গিক তথ্যটি supporting evidence-এ রয়েছে; বিস্তারিত অংশটি নিচে citation-এ দেখুন।"
    return "Relevant information supporting evidence-e ache; details citation-e dekhun."


def evidence_value(field: str, excerpts: Sequence[str]) -> str | None:
    course_code = r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?"
    patterns: Mapping[str, tuple[str, ...]] = {
        "credits": (r"(?:credit(?:\s+value)?|credits?)\s*[:=-]?\s*(\d+(?:\.\d+)?)",),
        "percentage": (r"(\d+(?:\.\d+)?)\s*%",),
        "attendance": (r"attendance[^.\n]{0,100}?(\d+(?:\.\d+)?\s*%)", r"(\d+(?:\.\d+)?\s*%)[^.\n]{0,100}?attendance"),
        "email": (r"([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",),
        "course_code": (r"course\s+(?:no\.?\s*/\s*)?(?:course\s+)?code\s*:\s*([A-Za-z][A-Za-z ()-]*\d{2,4}(?:\([A-Za-z0-9]+\))?)",),
        "course_title": (r"course\s+title\s*:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|\s+course\s+(?:type|credit)|$)",),
        "course_type": (r"course\s+type\s*:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|\s+credit|$)",),
        "prerequisite": (rf"pre\s*[- ]?\s*requisites?\s*[:=-]\s*((?:Nil|None|N/A|{course_code})(?:\s*,\s*{course_code})*)",),
        "grade": (r"(?:grade\s*[:=-]?\s*)([A-F][+-]?)",),
        "duration": (r"(\d+(?:\.\d+)?\s*(?:years?|months?|weeks?|days?))",),
        "semester": (r"((?:spring|summer|fall|autumn|winter)\s+\d{4}|(?:semester\s+\d{1,2}|\d{1,2}(?:st|nd|rd|th)\s+semester))",),
        "date": (r"((?:19|20)\d{2}|\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4}))",),
    }
    values: list[str] = []
    for excerpt in excerpts:
        for pattern in patterns.get(field, ()):
            match = re.search(pattern, excerpt, re.I)
            if match:
                value = " ".join(match.group(1).split()).strip(" .;:")
                if value and value.casefold() not in {item.casefold() for item in values}:
                    values.append(value)
    return values[0] if len(values) == 1 else None


__all__ = [
    "AnswerStrategy", "EXACT_FIELDS", "LANGUAGE_STYLES", "LanguageStyle", "effective_answer_field", "evidence_value",
    "format_exact_fact", "format_structured_text", "select_answer_strategy",
]
