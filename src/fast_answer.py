from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Sequence

from .answer_policy import format_structured_text
from .language_detector import Language
from .language_validator import unsupported_answer
from .evidence import identify_entities
from .query_normalization import extract_course_entities, normalize_retrieval_text

RuntimeIntent = Literal[
    "topics",
    "clo",
    "weekly_content",
    "course_objective",
    "assessment",
    "final_exam_marks",
    "mark_distribution",
    "prerequisite",
    "course_code",
    "course_title",
    "course_type",
    "course_credit",
    "course_metadata",
    "unknown",
]

STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "course", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "why",
}

# Kept as empty compatibility symbols for callers from earlier versions. Runtime
# routing no longer maps courses to a particular PDF filename.
DEFAULT_COURSE_SOURCE = ""

WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


def _strip_markdown_list_marker(text: str) -> str:
    return re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", text).strip()


def _split_answer_items(text: str) -> list[str]:
    cleaned_text = " ".join(str(text).split()).strip().rstrip(".")
    if not cleaned_text:
        return []

    bullet_lines = [
        _strip_markdown_list_marker(line)
        for line in str(text).splitlines()
        if re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", line)
    ]
    if bullet_lines:
        return [item.rstrip(" .") for item in bullet_lines if item]

    if ":" in cleaned_text:
        cleaned_text = cleaned_text.split(":", 1)[1].strip()

    cleaned_text = re.sub(r"^(?:topics?\s+)?(?:include|includes|included)\s+", "", cleaned_text, flags=re.IGNORECASE)
    parts = re.split(r"\s*,\s*|\s*;\s*|\s+\band\b\s+|\s+\u2022\s+", cleaned_text, flags=re.IGNORECASE)
    items = []
    for part in parts:
        item = re.sub(r"^(?:and|or)\s+", "", part.strip(" ."), flags=re.IGNORECASE).strip()
        if item:
            items.append(item)
    return items


def _format_items(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _clean_topic_item(item: str) -> str:
    cleaned = item.strip(" .;")
    cleaned = re.sub(r"^(?:topics?\s+)?(?:such\s+as|include|includes|included)\s+", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if cleaned[0].islower():
        return cleaned[0].upper() + cleaned[1:]
    return cleaned


def _normalize_topic_items(subject: str, items: Sequence[str]) -> list[str]:
    normalized = []
    for item in items:
        cleaned = _clean_topic_item(item)
        if not cleaned:
            continue
        if subject.lower() == "boolean algebra":
            if cleaned.lower() == "multiplication":
                cleaned = "Boolean multiplication"
            if cleaned.lower() in {"truth tables", "boolean expression simplification"}:
                continue
        normalized.append(cleaned)
    return normalized


def _normalized_query(text: str) -> str:
    return normalize_retrieval_text(text)


def detect_course_code(question: str) -> str:
    entities = extract_course_entities(question)
    return entities[0].canonical if entities else ""


def _intent_patterns(text: str) -> dict[str, bool]:
    return {
        "clo": bool(re.search(r"\bclo\s*\d*\b|course learning outcome", text)),
        "objective": bool(re.search(r"\b(objective|objectives|purpose|aim|aims|udd?esh|uddeshyo|main objective)\b|\u0989\u09a6\u09cd\u09a6\u09c7\u09b6\u09cd\u09af", text)),
        "weekly": bool(re.search(r"\bweek\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b|\b\d+(?:st|nd|rd|th|\u09ae)?\s+(?:week|\u09b8\u09aa\u09cd\u09a4\u09be\u09b9)\b|\u09b8\u09aa\u09cd\u09a4\u09be\u09b9\s*\d+|\bsoptah\b|\bsoptaho\b", text)),
        "final_exam": bool(re.search(r"\bfinal(?:\s+exam(?:ination)?)?\b|\bterm\s+examination\b|\u09ab\u09be\u0987\u09a8\u09be\u09b2", text)),
        "mark_distribution": bool(re.search(r"mark\s+distribution|marks?\s+distribution|weighting|allocated|boraddo|percentage|percent|shotangsho|শতাংশ|বরাদ্দ", text)),
        "assessment": bool(re.search(r"\bassessment\b|class\s+tests?|quizz?es?|assignment|presentation|attendance|grade|grading|score|marks?\b|পরীক্ষা|কুইজ|গ্রেড", text)),
        "prerequisite": bool(re.search(r"\bpre[- ]?requisites?\b|prereq|পূর্বশর্ত", text)),
        "credit": bool(re.search(r"\bcredits?\b|ক্রেডিট", text)),
        "title": bool(re.search(r"\btitle\b|শিরোনাম", text)),
        "code": bool(re.search(r"\b(code|number)\b|কোড", text)),
        "type": bool(re.search(r"\btype\b|\bkind\b|ধরন", text)),
        "topic": bool(re.search(r"\btopics?\b|content|covered|included|under|porano|include|শেখানো|পড়ানো|পড়ানো|বিষয়|বিষয়|অন্তর্ভুক্ত", text)),
    }


def detect_runtime_intent(question: str) -> RuntimeIntent:
    text = _normalized_query(question)
    patterns = _intent_patterns(text)

    if patterns["clo"]:
        return "clo"
    if patterns["objective"]:
        return "course_objective"
    if patterns["weekly"]:
        return "weekly_content"
    if patterns["final_exam"] and patterns["mark_distribution"]:
        return "final_exam_marks"
    if patterns["mark_distribution"] and not patterns["final_exam"]:
        return "mark_distribution"
    if patterns["assessment"] and not patterns["final_exam"]:
        return "assessment"
    if patterns["prerequisite"]:
        return "prerequisite"
    if patterns["credit"]:
        return "course_credit"
    if patterns["code"] and patterns["title"]:
        return "course_metadata"
    if patterns["title"]:
        return "course_title"
    if patterns["code"]:
        return "course_code"
    if patterns["type"]:
        return "course_type"
    if patterns["topic"]:
        return "topics"
    return "course_metadata" if detect_course_code(question) else "unknown"


def _infer_topic_subject(question: str) -> str:
    stripped = " ".join(str(question).split()).strip(" ?।")
    patterns = [
        r"^(.+?)-er\s+under-e\b",
        r"^(.+?)-এর\s+অধীনে(?=\s|$)",
        r"\bunder\s+(?:the\s+)?(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" :")
    return ""


def _normal_subject_pattern(subject: str) -> str:
    return r"\s+".join(re.escape(part) for part in subject.split())


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u0980-\u09ff]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def retrieve_lexical(question: str, metadata: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    requested_course = detect_course_code(question)
    requested_topic = _infer_topic_subject(question)
    requested_week = extract_week_number(question)
    intent = detect_runtime_intent(question)
    ranked = []

    for item in metadata:
        text = str(item.get("text", ""))
        source = str(item.get("source", ""))
        searchable = f"{text} {source}".lower()
        score = 0.0
        for token in query_tokens:
            lowered = token.lower()
            if lowered in searchable:
                score += 1.0
            if re.search(rf"\b{re.escape(lowered)}\b", searchable):
                score += 1.0

        if requested_course and re.search(re.escape(requested_course).replace(r"\ ", r"\s*"), searchable, flags=re.IGNORECASE):
            score += 6.0

        if intent == "clo":
            match = re.search(r"\bclo\s*(\d+)\b", question, flags=re.IGNORECASE)
            if match and re.search(rf"\bCLO\s*{match.group(1)}\b", text, flags=re.IGNORECASE):
                score += 18.0
            elif "course learning outcomes" in searchable:
                score += 8.0
        elif intent == "course_objective" and "course objectives" in searchable:
            score += 16.0
        elif intent == "topics":
            if "course content" in searchable:
                score += 10.0
            if "alignment of topics" in searchable:
                score += 6.0
        elif intent == "weekly_content":
            if "weekly plan" in searchable or "week " in searchable:
                score += 10.0
            if requested_week and re.search(rf"\bWeek\s+{requested_week}\b", text, flags=re.IGNORECASE):
                score += 18.0
        elif intent in {"assessment", "final_exam_marks", "mark_distribution"}:
            if "assessment" in searchable or "evaluation policy" in searchable or "weighting" in searchable:
                score += 12.0
            if intent == "final_exam_marks" and "final exam" in searchable:
                score += 10.0
            if intent == "final_exam_marks" and re.search(r"\b(?:Final Exam|Term Examination)\b[^\d%]{0,40}\d+(?:\.\d+)?\s*%", text, flags=re.IGNORECASE):
                score += 20.0
            if intent == "mark_distribution" and re.search(r"\b(?:Class Tests|Assessment|Mid-Term Examination)\b[^\d%]{0,40}\d+(?:\.\d+)?\s*%", text, flags=re.IGNORECASE):
                score += 18.0

        if requested_topic:
            topic_pattern = _normal_subject_pattern(requested_topic)
            if re.search(rf"\b{topic_pattern}\s*:", text, flags=re.IGNORECASE):
                score += 12.0
            elif re.search(rf"\b{topic_pattern}\b", text, flags=re.IGNORECASE):
                score += 3.0

        if score > 0:
            ranked.append({**item, "score": score})

    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ranked[:top_k]


def _extract_field(text: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|$)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_course_code(text: str) -> str:
    match = re.search(
        r"Course\s+No\.\s*/\s*Course\s+Code:\s*(.*?)(?=\s+2\.\s+Course\s+Title:)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_course_table_row(text: str, course: str) -> tuple[str, str, str] | None:
    """Extract title, credit, and prerequisite from a generic curriculum table row."""
    if not course:
        return None
    course_pattern = r"\s*".join(re.escape(part) for part in re.findall(r"[A-Za-z0-9]+", course))
    next_code = r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?"
    match = re.search(
        rf"\b{course_pattern}\b\s+(.+?)\s+(\d+(?:\.\d+)?)\s+(.+?)(?=\s+{next_code}\b|\s+Total\b|\s+(?:First|Second|Third|Fourth)\s+Year\b|$)",
        re.sub(r"\s+", " ", text),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return tuple(_clean_statement(value) for value in match.groups())


def _clean_statement(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .;:-")


def _extract_course_learning_outcome(text: str, outcome_number: str) -> str:
    match = re.search(
        rf"\bCLO\s*{re.escape(outcome_number)}\b\s*(.*?)(?=\s+CLO\s*\d+\b|\s+13\.\s+Mapping\b|\s+Part\s+B\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_statement(match.group(1)) if match else ""


def _extract_objectives(text: str) -> str:
    match = re.search(
        r"Course Objectives and Course Summary:\s*(.*?)(?=\s+This is a core course\b|\s+12\.\s+Course Learning Outcomes\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_statement(match.group(1)) if match else ""


def _extract_course_content(text: str) -> str:
    match = re.search(
        r"14\.\s+Course Content:\s*(.*?)(?=\s+15\.\s+Alignment|\s+16\.\s+Class Schedule|$)",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_statement(match.group(1)) if match else ""


def _extract_topics_for_subject(text: str, subject: str) -> list[str]:
    if not subject:
        return []

    normalized_text = re.sub(r"\s+", " ", str(text)).strip()
    subject_pattern = _normal_subject_pattern(subject)
    topic_heading = re.search(
        rf"\b{subject_pattern}\s*:\s*(.+?)(?=\s+[A-Z][A-Za-z /&-]{{2,55}}\s*:|\s+(?:Week|CT\d|MID-TERM|FINAL|Part\s+C)\b|$)",
        normalized_text,
        flags=re.IGNORECASE,
    )
    if not topic_heading:
        return []

    raw_items = topic_heading.group(1)
    raw_items = re.sub(r"\b(?:and\s+)?their\s+conversions\b.*$", "", raw_items, flags=re.IGNORECASE).strip(" ,;.")
    return _normalize_topic_items(subject, _split_answer_items(raw_items))


def _extract_alignment_topics(text: str) -> list[str]:
    match = re.search(
        r"15\.\s+Alignment of topics.*?(?=\s+16\.\s+Class Schedule|\s+Part\s+C|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    section = re.sub(r"\s+", " ", match.group(0))
    section = re.sub(r"^.*?Course (?:Learning )?Outcome(?: \(CLO\))?\s*", "", section, flags=re.IGNORECASE)
    rows = re.split(r"\s+(?=\d+\.?\s+[A-Z])", section)
    items = []
    for row in rows:
        row = re.sub(r"^\d+\.?\s+", "", row).strip()
        row = re.sub(r"\s+CLO\d(?:,\s*CLO\d)*.*$", "", row, flags=re.IGNORECASE).strip(" .;")
        if row and not row.lower().startswith("alignment of topics"):
            items.append(row)
    return items


def _structured_topic_answer(text: str) -> tuple[str, list[str]] | None:
    cleaned_text = " ".join(str(text).split()).strip().rstrip(".")
    patterns = [
        r"^(?:Under|In|For)\s+(.+?),?\s+(?:the\s+)?topics?\s+(?:include|includes|included)\s+(.+)$",
        r"^(.+?)\s+(?:topics?\s+)?(?:include|includes|included)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned_text, flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip(" :")
        if subject.lower() in {"topic", "topics", "course topics"}:
            return None
        items = _normalize_topic_items(subject, _split_answer_items(match.group(2)))
        if subject and items:
            return subject, items
    return None


def build_topic_answer(question: str, retrieved: Sequence[Dict[str, Any]], language: Language) -> str | None:
    topic_subject = _infer_topic_subject(question)
    if topic_subject:
        for item in retrieved:
            topic_items = _extract_topics_for_subject(str(item.get("text", "")), topic_subject)
            if topic_items:
                return format_answer_for_language(
                    f"Under {topic_subject}, topics include {', '.join(topic_items)}.",
                    language,
                )

    for item in retrieved:
        course_content = _extract_course_content(str(item.get("text", "")))
        if course_content:
            return format_answer_for_language(f"Topics include {course_content}.", language)

    for item in retrieved:
        topics = _extract_alignment_topics(str(item.get("text", "")))
        if topics:
            return format_answer_for_language(f"Topics include {', '.join(topics)}.", language)
    return None


def extract_week_number(question: str) -> int | None:
    text = _normalized_query(question)
    match = re.search(r"\bweek\s*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th|\u09ae)?\s+(?:week|\u09b8\u09aa\u09cd\u09a4\u09be\u09b9)\b|\u09b8\u09aa\u09cd\u09a4\u09be\u09b9\s*(\d{1,2})", text)
    if match:
        return int(match.group(1) or match.group(2) or match.group(3))
    for word, number in WORD_NUMBERS.items():
        if re.search(rf"\bweek\s+{word}\b|\b{word}\s+week\b", text):
            return number
    return None


def _extract_week_content(text: str, week_number: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    pattern = rf"(.{{0,260}}?)\bWeek\s+{week_number}(?:-\d+)?\b(.{{0,180}})"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if not match:
        return ""

    before = match.group(1)
    after = match.group(2)
    before = re.split(r"\b(?:Week\s+\d+(?:-\d+)?|CT\d|MID-TERM EXAMINATION|FINAL EXAMINATION)\b", before, flags=re.IGNORECASE)[-1]
    before = re.sub(r"^.*\bCLO\d(?:,\s*CLO\d)*\s+", "", before, flags=re.IGNORECASE)
    before = re.sub(r"\s+(?:gain|understand|apply|relate|learning|constructing|identifying|demonstrating)\b.*$", "", before, flags=re.IGNORECASE)
    topic = _clean_statement(before)
    if re.search(r"exam", topic, flags=re.IGNORECASE):
        topic = "MID-TERM EXAMINATION" if week_number != 15 else "FINAL EXAMINATION"

    if not topic:
        first_words = re.split(r"\s+(?:Class notes|Practice|Assignment|Lecture|Multimedia|CLO\d)\b", after, flags=re.IGNORECASE)[0]
        topic = _clean_statement(first_words)
    return topic


def _extract_assessment_percentage(text: str, assessment_name: str) -> str:
    patterns = [
        rf"\b{assessment_name}\b\s*\((\d+(?:\.\d+)?)\s*%\)",
        rf"\b{assessment_name}\b[^\d%]{{0,40}}(\d+(?:\.\d+)?)\s*%",
        rf"\d+\.\s*{assessment_name}s?\s+(\d+(?:\.\d+)?)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_mark_distribution(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    items = []
    for label in ("Final Exam", "Term Examination", "Mid-Term Examination", "Mid Term", "Assessment", "Class Tests"):
        percentage = _extract_assessment_percentage(normalized, label)
        if percentage:
            display = "Mid-Term Examination" if label == "Mid Term" else label
            items.append(f"{display}: {percentage}%")
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _extract_grade(text: str, question: str) -> str:
    if not re.search(r"\bgrade\b|গ্রেড", question, flags=re.IGNORECASE):
        return ""
    match = re.search(r"80%\s+and\s+above\s+(A\+)\s+4\.00", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def format_answer_for_language(answer: str, language: Language) -> str:
    cleaned_answer = " ".join(str(answer).split()).strip()
    if not cleaned_answer:
        return cleaned_answer

    topic_answer = _structured_topic_answer(cleaned_answer)
    if topic_answer is not None:
        subject, items = topic_answer
        bullet_list = _format_items(items)
        if language == "bangla":
            return f"{subject}-এর অধীনে বিষয়গুলো:\n{bullet_list}"
        if language == "banglish":
            return f"{subject}-er under-e topics:\n{bullet_list}"
        return f"Topics under {subject}:\n{bullet_list}"

    return _finalize_answer_for_language(cleaned_answer, language)


def _translate_english_to_banglish(text: str) -> str:
    return format_structured_text(text, "banglish")


def _finalize_answer_for_language(answer: str, language: Language) -> str:
    return format_structured_text(answer, language)


def _relevant_items(retrieved: Sequence[Dict[str, Any]], course: str) -> list[Dict[str, Any]]:
    return list(retrieved)


def build_extractive_answer(
    question: str,
    retrieved: Sequence[Dict[str, Any]],
    language: Language,
    intent: RuntimeIntent | None = None,
) -> str:
    if not retrieved:
        return unsupported_answer(language)

    runtime_intent = intent or detect_runtime_intent(question)
    requested_course = detect_course_code(question)
    items = _relevant_items(retrieved, requested_course)
    top_text = re.sub(r"\s+", " ", str(items[0].get("text", ""))).strip()
    table_row = None
    for item in items:
        table_row = _extract_course_table_row(str(item.get("text", "")), requested_course)
        if table_row:
            break

    if runtime_intent == "clo":
        clo_match = re.search(r"\bCLO\s*(\d+)\b", question, flags=re.IGNORECASE)
        if clo_match:
            for item in items:
                outcome = _extract_course_learning_outcome(str(item.get("text", "")), clo_match.group(1))
                if outcome:
                    return _finalize_answer_for_language(f"CLO {clo_match.group(1)}: {outcome}.", language)
        return unsupported_answer(language)

    if runtime_intent == "course_objective":
        for item in items:
            objectives = _extract_objectives(str(item.get("text", "")))
            if objectives:
                return _finalize_answer_for_language(objectives + ".", language)
        return unsupported_answer(language)

    if runtime_intent == "topics":
        topic_answer = build_topic_answer(question, items, language)
        return topic_answer or unsupported_answer(language)

    if runtime_intent == "weekly_content":
        week_number = extract_week_number(question)
        if week_number is None and re.search(r"mid.?term", question, flags=re.IGNORECASE):
            for item in items:
                text = str(item.get("text", ""))
                match = re.search(r"MID-TERM EXAMINATION\s+Week\s+(\d+)", text, flags=re.IGNORECASE)
                if match:
                    return _finalize_answer_for_language(f"Mid-Term Examination is scheduled in Week {match.group(1)}.", language)
        if week_number is not None:
            for item in items:
                content = _extract_week_content(str(item.get("text", "")), week_number)
                if content:
                    return _finalize_answer_for_language(f"Week {week_number}: {content}.", language)
        return unsupported_answer(language)

    if runtime_intent in {"assessment", "final_exam_marks", "mark_distribution"}:
        if runtime_intent == "final_exam_marks":
            for item in items:
                percentage = _extract_assessment_percentage(str(item.get("text", "")), "Final Exam")
                if percentage:
                    return _finalize_answer_for_language(f"Final Exam: {percentage}%.", language)
                percentage = _extract_assessment_percentage(str(item.get("text", "")), "Term Examination")
                if percentage:
                    return _finalize_answer_for_language(f"Term Examination: {percentage}%.", language)
            return unsupported_answer(language)

        if runtime_intent == "mark_distribution":
            for item in items:
                distribution = _extract_mark_distribution(str(item.get("text", "")))
                if distribution:
                    return _finalize_answer_for_language("Mark distribution:\n" + _format_items(distribution), language)
            return unsupported_answer(language)

        if re.search(r"\battendance\b|উপস্থিতি|হাজিরা", question, flags=re.IGNORECASE):
            attendance_sentences: list[str] = []
            for item in items:
                text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    if re.search(r"\battend(?:ance)?\b", sentence, re.I) and re.search(
                        r"\b(?:required|must|expected)\b|\d+(?:\.\d+)?\s*%",
                        sentence,
                        re.I,
                    ):
                        attendance_sentences.append(sentence.strip(" ."))
            if attendance_sentences:
                best = max(
                    attendance_sentences,
                    key=lambda sentence: (
                        bool(re.search(r"\d+(?:\.\d+)?\s*%", sentence)),
                        bool(re.search(r"\brequired\b|\bmust\b", sentence, re.I)),
                    ),
                )
                return _finalize_answer_for_language(best + ".", language)
            return unsupported_answer(language)

        for item in items:
            grade = _extract_grade(str(item.get("text", "")), question)
            if grade:
                return _finalize_answer_for_language(f"80% and above is awarded grade {grade}.", language)
        for item in items:
            text = str(item.get("text", ""))
            if "Assessment Strategy" in text:
                section = re.search(r"Assessment Strategy\s+(.*?)(?=\s+CIE-|\s+20\.\s+Evaluation Policy|$)", text, flags=re.IGNORECASE)
                if section:
                    return _finalize_answer_for_language(_clean_statement(section.group(1)) + ".", language)
        return unsupported_answer(language)

    if runtime_intent == "prerequisite":
        code_pattern = r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?"
        for item in items:
            labeled = re.search(
                rf"pre\s*[- ]?\s*requisites?\s*:\s*((?:Nil|None|{code_pattern})(?:\s*,\s*{code_pattern})*)",
                str(item.get("text", "")),
                re.IGNORECASE,
            )
            if labeled:
                return _finalize_answer_for_language(f"Prerequisite: {_clean_statement(labeled.group(1))}.", language)
        if table_row:
            return _finalize_answer_for_language(f"Prerequisite: {table_row[2]}.", language)
        query_tokens = set(_tokens(question))
        for item in items:
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            if re.search(r"\bpre\s*[- ]?\s*requisites?\b", text, flags=re.IGNORECASE):
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    sentence_tokens = set(_tokens(sentence))
                    if "prerequisite" in sentence.casefold() or query_tokens & sentence_tokens:
                        return _finalize_answer_for_language(sentence.strip(" .") + ".", language)
        return unsupported_answer(language)

    title = _extract_field(top_text, "Course Title")
    course_code = _extract_course_code(top_text)
    course_type = _extract_field(top_text, "Course Type")
    credit = _extract_field(top_text, "Credit Value")
    if table_row:
        title = title or table_row[0]
        credit = credit or table_row[1]

    if runtime_intent == "course_code" and course_code:
        return _finalize_answer_for_language(f"Course code: {course_code}.", language)
    if runtime_intent == "course_title" and title:
        return _finalize_answer_for_language(f"Course title: {title}.", language)
    if runtime_intent == "course_type" and course_type:
        return _finalize_answer_for_language(f"Course type: {course_type}.", language)
    if runtime_intent == "course_credit" and credit:
        return _finalize_answer_for_language(f"Credit value: {credit}.", language)
    if runtime_intent == "course_metadata" and title:
        details = []
        if course_code:
            details.append(f"Course code: {course_code}.")
        details.append(f"Course title: {title}.")
        asks_code_title = bool(re.search(r"\bcode\b|কোড", question, flags=re.IGNORECASE)) and bool(
            re.search(r"\btitle\b|শিরোনাম", question, flags=re.IGNORECASE)
        )
        if not asks_code_title and re.search(r"\bwhat\s+is\b|ki\b|কী|কি", question, flags=re.IGNORECASE):
            if course_type:
                details.append(f"Course type: {course_type}.")
            if credit:
                details.append(f"Credit value: {credit}.")
        return _finalize_answer_for_language(" ".join(details), language)

    query_tokens = set(_tokens(question))
    sentences: list[str] = []
    for item in items:
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
        return unsupported_answer(language)

    answer = " ".join(sentences)
    if language == "english":
        return "Relevant information from the available university documents:\n\n" + answer
    return _finalize_answer_for_language(answer, language)


__all__ = [
    "RuntimeIntent",
    "DEFAULT_COURSE_SOURCE",
    "build_extractive_answer",
    "build_topic_answer",
    "detect_course_code",
    "detect_runtime_intent",
    "extract_week_number",
    "retrieve_lexical",
]
