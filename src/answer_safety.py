"""Independent, conservative checks for requested relations and explicit facts.

These rules inspect only the question and verified evidence, never evaluation
references. Unknown relations are left to the existing grounding contract.
"""

from __future__ import annotations

import re
from typing import Sequence

from .query_normalization import extract_course_entities, is_mark_distribution_query, normalize_retrieval_text
from .evidence import detect_requested_field

EMAIL = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
EMAIL_LIST = re.compile(rf"\bE-?mail\s*:\s*((?:{EMAIL}(?:\s*[,;]+\s*|\s+and\s+)?)+)", re.I)
ABSENCE_CLAIM = re.compile(
    r"\b(?:not|never)\s+(?:explicitly\s+)?(?:specified|listed|mentioned|provided|given|found)\b"
    r"|\b(?:provide|deya|ullekh)\s+koreche\s+na\b"
    r"|উল্লেখিত\s+হয়নি|দেওয়া\s+হয়নি",
    re.I,
)
BOILERPLATE_WORDS = {
    "course", "code", "document", "evidence", "provided", "listed", "answer",
    "relevant", "information", "holo", "ache", "from", "with", "that", "this",
    "which", "what", "where", "when", "does", "such", "about", "area",
}


def _content_words(text: str) -> set[str]:
    return {
        word.casefold() for word in re.findall(r"[A-Za-z]{4,}|[\u0980-\u09ff]{3,}", text)
        if word.casefold() not in BOILERPLATE_WORDS
    }


def _association_anchor(question: str) -> set[str]:
    """Keep the subject of an explicit pairing tied to the cited passage.

    The question, rather than a dataset reference or a translated answer, supplies
    the subject. Non-Latin anchors are left to the existing evidence contract.
    """
    text = normalize_retrieval_text(question)
    paired = re.search(r"\bpaired\s+with\s+(.+?)[?.!]?$", text, re.I)
    companion = re.search(r"^(.+?)\s+(?:sathe|with)\b", text, re.I)
    phrase = paired.group(1) if paired else companion.group(1) if companion else ""
    return {term.casefold() for term in re.findall(r"[A-Za-z]{4,}", phrase)
            if term.casefold() not in BOILERPLATE_WORDS and term.casefold() not in {"paired", "course"}}


def requested_relation(question: str) -> str | None:
    """Resolve relation distinctions whose shared field admits incompatible facts."""
    if is_mark_distribution_query(question):
        return "mark_distribution"
    text = normalize_retrieval_text(question)
    if not re.search(r"\bsemesters?\b|সেমিস্টার", text, re.I):
        return None
    if re.search(r"\bclass(?:es)?\b|ক্লাস", text, re.I) and re.search(
        r"\bweeks?\b|সপ্তাহ|মেয়াদ|সময়কাল|duration|কত", text, re.I
    ):
        return "class_duration_weeks"
    if re.search(r"\bhow\s+long\b|\bduration\b|\blength\b|\bweeks?\b|মেয়াদ|সময়কাল|কতদিন|koto\s+din", text, re.I):
        return "regular_semester_duration"
    if re.search(r"\bnames?\b|\bwhich\b|\blist\b|নাম|কোন\s+কোন", text, re.I):
        return "semester_names"
    if re.search(r"\bhow\s+many\b|\bnumber\s+of\b|\bkoy(?:ta|ti)?\b|কয়টি|কয়|কতটি|কত", text, re.I):
        return "regular_semester_count"
    return None


def relation_compatible(wanted: str | None, extracted: str | None) -> bool:
    if not wanted or not extracted:
        return True
    return wanted == extracted


def explicit_email_values(question: str, excerpts: Sequence[str]) -> tuple[str, ...]:
    """Emails in the relevant labeled field, not neighboring departments."""
    q = normalize_retrieval_text(question).casefold()
    if not re.search(r"\be-?mail\b|ইমেইল", q, re.I):
        return ()
    if re.search(r"\b(?:any|one|single|either)\b|যেকোনো\s+একটি", q, re.I):
        return ()
    subject = re.search(r"([A-Za-z][A-Za-z ]{2,60}?)\s*(?:office|department)", question, re.I)
    subject_words = set(re.findall(r"[a-z]{3,}", subject.group(0).casefold())) if subject else set()
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for excerpt in excerpts:
        for match in EMAIL_LIST.finditer(excerpt):
            values = tuple(dict.fromkeys(value.casefold() for value in re.findall(EMAIL, match.group(1), re.I)))
            if not values:
                continue
            prefix = excerpt[max(0, match.start() - 300):match.start()].casefold()
            last_heading = max(prefix.rfind("department of"), prefix.rfind("admission office"))
            local = prefix[last_heading:] if last_heading >= 0 else prefix
            score = sum(word in local for word in subject_words)
            candidates.append((score, values))
    if not candidates:
        return ()
    best = max(score for score, _ in candidates)
    return tuple(dict.fromkeys(value for score, values in candidates if score == best for value in values))


def answer_safety_failures(question: str, answer: str, excerpts: Sequence[str], *,
                           extracted_relation: str | None = None,
                           relation_values: Sequence[str] = (),
                           cited_excerpt: str | None = None) -> tuple[str, ...]:
    failures: list[str] = []
    # A local passage cannot establish that a requested fact is absent from
    # the document. Treat such wording as an abstention attempt, not a fact.
    if ABSENCE_CLAIM.search(answer):
        failures.append("UNVERIFIED_ABSENCE_CLAIM")
    # A broad question needs a fact from its cited passage beyond words already
    # present in the question. An entity label alone is never that fact.
    requested_field = detect_requested_field(question)
    answer_field = detect_requested_field(answer)
    clo_missing = bool(re.search(r"\bCLOs?\b", question, re.I)
                       and not re.search(r"\bCLO\s*\d+\b", answer, re.I))
    if (requested_field != "general" and answer_field != "general"
            and answer_field != requested_field and not clo_missing):
        failures.append("WRONG_FIELD_ANSWER")
    cited_text = cited_excerpt if cited_excerpt is not None else " ".join(excerpts)
    if excerpts and requested_field == "general":
        novel_answer = _content_words(answer) - _content_words(question)
        source_content = _content_words(cited_text) - _content_words(question)
        if not novel_answer or len(novel_answer & source_content) < min(2, len(novel_answer)):
            failures.append("REQUESTED_FACT_NOT_IN_CITED_EVIDENCE")
        anchor = _association_anchor(question)
        if anchor and not anchor.intersection(_content_words(cited_text)):
            failures.append("REQUESTED_RELATION_NOT_ON_CITED_PASSAGE")
    # Asking for named outcome identifiers cannot be answered with a neighboring
    # week topic, even when the topic appears on the same page.
    if clo_missing:
        failures.append("REQUESTED_FIELD_MISSING")
    wanted = requested_relation(question)
    if not relation_compatible(wanted, extracted_relation):
        failures.append("RELATION_MISMATCH")
    emails = explicit_email_values(question, excerpts)
    if emails and not all(value in answer.casefold() for value in emails):
        failures.append("REQUIRED_VALUE_LOSS")
    # These extracted relations expose a bounded set of explicitly verified
    # components. Check each component, not just the first convenient value.
    if extracted_relation in {"initially_offered", "mark_distribution", "program_discipline_counts"}:
        number_words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        answer_folded = answer.casefold()
        if any(not any(candidate in answer_folded for candidate in
                       (str(value).casefold(), number_words.get(str(value).casefold(), "\0")))
               for value in relation_values):
            failures.append("REQUIRED_VALUE_LOSS")
    if re.search(r"\bpre\s*[- ]?\s*requisites?\b|পূর্বশর্ত", question, re.I):
        code_pattern = r"\b[A-Z]{2,8}\s*\d{2,4}(?:\s*\([A-Z]\))?\b"
        for excerpt in excerpts:
            match = re.search(r"\bpre\s*[- ]?\s*requisites?\s*:\s*([^.;\n]{1,100})", excerpt, re.I)
            if not match:
                continue
            # Restrict to the comma/and-delimited field prefix, never codes in
            # a following description or another course block.
            prefix = re.match(rf"\s*((?:{code_pattern}(?:\s*(?:,|/|and)\s*)?)+)", match.group(1), re.I)
            if prefix:
                required = {re.sub(r"[^A-Z0-9]", "", code.upper()) for code in re.findall(code_pattern, prefix.group(1), re.I)}
                present = {item.compact for item in extract_course_entities(answer)}
                if not required.issubset(present):
                    failures.append("REQUIRED_VALUE_LOSS")
                break
    requested_codes = {item.compact for item in extract_course_entities(question)}
    answer_codes = {item.compact for item in extract_course_entities(answer)}
    if requested_codes and answer_codes and not requested_codes.issubset(answer_codes):
        failures.append("ENTITY_MISMATCH")
    if extracted_relation == "attendance_exam_requirement":
        source = " ".join(excerpts)
        if re.search(r"\bat\s+least\b", source, re.I) and not re.search(
            r"\bat\s+least\b|\bkompakhe\b|অন্তত|কমপক্ষে", answer, re.I
        ):
            failures.append("ESSENTIAL_QUALIFIER_LOSS")
    if extracted_relation == "repeat_course_maximum" and not re.search(
        r"\bat\s+most\b|\bmaximum\b|\bup\s+to\b|\bshorboccho\b|সর্বোচ্চ", answer, re.I
    ):
        failures.append("ESSENTIAL_QUALIFIER_LOSS")
    return tuple(dict.fromkeys(failures))


__all__ = ["answer_safety_failures", "explicit_email_values", "relation_compatible", "requested_relation"]
