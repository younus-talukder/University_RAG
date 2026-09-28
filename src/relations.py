"""Small, evidence-derived relations and deterministic multilingual realization.

Every extractor reads verified passage text only. Failure to match means abstain;
it never promotes an unverified retrieved chunk to evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .document_metadata import extract_document_metadata, metadata_fields_for_query
from .evidence import EvidenceAssessment, SupportStatus
from .query_normalization import extract_course_entities, is_mark_distribution_query


@dataclass(frozen=True)
class EvidenceRelation:
    subject: str
    relation: str
    values: tuple[str, ...]
    source: str
    page: int | None
    chunk_id: str | int | None
    evidence_span: str
    target_language: str
    labels: tuple[str, ...] = ()
    continuation_chunk_ids: tuple[str, ...] = ()


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip(" ,.;:-")


def _subject(question: str, evidence: str) -> str:
    for token in re.findall(r"\b[A-Z]{2,8}\b", question):
        if token not in {"CLO", "CGPA", "GPA", "PDF"}:
            return token
    match = re.search(r"\(([A-Z]{2,8})\)", evidence)
    return match.group(1) if match else "the university"


def _asked(question: str, *patterns: str) -> bool:
    return any(re.search(pattern, question, re.I) for pattern in patterns)


def _relation(question: str, text: str, target_language: str, source: str,
              page: int | None, chunk_id: str | int | None,
              continuation_chunk_ids: tuple[str, ...]) -> EvidenceRelation | None:
    subject = _subject(question, text)

    def build(kind: str, values: Sequence[str], span: str, labels: Sequence[str] = ()) -> EvidenceRelation:
        return EvidenceRelation(subject, kind, tuple(_clean(value) for value in values),
                                source, page, chunk_id, " ".join(span.split()), target_language,
                                tuple(_clean(label) for label in labels), continuation_chunk_ids)

    requested_metadata = metadata_fields_for_query(question)
    if requested_metadata:
        metadata = extract_document_metadata(text)
        if all(metadata.get(field) for field in requested_metadata):
            if "publication_disclaimer" in requested_metadata:
                disclaimer = metadata["publication_disclaimer"]
                authority = re.search(r"(?:at\s+(?:the\s+)?discretion\s+of\s+([^.;]+)|at\s+([^.;]+?)'s\s+discretion)", disclaimer, re.I)
                change = re.search(r"\b(?:subject\s+to\s+change|may\s+change)\b", disclaimer, re.I)
                if authority and change:
                    return build("publication_disclaimer", [authority.group(1) or authority.group(2)], text)
                return None
            if "published_by" in requested_metadata:
                return build("published_by", [metadata["published_by"]], text)
            if "edition" in requested_metadata and "publication_date" in requested_metadata:
                return build("edition_publication_date", [metadata["edition"], metadata["publication_date"]], text)
            if "edition" in requested_metadata:
                return build("edition", [metadata["edition"]], text)
            if "publication_date" in requested_metadata:
                return build("publication_date", [metadata["publication_date"]], text)
        return None

    # A reconstructed table row is usable only when it names exactly the
    # requested course and owns both trailing cells in the same row.
    course_entities = extract_course_entities(question)
    if len(course_entities) == 1 and _asked(question, r"credits?|ক্রেডিট|prerequisite|পূর্বশর্ত"):
        row = re.fullmatch(
            r"\s*([A-Z]{2,5})\s*(\d{3})(?:\s*\(([A-Z])\))?\s+"
            r"([A-Za-z][A-Za-z: &-]{3,100}?)\s+(\d+(?:\.\d+)?)\s+"
            r"(Nil|None|[A-Z]{2,5}\s*\d{3}(?:\([A-Z]\))?)\s*",
            text, re.I,
        )
        if row:
            row_code = re.sub(r"[^A-Za-z0-9]", "", "".join(row.group(i) or "" for i in (1, 2, 3))).upper()
            if row_code == course_entities[0].compact:
                if _asked(question, r"credits?|ক্রেডিট"):
                    return build("course_credits", [row.group(5)], row.group(0), (course_entities[0].canonical,))
                return build("course_prerequisite", [row.group(6)], row.group(0), (course_entities[0].canonical,))

    if _asked(question, r"\bunder\b|\bact\b|\blaw\b|আইনের|আইন|অধীনে|odhine"):
        law = r"(?:[A-Z][A-Za-z-]*\s+){1,6}Act\s+\d{4}"
        match = re.search(rf"started\s+its\s+operation\s+(?:in\s+\d{{4}}\s+)?under\s+(?:the\s+)?({law})", text, re.I)
        if not match:
            match = re.search(rf"under\s+(?:the\s+)?({law})\s*,?\s*started\s+its\s+operation", text, re.I)
        if match:
            return build("started_under", [match.group(1)], match.group(0))

    if _asked(question, r"\binitial(?:ly)?\b|শুরুতে|shurute|প্রথম") and _asked(question, r"\bprograms?\b|প্রোগ্রাম|bachelor|ব্যাচেলর"):
        match = re.search(r"(?:started\s+its\s+operation|offered).{0,115}?Bachelor\s+Degree\s+Programs?\s+in\s+(.+?)\s+only\b", text, re.I)
        if match:
            parts = match.group(1).rsplit(" and ", 1)
            if len(parts) == 2 and all(1 <= len(part.split()) <= 8 and re.search(r"[A-Za-z]", part) for part in parts):
                return build("initially_offered", parts, match.group(0))

    if _asked(question, r"undergraduate|আন্ডারগ্র্যাজুয়েট|postgraduate|পোস্টগ্র্যাজুয়েট|disciplines?|ডিসিপ্লিন") and _asked(question, r"\bhow\s+many\b|\bkoy|\bkoto|কয়|কত"):
        match = re.search(r"undergraduate\s+programs?\s+in\s+(\w+)\s+disciplines?\s+and\s+post[- ]graduate\s+programs?\s+in\s+(\w+)\s+disciplines?", text, re.I)
        if match:
            return build("program_discipline_counts", match.groups(), match.group(0), ("undergraduate", "postgraduate"))

    if (_asked(question, r"\brepeat(?:ed|ing)?\b|পুনরায়|পুনঃপরীক্ষা")
            and _asked(question, r"\bcourses?\b|কোর্স")
            and _asked(question, r"\bhow\s+many\b|\bmaximum\b|\bkoyta\b|কয়টি|কত")):
        match = re.search(
            r"\brepeat\s+(?:up\s+to\s+|a\s+maximum\s+of\s+|at\s+most\s+)"
            r"(\d+|[A-Za-z]+)\s+courses?\b", text, re.I,
        )
        if match and (match.group(1).isdigit() or match.group(1).casefold() in _NUMBER_WORDS):
            return build("repeat_course_maximum", [match.group(1)], match.group(0))

    category_request = re.search(r"\bCategory\s*[- ]?\s*(\d{1,2})\b|ক্যাটাগরি\s*(\d{1,2})", question, re.I)
    if category_request:
        number = next(value for value in category_request.groups() if value)
        match = re.search(
            rf"\bCategory\s*[- ]?\s*{re.escape(number)}\s*:\s*(.+?[.!?])\s*"
            rf"(A\s+student\s+of\s+Category\s*[- ]?\s*{re.escape(number)}\s+.+?[.!?])",
            text, re.I,
        )
        if match:
            criteria, eligibility = map(_clean, match.groups())
            if (re.search(r"passed\s+all\s+the\s+(?:courses\s+)?prescribed\s+courses|passed\s+all\s+the\s+courses\s+prescribed", criteria, re.I)
                    and re.search(r"no\s+backlog", criteria, re.I)
                    and re.search(r"eligible\s+for\s+registration\s+in\s+all\s+courses", eligibility, re.I)):
                return build("category_definition", [criteria, eligibility], match.group(0), (number,))

    if is_mark_distribution_query(question):
        match = re.search(r"distribution\s+of\s+marks[^:]{0,100}:\s*(.+?)(?:\bTotal\s+\d+(?:\.\d+)?\s*%|$)", text, re.I)
        if match:
            pairs = re.findall(r"([A-Za-z][A-Za-z ]{1,35}?)\s+(\d+(?:\.\d+)?)\s*%", match.group(1))
            labels = tuple(_clean(label) for label, _ in pairs)
            values = tuple(number + "%" for _, number in pairs)
            if 2 <= len(pairs) <= 8 and len(set(label.casefold() for label in labels)) == len(labels):
                return build("mark_distribution", values, match.group(0), labels)

    if _asked(question, r"attendance|উপস্থিতি|hajira") and _asked(question, r"final\s+exam|ফাইনাল|বসতে|boshte"):
        match = re.search(r"required\s+to\s+attend\s+at\s+least\s+(\d+(?:\.\d+)?\s*%)\s+of\s+all\s+the\s+classes.{0,100}?sit\s+for\s+the\s+final\s+examination", text, re.I)
        if match:
            qualifiers = ("at_least",) + (("every_course",) if re.search(r"\bin\s+every\s+course\b", match.group(0), re.I) else ())
            return build("attendance_exam_requirement", [match.group(1)], match.group(0), qualifiers)

    if _asked(question, r"honors|সম্মান|honours") and _asked(question, r"CGPA|সিজিপিএ"):
        match = re.search(r"degree\s+with\s+honors\s+if\s+his/her\s+CGPA\s+is\s+(\d+(?:\.\d+)?)\s+or\s+above", text, re.I)
        if match:
            return build("honors_cgpa", [match.group(1)], match.group(0))

    if _asked(question, r"normal\s+progress|স্বাভাবিক\s+অগ্রগতি") and _asked(question, r"CGPA|সিজিপিএ"):
        match = re.search(r"normal\s+progress\s+towards\s+a\s+degree\s+if\s+his/her\s+CGPA.{0,60}?is\s+(\d+(?:\.\d+)?)\s+or\s+better", text, re.I)
        if match:
            return build("normal_progress_cgpa", [match.group(1)], match.group(0))

    if _asked(question, r"regular\s+semester|নিয়মিত\s+সেমিস্টার") and _asked(question, r"\bhow\s+many\b|\bkoy|\bkoto|কয়|কত") and not _asked(question, r"duration|কতদিন|class|ক্লাস|week|সপ্তাহ"):
        match = re.search(r"There\s+will\s+be\s+(\w+)\s+semesters?\s*[-–]\s*(\w+)\s+and\s+(\w+)\s+Semester\s+in\s+an\s+academic\s+year", text, re.I)
        if match:
            return build("regular_semester_count", match.groups(), match.group(0))

    if _asked(question, r"re[- ]?examination|re[- ]?scrutiny|পুনঃপরীক্ষা") and _asked(question, r"application|submit|আবেদন|working\s+day"):
        match = re.search(r"within\s+(\d+)\s*\([^)]*\)\s+working\s+days\s+from\s+the\s+publication\s+of\s+final\s+results", text, re.I)
        if match:
            return build("reexamination_deadline", [match.group(1)], match.group(0))
    return None


def extract_relation(question: str, assessment: EvidenceAssessment,
                     retrieved: Sequence[dict[str, Any]], target_language: str) -> EvidenceRelation | None:
    if assessment.status is not SupportStatus.SUPPORTED:
        return None
    for evidence in assessment.evidence:
        relation = _relation(question, evidence.excerpt, target_language, evidence.source,
                             evidence.page, evidence.chunk_id, evidence.continuation_chunk_ids)
        if relation:
            return relation
    return None


def validate_relation_grounding(question: str, item: EvidenceRelation,
                                assessment: EvidenceAssessment, answer: str = "") -> dict[str, Any]:
    """Re-extract from the cited verified span; never trust an answer alone."""
    for evidence in assessment.evidence:
        if (evidence.source != item.source or evidence.page != item.page
                or evidence.chunk_id != item.chunk_id):
            continue
        if item.evidence_span not in " ".join(evidence.excerpt.split()):
            continue
        repeated = _relation(question, item.evidence_span, item.target_language,
                             item.source, item.page, item.chunk_id,
                             item.continuation_chunk_ids)
        if (repeated and repeated.relation == item.relation
                and repeated.values == item.values and repeated.labels == item.labels):
            if answer and item.relation == "attendance_exam_requirement":
                required = {
                    "at_least": r"\bat\s+least\b|\bkompakhe\b|অন্তত|কমপক্ষে",
                    "every_course": r"\b(?:every|each)\s+course\b|\bprotiti\s+course\b|\bproti\s+course\b|প্রতিটি\s+কোর্স|প্রত্যেক\s+কোর্স",
                }
                missing = [label for label in item.labels if label in required and not re.search(required[label], answer, re.I)]
                if missing:
                    return {"grounding_validation_passed": False,
                            "grounding_validation_reason": "MISSING_REQUIRED_QUALIFIER",
                            "unsupported_facts": missing}
            if answer and item.relation == "repeat_course_maximum" and not re.search(
                r"\bat\s+most\b|\bmaximum\b|\bup\s+to\b|\bshorboccho\b|সর্বোচ্চ", answer, re.I
            ):
                return {"grounding_validation_passed": False,
                        "grounding_validation_reason": "MISSING_REQUIRED_QUALIFIER",
                        "unsupported_facts": ["maximum"]}
            return {"grounding_validation_passed": True,
                    "grounding_validation_reason": "RELATION_EVIDENCE_VERIFIED",
                    "unsupported_facts": []}
    return {"grounding_validation_passed": False,
            "grounding_validation_reason": "RELATION_EVIDENCE_MISMATCH",
            "unsupported_facts": list(item.values)}


_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def realize_relation(item: EvidenceRelation) -> str:
    language, subject, values = item.target_language, item.subject, item.values
    if language not in {"english", "bangla", "banglish"} or not values:
        return ""
    relation = item.relation
    if relation in {"course_credits", "course_prerequisite"} and item.labels:
        course = item.labels[0]
        if relation == "course_credits":
            return {"english": f"{course} carries {values[0]} credits.",
                    "bangla": f"{course} কোর্সের ক্রেডিট {values[0]}।",
                    "banglish": f"{course} course-er credit {values[0]}."}[language]
        value = "none listed" if values[0].casefold() in {"nil", "none"} else values[0]
        return {"english": f"The prerequisite for {course} is {value}.",
                "bangla": f"{course} কোর্সের পূর্বশর্ত: {value}।",
                "banglish": (f"{course} course-er kono prerequisite deya nei."
                            if values[0].casefold() in {"nil", "none"}
                            else f"{course} course-er prerequisite {value}.")}[language]
    if relation == "published_by":
        return {"english": f"The prospectus was published by {values[0]}.",
                "bangla": f"প্রসপেক্টাসটি প্রকাশ করেছে {values[0]}।",
                "banglish": f"Prospectus-ta {values[0]} publish koreche."}[language]
    if relation == "publication_disclaimer":
        bangla_authority = re.sub(r"^the\s+", "", values[0], flags=re.I)
        return {"english": f"The disclaimer says the prospectus information may change at the discretion of {values[0]}.",
                "bangla": f"ডিসক্লেইমার অনুযায়ী, {bangla_authority}-এর বিবেচনায় প্রসপেক্টাসের তথ্য পরিবর্তন হতে পারে।",
                "banglish": f"Disclaimer onujayi, {values[0]}-er discretion-e prospectus-er information change hote pare."}[language]
    if relation in {"edition_publication_date", "edition", "publication_date"}:
        if relation == "edition_publication_date":
            return {"english": f"The prospectus lists {values[0]} and the date {values[1]}.",
                    "bangla": f"প্রসপেক্টাসে {values[0]} সংস্করণ এবং {values[1]} তারিখ দেওয়া আছে।",
                    "banglish": f"Prospectus-e {values[0]} ar {values[1]} date deya ache."}[language]
        noun = "edition" if relation == "edition" else "date"
        return {"english": f"The prospectus {noun} is {values[0]}.",
                "bangla": f"প্রসপেক্টাসের {'সংস্করণ' if noun == 'edition' else 'তারিখ'} {values[0]}।",
                "banglish": f"Prospectus-er {noun} {values[0]}."}[language]
    if relation == "started_under":
        return {"english": f"{subject} started its operation under {values[0]}.",
                "bangla": f"{subject} {values[0]}-এর অধীনে কার্যক্রম শুরু করেছিল।",
                "banglish": f"{subject} {values[0]}-er odhine operation start korechilo."}[language]
    if relation == "initially_offered" and len(values) == 2:
        return {"english": f"{subject} initially offered the bachelor degree programs {values[0]} and {values[1]}.",
                "bangla": f"{subject} শুরুতে {values[0]} এবং {values[1]}—এই দুইটি ব্যাচেলর ডিগ্রি প্রোগ্রাম চালু করেছিল।",
                "banglish": f"{subject} shurute {values[0]} ar {values[1]}—ei duita bachelor degree program offer korechilo."}[language]
    if relation == "program_discipline_counts" and len(values) == 2:
        first, second = (_NUMBER_WORDS.get(value.casefold(), value) for value in values)
        return {"english": f"{subject} offers undergraduate programs in {first} disciplines and postgraduate programs in {second} disciplines.",
                "bangla": f"{subject} {first}টি আন্ডারগ্র্যাজুয়েট এবং {second}টি পোস্টগ্র্যাজুয়েট ডিসিপ্লিনে প্রোগ্রাম অফার করে।",
                "banglish": f"{subject} {first}ta undergraduate ar {second}ta postgraduate discipline-e program offer kore."}[language]
    if relation == "category_definition" and item.labels and len(values) == 2:
        label = "Category " + item.labels[0]
        return {"english": f"{label}: {values[0]}. {values[1]}.",
                "bangla": f"{label}-এ ওই সেমিস্টারের নির্ধারিত সব কোর্স পাস করা এবং কোনো কোর্স বাকি না থাকা শিক্ষার্থীরা পড়ে। তারা পরবর্তী সেমিস্টারের নির্ধারিত সব কোর্সে নিবন্ধন করতে পারে।",
                "banglish": f"{label}-te oi semester-er shob prescribed course pass kora, kono backlog na thaka students-ra pore. Tara next semester-er shob prescribed course-e registration korte pare."}[language]
    if relation == "mark_distribution" and len(values) == len(item.labels):
        translations = {"assessment": ("মূল্যায়ন", "assessment"), "mid semester": ("মধ্য সেমিস্টার", "mid semester"),
                        "final exam": ("চূড়ান্ত পরীক্ষা", "final exam")}
        parts = []
        for label, value in zip(item.labels, values):
            translated = translations.get(label.casefold(), (label, label))
            label_text = label if language == "english" else translated[0 if language == "bangla" else 1]
            parts.append(f"{label_text} {value}")
        joined = ", ".join(parts)
        return {"english": f"The mark distribution is {joined}.",
                "bangla": f"নম্বর বণ্টন হলো: {joined}।",
                "banglish": f"Marks distribution holo: {joined}."}[language]
    if relation == "attendance_exam_requirement":
        every = "every_course" in item.labels
        return {"english": f"A student must attend at least {values[0]} of classes{' in every course' if every else ''} to sit for the final examination.",
                "bangla": f"চূড়ান্ত পরীক্ষায় বসতে {'প্রতিটি কোর্সে ' if every else ''}অন্তত {values[0]} ক্লাসে উপস্থিত থাকতে হবে।",
                "banglish": f"Final exam-e boshte {'protiti course-e ' if every else ''}kompakhe {values[0]} class-e attend korte hobe."}[language]
    if relation == "repeat_course_maximum":
        count = _NUMBER_WORDS.get(values[0].casefold(), values[0])
        return {"english": f"A student may repeat at most {count} courses for grade improvement.",
                "bangla": f"গ্রেড উন্নয়নের জন্য একজন শিক্ষার্থী সর্বোচ্চ {count}টি কোর্স পুনরায় নিতে পারে।",
                "banglish": f"Grade improvement-er jonno ekjon student shorboccho {count}ta course repeat korte pare."}[language]
    if relation in {"honors_cgpa", "normal_progress_cgpa"}:
        honors = relation == "honors_cgpa"
        return {"english": f"{'A degree with honors requires' if honors else 'Normal progress toward a degree requires'} a CGPA of {values[0]} or above.",
                "bangla": f"{'সম্মানসহ ডিগ্রির' if honors else 'ডিগ্রির পথে স্বাভাবিক অগ্রগতির'} জন্য CGPA অন্তত {values[0]} হতে হবে।",
                "banglish": f"{'Honors degree-r' if honors else 'Degree-r normal progress-er'} jonno CGPA kompakhe {values[0]} lagbe."}[language]
    if relation == "regular_semester_count" and len(values) == 3:
        count = _NUMBER_WORDS.get(values[0].casefold(), values[0])
        return {"english": f"An academic year has {count} regular semesters: {values[1]} and {values[2]}.",
                "bangla": f"এক শিক্ষাবর্ষে {count}টি নিয়মিত সেমিস্টার আছে: {values[1]} ও {values[2]}।",
                "banglish": f"Ek academic year-e {count}ta regular semester ache: {values[1]} ar {values[2]}."}[language]
    if relation == "reexamination_deadline":
        return {"english": f"The re-examination application must be submitted within {values[0]} working days after publication of final results.",
                "bangla": f"চূড়ান্ত ফল প্রকাশের {values[0]} কর্মদিবসের মধ্যে পুনঃপরীক্ষার আবেদন জমা দিতে হবে।",
                "banglish": f"Final result publish howar {values[0]} working day-er moddhe re-examination application submit korte hobe."}[language]
    return ""


__all__ = ["EvidenceRelation", "extract_relation", "realize_relation"]
