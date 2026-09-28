from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Sequence

from .language_detector import Language
from .query_normalization import extract_course_entities, is_mark_distribution_query, normalize_retrieval_text
from .document_metadata import extract_document_metadata, metadata_fields_for_query
from .course_rows import course_field_value


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RequestedEntity:
    kind: str
    value: str
    normalized: str


@dataclass(frozen=True)
class QueryRequest:
    entities: tuple[RequestedEntity, ...]
    requested_field: str
    ambiguous: bool = False
    ambiguity_reason: str = ""

    @property
    def primary_entity(self) -> str | None:
        return self.entities[0].value if self.entities else None


@dataclass(frozen=True)
class Evidence:
    document_id: str | None
    source: str
    relative_path: str | None
    page: int | None
    chunk_id: str | int | None
    text: str
    excerpt: str
    retrieval_score: float
    detected_entity: str | None
    matched_entity: str | None
    requested_field: str
    matched_field: str | None
    support_status: str
    continuation_chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAssessment:
    request: QueryRequest
    status: SupportStatus
    evidence: tuple[Evidence, ...]
    reason: str
    conflicting_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConflictIdentity:
    """Semantic identity of a claim; provenance is retained, not arbitrated."""

    entity_id: str
    field: str
    relation: str
    scope: str
    qualifiers: tuple[str, ...]
    document_context: str = ""

    @property
    def comparison_key(self) -> tuple[str, str, str, str, tuple[str, ...]]:
        # Different documents or editions may genuinely disagree. Neither
        # source identity nor recency is a reason to suppress the conflict.
        return self.entity_id, self.field, self.relation, self.scope, self.qualifiers


# Patterns describe document language, not a particular dataset, PDF, or course.
FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "credits": (r"\bcredits?\b", r"credit\s+value", r"ক্রেডিট"),
    "prerequisite": (r"\bpre\s*[- ]?\s*requisites?\b", r"\bprereq\b", r"পূর্বশর্ত"),
    "course_title": (r"course\s+title", r"\btitle\b", r"শিরোনাম", r"নাম"),
    "course_code": (r"course\s+(?:no\.?|code|number)", r"\bcode\b", r"কোড"),
    "semester": (r"\bsemesters?\b", r"\bterm\b", r"সেমিস্টার"),
    "attendance": (r"\battendance\b", r"উপস্থিতি", r"হাজিরা"),
    "grade": (r"\bgrades?\b", r"\bcgpa\b", r"গ্রেড"),
    "percentage": (r"\bpercent(?:age)?\b", r"\d+(?:\.\d+)?\s*%", r"শতাংশ"),
    "email": (r"\be-?mail\b", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", r"ইমেইল"),
    "requirement": (r"\brequirements?\b", r"\brequired\b", r"যোগ্যতা", r"প্রয়োজন"),
    "duration": (r"\bduration\b", r"\b(?:years?|months?|weeks?|days?)\b", r"মেয়াদ", r"সময়কাল"),
    "topic": (r"\btopics?\b", r"course\s+content", r"\bcontent\b", r"বিষয়", r"পড়ানো", r"পড়ানো"),
    "policy": (r"\bpolic(?:y|ies)\b", r"\brules?\b", r"regulations?", r"নীতি", r"নিয়ম"),
    "publication": (r"\bpublish(?:ed|er|ing)?\b", r"publication", r"প্রকাশ"),
    "date": (r"\bdates?\b", r"\bwhen\b", r"তারিখ", r"কবে"),
    "course_type": (r"course\s+type", r"\btype\b", r"ধরন"),
    "objective": (r"\bobjectives?\b", r"course\s+summary", r"উদ্দেশ্য", r"uddesh"),
    "learning_outcome": (r"\bclo\s*\d*\b", r"course\s+learning\s+outcomes?"),
    "weekly_content": (r"\bweek\s*\d+\b", r"weekly\s+plan", r"সপ্তাহ"),
    "assessment": (r"\bassessment\b", r"\bexam(?:ination)?\b", r"\bmarks?\b", r"পরীক্ষা", r"নম্বর"),
}


FIELD_QUERY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("learning_outcome", FIELD_PATTERNS["learning_outcome"]),
    ("prerequisite", FIELD_PATTERNS["prerequisite"]),
    ("attendance", FIELD_PATTERNS["attendance"]),
    ("email", FIELD_PATTERNS["email"]),
    ("credits", FIELD_PATTERNS["credits"]),
    ("course_title", FIELD_PATTERNS["course_title"]),
    ("course_code", FIELD_PATTERNS["course_code"]),
    ("course_type", FIELD_PATTERNS["course_type"]),
    ("semester", FIELD_PATTERNS["semester"]),
    ("grade", FIELD_PATTERNS["grade"]),
    ("percentage", FIELD_PATTERNS["percentage"]),
    ("weekly_content", FIELD_PATTERNS["weekly_content"]),
    ("duration", FIELD_PATTERNS["duration"]),
    ("objective", FIELD_PATTERNS["objective"]),
    ("topic", FIELD_PATTERNS["topic"] + (r"\bcovered\b", r"\bporano\b")),
    ("publication", FIELD_PATTERNS["publication"] + (r"\bwho\b", r"কে\s+প্রকাশ")),
    ("requirement", FIELD_PATTERNS["requirement"]),
    ("policy", FIELD_PATTERNS["policy"]),
    ("date", FIELD_PATTERNS["date"]),
    ("assessment", FIELD_PATTERNS["assessment"]),
)


ENTITY_DEPENDENT_FIELDS = {
    "credits", "prerequisite", "course_title", "course_code", "course_type",
    "learning_outcome", "objective", "topic", "weekly_content",
    "publication", "attendance", "percentage",
}


def _normal(text: str) -> str:
    text = normalize_retrieval_text(text)
    text = re.sub(r"[^a-z0-9\u0980-\u09ff@.+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _course_code_parts(text: str) -> Iterable[tuple[str, str, tuple[int, int]]]:
    for entity in extract_course_entities(text):
        yield entity.raw, entity.compact, entity.span


def identify_entities(question: str) -> tuple[RequestedEntity, ...]:
    found: list[RequestedEntity] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, normalized: str | None = None) -> None:
        key = (kind, normalized or _normal(value))
        if key[1] and key not in seen:
            seen.add(key)
            found.append(RequestedEntity(kind, value.strip(), key[1]))

    course_spans: list[tuple[int, int]] = []
    for raw, compact, span in _course_code_parts(question):
        add("course_code", raw, compact)
        course_spans.append(span)
    for match in re.finditer(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", question):
        add("email", match.group(0))
    for match in re.finditer(r"\b(?:spring|summer|fall|autumn|winter)\s+\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+semester\b|\bsemester\s+\d{1,2}\b|\b\d{1,2}\s*তম\s*সেমিস্টার\b", question, re.I):
        add("semester", match.group(0))
    for match in re.finditer(r"\bsection\s+[A-Za-z0-9-]+\b|\bসেকশন\s+[A-Za-z0-9-]+\b", question, re.I):
        add("section", match.group(0))
    for match in re.finditer(r"\bgrade\s+[A-F](?:[+-])?\b|\bগ্রেড\s+[A-F](?:[+-])?\b", question, re.I):
        add("grade", match.group(0))
    for match in re.finditer(
        r"\b(?:B\.?Sc\.?|M\.?Sc\.?|Bachelor(?:'s)?|Master(?:'s)?|Ph\.?D\.?)\s+(?:degree\s+)?(?:in|of)\s+[A-Za-z][A-Za-z &-]{2,80}",
        question,
        re.I,
    ):
        add("program", match.group(0).strip(" ?.,"))
    for match in re.finditer(r"\b[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){1,5}\s+(?:Policy|Rules|Regulations)\b", question):
        add("policy", match.group(0))
    for match in re.finditer(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b", question):
        add("date", match.group(0))
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:credits?|years?|months?|weeks?|days?|%)(?!\w)", question, re.I):
        if not any(match.start() < end and match.end() > start for start, end in course_spans):
            add("value", match.group(0))
    return tuple(found)


def detect_requested_field(question: str) -> str:
    normalized = normalize_retrieval_text(question)
    if is_mark_distribution_query(normalized):
        return "assessment"
    for field, patterns in FIELD_QUERY_PATTERNS:
        if any(re.search(pattern, normalized, re.I) for pattern in patterns):
            return field
    return "general"


def analyze_query(question: str) -> QueryRequest:
    normalized = normalize_retrieval_text(question)
    entities = identify_entities(normalized)
    requested_field = detect_requested_field(normalized)
    has_subject_context = bool(
        re.search(r"\b(?:program(?:me)?|prospectus|university|department|policy|rule|semester|exam(?:ination)?)\b|বিশ্ববিদ্যালয়|বিভাগ|নীতি|নিয়ম|পরীক্ষা|প্রসপেক্টাস", normalized, re.I)
    )
    # A named policy/rule can identify the fact without identifying a course.
    # Generic pronouns and bare field names still need a subject.
    relation_sufficient = bool(
        requested_field == "credits" and re.search(
            r"theoretical\s+courses?|theory\s+courses?|থিওরেটিক্যাল\s+কোর্স|তাত্ত্বিক\s+কোর্স", normalized, re.I
        ) and re.search(r"assign|নির্ধারিত|নির্ধারণ|কীভাবে|কিভাবে|kivabe", normalized, re.I)
        or requested_field == "percentage" and re.search(
            r"\bseats?\b|আসন", normalized, re.I
        ) and re.search(r"reserv|সংরক্ষিত|সংরক্ষণ", normalized, re.I)
        or requested_field == "attendance" and re.search(
            r"\bpolicy\b|\brule\b|নীতি|নিয়ম", normalized, re.I
        )
    )
    ambiguous = requested_field in ENTITY_DEPENDENT_FIELDS and not entities and not has_subject_context and not relation_sufficient
    reason = "The requested field needs a specific entity or context." if ambiguous else ""
    return QueryRequest(entities, requested_field, ambiguous, reason)


def _entity_occurs(entity: RequestedEntity, text: str) -> bool:
    if entity.kind == "course_code":
        compact_text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return entity.normalized in compact_text
    return entity.normalized in _normal(text)


def _course_is_subject(entity: RequestedEntity, text: str, metadata: dict[str, Any] | None = None) -> bool:
    requested = entity.normalized
    if metadata and metadata.get("entity_type") == "course" and metadata.get("entity_id"):
        structured_entity = re.sub(r"[^A-Za-z0-9]", "", str(metadata["entity_id"])).upper()
        if structured_entity == requested and _entity_occurs(entity, text):
            return True
    code_pattern = r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?"

    # A labeled course block establishes ownership until another labeled block.
    labeled = re.findall(
        rf"course\s+(?:no\.?\s*/\s*course\s+code|code)\s*:\s*({code_pattern})",
        text,
        re.I,
    )
    if labeled:
        return any(re.sub(r"[^A-Za-z0-9]", "", value).upper() == requested for value in labeled)

    # Curriculum tables establish ownership through a row: code, title, credit,
    # prerequisite. The negative lookahead prevents a prerequisite mention from
    # borrowing the following course's row.
    requested_pattern = r"\s*".join(re.escape(part) for part in re.findall(r"[A-Za-z0-9]+", entity.value))
    has_table_header = bool(re.search(r"course\s+code\s+course\s+title\s+credits?", text, re.I))
    if has_table_header and re.search(
        rf"\b{requested_pattern}\b\s+(?!{code_pattern}\b).{{2,140}}?\s+\d+(?:\.\d+)?\s+(?:Nil|None|{code_pattern})",
        re.sub(r"\s+", " ", text),
        re.I,
    ):
        return True

    # Outside a recognized block/table, presence alone does not prove that the
    # surrounding facts belong to the requested course.
    return False


def _field_occurs(field: str, text: str, question: str) -> bool:
    if field == "general":
        query_terms = {
            token for token in re.findall(r"[A-Za-z\u0980-\u09ff]{3,}", _normal(question))
            if token not in {"what", "which", "where", "when", "who", "how", "university", "document"}
        }
        body = _normal(text)
        return sum(term in body for term in query_terms) >= min(2, len(query_terms)) if query_terms else False
    return any(re.search(pattern, text, re.I) for pattern in FIELD_PATTERNS.get(field, ()))


def _structured_field_occurs(field: str, text: str, metadata: dict[str, Any]) -> bool:
    hints = set(metadata.get("field_types") or [])
    if field not in hints:
        return False
    if field == "credits":
        return bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
    if field == "prerequisite":
        return bool(re.search(r"\b(?:Nil|None|[A-Za-z]{2,12}\s*\d{2,4})\b", text, re.I))
    if field == "course_title":
        title = str(metadata.get("course_title") or "").strip()
        return bool(title and _normal(title) in _normal(text))
    if field == "course_code":
        return bool(metadata.get("entity_id"))
    return _field_occurs(field, text, "")


def _support_excerpt(text: str, field: str, entities: Sequence[RequestedEntity], limit: int = 420) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if len(clean) <= limit:
        return clean
    anchors = [re.escape(entity.value) for entity in entities]
    anchors.extend(FIELD_PATTERNS.get(field, ()))
    position = 0
    for anchor in anchors:
        match = re.search(anchor, clean, re.I)
        if match:
            position = match.start()
            break
    start = max(0, position - limit // 3)
    end = min(len(clean), start + limit)
    excerpt = clean[start:end].strip()
    if start:
        excerpt = "…" + excerpt
    if end < len(clean):
        excerpt += "…"
    return excerpt


def _fact_values(field: str, excerpt: str) -> set[str]:
    patterns: dict[str, tuple[str, ...]] = {
        "credits": (r"(?:credit(?:\s+value)?|credits?)\s*[:=-]?\s*(\d+(?:\.\d+)?)",),
        "percentage": (r"(\d+(?:\.\d+)?)\s*%",),
        "attendance": (r"attendance[^.\n]{0,100}?(\d+(?:\.\d+)?)\s*%", r"(\d+(?:\.\d+)?)\s*%[^.\n]{0,100}?attendance"),
        "email": (r"([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",),
        "course_code": (r"course\s+(?:no\.?\s*/\s*)?(?:course\s+)?code\s*:\s*([A-Za-z][A-Za-z ()-]*\d{2,4}(?:\([A-Za-z0-9]+\))?)",),
        "course_title": (r"course\s+title\s*:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|\s+course\s+(?:type|credit)|$)",),
        "course_type": (r"course\s+type\s*:\s*(.*?)(?=\s+\d+\.\s+[A-Z]|\s+credit|$)",),
        "prerequisite": (
            r"pre\s*[- ]?\s*requisites?\s*[:=-]\s*(None|Nil|No\s+prerequisite|Not\s+applicable|N/?A|[A-Z]{2,8}\s*\d{2,4}(?:\s*,\s*[A-Z]{2,8}\s*\d{2,4})*)\b",
            r"\b\d+(?:\.\d+)?\s+(Nil|None)\b",
        ),
        "grade": (r"(?:grade\s*[:=-]?\s*)([A-F][+-]?)",),
        "duration": (r"(\d+(?:\.\d+)?\s*(?:years?|months?|weeks?|days?))",),
        "semester": (r"((?:spring|summer|fall|autumn|winter)\s+\d{4}|(?:semester\s+\d{1,2}|\d{1,2}(?:st|nd|rd|th)\s+semester))",),
        "date": (r"((?:19|20)\d{2}|\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4}))",),
    }
    values: set[str] = set()
    for pattern in patterns.get(field, ()):
        for match in re.finditer(pattern, excerpt, re.I):
            raw = match.group(1).strip()
            value = _normal(raw).strip(" .;:")
            # N/A is equivalent only after a verified prerequisite label or
            # row has bound the value to that field. Never normalize it globally.
            if field == "prerequisite" and re.fullmatch(
                r"(?:none|nil|no\s+pre\s*[- ]?\s*requisite|not\s+applicable|n/?a)", raw, re.I
            ):
                value = "NO_PREREQUISITE"
            values.add(value)
    return {value for value in values if value}


def _material_qualifiers(text: str) -> tuple[str, ...]:
    qualifiers: list[str] = []
    category = re.search(r"\bcategory\s*[- ]?\s*(\d+)\b|ক্যাটাগরি\s*(\d+)", text, re.I)
    if category:
        qualifiers.append("category:" + next(value for value in category.groups() if value))
    for label, pattern in (
        ("undergraduate", r"\bundergraduate\b|আন্ডারগ্র্যাজুয়েট"),
        ("postgraduate", r"\bpostgraduate\b|পোস্টগ্র্যাজুয়েট"),
        ("every_course", r"\b(?:every|each)\s+course\b|প্রতিটি\s+কোর্স|প্রত্যেক\s+কোর্স"),
        ("per_semester", r"\bper\s+semester\b|প্রতি\s+সেমিস্টার"),
    ):
        if re.search(pattern, text, re.I):
            qualifiers.append(label)
    return tuple(qualifiers)


def _credit_relation(text: str, *, question: bool = False) -> str:
    theory = re.search(r"\btheoretical\s+courses?\b|\btheory\s+courses?\b|থিওরেটিক্যাল\s+কোর্স|তাত্ত্বিক\s+কোর্স", text, re.I)
    assignment = re.search(r"\bassign(?:ed|ment)?\b|\bone\s+credit\b|\b1\s+credit\b|কীভাবে|কিভাবে|নির্ধারিত|kivabe", text, re.I)
    theory_heading = re.search(r"\btheoretical\s+courses?\s*:", text, re.I)
    if theory and (bool(assignment) if question else bool(theory_heading)):
        return "theoretical_course_credit_rule"
    lab = re.search(r"\b(?:laboratory|lab)\s*(?:/field)?\s*(?:courses?|work)?\b|ল্যাব\s+কোর্স", text, re.I)
    if lab and (bool(assignment) if question else bool(re.search(r"\b(?:laboratory|lab)(?:/field)?(?:/design)?\s*(?:courses?|work)?\s*:", text, re.I))):
        return "lab_course_credit_rule"
    if re.search(r"\b(?:semester\s+total|total\s+(?:semester\s+)?credits?)\b|মোট\s+ক্রেডিট", text, re.I):
        return "semester_credit_total"
    return "course_credit_value" if question or re.search(r"\b(?:course\s+code|credits?\s*[:=-]|credit\s+value)\b", text, re.I) else "credit_other"


def _theoretical_rule_value(text: str) -> str | None:
    match = re.search(
        r"(?:theoretical\s+courses?|theory\s+courses?)\s*:\s*(.+?)"
        r"(?=\b(?:laboratory|lab)\s*(?:/field)?(?:/design)?\b|\btype\s+of\s+courses\b|$)",
        text, re.I | re.S,
    )
    if not match:
        return None
    rule = " ".join(match.group(1).split())
    # Compare the assignment rule, not an illustrative three-credit course.
    clause = re.split(r"\bThus\b|[.;]", rule, maxsplit=1, flags=re.I)[0]
    numbered = re.sub(r"\bone\b", "1", clause, flags=re.I)
    numbered = re.sub(r"\btwo\b", "2", numbered, flags=re.I)
    numbered = re.sub(r"\bthree\b", "3", numbered, flags=re.I)
    forward = re.search(
        r"(\d+)\s+(lecture|hour|class)(?:s)?\s+per\s+week"
        r"(?:\s+per\s+semester)?.{0,75}?equivalent\s+to\s+(\d+)\s+credits?",
        numbered, re.I,
    )
    reverse = re.search(
        r"(\d+)\s+credits?\s*(?:=|equals?|(?:is\s+)?equivalent\s+to)\s*"
        r"(\d+)\s+(lecture|hour|class)(?:s)?(?:\s+per\s+week)?",
        numbered, re.I,
    )
    if forward:
        amount, unit, credits = forward.groups()
    elif reverse:
        credits, amount, unit = reverse.groups()
    else:
        return None
    return f"{amount}:{unit.casefold()}_per_week:{credits}:credit"


def _credit_claim_relation(text: str, requested: str) -> tuple[str, str | None]:
    if requested == "theoretical_course_credit_rule":
        value = _theoretical_rule_value(text)
        return requested, value
    observed = _credit_relation(text)
    if observed != requested:
        return observed, None
    return observed, None


def _conflict_identity(question: str, request: QueryRequest, evidence: Evidence,
                       item: dict[str, Any]) -> tuple[ConflictIdentity, set[str]] | None:
    field = request.requested_field
    requested_relation = _credit_relation(question, question=True) if field == "credits" else field
    value: str | None = None
    if field == "credits":
        observed_relation = _credit_relation(evidence.text)
        if requested_relation == "course_credit_value" and observed_relation == "semester_credit_total":
            return None
        owned = (course_field_value(question, "credits", [evidence.text])
                 if requested_relation == "course_credit_value" and request.primary_entity else None)
        relation, value = (("course_credit_value", owned) if owned is not None else
                           _credit_claim_relation(evidence.text, requested_relation))
        if relation != requested_relation:
            return None
        if requested_relation == "theoretical_course_credit_rule" and not value:
            return None
    elif field == "prerequisite":
        relation = "course_prerequisite"
        owned = course_field_value(question, "prerequisite", [evidence.text]) if request.primary_entity else None
        if owned is not None:
            value = "NO_PREREQUISITE" if owned.casefold() in {"nil", "none", "n/a"} else _normal(owned)
    elif field == "attendance":
        relation = "attendance_requirement"
    elif field == "percentage":
        # Percentages from unrelated policy domains are not comparable.
        def domain(text: str) -> str:
            if re.search(r"\battendance\b|উপস্থিতি", text, re.I):
                return "attendance_requirement"
            if re.search(r"\bseats?\b.{0,50}\breserv|\breserv.{0,50}\bseats?\b|আসন", text, re.I):
                return "seat_reservation_percentage"
            if re.search(r"\bmarks?\b|নম্বর|মার্কস", text, re.I):
                return "marks_percentage"
            if re.search(r"\badmission\b|ভর্তি", text, re.I):
                return "admission_percentage"
            return "policy_percentage"
        relation = domain(evidence.text)
        if relation != domain(question):
            return None
    else:
        relation = field
    qualifiers = _material_qualifiers(evidence.text)
    requested_qualifiers = _material_qualifiers(question)
    for qualifier in requested_qualifiers:
        if qualifier.startswith("category:") and qualifier not in qualifiers:
            return None
    # Theoretical/lab scope is captured in the relation. Generic descriptions
    # may omit qualifiers; distinct *explicit* scopes must never conflict.
    scope = ("theoretical_courses" if relation == "theoretical_course_credit_rule" else
             "laboratory_courses" if relation == "lab_course_credit_rule" else
             "semester" if relation == "semester_credit_total" else
             "course" if request.primary_entity else "policy")
    context = str(item.get("edition") or item.get("effective_date") or item.get("publication_year") or "")
    identity = ConflictIdentity(request.primary_entity or "", field, relation, scope, qualifiers, context)
    values = {value} if value else _fact_values(field, evidence.excerpt)
    if field == "credits" and relation == "course_credit_value":
        canonical: set[str] = set()
        for raw in values:
            try:
                number = format(Decimal(raw).normalize(), "f")
                canonical.add(number if "." in number else number + ".0")
            except InvalidOperation:
                canonical.add(raw)
        values = canonical
    return identity, values


def assess_evidence(question: str, retrieved: Sequence[dict[str, Any]]) -> EvidenceAssessment:
    metadata_fields = metadata_fields_for_query(question)
    if metadata_fields:
        request = QueryRequest((), "document_metadata")
        verified: list[Evidence] = []
        fact_signatures: set[tuple[str, ...]] = set()
        for item in retrieved:
            text = str(item.get("text", ""))
            values = extract_document_metadata(text)
            if not all(values.get(field) for field in metadata_fields):
                continue
            fact_signatures.add(tuple(values[field].casefold() for field in metadata_fields))
            verified.append(Evidence(
                document_id=str(item.get("document_id")) if item.get("document_id") else None,
                source=str(item.get("source") or "unknown"),
                relative_path=str(item.get("relative_path")) if item.get("relative_path") else None,
                page=item.get("page"), chunk_id=item.get("chunk_id"), text=text,
                excerpt=" ".join(text.split())[:1200], retrieval_score=float(item.get("score", 0.0)),
                detected_entity=None, matched_entity=None,
                requested_field="document_metadata", matched_field="document_metadata",
                support_status=SupportStatus.SUPPORTED.value,
            ))
        if len(fact_signatures) > 1:
            conflicting = tuple(Evidence(**{**asdict(item), "support_status": SupportStatus.CONFLICTING.value}) for item in verified)
            return EvidenceAssessment(request, SupportStatus.CONFLICTING, conflicting,
                                      "Labeled document metadata contains conflicting values.")
        if verified:
            return EvidenceAssessment(request, SupportStatus.SUPPORTED, tuple(verified),
                                      "Labeled document metadata contains every requested field.")
        return EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (),
                                  "No labeled document metadata contained every requested field.")
    request = analyze_query(question)
    if request.ambiguous:
        return EvidenceAssessment(request, SupportStatus.AMBIGUOUS, (), request.ambiguity_reason)

    verified: list[Evidence] = []
    verified_items: list[dict[str, Any]] = []
    for item in retrieved:
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        matched_entities = [
            entity
            for entity in request.entities
            if (
                _course_is_subject(entity, text, item)
                if entity.kind == "course_code"
                else _entity_occurs(entity, text)
            )
        ]
        if request.entities and len(matched_entities) != len(request.entities):
            continue
        if request.requested_field == "general" and request.entities:
            field_matches = True
        else:
            field_matches = _field_occurs(request.requested_field, text, question) or _structured_field_occurs(
                request.requested_field, text, item
            )
        if not field_matches:
            continue
        excerpt = _support_excerpt(text, request.requested_field, request.entities)
        verified.append(
            Evidence(
                document_id=str(item.get("document_id")) if item.get("document_id") else None,
                source=str(item.get("source") or "unknown"),
                relative_path=str(item.get("relative_path")) if item.get("relative_path") else None,
                page=item.get("page"),
                chunk_id=item.get("chunk_id"),
                text=text,
                excerpt=excerpt,
                retrieval_score=float(item.get("score", 0.0)),
                detected_entity=request.primary_entity,
                matched_entity=matched_entities[0].value if matched_entities else None,
                requested_field=request.requested_field,
                matched_field=request.requested_field,
                support_status=SupportStatus.SUPPORTED.value,
            )
        )
        verified_items.append(item)

    if not verified:
        reason = "No retrieved passage matched both the requested entity and requested field."
        return EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (), reason)

    claims = [_conflict_identity(question, request, evidence, item)
              for evidence, item in zip(verified, verified_items)]
    if request.requested_field == "credits" and _credit_relation(question, question=True) == "theoretical_course_credit_rule":
        relevant = [(evidence, item, claim) for evidence, item, claim in zip(verified, verified_items, claims)
                    if claim is not None]
        if not relevant:
            return EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (),
                                      "No passage verified the requested theoretical-course credit rule.")
        verified = [evidence for evidence, _, _ in relevant]
        verified_items = [item for _, item, _ in relevant]
        claims = [claim for _, _, claim in relevant]

    grouped: dict[tuple[str, str, str, str, tuple[str, ...]], list[tuple[int, str]]] = {}
    for index, claim in enumerate(claims):
        if claim is None:
            continue
        identity, values = claim
        if len(values) == 1:
            grouped.setdefault(identity.comparison_key, []).append((index, next(iter(values))))
    conflicts = [entries for entries in grouped.values()
                 if len({value for _, value in entries}) > 1]
    if conflicts and request.requested_field in {
        "credits", "percentage", "attendance", "email", "course_code", "course_title",
        "course_type", "prerequisite", "grade", "duration", "semester", "date",
    }:
        conflicting_indices = {index for entries in conflicts for index, _ in entries}
        conflicting_values = {value for entries in conflicts for _, value in entries}
        conflicting = tuple(
            Evidence(**{**asdict(evidence), "support_status": SupportStatus.CONFLICTING.value})
            for index, evidence in enumerate(verified) if index in conflicting_indices
        )
        return EvidenceAssessment(
            request, SupportStatus.CONFLICTING, conflicting,
            "Retrieved passages contain different values for the same entity, relation, and scope.",
            tuple(sorted(conflicting_values)),
        )

    return EvidenceAssessment(
        request, SupportStatus.SUPPORTED, tuple(verified),
        "Retrieved evidence matches the requested entity and field.",
    )


def response_for_status(status: SupportStatus, language: Language) -> str:
    messages = {
        SupportStatus.INSUFFICIENT: {
            "english": "I couldn't find enough information in the provided university documents to answer this reliably.",
            "bangla": "প্রদত্ত বিশ্ববিদ্যালয়ের নথিতে নির্ভরযোগ্যভাবে উত্তর দেওয়ার মতো পর্যাপ্ত তথ্য পাওয়া যায়নি।",
            "banglish": "Provided university document-e reliable answer deyar moto enough information paoa jayni.",
        },
        SupportStatus.CONFLICTING: {
            "english": "The university documents contain conflicting evidence for this item.",
            "bangla": "বিশ্ববিদ্যালয়ের নথিতে এই বিষয়ে পরস্পরবিরোধী তথ্য রয়েছে।",
            "banglish": "University document-gulote ei bishoye conflicting information ache.",
        },
        SupportStatus.AMBIGUOUS: {
            "english": "Please specify the course, program, policy, or other item you mean so I can answer reliably.",
            "bangla": "নির্ভরযোগ্যভাবে উত্তর দেওয়ার জন্য অনুগ্রহ করে কোন কোর্স, প্রোগ্রাম, নীতি বা বিষয় বোঝাচ্ছেন তা নির্দিষ্ট করুন।",
            "banglish": "Reliable answer-er jonno kon course, program, policy, ba item bujhacchen seta specify korun.",
        },
    }
    return messages.get(status, messages[SupportStatus.INSUFFICIENT])[language]


def select_primary_evidence(answer: str, evidence: Sequence[Evidence]) -> Evidence | None:
    if not evidence:
        return None
    answer_tokens = set(re.findall(r"[A-Za-z0-9@.+-]{2,}", answer.casefold()))
    return max(
        evidence,
        key=lambda item: len(answer_tokens & set(re.findall(r"[A-Za-z0-9@.+-]{2,}", item.excerpt.casefold()))),
    )


def answer_claim_is_bound(answer: str, evidence: Sequence[Evidence]) -> bool:
    context = " ".join(item.excerpt for item in evidence).casefold()
    critical = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\d+(?:\.\d+)?%?|[A-Za-z]{2,12}\s*\d{2,4}", answer, re.I))
    if not all(re.sub(r"\s+", "", token.casefold()) in re.sub(r"\s+", "", context) for token in critical):
        return False

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "from",
        "in", "on", "and", "or", "this", "that", "course", "answer", "value", "holo",
        "credit", "credits", "prerequisite", "title", "code", "type", "grade", "semester",
        "attendance", "percentage", "duration", "topic", "topics", "policy", "requirement",
        "er", "ei", "উত্তর", "হলো", "কোর্স", "মান", "ক্রেডিট", "পূর্বশর্ত", "গ্রেড",
    }
    claim_tokens = {
        token for token in re.findall(r"[A-Za-z\u0980-\u09ff]{3,}", answer.casefold())
        if token not in stopwords
    }
    if not claim_tokens:
        return bool(critical)
    context_tokens = set(re.findall(r"[A-Za-z\u0980-\u09ff]{3,}", context))
    grounded = len(claim_tokens & context_tokens)
    return grounded >= 1 and grounded / len(claim_tokens) >= 0.5


__all__ = [
    "Evidence", "EvidenceAssessment", "QueryRequest", "RequestedEntity", "SupportStatus",
    "analyze_query", "answer_claim_is_bound", "assess_evidence", "detect_requested_field",
    "identify_entities", "response_for_status", "select_primary_evidence",
]
