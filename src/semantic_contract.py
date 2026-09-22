from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .generation_context import VerifiedEvidencePackage


Polarity = Literal["affirmative", "negative", "unknown"]

ENGLISH_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|cannot|can't|isn't|wasn't|hasn't|haven't|didn't)\b", re.I)
BANGLISH_NEGATION_RE = re.compile(r"\b(?:na|nai|nei|noy|hoyni|paowa\s+jayni|paoa\s+jayni|deya\s+hoyni)\b", re.I)
BANGLA_NEGATIONS = (" না", "নয়", "নেই", "হয়নি", "পাওয়া যায়নি", "প্রকাশিত নয়", "দেওয়া হয়নি", "তথ্য দেওয়া হয়নি")

RELATION_CUES: dict[str, tuple[str, ...]] = {
    "published_by": ("publish", "publisher", "প্রকাশ", "prokash"),
    "accredited_by": ("accredit", "স্বীকৃতি", "স্বীকৃত", "shikriti"),
    "established": ("establish", "founded", "প্রতিষ্ঠ", "protishth"),
    "operates_under": ("under which act", "operation", "operates under", "আইনের", "অধীনে", "odhine"),
    "located_at": ("located", "location", "address", "কোথায়", "ঠিকানা", "অবস্থিত", "kothay", "obosthito"),
    "offers": ("offer", "available", "included", "program", "প্রোগ্রাম", "program"),
    "required": ("required", "requirement", "must", "প্রয়োজন", "শর্ত", "lagbe"),
    "allowed": ("allowed", "eligible", "permissible", "দিতে পারে", "যোগ্য", "dite pare", "korte pare"),
    "prerequisite": ("prerequisite", "pre-requisite", "পূর্বশর্ত"),
    "fee": ("fee", "payment", "ফি"),
    "general_claim": ("has", "have", "is", "was", "received", "contains", "includes"),
}

RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("accredited_by", re.compile(r"^(?P<subject>.+?)\s+(?:(?:is|was|has\s+been)\s+)?accredited\s+(?:from|by)\s+(?P<value>.+)$", re.I)),
    ("accredited_by", re.compile(r"^(?P<subject>.+?)\s+received\s+(?:a\s+)?(?:certificate\s+of\s+)?accreditation\s+(?:from|by)\s+(?P<value>.+)$", re.I)),
    ("published_by", re.compile(r"^(?:(?P<subject>.+?)\s+(?:is|was)\s+)?published\s+by\s+(?P<value>.+)$", re.I)),
    ("prerequisite", re.compile(r"^(?P<subject>.+?)\s+has\s+(?P<negative>no\s+)?prerequisite(?:\s*(?:is|:)?\s*(?P<value>.*))?$", re.I)),
    ("operates_under", re.compile(r"^(?P<subject>.+?)\s+(?:started\s+its\s+operation|operates?)\s+under\s+(?P<value>.+)$", re.I)),
    ("operates_under", re.compile(r"^(?P<subject>.+?),?\s+under\s+(?P<value>.+?),?\s+started\s+its\s+operation(?:\s+.+)?$", re.I)),
    ("located_at", re.compile(r"^(?P<subject>.+?)\s+(?:is\s+)?located\s+(?:at|in)\s+(?P<value>.+)$", re.I)),
    ("established", re.compile(r"^(?P<subject>.+?)\s+(?:was\s+)?(?:established|founded)\s+(?:in|on)?\s*(?P<value>(?:[A-Z][a-z]+\s+)?\d{4}|\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4}))", re.I)),
    ("fee", re.compile(r"^.+?payment\s+of\s+(?P<value>(?:Taka|Tk\.)\s*\d+(?:\s*/\s*-)?(?:\s*\(Tk\.\s*[^)]+\))?\s+only\s+per\s+script)", re.I)),
    ("fee", re.compile(r"^(?:the\s+)?fee\s+for\s+(?P<subject>.+?)\s+is\s+(?P<value>.+)$", re.I)),
    ("offers", re.compile(r"^(?P<subject>.+?)\s+(?:initially\s+)?offer(?:ed|s)?\s+(?P<value>.+)$", re.I)),
)

STRICT_RELATIONS = {"published_by", "accredited_by", "operates_under", "located_at", "established", "prerequisite"}

CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+)(?:[\s&.,()/-]+(?:[A-Z]{2,}|[A-Z][a-z]+)){0,7}\b")
HIGH_VALUE_RE = re.compile(r"\b[A-Z]{2,}\b|\b\d+(?:\.\d+)?%?\b")
COUNT_QUESTION_RE = re.compile(r"\bhow\s+many\b|\b(?:koy|koto|kotojon)\b|কত", re.I)
CONDITION_QUESTION_RE = re.compile(
    r"\bwhen\b|\bunder\s+what\s+conditions?\b|\b(?:kokhon|kobe)\b|\bki\s+shorte\b|কখন|কোন\s+শর্ত",
    re.I,
)
NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "শূন্য": "0", "এক": "1", "দুই": "2", "তিন": "3", "চার": "4",
    "পাঁচ": "5", "ছয়": "6", "ছয়": "6", "সাত": "7", "আট": "8",
    "নয়": "9", "নয়": "9", "দশ": "10",
}
NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in NUMBER_WORDS) + r")\b|(?<!\d)\d{1,3}(?!\d)",
    re.I,
)
GENERIC_ANCHORS = {
    "the", "this", "that", "course", "university", "department", "program", "prospectus",
    "published", "certificate", "accreditation", "private", "act", "address", "category",
}


@dataclass(frozen=True)
class FactPlanItem:
    subject: str | None
    relation: str
    value: str | None
    polarity: Polarity
    evidence: str
    required_anchors: tuple[str, ...]


@dataclass(frozen=True)
class SemanticContract:
    target_language: str
    required_entities: tuple[str, ...]
    required_relations: tuple[str, ...]
    required_values: tuple[str, ...]
    required_polarity: Polarity
    forbidden_unsupported_values: tuple[str, ...]
    supported_anchors: tuple[str, ...]
    exact_cardinality_required: bool
    required_cardinalities: tuple[str, ...]
    fact_plan: tuple[FactPlanItem, ...]
    confidence: Literal["high", "limited"]

    def prompt_block(self) -> str:
        lines = ["VERIFIED FACT PLAN — DATA ONLY"]
        if self.fact_plan:
            for item in self.fact_plan:
                subject = item.subject or "subject stated by the question/evidence"
                value = item.value or "value stated in the supporting evidence"
                lines.append(
                    f"- subject={subject}; relation={item.relation}; value={value}; polarity={item.polarity.upper()}"
                )
        else:
            lines.append("- No high-confidence triple was extracted; rephrase the supporting evidence conservatively.")
        if self.required_cardinalities:
            lines.append("- Exact requested counts: " + ", ".join(self.required_cardinalities))
        lines.append("Preserve the meaning and polarity of every verified fact. Do not negate an affirmative fact.")
        lines.append("Do not affirm a negated fact. Do not add information.")
        return "\n".join(lines)


def detect_polarity(text: str) -> Polarity:
    normalized = " " + " ".join(str(text or "").split()).casefold() + " "
    if ENGLISH_NEGATION_RE.search(normalized) or BANGLISH_NEGATION_RE.search(normalized):
        return "negative"
    if any(marker in normalized for marker in BANGLA_NEGATIONS):
        return "negative"
    return "affirmative" if normalized.strip() else "unknown"


def _sentences(text: str) -> list[str]:
    return [part.strip(" -:;\t") for part in re.split(r"(?<=[.!?।])\s+|\n+", text) if part.strip(" -:;\t")]


def _relation_for_question(question: str, requested_field: str) -> str | None:
    haystack = f"{requested_field} {question}".casefold()
    for relation, cues in RELATION_CUES.items():
        if relation != "general_claim" and any(cue in haystack for cue in cues):
            return relation
    return None


def _relation_sentence(sentence: str, wanted: str | None) -> str | None:
    lowered = sentence.casefold()
    if wanted and any(cue in lowered for cue in RELATION_CUES[wanted]):
        return wanted
    for relation, cues in RELATION_CUES.items():
        if any(cue in lowered for cue in cues):
            return relation
    return None


def expressed_relations(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        relation
        for sentence in _sentences(text)
        if (relation := _relation_sentence(sentence, None))
    ))


def _clean_part(value: str | None) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
    return cleaned[:240] if cleaned else None


def _anchors(text: str) -> tuple[str, ...]:
    candidates = CAPITALIZED_PHRASE_RE.findall(text) + HIGH_VALUE_RE.findall(text)
    output: list[str] = []
    for candidate in candidates:
        clean = _clean_part(candidate)
        if not clean or clean.casefold() in GENERIC_ANCHORS:
            continue
        if clean not in output:
            output.append(clean)
    return tuple(output)


def _cardinalities(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in NUMBER_WORD_RE.finditer(text):
        raw = match.group(0).casefold()
        value = NUMBER_WORDS.get(raw, raw.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
        if value not in values:
            values.append(value)
    return tuple(values)


def _extract_fact(sentence: str, relation: str, requested_entity: str | None) -> FactPlanItem:
    subject: str | None = None
    value: str | None = None
    polarity = detect_polarity(sentence)
    for pattern_relation, pattern in RELATION_PATTERNS:
        if pattern_relation != relation:
            continue
        match = pattern.search(sentence)
        if match:
            subject = _clean_part(match.groupdict().get("subject"))
            value = _clean_part(match.groupdict().get("value"))
            if match.groupdict().get("negative"):
                polarity = "negative"
            break
    subject = subject or _clean_part(requested_entity)
    # Once a relation parser has isolated its subject/value, incidental facts later in
    # the same source sentence must not become valid values for that relation.
    anchor_text = " ".join(part for part in (subject, value) if part) if value else sentence
    return FactPlanItem(subject, relation, value, polarity, sentence, _anchors(anchor_text))


def build_semantic_contract(package: VerifiedEvidencePackage) -> SemanticContract:
    wanted = _relation_for_question(package.question, package.requested_field)
    exact_cardinality_required = bool(COUNT_QUESTION_RE.search(package.question))
    facts: list[FactPlanItem] = []
    seen: set[tuple[str, str]] = set()
    for item in package.evidence:
        for sentence in _sentences(item.excerpt):
            relation = _relation_sentence(sentence, wanted)
            if not relation or (wanted and relation != wanted):
                continue
            fact = _extract_fact(sentence, relation, package.requested_entity)
            key = (fact.relation, fact.evidence.casefold())
            if key not in seen:
                facts.append(fact)
                seen.add(key)
            if len(facts) >= 4:
                break
        if len(facts) >= 4:
            break
    polarities = {fact.polarity for fact in facts if fact.polarity != "unknown"}
    required_polarity: Polarity = polarities.pop() if len(polarities) == 1 else "unknown"
    requested_entity = _clean_part(package.requested_entity)
    # Document-type subjects are often omitted by the entity parser, especially
    # in Bangla/Banglish. Preserve this generic subject boundary for relation checks.
    document_subject = (
        "prospectus"
        if package.requested_field in {"publication", "publisher"}
        and re.search(r"prospectus|প্রসপেক্টাস", package.question, re.I)
        else None
    )
    required_entities = tuple(dict.fromkeys(filter(None, (requested_entity, document_subject))))
    required_values = tuple(dict.fromkeys(anchor for fact in facts for anchor in fact.required_anchors))
    full_evidence = "\n".join(item.excerpt for item in package.evidence)
    relations = tuple(dict.fromkeys(fact.relation for fact in facts))
    relation_set = {fact.relation for fact in facts}
    required_cardinalities = (
        tuple(dict.fromkeys(value for fact in facts[:1] for value in _cardinalities(fact.value or fact.evidence)))
        if exact_cardinality_required or CONDITION_QUESTION_RE.search(package.question)
        else ()
    )
    parsed_values = all(fact.value for fact in facts) if relation_set.intersection(STRICT_RELATIONS) else True
    relation_atoms = _atomic_anchors(" ".join(
        part
        for fact in facts
        if fact.relation in STRICT_RELATIONS
        for part in (fact.subject or "", fact.value or "")
    ))
    relation_atoms.update(_atomic_anchors(" ".join(required_entities)))
    forbidden_values = (
        tuple(sorted(_atomic_anchors(full_evidence) - relation_atoms))
        if relation_set.intersection(STRICT_RELATIONS) and parsed_values
        else ()
    )
    high_confidence = (
        bool(facts) and bool(wanted) and len(relation_set) == 1
        and required_polarity != "unknown" and parsed_values
    )
    return SemanticContract(
        target_language=package.original_language,
        required_entities=required_entities,
        required_relations=relations,
        required_values=required_values,
        required_polarity=required_polarity,
        forbidden_unsupported_values=forbidden_values,
        supported_anchors=tuple(sorted(_atomic_anchors(full_evidence))),
        exact_cardinality_required=exact_cardinality_required,
        required_cardinalities=required_cardinalities,
        fact_plan=tuple(facts),
        confidence="high" if high_confidence else "limited",
    )


def _normalized_anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0980-\u09ff]+", "", value.casefold())


def _atomic_anchors(text: str) -> set[str]:
    atoms = re.findall(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]{2,}\b|\b\d+(?:\.\d+)?%?\b", text)
    return {
        _normalized_anchor(value) for value in atoms
        if value.casefold() not in GENERIC_ANCHORS and _normalized_anchor(value)
    }


def validate_semantic_contract(answer: str, contract: SemanticContract) -> dict[str, Any]:
    if not answer.strip():
        return {"passed": False, "reason": "EMPTY_ANSWER", "repairable": False, "details": []}
    details: list[str] = []
    answer_polarity = detect_polarity(answer)
    if contract.required_polarity != "unknown" and answer_polarity != contract.required_polarity:
        details.append(f"expected={contract.required_polarity};answer={answer_polarity}")
        return {"passed": False, "reason": "WRONG_POLARITY", "repairable": True, "details": details}

    compact_answer = _normalized_anchor(answer)
    # Only enforce the inferred document subject when the evidence relation was
    # too weak to parse confidently. Explicit requested entities remain governed
    # by the existing evidence/anchor checks to preserve prior behavior.
    missing_entities = [
        entity for entity in contract.required_entities
        if contract.confidence == "limited"
        and entity == "prospectus"
        and _normalized_anchor(entity) not in compact_answer
    ]
    if missing_entities:
        return {
            "passed": False, "reason": "MISSING_REQUIRED_ENTITY", "repairable": True,
            "details": [f"missing entity: {value}" for value in missing_entities],
        }

    if contract.exact_cardinality_required and not contract.required_cardinalities:
        return {
            "passed": False,
            "reason": "SEMANTIC_CONTRACT_UNCERTAIN",
            "repairable": False,
            "details": ["exact count requested but no reliable evidence-derived count was extracted"],
        }

    if contract.required_cardinalities:
        answer_cardinalities = set(_cardinalities(answer))
        missing_cardinalities = [value for value in contract.required_cardinalities if value not in answer_cardinalities]
        if missing_cardinalities:
            return {
                "passed": False,
                "reason": "MISSING_REQUIRED_ENTITY",
                "repairable": True,
                "details": [f"missing exact count: {value}" for value in missing_cardinalities],
            }

    if (
        contract.target_language != "english"
        and contract.confidence == "limited"
        and any(relation in STRICT_RELATIONS or relation == "general_claim" for relation in contract.required_relations)
    ):
        return {
            "passed": False,
            "reason": "SEMANTIC_CONTRACT_UNCERTAIN",
            "repairable": False,
            "details": list(contract.required_relations),
        }

    answer_anchors = _atomic_anchors(answer)
    evidence_anchors = set(contract.supported_anchors)
    wrong_relation_values = sorted(answer_anchors.intersection(contract.forbidden_unsupported_values))
    if wrong_relation_values:
        return {
            "passed": False,
            "reason": "WRONG_RELATION_VALUE",
            "repairable": False,
            "details": wrong_relation_values,
        }
    unsupported = sorted(value for value in answer_anchors if value and value not in evidence_anchors)
    if contract.fact_plan and unsupported:
        return {"passed": False, "reason": "WRONG_RELATION_VALUE", "repairable": False, "details": unsupported}

    if contract.confidence == "high" and contract.required_values:
        required = _atomic_anchors(" ".join(contract.required_values))
        # A relation may expose several supporting anchors; require at least one rather than every incidental name.
        if required and not answer_anchors.intersection(required):
            return {"passed": False, "reason": "MISSING_REQUIRED_ENTITY", "repairable": True, "details": sorted(required)}
    return {"passed": True, "reason": "SEMANTIC_CONTRACT_SATISFIED", "repairable": False, "details": []}


def repair_instruction(reason: str) -> str:
    if reason == "WRONG_POLARITY":
        return "Your previous answer reversed the evidence meaning. Rewrite it without changing affirmative or negative meaning. Do not add facts."
    if reason == "MISSING_REQUIRED_ENTITY":
        return "Your previous answer omitted a required entity or value from the verified fact plan. Include it exactly and add no facts."
    if reason in {"BANGLISH_BODY_PURE_ENGLISH", "BANGLISH_CONTAINS_BENGALI_SCRIPT", "BANGLA_BODY_TOO_ENGLISH"}:
        return "Rewrite only in the required language and preserve every verified fact and its polarity."
    return "Repair only the stated formatting or language problem. Preserve every fact and its polarity."


__all__ = [
    "FactPlanItem", "SemanticContract", "build_semantic_contract", "detect_polarity",
    "expressed_relations", "repair_instruction", "validate_semantic_contract",
]
