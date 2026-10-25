from src.evidence import SupportStatus, assess_evidence
from src.pipeline import _widen_bounded_evidence
from src.course_rows import course_field_value
from src.answer_policy import evidence_value


def _chunk(code: str, value: str, index: int) -> dict:
    return {
        "source": f"document-{index}.pdf",
        "document_id": f"document-{index}",
        "page": 1,
        "chunk_id": f"chunk-{index}",
        "text": f"Course Code: {code} Credit Hour: {value}",
        "entity_type": "course",
        "entity_id": code,
        "field_types": ["course_code", "credits"],
        "score": 0.1,
    }


def test_verified_course_field_below_top_three_can_be_used():
    question = "How many credits does CSE 305 have?"
    pool = [_chunk("CSE 202", "3", i) for i in range(3)] + [_chunk("CSE 305", "1.5", 3)]
    initial = assess_evidence(question, pool[:3])
    result, selected, widened = _widen_bounded_evidence(question, initial, pool[:3], pool)
    assert initial.status is SupportStatus.INSUFFICIENT
    assert widened is True
    assert result.status is SupportStatus.SUPPORTED
    assert result.evidence[0].chunk_id == "chunk-3"
    assert selected == pool


def test_other_course_or_blank_field_cannot_be_promoted():
    question = "How many credits does CSE 305 have?"
    pool = [_chunk("CSE 202", "3", i) for i in range(4)]
    initial = assess_evidence(question, pool[:3])
    result, selected, widened = _widen_bounded_evidence(question, initial, pool[:3], pool)
    assert widened is False
    assert result is initial
    assert selected == pool[:3]


def test_conflicting_values_cannot_be_promoted():
    question = "How many credits does CSE 305 have?"
    pool = [_chunk("CSE 202", "3", i) for i in range(3)]
    pool += [_chunk("CSE 305", "1.5", 3), _chunk("CSE 305", "3", 4)]
    initial = assess_evidence(question, pool[:3])
    result, selected, widened = _widen_bounded_evidence(question, initial, pool[:3], pool)
    assert widened is False
    assert result is initial
    assert selected == pool[:3]


def test_ambiguous_question_cannot_be_promoted():
    question = "What are the course credits?"
    pool = [_chunk("CSE 305", "1.5", i) for i in range(4)]
    initial = assess_evidence(question, pool[:3])
    result, selected, widened = _widen_bounded_evidence(question, initial, pool[:3], pool)
    assert initial.status is SupportStatus.AMBIGUOUS
    assert widened is False
    assert result is initial
    assert selected == pool[:3]


def test_labeled_credit_hour_is_owned_by_requested_course():
    question = "CSE 314 কোর্সের ক্রেডিট কী দেওয়া আছে?"
    excerpt = "Course Code: CSE 314 Semester: Spring 2025 Credit Hour: 1.5"
    assert course_field_value(question, "credits", [excerpt]) == "1.5"
    assert evidence_value("credits", [excerpt]) == "1.5"
    assert course_field_value("CSE 305 credit কত?", "credits", [excerpt]) is None
    assert evidence_value("credits", ["Credit Hour: "]) is None
