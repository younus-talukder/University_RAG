from src.evidence import SupportStatus, analyze_query, assess_evidence
from src.query_normalization import extract_course_entities


def _codes(text):
    return [entity.compact for entity in extract_course_entities(text)]


def test_flag_off_keeps_v7_behaviour(monkeypatch):
    monkeypatch.delenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", raising=False)
    assert _codes("Which topics are assessed in Quiz 04?") == ["QUIZ04"]


def test_assessment_tokens_are_not_course_codes(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", "1")
    for text in ("Which topics are assessed in Quiz 04?", "What happens in Lab 10?", "Which CLO does CT 02 cover?",
                 "Quiz-04 e kon topic ache?", "Part 2 of the outline"):
        assert _codes(text) == [], text


def test_real_course_codes_survive_next_to_assessment_tokens(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", "1")
    assert _codes("In CSE 305, which topics does Quiz 04 assess?") == ["CSE305"]
    assert _codes("PHY 101 er credit koto?") == ["PHY101"]
    assert _codes("ENG (CSE) 101 course outline") == ["ENGCSE101"]


def test_language_parity_for_course_codes(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", "1")
    variants = ("Which topic does Quiz 04 of CSE 305 assess?",
                "CSE 305-এর Quiz 04-এ কোন বিষয় মূল্যায়ন করা হয়?",
                "CSE 305-er Quiz 04-e kon topic assess kora hoy?")
    assert {tuple(_codes(text)) for text in variants} == {("CSE305",)}


def test_assessment_item_must_be_named_by_the_passage(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", "1")
    question = "Which topic is assessed by Class Test/Quiz-04 and the final exam?"
    assert [(e.kind, e.normalized) for e in analyze_query(question).entities] == [("assessment_item", "quiz:4")]
    other_course = [{"chunk_id": "x", "text": "Topics Assessment Tools CLO1 Lecture Quiz, Written exam"}]
    owning_row = [{"chunk_id": "y", "text": "Topics / Content Assessment Techniques 4. Feasibility Study Class Test/Quiz-04, Final Exam"}]
    assert assess_evidence(question, other_course).status is SupportStatus.INSUFFICIENT
    assert assess_evidence(question, owning_row).status is SupportStatus.SUPPORTED
    assert assess_evidence(question.replace("Quiz-04", "Quiz 004"), owning_row).status is SupportStatus.SUPPORTED
    assert assess_evidence(question, [{"chunk_id": "z", "text": "Topics / Content Assessment Techniques 1. SDLC Class Test/Quiz-01, Final Exam"}]).status is SupportStatus.INSUFFICIENT


def test_single_digit_items_keep_baseline_behaviour(monkeypatch):
    # "CT2" was never a course-code candidate, so R0 must not turn it into an entity.
    monkeypatch.setenv("RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR", "1")
    question = "Which class-test result is chosen between CT2 and CT3?"
    assert analyze_query(question).entities == ()
