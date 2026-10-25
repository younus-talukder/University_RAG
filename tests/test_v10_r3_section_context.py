from src.evidence import SupportStatus, assess_evidence
from src.section_context import annotate_section_context, governing_section


def _chain(*texts, document="doc-a", pages=None):
    chunks, previous = [], None
    for index, text in enumerate(texts):
        chunk = {"chunk_id": f"{document}-c{index}", "document_id": document, "page": (pages or [1] * len(texts))[index],
                 "text": text, "previous_chunk_id": previous}
        chunks.append(chunk)
        previous = chunk["chunk_id"]
    return chunks


def test_row_on_later_page_inherits_column_header_and_keeps_both_ids():
    chunks = _chain("16. Class Schedule/Lesson Plan/Weekly plan: Topics Specific Outcome(s) Time Frame Teaching Strategy(s)",
                    "Introduction Week 1 Lecture CLO1", "Joins Week 5 Practice CLO2", "Exam Week 7 Lab exam CLO1, CLO2",
                    pages=[3, 3, 4, 5])
    label, header_id = governing_section(chunks[3], {c["chunk_id"]: c for c in chunks})
    assert label.startswith("16. Class Schedule/Lesson Plan/Weekly plan:") and "Teaching Strategy(s)" in label
    assert header_id == chunks[0]["chunk_id"]
    annotated = annotate_section_context([chunks[3]], chunks)[0]
    assert annotated["chunk_id"] == chunks[3]["chunk_id"]
    assert annotated["section_header_chunk_id"] == chunks[0]["chunk_id"]
    assert annotated["text"].endswith(chunks[3]["text"])


def test_sentence_tail_is_not_treated_as_column_labels():
    chunks = _chain("11. Course Objectives and Course Summary: The course focuses on databases. The objectives are to:",
                    "1. Introduce data models. 2. Teach DDL and DML operations.")
    label, _ = governing_section(chunks[1], {c["chunk_id"]: c for c in chunks})
    assert label == "11. Course Objectives and Course Summary:"


def test_header_is_never_taken_from_another_document_or_too_far_back():
    other = _chain("18. Teaching-Learning Strategies: Strategies Topics", document="doc-b")
    row = {"chunk_id": "doc-a-row", "document_id": "doc-a", "page": 4, "text": "Problem-Based Learning Feasibility Study",
           "previous_chunk_id": other[0]["chunk_id"]}
    assert governing_section(row, {c["chunk_id"]: c for c in [*other, row]}) is None
    far = _chain("18. Teaching-Learning Strategies: Strategies Topics", "Problem-Based Learning", pages=[1, 4])
    assert governing_section(far[1], {c["chunk_id"]: c for c in far}) is None


def test_course_label_heading_never_becomes_a_section_label():
    chunks = _chain("1. Course No. / Course Code: CSE 212", "Students learn joins and views.")
    assert governing_section(chunks[1], {c["chunk_id"]: c for c in chunks}) is None


def test_chunk_with_its_own_heading_gets_no_label():
    chunks = _chain("18. Teaching-Learning Strategies: Strategies Topics", "19. Assessment Techniques: SL Topics")
    assert governing_section(chunks[1], {c["chunk_id"]: c for c in chunks}) is None


def test_section_label_does_not_grant_course_ownership():
    chunks = _chain("11. Course Objectives and Course Summary:", "1. Teach students DDL and DML operations.")
    annotated = annotate_section_context([chunks[1]], chunks)
    result = assess_evidence("Which database operations are named in the CSE 212 objectives?", annotated)
    assert result.status is SupportStatus.INSUFFICIENT


def test_general_questions_get_heading_for_the_prompt_but_not_for_the_gate():
    chunks = _chain("16. Class Schedule/Lesson Plan/Weekly plan: Topics Specific Outcome(s) Teaching Strategy(s)",
                    "ER Diagram Week 6 Lecture, multimedia CLO3")
    plain = annotate_section_context([chunks[1]], chunks, label_text=False)[0]
    assert plain["text"] == chunks[1]["text"] and plain["heading"].startswith("16. Class Schedule")
    question = "Feasibility study-r sathe kon teaching strategy ache?"
    assert assess_evidence(question, [plain]).status is SupportStatus.INSUFFICIENT
