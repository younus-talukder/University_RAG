import pickle

from src.config import VECTOR_DB_DIR
from src.evidence import SupportStatus
from src.same_page_evidence import assess_unique_same_page_labeled_field_passage
from src.section_context import annotate_section_context

QUESTION = "Which database operations are named in the CSE 212 objectives?"


def _index():
    return pickle.loads((VECTOR_DB_DIR / "metadata.pkl").read_bytes())


def _objective_page(metadata):
    passage = next(item for item in metadata if "different constraints, DDL and DML operations" in item["text"])
    page = [item for item in metadata if item["document_id"] == passage["document_id"] and item["page"] == passage["page"]]
    return passage, page


def _synthetic_page():
    header = {"document_id": "d", "page": 3, "chunk_id": "d-h", "text": "1. Course No. / Course Code: CSE 212"}
    row = {"document_id": "d", "page": 3, "chunk_id": "d-r", "text": "CLO2 Implementation of DML commands assessed in the midterm"}
    other = {"document_id": "d", "page": 3, "chunk_id": "d-o", "text": "Lecture, multimedia, discussions"}
    return [header, row, other]


def test_labeled_field_passage_binds_to_unique_same_page_header_and_keeps_both_ids():
    page = _synthetic_page()
    result = assess_unique_same_page_labeled_field_passage(
        "Which CLO covers the implementation of DML commands in CSE 212?", page, page)
    assert result is not None
    assessment, selected = result
    assert assessment.status is SupportStatus.SUPPORTED
    assert selected["chunk_id"] == "d-r" and selected["same_page_owner_chunk_id"] == "d-h"
    assert assessment.evidence[0].continuation_chunk_ids == ("d-h",)


def test_with_section_labels_the_answer_bearing_objective_list_wins():
    # Only candidates that pass the unchanged gate compete; the labeled objective
    # list matches more distinctive terms than the summary paragraph before it.
    metadata = _index()
    passage, page = _objective_page(metadata)
    result = assess_unique_same_page_labeled_field_passage(QUESTION, annotate_section_context(page, metadata), metadata)
    assert result is not None
    assessment, selected = result
    assert selected["chunk_id"] == passage["chunk_id"]
    assert "DDL and DML" in assessment.evidence[0].text


def test_identifier_family_named_by_the_question_must_be_in_the_passage():
    page = _synthetic_page()
    # The row names CLO2 but no PLO, so a PLO question must not bind it.
    assert assess_unique_same_page_labeled_field_passage(
        "Which PLO is mapped to the CLO covering DML commands implementation in CSE 212?", page, page) is None


def test_general_field_questions_are_left_to_the_v7_join():
    metadata = _index()
    _, page = _objective_page(metadata)
    assert assess_unique_same_page_labeled_field_passage(
        "What real-life task does CSE 212 emphasize?", annotate_section_context(page, metadata), metadata) is None


def test_competing_course_header_on_the_page_blocks_binding():
    metadata = _index()
    passage, page = _objective_page(metadata)
    header = next(item for item in page if "Course Code:" in item["text"])
    competitor = {**header, "chunk_id": "competing-header", "text": "Course Code: MAT 742"}
    result = assess_unique_same_page_labeled_field_passage(QUESTION, page, [*metadata, competitor])
    assert result is None


def test_wrong_course_code_never_binds():
    metadata = _index()
    _, page = _objective_page(metadata)
    assert assess_unique_same_page_labeled_field_passage(
        "Which database operations are named in the CSE 305 objectives?", annotate_section_context(page, metadata), metadata) is None
