"""Stage A v10 regressions on the real Stage A index metadata.

CONTROL_RULE (gate level): the prior false-supported variants may pass the
evidence gate only with evidence that contains the PDF-verified answer span on
the verified page (then a human reviews the final answer); any other pass fails.
"""

import pickle

import pytest

import src.pipeline as pipeline

from src.course_title_evidence import assess_course_title
from src.document_ownership import annotate_document_ownership, classify_documents
from src.evidence import SupportStatus, assess_evidence
from src.query_normalization import extract_course_entities
from src.same_page_evidence import (assess_unique_same_page_course_passage,
                                    assess_unique_same_page_labeled_field_passage)
from src.section_context import annotate_section_context
from src.config import VECTOR_DB_DIR

ALL_FLAGS = ("FIX_COURSE_CODE_EXTRACTOR", "SAME_PAGE_LABELED_FIELD_JOIN", "TABLE_HEADER_CONTEXT",
             "COURSE_ANCHORED_RETRIEVAL", "DEEPER_GATE_CHECKED_SELECTION", "SINGLE_COURSE_DOCUMENT_OWNERSHIP")
CONTROLS = {
    # question: (verified PDF, verified page, verified span)
    "Feasibility study-r sathe kon teaching strategy ache?": ("CSE 305.pdf", 4, "problem-based learning"),
    "CSE 212 real-life kon kaj-e focus kore?": ("CSE 212.pdf", 1, "database design and model"),
    "CSE 212-এর উদ্দেশ্যে কোন ডেটাবেস অপারেশনগুলোর নাম আছে?": ("CSE 212.pdf", 1, "ddl and dml"),
}


@pytest.fixture
def flags_on(monkeypatch):
    for flag in ALL_FLAGS:
        monkeypatch.setenv(f"RAG_REPAIR_{flag}", "1")


def _index():
    return pickle.loads((VECTOR_DB_DIR / "metadata.pkl").read_bytes())


def _unverified(assessment, source, page, span):
    """Chunk ids of passing evidence that lacks the verified span on the verified page."""
    if assessment is None or assessment.status is not SupportStatus.SUPPORTED:
        return set()
    return {e.chunk_id for e in assessment.evidence
            if not (e.source == source and e.page == page and span in " ".join(e.text.split()).lower())}


def _decisions(question, pool, metadata):
    prepared = pipeline._prepare_pool(question, [dict(item) for item in pool], metadata, {}, "original")
    joined = [join(question, prepared, metadata) for join in
              (assess_unique_same_page_course_passage, assess_unique_same_page_labeled_field_passage)]
    return [assess_evidence(question, prepared), *[j[0] if j else None for j in joined]]


@pytest.mark.parametrize("question", list(CONTROLS))
def test_flags_never_add_unverified_gate_passes_for_controls(monkeypatch, question):
    metadata = _index()
    source, page, span = CONTROLS[question]
    pools = [[item for item in metadata if item["source"] == name and item["page"] == number]
             for name, number in sorted({(item["source"], item["page"]) for item in metadata})]
    for pool in pools:
        for flag in ALL_FLAGS:
            monkeypatch.delenv(f"RAG_REPAIR_{flag}", raising=False)
        before = set().union(*(_unverified(d, source, page, span) for d in _decisions(question, pool, metadata)))
        for flag in ALL_FLAGS:
            monkeypatch.setenv(f"RAG_REPAIR_{flag}", "1")
        after = set().union(*(_unverified(d, source, page, span) for d in _decisions(question, pool, metadata)))
        assert after <= before, (pool[0]["source"], pool[0]["page"], sorted(after - before))


def test_table_row_carries_its_header_from_the_same_page(flags_on):
    metadata = _index()
    row = next(item for item in metadata if "Problem-Based Learning Project Scheduling" in item["text"])
    labeled = annotate_section_context([row], metadata)[0]
    assert labeled["heading"].startswith("18. Teaching-Learning Strategies:")
    assert labeled["section_header_chunk_id"] != row["chunk_id"]


def test_table_row_carries_its_header_from_an_earlier_page(flags_on):
    metadata = _index()
    row = next(item for item in metadata if "Conduct hands-on lab exam covering Labs 1-6" in item["text"])
    labeled = annotate_section_context([row], metadata)[0]
    header = next(item for item in metadata if item["chunk_id"] == labeled["section_header_chunk_id"])
    assert labeled["heading"].startswith("16. Class Schedule/Lesson Plan/Weekly plan:")
    assert header["document_id"] == row["document_id"] and header["page"] < row["page"]


def test_course_title_binding_is_unchanged_with_all_flags(flags_on):
    metadata = _index()
    question = "What is the course title of CSE 314?"
    pool = pipeline._prepare_pool(question, [item for item in metadata if item["source"] == "CSE 314.pdf" and item["page"] == 1], metadata, {}, "original")
    assessment = assess_course_title(question, pool, metadata)
    assert assessment.status is SupportStatus.SUPPORTED
    assert any("Software Engineering Lab" in e.text for e in assessment.evidence)


def test_language_parity_of_course_entity_extraction(flags_on):
    variants = ("What computer-related area is PHY 101 described as closely relevant to?",
                "PHY 101 কম্পিউটার-সংশ্লিষ্ট কোন ক্ষেত্রের সঙ্গে ঘনিষ্ঠভাবে প্রাসঙ্গিক?",
                "PHY 101 computer-er kon area-r sathe closely relevant?")
    assert {tuple(e.compact for e in extract_course_entities(v)) for v in variants} == {("PHY101",)}
