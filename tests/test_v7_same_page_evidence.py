import pickle

from src.config import VECTOR_DB_DIR
from src.evidence import SupportStatus
from src.evidence_spans import expand_assessment_evidence
from src.same_page_evidence import assess_unique_same_page_course_passage


def _index():
    return pickle.loads((VECTOR_DB_DIR / "metadata.pkl").read_bytes())


def test_page_header_binds_answer_passage_and_retains_both_chunk_ids():
    metadata = _index()
    # Use the document identity of the answer passage to avoid relying on a
    # filename or a hard-coded course identifier in the repair itself.
    passage = next(item for item in metadata if "closely relevant to computer hardware" in item["text"])
    own_header = next(item for item in metadata if item["document_id"] == passage["document_id"]
                      and item["page"] == passage["page"] and "Course Code:" in item["text"])
    result = assess_unique_same_page_course_passage(
        "What computer-related area is PHY 101 described as closely relevant to?",
        [own_header, passage], metadata)
    assert result is not None
    assessment, selected = result
    assert assessment.status is SupportStatus.SUPPORTED
    assert selected["chunk_id"] == passage["chunk_id"]
    expanded = expand_assessment_evidence(assessment, metadata)
    evidence = expanded.evidence[0]
    assert "computer hardware" in evidence.excerpt
    assert own_header["chunk_id"] in evidence.continuation_chunk_ids


def test_multiple_course_headers_on_page_block_binding():
    metadata = _index()
    passage = next(item for item in metadata if "closely relevant to computer hardware" in item["text"])
    own_header = next(item for item in metadata if item["document_id"] == passage["document_id"]
                      and item["page"] == passage["page"] and "Course Code:" in item["text"])
    competitor = {**own_header, "chunk_id": "competing-header", "text": "Course Code: MAT 742"}
    result = assess_unique_same_page_course_passage(
        "What computer-related area is PHY 101 described as closely relevant to?",
        [own_header, passage], [*metadata, competitor])
    assert result is None


def test_header_on_another_page_cannot_bind_passage():
    metadata = _index()
    passage = next(item for item in metadata if "closely relevant to computer hardware" in item["text"])
    header = next(item for item in metadata if item["document_id"] == passage["document_id"]
                  and item["page"] == passage["page"] and "Course Code:" in item["text"])
    moved_header = {**header, "page": 2}
    without_original = [item for item in metadata if item["chunk_id"] != header["chunk_id"]]
    result = assess_unique_same_page_course_passage(
        "What computer-related area is PHY 101 described as closely relevant to?",
        [passage], [*without_original, moved_header])
    assert result is None


def test_bounded_pool_can_recover_one_later_course_summary_passage():
    metadata = _index()
    passage = next(item for item in metadata if "database in a real-life scenario" in item["text"])
    header = next(item for item in metadata if item["document_id"] == passage["document_id"]
                  and item["page"] == passage["page"] and "Course Code:" in item["text"])
    result = assess_unique_same_page_course_passage(
        "What real-life task does CSE 212 emphasize?",
        [header, passage], metadata)
    assert result is not None
    assessment, selected = result
    assert assessment.status is SupportStatus.SUPPORTED
    assert selected["chunk_id"] == passage["chunk_id"]
    assert "database design and model" in expand_assessment_evidence(assessment, metadata).evidence[0].excerpt


def test_prior_false_supported_questions_do_not_gain_page_owner_support():
    metadata = _index()
    page_one = [item for item in metadata if item["page"] == 1]
    questions = (
        "Feasibility study-r sathe kon teaching strategy ache?",
        "CSE 212 real-life kon kaj-e focus kore?",
        "CSE 212-এর উদ্দেশ্যে কোন ডেটাবেস অপারেশনগুলোর নাম আছে?",
    )
    for question in questions:
        assert assess_unique_same_page_course_passage(question, page_one, metadata) is None
