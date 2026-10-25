from src.document_ownership import annotate_document_ownership, classify_documents, single_course_owner
from src.evidence import SupportStatus, assess_evidence


def _docs():
    return [
        {"document_id": "outline", "chunk_id": "o1", "page": 1, "text": "1. Course No. / Course Code: CSE 212"},
        {"document_id": "outline", "chunk_id": "o2", "page": 5, "text": "Week 7 Midterm lab exam CLO1, CLO2"},
        {"document_id": "outline", "chunk_id": "o3", "page": 2, "text": "Prerequisite: CSE 111 CLO1"},
        {"document_id": "curriculum", "chunk_id": "c1", "page": 3, "text": "Course Code Course Title Credits CSE 212 Database Lab 1.5"},
        {"document_id": "curriculum", "chunk_id": "c2", "page": 3, "text": "Course Code: CSE 212 Course Code: CSE 305"},
        {"document_id": "two-headers", "chunk_id": "t1", "page": 1, "text": "Course Code: CSE 101"},
        {"document_id": "two-headers", "chunk_id": "t2", "page": 4, "text": "Course Code: CSE 102"},
        {"document_id": "brochure", "chunk_id": "b1", "page": 1, "text": "University at a glance"},
    ]


def test_classification_is_conservative():
    result = classify_documents(_docs())
    assert result["outline"].status == "SINGLE_COURSE" and result["outline"].course_code == "CSE212"
    assert result["curriculum"].status == "MULTI_COURSE"
    assert result["two-headers"].status == "MULTI_COURSE"
    assert result["brochure"].status == "UNCERTAIN"
    assert single_course_owner("CSE 212", result) == "outline"
    assert single_course_owner("CSE 101", result) is None


def test_only_chunks_naming_no_other_course_inherit_the_owner():
    ownership = classify_documents(_docs())
    annotated = {item["chunk_id"]: item for item in annotate_document_ownership(_docs(), ownership)}
    assert annotated["o2"].get("document_owner_course") == "CSE212"
    assert "document_owner_course" not in annotated["o3"]      # names another course
    assert "document_owner_course" not in annotated["c1"]      # multi-course document
    assert "document_owner_course" not in annotated["b1"]


def test_flag_controls_the_subject_test(monkeypatch):
    ownership = classify_documents(_docs())
    pool = [item for item in annotate_document_ownership(_docs(), ownership) if item["chunk_id"] == "o2"]
    question = "Which CLOs does the CSE 212 week 7 midterm lab exam assess?"
    monkeypatch.delenv("RAG_REPAIR_SINGLE_COURSE_DOCUMENT_OWNERSHIP", raising=False)
    assert assess_evidence(question, pool).status is SupportStatus.INSUFFICIENT
    monkeypatch.setenv("RAG_REPAIR_SINGLE_COURSE_DOCUMENT_OWNERSHIP", "1")
    assert assess_evidence(question, pool).status is SupportStatus.SUPPORTED


def test_ownership_never_bypasses_the_field_gate(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_SINGLE_COURSE_DOCUMENT_OWNERSHIP", "1")
    ownership = classify_documents(_docs())
    header_only = [item for item in annotate_document_ownership(_docs(), ownership) if item["chunk_id"] == "o1"]
    result = assess_evidence("Which database operations are named in the CSE 212 objectives?", header_only)
    assert result.status is SupportStatus.INSUFFICIENT


def test_inherited_ownership_requires_a_distinctive_match():
    ownership = classify_documents(_docs())
    marked = lambda q: {item["chunk_id"] for item in annotate_document_ownership(_docs(), ownership, q) if item.get("document_owner_course")}
    assert "o2" in marked("Which CLOs does the CSE 212 week 7 midterm lab exam assess?")
    assert "o2" not in marked("Which CLOs does the CSE 212 week 9 midterm lab exam assess?")   # numbered identifier differs
    assert "o2" not in marked("Which PLO does the CSE 212 week 7 midterm lab exam assess?")    # identifier family absent
    assert "o2" not in marked("CSE 212 er CLO2 onujayi students ki korte parbe?")               # too few distinctive terms


def test_family_asked_without_a_number_needs_a_numbered_member():
    from src.same_page_evidence import distinctive_match
    clo_list = "12. Course Learning Outcomes: CLO 1 Develop applications. CLO 2 Develop well-designed applications using the OOP features. Mapped to PLO."
    mapping = "CLO2 Develop well-designed applications using OOP features PLO3 mapped"
    question = "Which PLO is mapped to CSE 202's OOP application outcome CLO2?"
    assert not distinctive_match(question, "cse 202", clo_list)
    assert distinctive_match(question, "cse 202", mapping)
