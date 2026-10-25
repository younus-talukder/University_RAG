import src.pipeline as pipeline


def _metadata():
    return [
        {"document_id": "owner", "chunk_id": "h", "page": 1, "text": "Course Code: CSE 212"},
        {"document_id": "owner", "chunk_id": "a", "page": 5, "text": "Week 7 lab exam"},
        {"document_id": "other", "chunk_id": "x", "page": 1, "text": "Course Code: CSE 305"},
        {"document_id": "other", "chunk_id": "y", "page": 5, "text": "Week 7 review"},
    ]


def _pool():
    return [{"document_id": "other", "chunk_id": "y", "score": 0.030},
            {"document_id": "other", "chunk_id": "x", "score": 0.025},
            {"document_id": "owner", "chunk_id": "a", "score": 0.020}]


def test_owner_document_is_boosted_not_filtered(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_COURSE_ANCHORED_RETRIEVAL", "1")
    trace = {}
    pool = pipeline._course_anchored_order("What happens in week 7 of CSE 212?", _pool(), _metadata(), trace, "original")
    assert [item["chunk_id"] for item in pool] == ["a", "y", "x"]
    assert trace["original_course_anchor_document"] == "owner"


def test_no_boost_without_a_unique_owner_or_with_flag_off(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_COURSE_ANCHORED_RETRIEVAL", "1")
    for question in ("What happens in week 7?", "What happens in week 7 of CSE 999?",
                     "Compare week 7 of CSE 212 and CSE 305."):
        trace = {}
        pool = pipeline._course_anchored_order(question, _pool(), _metadata(), trace, "original")
        assert [item["chunk_id"] for item in pool] == ["y", "x", "a"], question
    monkeypatch.delenv("RAG_REPAIR_COURSE_ANCHORED_RETRIEVAL")
    pool = pipeline._course_anchored_order("What happens in week 7 of CSE 212?", _pool(), _metadata(), {}, "original")
    assert [item["chunk_id"] for item in pool] == ["y", "x", "a"]


def test_pool_preparation_never_reorders(monkeypatch):
    monkeypatch.setenv("RAG_REPAIR_COURSE_ANCHORED_RETRIEVAL", "1")
    pool = pipeline._prepare_pool("What happens in week 7 of CSE 212?", _pool(), _metadata(), {}, "original")
    assert [item["chunk_id"] for item in pool] == ["y", "x", "a"]
