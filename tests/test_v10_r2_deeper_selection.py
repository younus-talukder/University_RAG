import pytest

import src.pipeline as pipeline

HEADER = {"document_id": "d1", "source": "X.pdf", "chunk_id": "h", "page": 1, "text": "1. Course No. / Course Code: CSE 212", "score": 0.05}
NOISE = [{"document_id": "d2", "source": "Y.pdf", "chunk_id": f"n{i}", "page": 2, "text": f"Unrelated text {i}", "score": 0.04} for i in range(16)]
OWNED = {"document_id": "d1", "source": "X.pdf", "chunk_id": "deep", "page": 1, "score": 0.01,
         "text": "Course Code: CSE 212 Credit Hour: 1.5"}
OTHER = {"document_id": "d2", "source": "Y.pdf", "chunk_id": "other", "page": 1, "score": 0.02,
         "text": "Course Code: CSE 305 Credit Hour: 3"}


class _Reached(BaseException):
    pass


def _run(monkeypatch, pool, flag):
    monkeypatch.setattr(pipeline, "load_index", lambda **kw: (None, [HEADER, *NOISE, OWNED, OTHER]))
    monkeypatch.setattr(pipeline, "_get_embedding_model", lambda: None)

    class FakeRetriever:
        def __init__(self, **kw):
            pass

        def retrieve(self, question, **kw):
            return [dict(item) for item in pool]

    monkeypatch.setattr(pipeline, "Retriever", FakeRetriever)
    if flag:
        monkeypatch.setenv("RAG_REPAIR_DEEPER_GATE_CHECKED_SELECTION", "1")
    else:
        monkeypatch.delenv("RAG_REPAIR_DEEPER_GATE_CHECKED_SELECTION", raising=False)
    seen = {}
    monkeypatch.setattr(pipeline, "_repair_trace_hook", lambda trace: seen.update(trace))
    result = pipeline._answer_question_impl("What is the credit hour of CSE 212?", top_k=3, use_generation=True)
    return result, seen


def test_deeper_candidate_must_pass_the_unchanged_gate(monkeypatch):
    result, trace = _run(monkeypatch, [*NOISE, OTHER, OWNED], flag=True)
    # Ranks 1-15 are already covered by the v7 bounded widening; R2 reaches rank 18.
    assert trace.get("deeper_selection_depth") == 18
    assert result["chunk_id"] == "deep"


def test_other_course_candidates_are_never_selected(monkeypatch):
    result, trace = _run(monkeypatch, [*NOISE, OTHER], flag=True)
    assert "deeper_selection_depth" not in trace
    assert result["answerability_status"] != "SUPPORTED"


def test_flag_off_keeps_v7_selection(monkeypatch):
    result, trace = _run(monkeypatch, [*NOISE, OTHER, OWNED], flag=False)
    assert "deeper_selection_depth" not in trace
