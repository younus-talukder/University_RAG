from pathlib import Path
import pytest

from src import page_reranker as module


class Scores:
    def __init__(self, values):
        self.values = values
        self.query = None
    def score_pairs(self, query, passages):
        self.query = query
        return self.values[:len(passages)]


def item(doc, page, text):
    return {'document_id': doc, 'source': doc + '.pdf', 'page': page, 'text': text}


def test_page_max_score_and_tie_preserves_fused_rank(monkeypatch):
    monkeypatch.setenv('RAG_RERANKER', 'bge_v2_m3')
    pool = [item('a', 1, 'low'), item('b', 2, 'medium'), item('a', 1, 'high'), item('c', 3, 'tie')]
    scorer = Scores([0.0, 0.8, 0.9, 0.8])
    result, trace = module.rerank_pages('query', pool, scorer)
    assert [row['text'] for row in result] == ['low', 'high', 'medium', 'tie']
    assert [row['page'] for row in trace['post_rerank_pages']] == [1, 2, 3]
    assert trace['rerank_candidate_depth'] == 4
    assert scorer.query == 'query'


def test_off_exact_order_and_no_model_load(monkeypatch):
    monkeypatch.setenv('RAG_RERANKER', 'off')
    pool = [item('a', 1, 'a'), item('b', 2, 'b')]
    result, trace = module.rerank_pages('query', pool)
    assert result == pool
    assert trace['rerank_latency_s'] == 0


def test_missing_model_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match='Pinned reranker missing'):
        module.PageReranker(tmp_path)


def test_scoring_query_rewrite_or_original(monkeypatch):
    monkeypatch.setenv('RAG_RERANKER', 'bge_v2_m3')
    pool = [item('a', 1, 'passage')]
    scorer = Scores([1.0])
    module.rerank_pages('valid English rewrite' or 'original', pool, scorer)
    assert scorer.query == 'valid English rewrite'
    module.rerank_pages('' or 'original', pool, scorer)
    assert scorer.query == 'original'


def test_default_flag_is_on(monkeypatch):
    monkeypatch.delenv('RAG_RERANKER', raising=False)
    assert module.enabled()
