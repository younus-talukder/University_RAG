from src.card_answer import render_card_answer
from src.card_router import CardIntent
from src.card_selector import CardSelection
import src.card_runtime as runtime


def test_fact_page_precedes_identity_page_in_citation():
    card = dict(card_id='synthetic', tier='atomic', course='EX 101', pdf='outline.pdf',
                page=3, pages=[3], field='credit', key='credit', value='3.00')
    intent = CardIntent('en', 'EX 101', 'credit', 'credit', '', 'lookup', 1, ())
    rendered = render_card_answer(intent, CardSelection('card', 'lookup', (card,), 1, 1, 'fixture'), (1,))
    assert rendered.pages == (3, 1)
    assert 'page 3, 1' in rendered.citation


def test_printed_nill_uses_none_template_verbatim():
    card = dict(card_id='synthetic', tier='atomic', course='EX 101', pdf='outline.pdf',
                page=1, pages=[1], field='prerequisite', key='prerequisite', value='Nill')
    for lang, phrase in [('en', 'has no prerequisite'),
                         ('bn', 'কোনো পূর্বশর্ত নেই'),
                         ('bl', 'kono prerequisite nai')]:
        intent = CardIntent(lang, 'EX 101', 'prerequisite', 'prerequisite', '', 'lookup', 1, ())
        result = render_card_answer(intent, CardSelection('card', 'lookup', (card,), 1, 1, 'fixture'))
        assert phrase in result.body and '"Nill"' in result.body


def test_rendered_language_failure_falls_back(monkeypatch):
    source = 'EX 101 Credit Value: 3.00'
    card = dict(card_id='synthetic', tier='atomic', course='EX 101', pdf='outline.pdf',
                page=1, pages=[1], field='credit', key='credit', value='3.00', plain_span=source,
                char_start=source.index('3.00'), char_end=len(source), span_status='found')

    class FakeIndex:
        def scores(self, vector, course):
            return {'synthetic': .9}

    class FakeModel:
        def embed(self, question):
            return object()

    seen = []
    def reject(text, target):
        seen.append((text, target))
        return {'validation_passed': False, 'response_language': 'english',
                'validation_reason': 'SYNTHETIC_LANGUAGE_FAILURE'}

    monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS', '1')
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    monkeypatch.setattr(runtime, '_bundle', lambda: ([card], FakeIndex(),
                                                     {'tau': .03, 'tau_rev': .03, 'disabled_fields': []}))
    monkeypatch.setattr(runtime, 'get_embedding_model', lambda: FakeModel())
    monkeypatch.setattr(runtime, '_plain_page', lambda pdf, page: source)
    monkeypatch.setattr(runtime, 'validate_language', reject)
    result, _ = runtime.attempt_card_answer('What is the credit of EX 101?')
    assert result is None
    assert seen and seen[0][0] != card['value'] and card['value'] in seen[0][0]


def test_card_response_has_normal_supported_fields_and_types(monkeypatch):
    from src.pipeline import answer_question
    monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS', '0')
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '0')
    normal = answer_question('What course title is listed for CSE 314?', use_generation=False)
    assert normal['answerability_status'] == 'SUPPORTED'

    source = 'EX 101 Credit Value: 3.00'
    card = dict(card_id='synthetic', tier='atomic', course='EX 101', pdf='outline.pdf',
                page=3, pages=[3], field='credit', key='credit', value='3.00', plain_span=source,
                char_start=source.index('3.00'), char_end=len(source), span_status='found')

    class FakeIndex:
        def scores(self, vector, course):
            return {'synthetic': .9}

    class FakeModel:
        def embed(self, question):
            return object()

    monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS', '1')
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    monkeypatch.setattr(runtime, '_bundle', lambda: ([card], FakeIndex(),
                                                     {'tau': .03, 'tau_rev': .03, 'disabled_fields': []}))
    monkeypatch.setattr(runtime, 'get_embedding_model', lambda: FakeModel())
    monkeypatch.setattr(runtime, '_plain_page', lambda pdf, page: source)
    result, _ = runtime.attempt_card_answer('What is the credit of EX 101?')
    assert result is not None
    assert result['answerability_status'] == 'SUPPORTED' and result['answer_returned'] is True
    assert result['page'] == 3
    assert set(normal) <= set(result)
    assert all(type(result[key]) is type(value) for key, value in normal.items())
