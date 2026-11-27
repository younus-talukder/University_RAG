from src import banglish_norm as norm
from src import card_runtime


def test_detector_is_query_only():
    assert norm.detected('EX 101-er CLO2 onujayi ki shikhbo?')
    assert not norm.detected('What is the credit of EX 101?')
    assert not norm.detected('EX 101')
    assert not norm.detected('EX 101-এর ক্রেডিট কত?')


def test_rewrite_preserves_identifiers_and_numbers():
    original = 'cse101-er week 9-e ki ache?'
    assert norm.valid_rewrite(original, 'What is in CSE 101 in week 9?')
    assert not norm.valid_rewrite(original, 'What is in CSE 102 in week 9?')
    assert not norm.valid_rewrite(original, 'What is in CSE 101?')


def test_fusion_preserves_original_and_is_deterministic():
    original = [{'chunk_id': 'a', 'source': 'a.pdf'}, {'chunk_id': 'b', 'source': 'b.pdf'}]
    rewritten = [{'chunk_id': 'c', 'source': 'c.pdf'}, {'chunk_id': 'a', 'source': 'a.pdf'}]
    one = norm.fuse(original, rewritten, 60, 3)
    assert one == norm.fuse(original, rewritten, 60, 3)
    assert {'a', 'b'} <= {x['chunk_id'] for x in one}


def test_original_anchor_protection_and_rewrite_rescue(monkeypatch):
    card = {'field': 'clo', 'value': 'Apply synthetic methods'}
    original = 'EX 101-er CLO2 onujayi students-ra ki korte parbe?'
    rewritten = 'According to EX 101 CLO2, what can students do?'
    assert card_runtime._distinctive_anchor_missing(original, card)
    assert not card_runtime._distinctive_anchor_missing(rewritten, card)
    assert norm.valid_rewrite(original, rewritten)
    assert norm.original_anchors_safe(original, card)
    card.update(field='book', key='book', row_key='book')
    named = 'EX 101-er Morgan book er naam ki?'
    book = {'field': 'book', 'value': 'Synthetic methods'}
    assert not norm.original_anchors_safe(named, book)
    assert card_runtime._distinctive_anchor_missing(named, book)


def test_off_disables_normalization(monkeypatch):
    monkeypatch.setenv('RAG_BANGLISH_NORM', 'off')
    assert not norm.enabled()
    monkeypatch.setenv('RAG_BANGLISH_NORM', 'fuse')
    assert norm.enabled()


def test_card_route_uses_valid_rewrite_without_losing_original_anchor(monkeypatch):
    from src.card_router import CardIntent
    from src.card_selector import CardSelection
    source = 'EX 101 CLO2 Apply synthetic methods'
    card = {'card_id': 'synthetic', 'tier': 'atomic', 'course': 'EX 101',
            'pdf': 'outline.pdf', 'page': 1, 'pages': [1], 'field': 'clo',
            'key': 'CLO 2', 'row_key': 'CLO 2', 'value': 'Apply synthetic methods',
            'plain_span': source, 'span_status': 'found', 'char_start': 12,
            'char_end': len(source)}
    original = 'EX 101-er CLO2 onujayi students-ra ki korte parbe?'
    rewritten = 'According to EX 101 CLO2, what can students do?'
    assert norm.valid_rewrite(original, rewritten)

    class Index:
        def scores(self, vector, course):
            return {'synthetic': .9}

    class Model:
        def embed(self, question):
            return object()

    class Values:
        def scores(self, *args):
            return {}

    monkeypatch.setenv('RAG_BANGLISH_NORM', 'fuse')
    monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS', '1')
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    monkeypatch.setattr(card_runtime, '_bundle', lambda: ([card], Index(), {'tau': .03, 'tau_rev': .03, 'disabled_fields': []}))
    monkeypatch.setattr(card_runtime, '_value_index', lambda: Values())
    monkeypatch.setattr(card_runtime, 'get_embedding_model', lambda: Model())
    monkeypatch.setattr(card_runtime, 'understand', lambda query, cards: CardIntent('en' if query.startswith(('According', 'Which')) else 'bl', 'EX 101', card['field'], card['key'], '', 'lookup', 1, ()))
    monkeypatch.setattr(card_runtime, 'select_cards', lambda *args, **kwargs: CardSelection('card', 'lookup', (card,), .9, .9, 'exact_field_key'))
    monkeypatch.setattr(card_runtime, 'select_item_by_value', lambda *args, **kwargs: CardSelection('fallback', 'lookup', (), 0, 0, 'none'))
    monkeypatch.setattr(card_runtime, '_field_policy', lambda: {(field, 'lookup'): {'enabled': 'true', 'exact_key_only': 'false'} for field in ('clo', 'book')})
    monkeypatch.setattr(card_runtime, '_span_verified', lambda card: True)
    monkeypatch.setattr(card_runtime, 'validate_grounding', lambda *args: {'grounding_validation_passed': True, 'grounding_validation_reason': 'synthetic'})
    seen = []
    def language(body, target):
        seen.append(target)
        return {'validation_passed': True, 'response_language': target, 'validation_reason': 'synthetic'}
    monkeypatch.setattr(card_runtime, 'validate_language', language)
    monkeypatch.setattr(card_runtime, 'answer_safety_failures', lambda *args, **kwargs: ())
    result, _ = card_runtime.attempt_card_answer(original, rewritten)
    assert result is not None and result['anchor_source'] == 'rewrite'
    assert seen == ['banglish']
    unnumbered, _ = card_runtime.attempt_card_answer('Which CLO do Class Tests 2 and 3 cover?')
    assert unnumbered is None
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'off')
    previous_unnumbered, _ = card_runtime.attempt_card_answer('Which CLO do Class Tests 2 and 3 cover?')
    assert previous_unnumbered is not None
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'on')
    card.update(key='CLO 8', row_key='CLO 8')
    blocked, _ = card_runtime.attempt_card_answer(original, rewritten)
    assert blocked is None  # selected key differs from requested CLO2
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'off')
    previous, _ = card_runtime.attempt_card_answer(original, rewritten)
    assert previous is not None and previous['card_id'] == 'synthetic'
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'on')
    card.update(key='CLO 2', row_key='CLO 2')
    card.update(field='book', key='book', row_key='book')
    named = 'EX 101-er Morgan book er naam ki?'
    result, _ = card_runtime.attempt_card_answer(named, 'Which book is used for EX 101?')
    assert result is None
    card.update(field='clo', key='CLO 2', row_key='CLO 2')
    monkeypatch.setenv('RAG_BANGLISH_NORM', 'off')
    result, _ = card_runtime.attempt_card_answer(original)
    assert result is None


def test_english_and_bangla_never_call_rewrite(monkeypatch):
    from src import pipeline
    monkeypatch.setenv('RAG_BANGLISH_NORM', 'fuse')
    monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS', '0')
    monkeypatch.setattr(pipeline, 'banglish_rewrite', lambda question: (_ for _ in ()).throw(AssertionError('rewrite called')))
    for question in ('What is the credit of EX 101?', 'EX 101 কোর্সের ক্রেডিট কত?'):
        result = pipeline.answer_question(question, use_generation=False)
        assert result['banglish_detected'] is False
        assert result['rewrite_text'] == ''
