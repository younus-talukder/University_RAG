from src.card_statement_guard import guard_cards, normalized_key, selected_key_matches


def card(key, value):
    return {'card_id': key + value[:5], 'tier': 'atomic', 'field': 'clo',
            'course': 'EX 101', 'key': key, 'row_key': key,
            'pdf': 'outline.pdf', 'page': 2, 'value': value}


def test_identifier_normalization_and_key_equality():
    assert normalized_key('clo-2') == normalized_key('CLO2') == normalized_key('CLO 2')
    assert selected_key_matches(card('CLO 2', 'Apply synthetic methods'), ['CLO2'])
    assert not selected_key_matches(card('CLO 8', 'Apply synthetic methods'), ['CLO2'])


def test_trim_full_list_reject_leading_foreign_and_keep_clean(monkeypatch):
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'on')
    full = card('CLO 1', 'At the end students can: CLO 1 Learn synthetic methods. CLO 2 Apply synthetic methods.')
    leading = card('CLO 1', 'CLO2 Apply synthetic methods.')
    clean = card('CLO 3', 'Evaluate synthetic methods.')
    cards = guard_cards([full, leading, clean], lambda pdf, page: full['value'] + '\n' + leading['value'] + '\n' + clean['value'])
    assert len(cards) == 2
    assert cards[0]['value'] == 'Learn synthetic methods'
    assert cards[1] is clean
    assert full['value'].startswith('At the end')  # immutable source card


def test_reject_short_nonverbatim_trim(monkeypatch):
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'on')
    short = card('CLO 1', 'CLO 1 Learn it. CLO 2 Apply methods.')
    absent = card('CLO 1', 'CLO 1 Learn synthetic methods. CLO 2 Apply methods.')
    assert guard_cards([short, absent], lambda pdf, page: 'unrelated page') == []


def test_off_returns_previous_cards_by_identity(monkeypatch):
    monkeypatch.setenv('RAG_CARD_STATEMENT_GUARD', 'off')
    items = [card('CLO 1', 'CLO 1 Learn methods. CLO 2 Apply methods.')]
    assert guard_cards(items, lambda pdf, page: '') is items
