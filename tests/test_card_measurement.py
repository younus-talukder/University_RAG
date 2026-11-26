from src.card_measurement import canonical_value, score_candidate


def test_none_like_forms_and_policy_scoring():
    for value in ('None', 'Nil', 'Nill', 'N/A', 'N/A.', 'No prerequisite',
                  'কোনো পূর্বশর্ত নেই', 'kono prerequisite nei'):
        assert canonical_value(value) == '<none>'
    assert score_candidate('None', 'Nill') == (True, True, 'CORRECT')
    assert score_candidate('alpha', 'beta') == (False, False, 'WRONG')
    assert score_candidate('alpha', 'beta', 'source_conflict') == (True, False, 'POLICY-CORRECT')
