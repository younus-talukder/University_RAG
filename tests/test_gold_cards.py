"""Opt-in acceptance test for generated cards against external hand-verified gold.

Set RAG_GOLD_TESTS=1, GOLD_CARDS_PATH, CARDS_GT_DOCS_PATH,
and GOLD_CHECK_SCRIPT to run.  The default-off suite collects no gold tests.
"""
import json,os,sys
from pathlib import Path

if os.getenv('RAG_GOLD_TESTS','').lower() in {'1','true','yes','on'} and os.getenv('GOLD_CARDS_PATH'):
    import importlib.util
    import pytest
    spec=importlib.util.spec_from_file_location('gold_check_external',os.environ['GOLD_CHECK_SCRIPT'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

    @pytest.fixture(scope='module')
    def generated():
        return [json.loads(x) for x in Path(os.environ['CARDS_GT_DOCS_PATH']).open(encoding='utf-8')]

    @pytest.fixture(scope='module')
    def gold_rows():
        return [json.loads(x) for x in Path(os.environ['GOLD_CARDS_PATH']).open(encoding='utf-8')]

    def test_gold_atomic_acceptance(generated,gold_rows):
        rows=module.check(generated)
        assert len(rows)==len(gold_rows)==193
        nonempty=[(g,r) for g,r in zip(gold_rows,rows) if g.get('value')]
        assert len(nonempty)==192
        atomic=sum(r['status']=='atomic' for _,r in nonempty)
        assert atomic>=173, f'atomic {atomic}/192; see gold_check.csv for field failures'
        assert all(r['extras_ok'] for g,r in zip(gold_rows,rows) if r['status']=='atomic' and any(k in g for k in ('weight_display','po_display','plo_display','clo_cells','co_cells','activities','teaching_strategy','clo_refs','co_refs','empty_section')))

    def test_hygiene_and_size(generated):
        nonempty=[c for c in generated if c.get('value')]
        assert all(any(ch.isalnum() for ch in c['value']) for c in nonempty)
        assert len({(c['pdf'],c['page'],module.normalize(c['value'])) for c in nonempty})==len(nonempty)
        assert sum(len(c['value'].split())<=80 for c in generated)/len(generated)>=.95

    def test_specific_gold_traps(generated):
        course=lambda name:[c for c in generated if c.get('course')==name]
        one=course('CSE 316');two=course('CSE 426')
        assert any(c.get('field')=='schedule' and c.get('row_key')=='Weeks 5–12' and c.get('activities')=='Circuit connection, Hands-on Lab practice, demonstration' for c in one)
        assert any(c.get('field')=='weight' and c.get('assessment')=='Assessment' and c.get('weight_display')=='30%' and '50%' in str(c.get('co_clo_cells')) for c in two)
        assert all(c.get('field') not in {'clo','co'} or len(c.get('value','').split())<=80 for c in generated)
        assert any(c.get('field')=='knowledge_profile' for c in one+two)
        assert any(c.get('field')=='weight_total' and c.get('clo_totals')=={'CLO 1':8,'CLO 2':20,'CLO 3':16,'CLO 4':16,'CLO 5':20,'CLO 6':20} for c in one)
        assert all(c.get('row_key','').startswith('Lecture ') for c in two if c.get('field')=='schedule')
