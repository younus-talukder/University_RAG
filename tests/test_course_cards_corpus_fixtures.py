"""One official-PDF fixture per detected atomic structure and outline family.

The manifest is a frozen Step 5A r5b artifact built from an automated corpus
scan. No held-out questions or answers are used.
"""
import csv,os
from functools import lru_cache
from pathlib import Path

if os.getenv('RAG_REPAIR_COURSE_CARDS','').lower() in {'1','true','yes','on'} and os.getenv('RAG_CORPUS_FIXTURES')=='1':
    import pytest
    from pypdf import PdfReader
    from src.course_cards_tiered import parse_tiered_pages

    manifest=Path(os.environ['RAG_FIXTURE_MANIFEST'])
    corpus=Path(os.environ['RAG_CORPUS_PDFS'])
    cases=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))

    @lru_cache(maxsize=None)
    def produced(pdf):
        reader=PdfReader(corpus/pdf)
        pages=[p.extract_text() or '' for p in reader.pages]
        layouts=[p.extract_text(extraction_mode='layout') or '' for p in reader.pages]
        return parse_tiered_pages(pdf,pages,layouts)

    @pytest.mark.parametrize('case',cases,ids=lambda x:f"{x['rule']}-{x['family']}")
    def test_corpus_rule(case):
        template,cards,_=produced(case['pdf'])
        assert template in {'A','B'}
        atomic=[x for x in cards if x['tier']=='atomic']
        expected=set(filter(None,case['expected_fields'].split('|')))
        if case['rule']=='exact_plain_span':
            assert any(x.get('span_status')=='found' and x.get('char_start') is not None for x in atomic)
        elif case['rule']=='weight_vs_column_pct':
            assert any(x['field']=='weight' and any('%' in str(v) for v in (x.get('co_clo_cells') or {}).values()) for x in atomic)
        else:
            assert any(x['field'] in expected for x in atomic),f"{case['rule']} structure detected in {case['pdf']} but no {sorted(expected)} card"
