"""Opt-in tests for schema-only atomic card normalization."""
import os

if os.getenv('RAG_REPAIR_CARD_ANSWERS','').lower() in {'1','true','yes','on'}:
    import pytest
    from src.card_schema import FIELD_SCHEMA,normalize_card,validate_card

    def base(field,value='A value'):
        return dict(card_id='id',course='EX 101',pdf='EX 101.pdf',page=1,pages=[1],field=field,
                    value=value,char_start=10,char_end=17,plain_span='A value',tier='atomic')

    def test_all_fields_have_a_schema():
        assert len(FIELD_SCHEMA)>=40
        assert {'schedule','weight','co_po_link','clo_plo_link','books'}<=set(FIELD_SCHEMA)

    def test_schedule_types_and_aliases_preserve_value():
        source=base('schedule','Original topic')
        source.update(row_key='Week 1',topic='Original topic',outcomes='Explain circuits',co_clo_refs='CLO1, CLO2')
        normalized=normalize_card(source)
        assert normalized['value']==source['value']
        assert normalized['outcomes']==['Explain circuits']
        assert normalized['clo_refs']=='CLO1, CLO2'
        assert normalized['key']=='Week 1'
        assert validate_card(normalized)==[]

    def test_weight_keeps_ambiguous_cell_keys_visible():
        source=base('weight','30%')
        source.update(assessment='Assessment',weight='30%',weight_display='30%',co_clo_cells={'outcome 1':'50%'})
        normalized=normalize_card(source)
        assert normalized['co_clo_cells']=={'outcome 1':'50%'}
        assert 'non_identifier_cell_key' in validate_card(normalized)
        assert 'co_cells' not in normalized

    def test_link_and_empty_section():
        link=base('clo_plo_link','a');link.update(row_key='CLO 1',po_cell_verbatim='a',plo_display='PLO (a)',header_text='PLO header')
        normalized=normalize_card(link)
        assert normalized['co_id']=='CLO 1'
        assert normalized['value']=='a'
        empty=base('books','');empty.update(empty_section=True,char_start=None,char_end=None)
        assert validate_card(normalize_card(empty))==[]

    def test_flagged_module_rejects_block_cards():
        card=base('course_title');card['tier']='block'
        with pytest.raises(ValueError):normalize_card(card)
