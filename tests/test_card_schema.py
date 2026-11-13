"""Opt-in tests for schema-only atomic card normalization."""
import os

if os.getenv('RAG_REPAIR_CARD_ANSWERS','').lower() in {'1','true','yes','on'}:
    import pytest
    from src.card_schema import FIELD_SCHEMA,normalize_card,populate_extras,validate_card

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

    def test_schedule_refs_use_full_plain_column_and_preserve_spacing():
        page='Lecture 2\nScaling and rotation.\nCO1,\nCO 2\nLecture, group work'
        card=base('schedule','Scaling and rotation.')
        card.update(row_key='Lecture 2',char_end=page.index('rotation.')+len('rotation.'),co_clo_refs='CO1',co_refs='CO1')
        revised=populate_extras(normalize_card(card),page)
        assert revised['co_refs']=='CO1, CO 2'
        assert revised['value']==card['value'] and revised['row_key']==card['row_key']

    def test_weight_cells_follow_header_and_row_not_prior_mention():
        page=('Presentation Followed by Viva appears in the schedule.\n'
              'Assessment Type % Weight CLO 1 CLO 2 CLO 3\n'
              'Lab Evaluation 20% 8 12 –\nPresentation Followed by Viva 20 – – 20\nTotal 40 8 12 20')
        card=base('weight','Presentation Followed by Viva 20 – – 20')
        card.update(assessment='Presentation Followed by Viva',char_start=page.rindex('Presentation Followed by Viva'),
                    weight_display='20%',co_clo_cells={})
        revised=populate_extras(normalize_card(card),page)
        assert revised['clo_cells']=={'CLO 3':'20'}
        assert revised['weight_display']=='20%'

    def test_existing_identifier_cells_win_over_compressed_column_guess():
        page='Assessment Type % weight CO1 CO2 CO3\nProject 20% 40%\nNext 10%'
        card=base('weight','Project 20% 40%')
        card.update(assessment='Project',char_start=page.index('Project'),weight_display='20%',
                    co_clo_cells={'CO3':'40%'})
        revised=populate_extras(normalize_card(card),page)
        assert revised['co_cells']=={'CO3':'40%'}
        assert revised['value']==card['value']
