"""Plain-text and layout-column fixtures from the three outline families."""
import os

if os.getenv('RAG_REPAIR_COURSE_CARDS','').lower() in {'1','true','yes','on'}:
    import pytest
    from src.course_cards_mapping import mapping_cards

    @pytest.mark.parametrize('kind,ident,cell,family',[
        ('B','CLO 1','a','PLO'),
        ('A-CO','CO 1','1','PO'),
        ('A-hybrid','CLO 1','PLO(b)','PLO'),
    ])
    def test_column_bands_three_families(kind,ident,cell,family):
        header=f'{ident.split()[0]:<8}{"Corresponding "+family+"s":<24}Bloom taxonomy'
        row=f'{ident:<8}{cell:<24}1/Apply'
        layout='Mapping of outcomes\n'+header+'\n'+row
        plain=f'Mapping of outcomes {ident} Explain the topic. {cell} 1/Apply'
        cards,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['row_key']==ident and x['value']==cell for x in cards)

    def test_split_prefix_number_cell():
        layout='Mapping\n'+f'{"CLO":<8}{"Corresponding PLOs":<24}Bloom taxonomy'+'\n'+f'{"CLO 1":<8}{"PLO":<24}1/Apply'+'\n'+f'{"":<8}{"1":<24}'
        plain='Mapping CLO 1 Explain the topic. PLO 1 1/Apply'
        cards,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['po_cell_verbatim']=='PLO 1' and x['plo_display']=='PLO 1' for x in cards)

    def test_cell_on_line_above_id_number():
        layout='Mapping\n'+f'{"CLO":<8}{"Corresponding PLOs":<24}Bloom taxonomy'+'\n'+f'{"CLO":<8}{"a":<24}1/Apply'+'\n'+f'{"1":<8}'
        plain='Mapping CLO 1 Explain the topic. a 1/Apply'
        cards,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['row_key']=='CLO 1' and x['plo_display']=='PLO (a)' for x in cards)
