"""Statement-source guard fixtures using plain outline table snippets."""
import os

if os.getenv('RAG_REPAIR_COURSE_CARDS','').lower() in {'1','true','yes','on'}:
    import pytest
    from src.course_cards_mapping import mapping_cards
    from src.course_cards_statements import statement_rejection

    @pytest.mark.parametrize('cell',['a','c','i','j','1','3','12'])
    def test_family_b_statement_and_link_are_separate(cell):
        statement='Explain circuit behavior using practical methods.'
        plain=('12. Course Learning Outcomes: CLO 1 '+statement+
               ' 13. Mapping / Alignment of CLOs with Program Learning Outcomes (PLO): '
               'CLO 1 '+cell+' 1/Apply')
        header=f'{"CLO":<8}{"Corresponding PLOs":<24}Bloom taxonomy'
        layout='13. Mapping / Alignment\n'+header+'\n'+f'{"CLO 1":<8}{cell:<24}1/Apply'
        links,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['field']=='clo_plo_link' and x['value']==cell for x in links)
        assert statement_rejection({'field':'clo','value':statement},'B',[plain]) is None
        assert statement_rejection({'field':'clo','value':cell},'B',[plain])=='identifier_only_cell'

    def test_family_a_co_statement_and_po_cell_are_separate():
        statement='Apply graphics methods to visualize geometric objects.'
        plain=('Course Outcomes (CO) and their mapping with Program Outcomes (PO): '
               'CO Statements: CO 1 '+statement+' PO2 Cognitive / Apply '
               'Weighting COs with Assessment methods:')
        header=f'{"CO":<8}{"CO Statements":<24}{"Corresponding POs":<24}Bloom taxonomy'
        layout='Course Outcomes (CO)\n'+header+'\n'+f'{"CO 1":<8}{statement[:20]:<24}{"PO2":<24}1/Apply'
        links,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['field']=='co_po_link' and x['value']=='PO2' for x in links)
        assert statement_rejection({'field':'co','value':statement},'A',[plain]) is None
        assert statement_rejection({'field':'co','value':'PO2'},'A',[plain])=='identifier_only_cell'

    def test_family_a_hybrid_statement_and_plo_cell_are_separate():
        statement='Analyze system requirements using engineering methods.'
        plain=('Course Learning Outcomes (CLO) and their mapping with Program Learning outcomes (PLO): '
               'CLO Statements: CLO 1 '+statement+' PLO(c) 1/Analyze '
               'Weighting COs with Assessment methods:')
        header=f'{"CLO":<8}{"CO Statements":<24}{"Corresponding PLOs":<24}Bloom taxonomy'
        layout='Course Learning Outcomes (CLO)\n'+header+'\n'+f'{"CLO 1":<8}{statement[:20]:<24}{"PLO(c)":<24}1/Analyze'
        links,_=mapping_cards('EX 101.pdf',[plain],[layout])
        assert any(x['field']=='clo_plo_link' and x['value']=='PLO(c)' for x in links)
        assert statement_rejection({'field':'clo','value':statement},'A',[plain]) is None
        assert statement_rejection({'field':'clo','value':'PLO(c)'},'A',[plain])=='identifier_only_cell'
