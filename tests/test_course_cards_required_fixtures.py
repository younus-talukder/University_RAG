"""Required Step 5A r5b fixtures from plain course-outline snippets."""
import os

if os.getenv('RAG_REPAIR_COURSE_CARDS','').lower() in {'1','true','yes','on'}:
    import pytest
    from src.course_cards_tiered import _segments,_supplement,_weight_cell_hints,parse_tiered_pages

    @pytest.mark.parametrize('family,heading',[('B','16. Class Schedule/Lesson Plan/Weekly plan:'),('A-hybrid','Course Content Outline and mapping with COs:')],ids=['B','A-hybrid'])
    def test_midrow_week_anchor_keeps_activity_cell(family,heading):
        plain=(heading+' Topics Specific Outcome(s) Time Frame '
               'Suggested Activities Teaching Strategy(s) Alignment with CLO '
               'Mini Projects • Implement hardware control programs '
               'Weeks 5–12 Circuit connection, Hands-on Lab practice, demonstration CLO1, CLO2 '
               '17. Teaching-Learning Strategies:')
        cards=_supplement('EX 101.pdf',[plain])
        assert any(c['field']=='schedule' and c['row_key']=='Weeks 5–12' and
                   c['activities']=='Circuit connection, Hands-on Lab practice, demonstration' for c in cards)

    def test_weight_and_outcome_percentage_are_distinct():
        plain='Weighting COs with Assessment methods: Assessment Type % weight CO1 CO2 CO3 Assessment 30% 50%'
        layout='Weighting COs with Assessment methods:\n'+f'{"Assessment Type":<40}{"% weight":<20}{"CO1":<16}{"CO2":<16}CO3\n'+f'{"Assessment":<40}{"30%":<20}{"":<16}{"50%":<16}'
        card=_weight_cell_hints({'field':'weight','assessment':'Assessment','weight_display':'30%'},layout,plain)
        assert card['weight_display']=='30%'
        assert card['co_clo_cells']=={'CO2':'50%'}

    @pytest.mark.parametrize('family,head,stop',[('B','Course Content:','15. Alignment:'),('A-CO','Course Synopsis:','Course Objectives:'),('A-hybrid','Course Synopsis:','Course Objectives:')],ids=['B','A-CO','A-hybrid'])
    def test_cross_page_list_continuation_stays_in_section(family,head,stop):
        pages=[head+' First topic;', 'Second topic. '+stop+' New section.']
        assert list(_segments(pages,head,stop))==[(1,'First topic;'),(2,'Second topic.')]

    @pytest.mark.parametrize('family,outcome',[('A-CO','Course Outcomes (CO):'),('A-hybrid','Course Learning Outcomes (CLO):')],ids=['A-CO','A-hybrid'])
    def test_empty_references_has_an_empty_section_card(family,outcome):
        plain='Course Outline Program: Example Course Title: Example Course Course Code: EX 101 '+outcome+' CO 1 Explain circuits. Required References: Special Instructions: Attend classes.'
        template,cards,_=parse_tiered_pages('EX 101.pdf',[plain])
        assert template=='A'
        assert any(c['tier']=='atomic' and c['field']=='books' and c.get('empty_section') and c['value']=='' for c in cards)
