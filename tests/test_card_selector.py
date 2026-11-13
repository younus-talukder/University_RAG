"""Opt-in card selection checks with synthetic scores and no model load."""
import os

if os.getenv('RAG_REPAIR_CARD_ANSWERS','').lower() in {'1','true','yes','on'}:
    from src.card_router import CardIntent
    from src.card_selector import infer_complete_fields,select_cards

    def card(key,field='schedule',ident=None):
        return {'card_id':ident or key,'course':'EX 101','field':field,'key':key,
                'row_key':key,'tier':'atomic','value':'Synthetic value'}

    def intent(field,key,mode='lookup'):
        return CardIntent('en','EX 101',field,key,'',mode,1.0,())

    def test_matching_field_and_week_inside_range():
        rows=[card('Weeks 5–12'),card('Week 14')]
        result=select_cards(intent('schedule','Week 7'),rows,{'Weeks 5–12':.82,'Week 14':.70},.2,.9)
        assert result.route=='card' and result.cards[0]['key']=='Weeks 5–12'

    def test_ambiguous_margin_falls_back():
        rows=[card('course_title','course_title'),card('credit','credit')]
        result=select_cards(intent('course_title',''),rows,{'course_title':.80,'credit':.79},.05,.9)
        assert result.route=='fallback'

    def test_list_and_count_require_complete_field():
        rows=[card('objective 1','objective'),card('objective 2','objective')]
        assert select_cards(intent('objective','','list'),rows,{},.1,.9).route=='fallback'
        routed=select_cards(intent('objective','','count'),rows,{},.1,.9,{'objective'})
        assert routed.route=='card' and len(routed.cards)==2

    def test_reverse_requires_own_threshold():
        rows=[card('Week 4')]
        assert select_cards(intent('schedule','','reverse'),rows,{'Week 4':.7},.1,.8).route=='fallback'
        assert select_cards(intent('schedule','','reverse'),rows,{'Week 4':.9},.1,.8).route=='card'

    def test_list_count_completeness_requires_contiguous_numbers():
        rows=[card('1','objective'),card('2','objective')]
        for number,row in enumerate(rows,1):row['list_index']=number
        assert infer_complete_fields(rows)=={'objective'}
        rows[1]['list_index']=3
        assert infer_complete_fields(rows)==set()
