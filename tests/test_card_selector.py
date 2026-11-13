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

    def test_unique_header_field_beats_repeated_course_code_embedding():
        rows=[card('course_code','course_code'),card('level_term','level_term')]
        result=select_cards(intent('level_term','level_term'),rows,{'course_code':.8,'level_term':.5},.03,.03)
        assert result.route=='card' and result.cards[0]['field']=='level_term'

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

    def test_value_anchored_item_requires_bm25_bge_agreement():
        from src.card_selector import CardValueIndex,select_item_by_value
        shared='standard academic reference materials practice overview instruction activity reading schedule '
        rows=[{**card('objective 1','objective'),'value':shared+'Quartz nebula calibration'},
              {**card('objective 2','objective'),'value':shared+'Harbor glacier assembly'}]
        lexical=CardValueIndex(rows)
        query='Which objective covers quartz nebula calibration?'
        picked=select_item_by_value(query,intent('objective',''),rows,lexical,
                                    {'objective 1':.9,'objective 2':.3},.03)
        assert picked.route=='card' and picked.cards[0]['key']=='objective 1'
        rejected=select_item_by_value(query,intent('objective',''),rows,lexical,
                                      {'objective 1':.3,'objective 2':.9},.03)
        assert rejected.route=='fallback'

    def test_value_anchored_course_requires_two_distinctive_tokens_and_model_agreement():
        from src.card_selector import CardValueIndex,resolve_course_by_value
        shared='standard academic reference materials practice overview instruction activity reading schedule '
        rows=[{**card('book 1','book','a'),'course':'EX 101','value':shared+'Quartz nebula calibration manual'},
              {**card('book 1','book','b'),'course':'AB 202','value':shared+'Harbor glacier assembly manual'},
              {**card('book 2','book','c'),'course':'CD 303','value':shared+'Common manual'}]
        lexical=CardValueIndex(rows)
        query='Which manual mentions quartz nebula calibration?'
        assert resolve_course_by_value(query,rows,lexical,{'a':.9,'b':.4,'c':.2},.03)=='EX 101'
        assert resolve_course_by_value(query,rows,lexical,{'a':.3,'b':.9,'c':.2},.03)==''
