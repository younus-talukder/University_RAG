"""Opt-in multilingual card rendering fixtures; no PDF fact in runtime code."""
import os

if os.getenv('RAG_REPAIR_CARD_ANSWERS','').lower() in {'1','true','yes','on'}:
    from src.card_answer import render_card_answer
    from src.card_router import CardIntent
    from src.card_selector import CardSelection
    from src.card_runtime import enabled
    import src.card_runtime as runtime

    def render(lang,field,value,**attributes):
        card=dict(card_id='synthetic',tier='atomic',course='EX 101',pdf='outline.pdf',page=2,pages=[2],
                  field=field,key=field,value=value,plain_span=value,**attributes)
        intent=CardIntent(lang,'EX 101',field,field,'','lookup',1,())
        selected=CardSelection('card','lookup',(card,),1,1,'fixture')
        return render_card_answer(intent,selected)

    def test_bangla_digits_only_in_numeric_slots_and_page():
        numeric=render('bn','credit','3.00')
        assert '৩.০০' in numeric.body and 'EX 101' in numeric.body and 'পৃষ্ঠা ২' in numeric.citation
        title=render('bn','course_title','Part 2, Edition 3')
        assert 'Part 2, Edition 3' in title.body
        assert 'EX 101' in title.body

    def test_none_variants_and_english_citation():
        answer=render('en','prerequisite','None')
        assert 'no prerequisite' in answer.body
        assert '(Source: outline.pdf, page 2)'==answer.citation

    def test_banglish_frame_and_identifier_preservation():
        answer=render('bl','clo','Explain examples',row_key='CLO 2')
        assert 'course er' in answer.body and 'CLO 2' in answer.body

    def test_bangla_frame_keeps_long_verbatim_english_title_meaningful():
        from src.language_validator import validate_language
        answer=render('bn','book','A Synthetic Reference Manual by Example Writers')
        assert validate_language(answer.body,'bangla')['validation_passed']
        assert 'A Synthetic Reference Manual' in answer.body

    def test_flag_requires_both_switches(monkeypatch):
        monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS','1')
        monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS','0')
        assert not enabled()
        monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS','1')
        assert enabled()

    def test_validated_card_path_and_failed_span_fallback(monkeypatch):
        source='EX 101 Credit Value: 3.00'
        card=dict(card_id='c1',tier='atomic',course='EX 101',pdf='outline.pdf',page=1,pages=[1],
                  field='credit',key='credit',value='3.00',plain_span=source,
                  char_start=source.index('3.00'),char_end=len(source),span_status='found')
        class FakeIndex:
            def scores(self,vector,course):return {'c1':.9}
        class FakeModel:
            def embed(self,question):return object()
        monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS','1')
        monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS','1')
        monkeypatch.setattr(runtime,'_bundle',lambda:([card],FakeIndex(),{'tau':.03,'tau_rev':.03,'disabled_fields':[]}))
        monkeypatch.setattr(runtime,'get_embedding_model',lambda:FakeModel())
        monkeypatch.setattr(runtime,'_plain_page',lambda pdf,page:source)
        result,extra=runtime.attempt_card_answer('What is the credit of EX 101?')
        assert result is not None and result['support_status']=='SUPPORTED'
        assert result['grounding_validation_passed'] and result['language_validation_passed']
        assert result['page']==1 and result['card_id']=='c1'
        monkeypatch.setattr(runtime,'_plain_page',lambda pdf,page:'EX 101 Credit Value: 4.00')
        result,_=runtime.attempt_card_answer('What is the credit of EX 101?')
        assert result is None

    def test_pipeline_flagged_short_circuit_does_not_load_faiss(monkeypatch):
        import src.pipeline as pipeline
        monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS','1')
        monkeypatch.setenv('RAG_REPAIR_CARD_ANSWERS','1')
        monkeypatch.setattr(runtime,'attempt_card_answer',lambda question:({'answer':'Verified synthetic answer.',
                                                                           'final_status':'ANSWER_RETURNED'},[]))
        monkeypatch.setattr(pipeline,'load_index',lambda **kwargs:(_ for _ in ()).throw(AssertionError('index loaded')))
        result=pipeline.answer_question('What is the credit of EX 101?')
        assert result['final_status']=='ANSWER_RETURNED'
