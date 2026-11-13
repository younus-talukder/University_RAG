"""Opt-in deterministic router fixtures; no dataset answer is used."""
import os

if os.getenv('RAG_REPAIR_CARD_ANSWERS','').lower() in {'1','true','yes','on'}:
    from src.card_router import understand,resolve_range

    def test_bengali_digits_and_department_spelling():
        intent=understand('সিএসই ৩১৬ কোর্সের ক্রেডিট কত?')
        assert intent.lang=='bn' and intent.course=='CSE 316'
        assert intent.field=='credit' and intent.mode=='lookup'

    def test_compact_course_and_schedule_key():
        intent=understand('cse316-er Week 4 er topic ki?')
        assert intent.course=='CSE 316' and intent.field=='schedule'
        assert intent.key=='Week 4'

    def test_week_range_and_resolution():
        assert understand('What is in Weeks 5-12 of CSE 316?').key=='Weeks 5–12'
        assert resolve_range('Week 7',['Week 1','Weeks 5–12'])=='Weeks 5–12'

    def test_two_courses_and_explanation_fall_back():
        assert understand('Compare CSE 316 and CSE 426').mode=='none'
        assert understand('Why is the credit of CSE 316 three?').mode=='none'

    def test_schedule_attribute_exception_falls_back():
        assert understand('What activities are in Week 4 of CSE 316?').mode=='none'

    def test_mapping_question_routes_to_link_field():
        intent=understand('Which PLO does CLO 2 map to in CSE 316?')
        assert intent.field=='clo_plo_link' and intent.key=='CLO 2'

    def test_count_list_reverse():
        assert understand('How many objectives does CSE 316 have?').mode=='count'
        assert understand('List all teachers of CSE 316').mode=='list'
        assert understand('Which week covers circuits in CSE 316?').mode=='reverse'
