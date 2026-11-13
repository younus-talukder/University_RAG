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

    def test_academic_session_is_distinct_from_year_and_semester():
        assert understand('What academic semester is listed for CSE 316?').field=='semester'
        assert understand('What academic level is listed for CSE 316?').field=='level_term'
        assert understand('CSE 316-er year ar semester ki?').field=='level_term'
        assert understand('EX 101-er semester ki deya ache?').field=='semester'
        assert understand('EX 101 কোর্সের শিক্ষাবর্ষের সেমিস্টার কী?').field=='semester'

    def test_count_list_reverse():
        assert understand('How many objectives does CSE 316 have?').mode=='count'
        assert understand('List all teachers of CSE 316').mode=='list'
        assert understand('Which week covers circuits in CSE 316?').mode=='reverse'

    def test_week_and_lecture_key_forms_across_languages():
        cases={
            'EX 101-এর ৪তম সপ্তাহে কী আছে?':'Week 4',
            'EX 101-এর ২য় সপ্তাহে কী আছে?':'Week 2',
            'EX 101-এর সপ্তাহ ৩-এ কী আছে?':'Week 3',
            'EX 101-এর লেকচার ৬ কী?':'Lecture 6',
            'EX 101-er week 5 er topic ki?':'Week 5',
            'EX 101-er 7th week er topic ki?':'Week 7',
            'EX 101-er soptaho 8 er topic ki?':'Week 8',
            'What is in the second week of EX 101?':'Week 2',
            'What is in week two of EX 101?':'Week 2',
        }
        for question,expected in cases.items():
            assert understand(question).key==expected,question

    def test_bangla_ordinal_words_one_through_sixteen_and_inflections():
        words=('প্রথম','দ্বিতীয়','তৃতীয়','চতুর্থ','পঞ্চম','ষষ্ঠ','সপ্তম','অষ্টম',
               'নবম','দশম','একাদশ','দ্বাদশ','ত্রয়োদশ','চতুর্দশ','পঞ্চদশ','ষোড়শ')
        for number,word in enumerate(words,1):
            for suffix in ('','এ','র','এর'):
                assert understand(f'EX 101-এর {word}{suffix} সপ্তাহে কী?').key==f'Week {number}'

    def test_weight_cues_and_course_scoped_assessment_name_ties():
        cards=[{'course':'EX 101','field':'weight','assessment':'Final Project'},
               {'course':'AB 202','field':'weight','assessment':'Final Project'}]
        assert understand('How much counts toward the final project of EX 101?',cards).field=='weight'
        assert understand('EX 101-এর final project কত শতাংশ?',cards).field=='weight'
        assert understand('EX 101-er final project koto marks?',cards).field=='weight'
        assert understand('How much is assigned to Final Project in EX 101?',cards).key=='Final Project'

    def test_term_in_assessment_is_not_course_level():
        for question in ('EX 101-er class test term-wise kibhabe divided?',
                         'How many CTs in each term of EX 101?',
                         'EX 101-এর টার্ম পরীক্ষার নিয়ম কী?'):
            intent=understand(question)
            assert intent.field!='level_term',question
        assert understand('What is the level/term of EX 101?').field=='level_term'

    def test_plo_po_cue_never_selects_statement():
        assert understand('Which PLO is CLO 2 in EX 101 linked to?').field=='clo_plo_link'
        assert understand('Which PO does CO 3 address in EX 101?').field=='co_po_link'
        assert understand('EX 101-এর প্রোগ্রাম আউটকাম কোনটি?').field=='co_po_link'

    def test_week_number_excludes_course_code_digits():
        assert understand('EX 108 week 6 topic').key=='Week 6'
        assert understand('EX 108 lecture 7 topic').key=='Lecture 7'
        assert understand('EX 108 week topic').key==''

    def test_named_other_document_falls_back():
        for name in ('prospectus','curriculum','syllabus book','infobook','alumni',
                     'প্রসপেক্টাস','কারিকুলাম','সিলেবাস'):
            intent=understand(f'According to the {name}, what is EX 101 credit?')
            assert intent.mode=='none' and intent.field=='',name
