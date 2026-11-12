"""Plain-text fixtures for the opt-in r4 extraction path."""
import os

if os.getenv('RAG_REPAIR_COURSE_CARDS','').lower() in {'1','true','yes','on'}:
    from src.course_cards_tiered import _anchor, _gap_fill, _segments, parse_tiered_pages

    def test_exact_span_uses_plain_offsets():
        raw='Course Outline\nProgram: Example\nCredit Hour: 1.5\nCourse Objectives: Learn systems.'
        card={'page':1,'value':'1.5','field':'credit','pdf':'EX 101.pdf'}
        found=_anchor(card,[raw])
        assert found['span_status']=='found'
        assert raw[found['char_start']:found['char_end']]=='1.5'
        assert 'Course Objectives' not in found['plain_span']

    def test_sections_continue_across_pages():
        pages=['Course Synopsis: First topic;','Second topic. Course Objectives: Begin.']
        segments=list(_segments(pages,r'Course Synopsis:',r'Course Objectives:'))
        assert segments==[(1,'First topic;'),(2,'Second topic.')]

    def test_numbered_objectives_continue_and_table_rows_are_distinct():
        pages=['11. Course Objectives and Course Summary: 1. Explain circuits.',
               '2. Build devices. 12. Course Learning Outcomes: CLO 1 Build devices. '
               '18. Assessment Techniques of each topic of the course:',
               'SL. No Topics / Content Assessment Techniques 1 Circuit basics Lab Task, Lab Evaluation Part C – Assessment and Evaluation']
        cards=_gap_fill('EX 101.pdf',pages)
        assert any(x['field']=='objective' and x['value']=='Build devices.' for x in cards)
        assert any(x['field']=='assessment_technique' and x['value']=='Lab Task, Lab Evaluation' for x in cards)

    def test_appendix_profile_and_grading_are_separate():
        pages=['Appendix-1: Washington Accord Program Learning Outcomes (PLO): (a) Apply engineering knowledge.',
               '(b) Analyze engineering problems. Appendix-2',
               'Ks Ps As CLO 1 a 1/Apply K3, K4 P1, P2 A1, A2 Total 100% 8 20 16 16 20 20',
               'UAP Grading Policy Numeric Grade Letter Grade Grade Point 80% and above A+ 4.00']
        cards=_gap_fill('EX 101.pdf',pages)
        assert any(x['field']=='plo_definition' and x['row_key']=='PLO (b)' for x in cards)
        assert any(x['field']=='knowledge_profile' and x['row_key']=='Ks' and x['course_level'] for x in cards)
        assert any(x['field']=='weight_total' and x['clo_totals']['CLO 6']==20 for x in cards)
        assert any(x['field']=='grading_scale' and not x['is_weight'] for x in cards)

    def test_two_tiers_and_not_found_exclusion():
        pages=['Course Outline Program: Example Course Title: Test Course Course Code: EX 101 '
               'Credit Hour: 1.5 Course Objectives: 1. Explain circuits. '
               'Course Outcomes (CO): CO 1 Explain circuits.']
        template,cards,rejected=parse_tiered_pages('EX 101.pdf',pages)
        assert template=='A'
        assert {'block','atomic'} <= {c['tier'] for c in cards}
        assert all(c['span_status']=='found' for c in cards if c['tier']=='atomic')
        assert all(c['span_status'] in {'not_found','statement_rejected'} for c in rejected)
        assert all(c.get('reject_reason') for c in rejected if c['span_status']=='statement_rejected')
