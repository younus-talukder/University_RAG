from src.course_cards import detect_template, match_text, parse_pages


def test_flag_off(monkeypatch):
    monkeypatch.delenv('RAG_REPAIR_COURSE_CARDS', raising=False)
    assert parse_pages('synthetic.pdf', ['Course Outline Program: X Course Title: Y']) == ('OFF', [])


def test_template_a_and_values(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    text = 'Course Outline\nProgram: Example\nCourse Code: ENG (CSE) 101\nLevel: 1/2\nClass Hours: Monday 9\nTuesday 10\nPre-requisite (if any): N/A\nLecture 1 Intro\nLecture 2 More'
    template, cards = parse_pages('synthetic.pdf', [text])
    assert template == 'A'
    assert any(c.field == 'course_code' and c.value == 'ENG (CSE) 101' for c in cards)
    assert any(c.field == 'level_term' and c.value == '1/2' for c in cards)
    assert any(c.field == 'class_hours' and 'Tuesday 10' in c.value for c in cards)
    assert any(c.field == 'prerequisite' and c.value == 'N/A' for c in cards)
    assert [c.row_key for c in cards if c.field == 'lecture'] == ['1', '2']


def test_template_b_blank_none_and_level(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', 'true')
    text = 'Course Outline Part A – Introduction 1. Course No. / Course Code: ABC 123 2. Course Title: Example 4. Level/Term and Section: First year first semester 7. Pre-requisite (If any): 8. Credit Value: 3.0 16. Weekly Plan: Topic A Week 1 Activity B Week 2'
    template, cards = parse_pages('synthetic.pdf', [text])
    assert template == 'B'
    assert any(c.field == 'level_term' and c.value == 'First year first semester' for c in cards)
    assert any(c.field == 'prerequisite' and c.value == 'BLANK' for c in cards)
    assert any(c.field == 'weekly_topic' and 'Topic A' in c.value for c in cards)
    text2 = text.replace('7. Pre-requisite (If any): 8.', '7. Pre-requisite (If any): None 8.')
    assert any(c.field == 'prerequisite' and c.value == 'None' for c in parse_pages('synthetic.pdf', [text2])[1])


def test_matching_only_normalizes():
    assert match_text('8 th Semester') == match_text('8th Semester')
    assert match_text('A\u200bB') == 'ab'


def test_stage_a_deferred_card_facts(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    # Short snippets are copied from pypdf plain extraction, not dataset answers.
    phy = ('Course Outline – Physics for Computer Science Part A – Introduction '
           '1. Course No. / Course Code: PHY 101 4. Level/Term and Section: 1st Year 1st Semester')
    _, cards = parse_pages('PHY 101.pdf', [phy])
    assert any(c.field == 'level_term' and c.value == '1st Year 1st Semester' for c in cards)  # Q 51
    oop = ('Course Outline Part A – Introduction 1. Course No. / Course Code: CSE 202 '
           '4. Level/Term and Section: 3rd Semester (2nd Year/1st Semester)')
    _, cards = parse_pages('CSE 202.pdf', [oop])
    assert any(c.field == 'level_term' and '2nd Year/1st Semester' in c.value for c in cards)  # Q 143
    db = ('Course Outline Part A – Introduction 1. Course No. / Course Code: CSE 212 '
          '4. Level/Term and Section: 4th Semester (2nd Year 2nd Semester) '
          '11. Course Objectives and Course Summary: The objectives of this course are to: '
          'normalization of relations and DDL and DML.')
    _, cards = parse_pages('CSE 212.pdf', [db])
    assert any(c.field == 'level_term' and '2nd Year 2nd Semester' in c.value for c in cards)  # Q 188
    assert any(c.field == 'objectives' and 'normalization' in c.value for c in cards)  # Q 197
    ref = ('Course Outline Part A – Introduction 1. Course No. / Course Code: PHY 101 '
           '22. Recommended Reference(s): 4. “Fundamental of Optics” by Francis A. Jenkins & Harvey E. White; 4th edition.')
    _, cards = parse_pages('PHY 101.pdf', [ref])
    assert any(c.field == 'books' and 'Fundamental of Optics' in c.value for c in cards)  # Q 249


def test_numbered_policy_without_separator_and_week_topic(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    text = ('Course Outline Part A – Introduction 1. Course No. / Course Code: MTH 101 '
            '16. Class Schedule/Lesson Plan/Weekly plan: Limits and continuity Week 1 '
            '20. Evaluation Policy: 1. Class Tests 30% 3. Mid-Term Examination 20% '
            '21. Text Book, Reference Books & Materials: Calculus')
    _, cards = parse_pages('synthetic.pdf', [text])
    assert any(c.field == 'weekly_topic' and 'Limits and continuity' in c.value for c in cards)
    assert any(c.field == 'evaluation_policy' and 'Mid-Term Examination 20%' in c.value for c in cards)


def test_template_a_weights_and_co_po_links(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    page = ('Course Outline Program: Computer Science and Engineering Course Code: CSE 426 '
            'CO No. CO Statements Corresponding POs Bloom’s taxonomy domain/level '
            'CO 1 Understand the objectives and terminology associated with Computer Graphics. '
            '1 Cognitive / Understand Lecture, Group discussion Quiz '
            'CO 2 Apply the techniques and algorithms of Computer Graphics. 5 Cognitive / Apply '
            'Weighting COs with Assessment methods: Assessment Type % weight CO1 CO2 CO3 '
            'Assessment 30% 50% Assignment 10% Mid 20% Project 20% 40% Report and Viva 10%')
    template, cards = parse_pages('synthetic.pdf', [page])
    assert template == 'A'
    assert any(c.field == 'co_po_link' and c.row_key == 'CO 1' and 'Cognitive / Understand' in c.value for c in cards)
    assert any(c.field == 'assessment_weight' and '20%' in c.value for c in cards)


def test_template_b_weights_and_clo_plo_links(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    page = ('Course Outline Part A – Introduction 1. Course No. / Course Code: CSE 316 '
            '13. Mapping / Alignment of CLOs with Program Learning Outcomes (PLO): '
            'CLO No. Corresponding PLOs Bloom’s taxonomy domain/level Delivery methods and activities '
            'CLO 1 a 1/Apply Lab demonstrations Lab Evaluation '
            'CLO 2 c 1/Apply Programming exercises Lab Evaluation '
            'Part C – Assessment and Evaluation Assessment Type % Weight CLO 1 CLO 2 '
            'Lab Evaluation (Assembly + Mini Projects) 20% 8 12 '
            '20. Evaluation Policy 1. Lab Evaluation 20% 2. Project 80%')
    template, cards = parse_pages('synthetic.pdf', [page])
    assert template == 'B'
    assert any(c.field == 'clo_plo_link' and c.row_key == 'CLO 1' and '1/Apply' in c.value for c in cards)
    assert any(c.field == 'assessment_weight' and '20%' in c.value for c in cards)
    assert any(c.field == 'assessment_weight' and '80%' in c.value for c in cards)


def test_three_layout_families_by_features(monkeypatch):
    from src.course_cards import detect_layout_family
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    b = 'Course Outline Part A – Introduction 1. Course No. / Course Code: ABC 123 12. Course Learning Outcomes: CLO 1 Apply'
    co = 'Course Outline Program: Example Course Code: ABC 123 CO No. CO 1 Do something 1 Cognitive'
    hybrid = 'Course Outline Program: Example Course Code: ABC 123 CLO 1 Do something PLO(a)'
    assert detect_layout_family([b]) == 'B numbered Part A-D'
    assert detect_layout_family([co]) == 'A label:value + CO/PO'
    assert detect_layout_family([hybrid]) == 'A label:value + CLO/PLO (hybrid)'


def test_bare_week_range_and_week_lecture_rows(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    text = ('Course Outline Program: Example Course Code: ABC 123 '
            'Course Content Outline and mapping with COs Weeks Topics / Content Course Outcome '
            '1 Introduction to chemistry CO1 Lecture, multimedia Supplied Content '
            '2-3 Atomic structure and models CO1 Lecture, multimedia Reference Books')
    _, cards = parse_pages('synthetic.pdf', [text])
    keys = [c.row_key for c in cards if c.field == 'schedule_row']
    assert 'week 1' in keys and 'week 2-3' in keys
    mixed = ('Course Outline Program: Example Course Code: ABC 123 '
             'Lecture Schedule (Tentative) Weeks Lecture # Topics Course Outcome '
             '2 3, 4 Coordinates and Transformations CO1 Lecture, Problem Solving')
    _, cards = parse_pages('synthetic.pdf', [mixed])
    assert any(c.field == 'schedule_row' and c.row_key == 'week 2 | lecture 3, 4' for c in cards)


def test_alignment_table_is_not_schedule(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    text = ('Course Outline Part A – Introduction 1. Course No. / Course Code: ABC 123 '
            '15. Alignment of topics with CLO SL. No Topics CLO 1 Topic one CLO1')
    assert not any(c.field == 'schedule_row' for c in parse_pages('synthetic.pdf', [text])[1])


def test_weight_traps_are_not_assessment_weight_cards(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    prefix = 'Course Outline Program: Example Course Code: ABC 123 '
    for trap in ['80% and above A+ 4.00', 'Late submission will result in a 50% deduction in score.',
                 'Minimum attendance: 70% class attendance is mandatory.']:
        text = prefix + 'Weighting COs with Assessment methods: Assessment Type % weight Quiz 10% ' + trap
        _, cards = parse_pages('synthetic.pdf', [text])
        weights = [c.value for c in cards if c.field == 'assessment_weight']
        assert weights and all(trap not in value for value in weights)


def test_weight_table_can_continue_without_percent_sign(monkeypatch):
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    pages = [
        'Course Outline Program: Example Course Code: ABC 123 Weighting COs with Assessment methods:',
        'Assessment Type % weight CO1 CO2 CO3 Lab Final Exam 40 20 30 0 Assignment 40 - 40 - Total 100',
    ]
    _, cards = parse_pages('synthetic.pdf', pages)
    assert any(c.field == 'assessment_weight' and c.page == 2 and 'Lab Final Exam 40' in c.value for c in cards)


def test_hybrid_numbered_direct_course_code(monkeypatch):
    from src.course_cards import detect_layout_family
    monkeypatch.setenv('RAG_REPAIR_COURSE_CARDS', '1')
    text = ('Course Outline: Project Strategy Part A – Introduction 1. Course Code: ABC 123 '
            '2. Course Title: Project Strategy 12. Course Learning Outcomes: CLO 1 Apply PLO(a)')
    assert detect_layout_family([text]) == 'A label:value + CLO/PLO (hybrid)'
    assert parse_pages('synthetic.pdf', [text])[0] == 'A'
