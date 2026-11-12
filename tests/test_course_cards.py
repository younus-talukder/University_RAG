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
