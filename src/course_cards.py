"""Opt-in course-outline fact cards from pypdf plain text only.

This module never changes the answer path.  Extraction is enabled only by
RAG_REPAIR_COURSE_CARDS=1; callers provide physical page text in order.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

FLAG = 'RAG_REPAIR_COURSE_CARDS'
_DROP = dict.fromkeys(map(ord, '\u200b\u200c\u200d\u2060\ufeff\u00ad'))
_REPL = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-', '−': '-'})


def enabled() -> bool:
    return os.getenv(FLAG, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def match_text(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).translate(_DROP).translate(_REPL).casefold()
    value = re.sub(r'\b(\d+)\s+(st|nd|rd|th)\b', r'\1\2', value)
    return re.sub(r'[^\w%]+', ' ', value).strip()


def _compact(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """Whitespace-compacted plain text and original offsets for each token."""
    found = list(re.finditer(r'\S+', raw))
    return ' '.join(m.group() for m in found), [(m.start(), m.end()) for m in found]


def _raw_span(raw: str, offsets: list[tuple[int, int]], compact: str, start: int, end: int) -> str:
    if not offsets or start >= end:
        return ''
    left = compact[:start].count(' ')
    right = compact[:end].count(' ')
    left = min(left, len(offsets) - 1)
    right = min(right, len(offsets) - 1)
    return raw[offsets[left][0]:offsets[right][1]]


@dataclass(frozen=True)
class Card:
    course: str
    field: str
    value: str
    pdf: str
    page: int
    plain_span: str
    template: str
    list_index: int | None = None
    row_key: str | None = None

    def record(self) -> dict:
        result = asdict(self)
        result['text'] = f'{self.course} | {self.field} | {self.value} | p{self.page}'
        return result


# Families are generic label patterns.  Numbered template B prefixes are optional.
_LABELS = [
    ('program', r'Program'),
    ('course_code', r'(?:Course\s+(?:No\.?\s*/\s*)?Code|Course\s+No\.)'),
    ('course_title', r'Course\s+Title'),
    ('course_type', r'Course\s+Type'),
    ('semester', r'(?:Academic\s+Session|Semester)'),
    ('level_term', r'(?:Level\s*/\s*Term(?:\s+and\s+Section)?|Level)'),
    ('credit', r'(?:Credit\s+Hour|Credit\s+Value)'),
    ('teacher', r'(?:Name\s*&\s*Designation\s+of\s+Teacher|Course\s+Instructor)'),
    ('office', r'Office\s*/\s*Room'),
    ('class_hours', r'Class\s+Hours'),
    ('consultation_hours', r'Consultation\s+Hours'),
    ('email', r'E\s*-?\s*mail|Email'),
    ('mobile', r'Mobile'),
    ('rationale', r'Rationale'),
    ('prerequisite', r'Pre\s*-?\s*requisite(?:\s*\(\s*If\s+any\s*\))?'),
    ('contact_hours', r'Contact\s+Hours'),
    ('total_marks', r'Total\s+Marks'),
    ('synopsis', r'Course\s+Synopsis'),
    ('objectives', r'Course\s+Objectives(?:\s+and\s+Course\s+Summary)?'),
    ('clo', r'Course\s+Learning\s+Outcomes?'),
    ('co_mapping', r'(?:CLO\s*(?:to|[-–>])\s*PLO|CO\s*(?:to|[-–>])\s*PO|Mapping\s*(?:/|of)?\s*Alignment\s+of\s+CLOs?\s+with\s+Program\s+Learning\s+Outcomes?\s*\(PLO\)|Mapping\s+of\s+CLO)'),
    ('content', r'Course\s+Content'),
    ('topic_clo', r'(?:Topic\s*(?:to|[-–>])\s*CLO|CLO\s+Mapping)'),
    ('weekly_plan', r'(?:Class\s+Schedule\s*/\s*Lesson\s+Plan\s*/\s*Weekly\s+plan|Weekly\s+Plan)'),
    ('assessment', r'Assessment\s+Techniques?'),
    ('assessment_strategy', r'Assessment\s+Strateg(?:y|ies)'),
    ('evaluation_policy', r'(?:Evaluation\s+Policy|Evaluation\s+Scheme)'),
    ('weights', r'Weighting\s+(?:COs?|of\s+COs?)'),
    ('books', r'(?:Text\s*Book(?:s)?|Required\s+Reference(?:s|\(s\))?|Recommended\s+Reference(?:s|\(s\))?|Reference\s+Books?(?:\s*&\s*Materials)?)'),
    ('special_instructions', r'Special\s+Instructions?'),
    ('attendance', r'Minimum\s+Attendance'),
    ('grading', r'Grading'),
]
_LABEL_RE = re.compile(r'(?<!\w)(?:(?P<number>\d{1,2})\s*\.\s*)?(?P<label>' + '|'.join('(?:' + p + ')' for _, p in _LABELS) + r')\s*:\s*', re.I)
_PATTERN_FIELDS = [(re.compile(r'^(?:' + p + r')$', re.I), field) for field, p in _LABELS]
_LECTURE = re.compile(r'\bLecture\s+(\d+)\b', re.I)
_WEEK = re.compile(r'\bWeeks?\s+(\d+)(?:\s*[-–]\s*(\d+))?\b', re.I)
_LIST = re.compile(r'(?<!\w)(?:CLO|CO)\s*[-:]?\s*(\d+)\b', re.I)
_SCHEDULE_HEAD = re.compile(
    r'\b(?:Weeks?\s+(?:Lecture\s*#\s+)?Topics?(?:\s*/\s*Content)?|'
    r'Lecture\s+(?:#\s+)?Topics?(?:\s*/\s*Content)?|'
    r'Lecture\s+Schedule\s*(?:\(Tentative\))?\s+Weeks?)\b', re.I)
_SCHEDULE_STOP = re.compile(
    r'\b(?:Part\s+C\b|Weighting\s+COs?\b|Required\s+References?\b|'
    r'Recommended\s+References?\b|Special\s+Instructions?\b|'
    r'Minimum\s+attendance\b|Grading\s+(?:Policy|System)\b|'
    r'\d{1,2}\s*\.\s*Assessment\s+Techniques?\b)', re.I)
_BARE_ROW = re.compile(r'(?<!\w)(\d{1,2}(?:\s*[-–]\s*\d{1,2})?)\s+(?=[A-Z(])')
_WEEK_LECTURE_ROW = re.compile(r'(?<!\w)(\d{1,2})\s+(\d{1,2}(?:\s*,\s*\d{1,2})*)\s+(?=[A-Z(])')


def _field(label: str) -> str | None:
    return next((field for pattern, field in _PATTERN_FIELDS if pattern.fullmatch(label)), None)


def detect_template(first_page: str) -> str:
    compact, _ = _compact(first_page)
    if re.search(r'\bPart\s+A\b', compact, re.I) and re.search(r'\bCourse\s+No\.?\s*/\s*Course\s+Code\s*:', compact, re.I):
        return 'B'
    if re.search(r'\bCourse\s+Outline\b', compact, re.I) and (
        re.search(r'\bProgram\s*:', compact, re.I)
        or re.search(r'\b1\s*\.\s*Course\s+Code\s*:', compact, re.I)
    ):
        return 'A'
    return 'UNKNOWN'


def detect_layout_family(pages: Iterable[str]) -> str:
    """Identify three outline families from document features, not filenames."""
    pages = list(pages)
    if not pages:
        return 'non-course'
    template = detect_template(pages[0])
    if template == 'B':
        return 'B numbered Part A-D'
    if template != 'A':
        return 'non-course'
    all_text = ' '.join(_compact(page)[0] for page in pages)
    if re.search(r'(?<!\w)CLO\s*\d+\b', all_text, re.I):
        return 'A label:value + CLO/PLO (hybrid)'
    return 'A label:value + CO/PO'


def parse_pages(pdf: str, pages: Iterable[str]) -> tuple[str, list[Card]]:
    """Return a template and cards; input pages must be pypdf plain extract_text()."""
    pages = list(pages)
    if not enabled() or not pages:
        return 'OFF', []
    template = detect_template(pages[0])
    if template == 'UNKNOWN':
        return template, []
    course = Path(pdf).stem
    cards: list[Card] = []
    schedule_active = False
    weight_active = False
    for page_number, raw in enumerate(pages, 1):
        flat, offsets = _compact(raw)
        matches = [m for m in _LABEL_RE.finditer(flat) if _field(m.group('label'))]
        for i, m in enumerate(matches):
            field = _field(m.group('label'))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(flat)
            if template == 'A':
                next_lecture = _LECTURE.search(flat, m.end(), end)
                if next_lecture:
                    end = next_lecture.start()
            value = flat[m.end():end].strip(' \t\n:')
            # A label followed immediately by a new label is BLANK, not None.
            if not value:
                value = 'BLANK'
                span = _raw_span(raw, offsets, flat, max(0, m.start() - 90), min(len(flat), m.end() + 90))
            else:
                span = _raw_span(raw, offsets, flat, max(0, m.start() - 90), min(len(flat), end + 90))
            cards.append(Card(course, field, value, pdf, page_number, span, template))
            if field == 'teacher' and value != 'BLANK':
                people = [x.strip() for x in (value.split(';') if ';' in value else value.split(',')) if x.strip()]
                for j, person in enumerate(people, 1):
                    pos = flat.find(person, m.end(), end)
                    if pos >= 0:
                        local_span = _raw_span(raw, offsets, flat, max(0, pos - 80), min(len(flat), pos + len(person) + 80))
                        cards.append(Card(course, 'teacher_person', person, pdf, page_number, local_span, template, list_index=j))
        # Explicit lecture anchors precede content; explicit week anchors follow topics.
        if template == 'A':
            anchors = list(_LECTURE.finditer(flat))
            for i, a in enumerate(anchors):
                end = anchors[i + 1].start() if i + 1 < len(anchors) else len(flat)
                content = flat[a.end():end].strip()
                if content:
                    cards.append(Card(course, 'lecture', content, pdf, page_number,
                                      _raw_span(raw, offsets, flat, a.start(), end), template,
                                      row_key=a.group(1)))
        else:
            anchors = list(_WEEK.finditer(flat))
            previous = 0
            for a in anchors:
                start = max(previous, a.start() - 250)
                topic = flat[start:a.start()].strip()
                if topic:
                    cards.append(Card(course, 'weekly_topic', topic, pdf, page_number,
                                      _raw_span(raw, offsets, flat, start, a.start()), template,
                                      row_key=a.group(0)))
                previous = a.end()
        if template == 'A':
            anchors = list(_WEEK.finditer(flat))
            previous = 0
            for a in anchors:
                start = max(previous, a.start() - 250)
                topic = flat[start:a.start()].strip()
                if topic:
                    cards.append(Card(course, 'weekly_topic', topic, pdf, page_number,
                                      _raw_span(raw, offsets, flat, start, a.start()), template,
                                      row_key=a.group(0)))
                previous = a.end()
        # Bare week/lecture numbers are rows only under a schedule table heading.
        # This excludes Part B topic-alignment tables headed "SL. No".
        heading = _SCHEDULE_HEAD.search(flat)
        if heading:
            schedule_active = True
            schedule_start = heading.end()
        else:
            schedule_start = 0
        if schedule_active:
            stop_match = _SCHEDULE_STOP.search(flat, schedule_start)
            schedule_end = stop_match.start() if stop_match else len(flat)
            region = flat[schedule_start:schedule_end]
            mixed = list(_WEEK_LECTURE_ROW.finditer(region)) if re.search(r'Weeks?\s+Lecture\s*#', flat[max(0, schedule_start-80):schedule_start+80], re.I) else []
            candidates = mixed or list(_BARE_ROW.finditer(region))
            valid: list[tuple[re.Match[str], str]] = []
            for candidate in candidates:
                number = int(re.split(r'\s*[-–]', candidate.group(1))[0])
                if number > 35:
                    continue
                following = region[candidate.end():candidate.end()+350]
                if not re.search(r'(?<!\w)(?:CLO|CO)\s*\d+\b', following, re.I):
                    continue
                kind = 'lecture' if re.search(r'Lecture\s+Topics', flat[max(0, schedule_start-80):schedule_start+80], re.I) else 'week'
                key = (f'week {candidate.group(1)} | lecture {candidate.group(2)}' if mixed else
                       f'{kind} {candidate.group(1)}')
                valid.append((candidate, key))
            for i, (candidate, key) in enumerate(valid):
                start = schedule_start + candidate.start()
                end = schedule_start + (valid[i + 1][0].start() if i + 1 < len(valid) else len(region))
                value = flat[start:end].strip()
                if value:
                    cards.append(Card(course, 'schedule_row', value, pdf, page_number,
                                      _raw_span(raw, offsets, flat, start, end), template,
                                      list_index=i+1, row_key=key))
            if stop_match:
                schedule_active = False
        # Generic numbered book items, preserving each item's full plain-text value.
        book_head = re.search(r'\b(?:Text\s*Book|Required\s+Reference(?:s|\(s\))?|Recommended\s+Reference(?:s|\(s\))?)\s*:?', flat, re.I)
        if book_head:
            section_end = re.search(r'\b(?:Appendix|Special\s+Instructions|Grading\s+Policy|Part\s+[A-D])\b', flat[book_head.end():], re.I)
            stop = book_head.end() + section_end.start() if section_end else len(flat)
            region = flat[book_head.end():stop]
            items = list(re.finditer(r'(?<!\w)(\d{1,2})\s*\.\s+(?=[A-Z“"(])', region))
            for j, item in enumerate(items):
                start = book_head.end() + item.start()
                end = book_head.end() + (items[j + 1].start() if j + 1 < len(items) else len(region))
                value = flat[book_head.end() + item.end():end].strip()
                if value:
                    cards.append(Card(course, 'book_item', value, pdf, page_number,
                                      _raw_span(raw, offsets, flat, max(0, start - 65), min(len(flat), end + 65)),
                                      template, list_index=int(item.group(1))))
        # Assessment weights are bounded by their table/policy heading.  A table may
        # start at the bottom of one page and continue on the next page.
        weight_heads = list(re.finditer(
            r'\b(?:Weighting\s+COs?\s+with\s+Assessment\s+methods|'
            r'Part\s+C\s*[-–]\s*Assessment\s+and\s+Evaluation|'
            r'Evaluation\s+Policy|Assessment\s+Type\s*%\s*[Ww]eight)\b', flat, re.I))
        starts = [h.end() for h in weight_heads] if weight_heads else ([0] if weight_active else [])
        weight_stop = re.compile(
            r'\b(?:Grading\s+(?:Policy|System|Scale)|Course\s+Content|'
            r'Lecture\s+Schedule|Part\s+D|Text\s*Book|Appendix|'
            r'Minimum\s+attendance|Special\s+Instructions?|Student.s\s+responsibilities|'
            r'Late\s+submission|Penalty|Deduction)\b', re.I)
        for j, start_at in enumerate(starts):
            next_head = weight_heads[j + 1].start() if weight_heads and j + 1 < len(weight_heads) else len(flat)
            stop_match = weight_stop.search(flat, start_at, next_head)
            stop = stop_match.start() if stop_match else next_head
            region = flat[start_at:stop]
            for k, pct in enumerate(re.finditer(r'(?<!\w)\d+(?:\.\d+)?\s*%', region), 1):
                before = region[max(0, pct.start()-65):pct.start()]
                after = region[pct.end():pct.end()+30]
                if (re.search(r'\b(?:attendance|deduction|penalty|above|grade|total)\b', before, re.I)
                    or re.match(r'\s+and\s+above\b', after, re.I)):
                    continue
                left = max(0, pct.start()-65)
                column_numbers = re.match(r'(?:\s+[-–]?\d+(?!%)){1,8}', after)
                right = pct.end() + (column_numbers.end() if column_numbers else 0)
                value = region[left:right].strip()
                cards.append(Card(course, 'assessment_weight', value, pdf, page_number,
                                  _raw_span(raw, offsets, flat, start_at+left, start_at+right),
                                  template, list_index=k, row_key=f'weight-{k}'))
            # Some tables omit the percent glyph but have an explicit "% weight" header.
            if not re.search(r'\d\s*%', region) and re.search(r'%\s*[Ww]eight|Weighting\s+COs?', flat[:start_at], re.I):
                rows = list(re.finditer(r'(?<!\w)([A-Za-z][A-Za-z /&()+-]{3,60})\s+(\d{1,3})(?=\s|$)', region))
                for k, row in enumerate(rows, 1):
                    value = row.group(0).strip()
                    if re.search(r'\b(?:attendance|deduction|penalty|grade|total)\b', value, re.I):
                        continue
                    cards.append(Card(course, 'assessment_weight', value, pdf, page_number,
                                      _raw_span(raw, offsets, flat, start_at+row.start(), start_at+row.end()),
                                      template, list_index=k, row_key=f'weight-{k}'))
            if stop_match:
                weight_active = False
            else:
                weight_active = True
        # A CO→PO or B CLO→PLO mapping row is kept as one ordered link card.
        map_head = re.search(r'\b(?:CO\s+No\.?|Mapping\s*/\s*Alignment\s+of\s+CLOs?\s+with\s+Program\s+Learning\s+Outcomes?)', flat, re.I)
        if map_head:
            limit_match = re.search(r'\b(?:Weighting\s+COs?|Part\s+[BCD]|\d{1,2}\.\s+(?:Course\s+Content|Class\s+Schedule))\b', flat[map_head.end():], re.I)
            stop = map_head.end() + limit_match.start() if limit_match else len(flat)
            region = flat[map_head.end():stop]
            links = list(re.finditer(r'(?<!\w)(CLO|CO)\s*(\d+)\b', region, re.I))
            for j, link in enumerate(links):
                start = map_head.end() + link.start()
                end = map_head.end() + (links[j + 1].start() if j + 1 < len(links) else len(region))
                value = flat[start:end].strip()
                if value and (re.search(r'\b(?:PLO|PO)\s*\d+\b|\b\d+\s*/\s*(?:Apply|Understand|Analyze|Create|Evaluate|Remember)\b', value, re.I) or re.search(r'\b(?:CLO\s*\d+\s+[a-z]\b|CO\s*\d+.+?\b\d+\s+Cognitive)', value, re.I)):
                    field = 'clo_plo_link' if link.group(1).upper() == 'CLO' else 'co_po_link'
                    cards.append(Card(course, field, value, pdf, page_number,
                                      _raw_span(raw, offsets, flat, max(0, start - 60), min(len(flat), end + 60)),
                                      template, list_index=int(link.group(2)), row_key=link.group(0)))
        # Keep CO/CLO ordering with one card per numbered statement fragment.
        rows = list(_LIST.finditer(flat))
        for i, row in enumerate(rows):
            end = rows[i + 1].start() if i + 1 < len(rows) else min(len(flat), row.end() + 450)
            value = flat[row.end():end].strip()
            if value:
                cards.append(Card(course, 'clo' if row.group(0).upper().startswith('CLO') else 'co',
                                  value, pdf, page_number, _raw_span(raw, offsets, flat, row.start(), end),
                                  template, list_index=int(row.group(1)), row_key=row.group(0)))
    return template, cards
