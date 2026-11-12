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
    ('co_mapping', r'(?:CLO\s*(?:to|[-–>])\s*PLO|CO\s*(?:to|[-–>])\s*PO|Mapping\s+of\s+CLO)'),
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


def _field(label: str) -> str | None:
    return next((field for pattern, field in _PATTERN_FIELDS if pattern.fullmatch(label)), None)


def detect_template(first_page: str) -> str:
    compact, _ = _compact(first_page)
    if re.search(r'\bPart\s+A\b', compact, re.I) and re.search(r'\bCourse\s+(?:No\.?\s*/\s*)?Code\s*:', compact, re.I):
        return 'B'
    if re.search(r'\bCourse\s+Outline\b', compact, re.I) and re.search(r'\bProgram\s*:', compact, re.I):
        return 'A'
    return 'UNKNOWN'


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
                span = _raw_span(raw, offsets, flat, m.start(), m.end())
            else:
                span = _raw_span(raw, offsets, flat, m.end(), end)
            cards.append(Card(course, field, value, pdf, page_number, span, template))
        # Row anchors: A lecture content follows its anchor; B weekly topic precedes it.
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
