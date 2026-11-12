"""Two-tier course cards from plain PDF text; no answer-path integration.

The block tier is the existing r2 parser. Atomic supplements use generic
section and table patterns. Every quoted value is re-anchored to plain text.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .course_cards import enabled, parse_pages
from .course_cards_atomic import parse_atomic_pages

DROP = dict.fromkeys(map(ord, '\u200b\u200c\u200d\u2060\ufeff\u00ad'))
REPL = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-', '−': '-'})
WORD = re.compile(r'[\w%]+', re.U)
ORDINAL = {'st', 'nd', 'rd', 'th'}
SECTION = re.compile(r'(?<!\w)(?:Part\s+[A-D]\b|(?:1[1-9]|20|21)\s*\.\s*(?:Course\s+Content|Alignment|Class\s+Schedule|Teaching|Assessment|Evaluation|Text\s*Book|Mapping)|Course\s+(?:Objectives|Outcomes|Content\s+Outline|Synopsis)|Weighting\s+COs|Special\s+Instructions|Required\s+References|Recommended\s+References|Prepared\s+by|Appendix\s*[-–]?\s*\d*)\b', re.I)
ROW_END = re.compile(r'(?<!\w)(?:CLO|CO)\s*\d+(?:\s*,\s*(?:CLO|CO)?\s*\d+)*\b', re.I)
WEEK = re.compile(r'\bWeeks?\s+\d+(?:\s*[-–]\s*\d+)?\b', re.I)
LECTURE = re.compile(r'\bLecture\s+\d+\b', re.I)


def compact(text: str) -> str:
    return ' '.join(text.split())


def _tokens(text: str) -> list[tuple[str, int, int]]:
    found = []
    for match in WORD.finditer(text):
        word = unicodedata.normalize('NFKC', match.group()).translate(DROP).translate(REPL).casefold()
        word = ''.join(x for x in word if x.isalnum() or x == '%')
        if word:
            if found and word in ORDINAL and found[-1][0].isdigit():
                prior, start, _ = found.pop()
                found.append((prior + word, start, match.end()))
            else:
                found.append((word, match.start(), match.end()))
    return found


def _locate(value: str, pages: list[str]) -> tuple[int, int, int, list[int]] | None:
    target = [x[0] for x in _tokens(value)]
    if not target:
        return None
    for page, raw in enumerate(pages, 1):
        tokens = _tokens(raw)
        words = [x[0] for x in tokens]
        first = target[0]
        for i, word in enumerate(words):
            if word == first and words[i:i + len(target)] == target:
                return page, tokens[i][1], tokens[i + len(target) - 1][2], [page]
    # A list or table row can continue across one physical page boundary.
    for i in range(len(pages) - 1):
        left, right = _tokens(pages[i]), _tokens(pages[i + 1])
        joined = [x[0] for x in left + right]
        for j in range(max(0, len(left) - len(target)), len(left)):
            if joined[j:j + len(target)] == target and j + len(target) > len(left):
                return i + 1, left[j][1], right[j + len(target) - len(left) - 1][2], [i + 1, i + 2]
    return None


def _anchor(card: dict, pages: list[str]) -> dict | None:
    if card.get('empty_section'):
        card = dict(card, tier='atomic', span_status='empty_section', char_start=None, char_end=None)
        return card
    value = str(card.get('value', ''))
    hit = _locate(value, pages)
    if hit is None:
        return None
    page, start, end, span_pages = hit
    # Prefer the card's stated page if the same short value occurs repeatedly.
    if card.get('page') != page:
        same = _locate(value, [pages[card['page'] - 1]]) if 1 <= card.get('page', 0) <= len(pages) else None
        if same is not None:
            page, start, end, span_pages = card['page'], same[1], same[2], [card['page']]
    raw = pages[page - 1]
    tokens = _tokens(raw)
    left = max(0, next((i for i, x in enumerate(tokens) if x[1] <= start < x[2]), 0) - 20)
    right = min(len(tokens), next((i for i, x in enumerate(tokens) if x[1] < end <= x[2]), len(tokens) - 1) + 21)
    context_start = tokens[left][1]
    context_end = tokens[right - 1][2]
    # The short citation context stays within a section. The exact value is
    # preserved even when a heading pattern occurs inside a table cell.
    headings = list(SECTION.finditer(raw))
    for heading in headings:
        if context_start < heading.start() < start:
            context_start = heading.start()
        if end < heading.start() < context_end:
            context_end = heading.start()
            break
    updated = dict(card, tier='atomic', page=page, pages=span_pages,
                   char_start=start, char_end=end,
                   plain_span=raw[context_start:context_end], span_status='found')
    return updated


def _new(pdf: str, page: int, field: str, value: str, **extra) -> dict:
    value = value.strip(' \t\r\n:;')
    return dict(course=Path(pdf).stem, pdf=pdf, page=page, pages=[page],
                field=field, value=value, card_id=hashlib.sha256(f'{pdf}|{page}|{field}|{value}'.encode()).hexdigest()[:20],
                text=f'{Path(pdf).stem} | {field} | {value} | p{page}', **extra)


def _section(flat: str, head: str, stop: str) -> str:
    start = re.search(head, flat, re.I)
    if not start:
        return ''
    end = re.search(stop, flat[start.end():], re.I)
    return flat[start.end():start.end() + end.start() if end else len(flat)].strip()


def _sentences(text: str) -> list[str]:
    # A semicolon also separates topic segments in course synopses.
    parts = re.split(r'(?<=[.!?;])\s+(?=[A-Z(\d])', text)
    return [x.strip() for x in parts if x.strip()]


def _supplement(pdf: str, pages: list[str]) -> list[dict]:
    out = []
    flats = [compact(p) for p in pages]
    for page, flat in enumerate(flats, 1):
        # Section prose, including sections that begin at the top of the next
        # page, is split into short verbatim sentences or topic segments.
        for field, head, stop in [
            ('content', r'\b(?:14\s*\.\s*)?Course\s+Content\s*:', r'\b15\s*\.\s*Alignment|\bPart\s+C\b'),
            ('synopsis', r'\bCourse\s+Synopsis\s*:', r'\bCourse\s+Objectives\s*:'),
            ('teaching_strategy', r'\b17\s*\.\s*Teaching\s*[-–]?\s*Learning\s+Strategies\s*:', r'\b18\s*\.\s*Assessment')]:
            body = _section(flat, head, stop)
            for index, sentence in enumerate(_sentences(body), 1):
                if len(sentence.split()) <= 60:
                    out.append(_new(pdf, page, field, sentence, list_index=index))
        # Mid-row week anchors: the previous CO/CLO terminator is the start
        # boundary and the next CO/CLO terminator is the end boundary.
        if re.search(r'\b(?:Class\s+Schedule|Weekly\s+plan|Topics\s+Specific\s+Outcome)', flat, re.I):
            region = _section(flat, r'\b(?:16\s*\.\s*)?Class\s+Schedule[^:]*:', r'\b17\s*\.\s*Teaching|\b18\s*\.\s*Assessment|\bPart\s+C') or flat
        else:
            region = flat if WEEK.search(flat) and ROW_END.search(flat) else ''
        anchors = list(WEEK.finditer(region))
        ends = list(ROW_END.finditer(region))
        prior_end = 0
        for anchor in anchors:
            terminator = next((x for x in ends if x.start() > anchor.end()), None)
            if terminator is None:
                continue
            before = region[prior_end:anchor.start()].strip()
            after = region[anchor.end():terminator.start()].strip()
            # Keep the topic at the start of the row; outcomes are bullet
            # clauses between the topic and the week anchor.
            topic = re.split(r'\s+[•●▪]', before, maxsplit=1)[0].strip()
            if not topic:
                prior_end = terminator.end()
                continue
            strategy = ''
            strategy_start = re.search(r'\b(?:Lecture|Group\s+work|Presentation|Discussion)\b', after, re.I)
            activities = after[:strategy_start.start()].strip() if strategy_start else after
            if strategy_start:
                strategy = after[strategy_start.start():].strip()
            out.append(_new(pdf, page, 'schedule', topic, row_key=anchor.group(), topic=topic,
                            outcomes=[x.strip() for x in re.split(r'[•●▪]',before)[1:] if x.strip()],
                            activities=activities, teaching_strategy=strategy,
                            co_clo_refs=terminator.group(), clo_refs=terminator.group(), reading=''))
            prior_end = terminator.end()
        # A lecture anchor begins its row; the first outcome-reference list
        # closes the topic cell before delivery and reading columns.
        if re.search(r'\b(?:Course\s+Content\s+Outline|Lecture\s+Topic)',flat,re.I) or (LECTURE.search(flat) and ROW_END.search(flat)):
            lecture_anchors = list(LECTURE.finditer(flat))
            for i, anchor in enumerate(lecture_anchors):
                end = lecture_anchors[i+1].start() if i+1 < len(lecture_anchors) else len(flat)
                row = flat[anchor.end():end]
                reference = ROW_END.search(row)
                if reference:
                    topic = row[:reference.start()].strip()
                    if topic:
                        out.append(_new(pdf,page,'schedule',topic,row_key=anchor.group(),topic=topic,
                                        co_clo_refs=reference.group(),clo_refs=reference.group(),
                                        outcomes=[],activities='',teaching_strategy='',reading=''))
        for exam in re.finditer(r'\b(?:Mid\s*[-–]?\s*Term|Midterm|Final)\s+Examination\b', flat, re.I):
            out.append(_new(pdf,page,'exam_event',exam.group(),row_key=exam.group()))
        # Numbered alignment and assessment-technique rows are table rows,
        # never CO/CLO statement cards.
        for field, head, stop in [
            ('topic_clo',r'\b15\s*\.\s*Alignment[^:]*:',r'\b16\s*\.\s*Class\s+Schedule'),
            ('assessment_technique',r'\b18\s*\.\s*Assessment\s+Techniques?[^:]*:',r'\bPart\s+C')]:
            body = _section(flat,head,stop)
            matches = list(re.finditer(r'(?<!\w)(\d{1,2})\s+(?=[A-Z])',body))
            for i, match in enumerate(matches):
                row = body[match.end():matches[i+1].start() if i+1<len(matches) else len(body)].strip()
                if field=='topic_clo':
                    refs=list(ROW_END.finditer(row));last=refs[-1] if refs else None
                    value=row[:refs[0].start()].strip() if refs else row
                    extra=dict(co_clo_refs=last.group(),clo_refs=last.group()) if last else {}
                else:
                    # The final assessment-cell label follows the topic text.
                    split=re.search(r'\b(?:Lab\s+Task|Lab\s+Evaluation|Continuous\s+Project\s+Evaluation|Project\s+Report|Presentation\s+Followed\s+by\s+Viva)\b',row,re.I)
                    value=row[split.start():].strip() if split else row
                    extra=dict(topic=row[:split.start()].strip()) if split else {}
                if value and len(value.split())<=80:out.append(_new(pdf,page,field,value,row_key=f'topic {match.group(1)}',**extra))
        # Strategy paragraphs have a label and a prose body.
        strategy = _section(flat,r'\b19\s*\.\s*Assessment\s+Strategy\b',r'\b20\s*\.\s*Evaluation\s+Policy')
        labels=list(re.finditer(r'\b(?:Lab\s+Exercises|Mini\s+Projects|Final\s+Project)\s*:',strategy,re.I))
        for i,label in enumerate(labels):
            body=strategy[label.end():labels[i+1].start() if i+1<len(labels) else len(strategy)].strip()
            body=re.split(r'\b(?:Lab\s+Exercises\s+and\s+Mini\s+Projects|Final\s+Project)\s*\(\d+\s+marks\)',body,maxsplit=1,flags=re.I)[0].strip()
            if body:out.append(_new(pdf,page,'assessment_strategy',body,row_key=label.group().rstrip(': ')))
        # Policy list items and grade-scale rows have their own fields.
        policy=_section(flat,r'\b20\s*\.\s*Evaluation\s+Policy\b',r'\bUAP\s+Grading\s+Policy|\bPart\s+D')
        items=list(re.finditer(r'(?<!\w)\d+\s*\.\s*[A-Z][A-Za-z ]+\d+(?:\.\d+)?%',policy))
        for item in items:out.append(_new(pdf,page,'evaluation_policy',item.group(),is_weight=True))
        grading=_section(flat,r'\b(?:UAP\s+)?Grading\s+Policy\s+Numeric\s+Grade',r'\bPart\s+D|\bAppendix')
        for row in re.finditer(r'\d+%\s+(?:and\s+above|to\s+less\s+than\s+\d+%)\s+[A-F][+-]?\s+\d+\.\d+',grading,re.I):
            out.append(_new(pdf,page,'grading_scale',row.group(),is_weight=False))
        # Appendix PLO items and PO numbered tables.
        appendix=_section(flat,r'\bAppendix\s*[-–]?\s*\d*\s*:\s*Washington\s+Accord[^:]*:',r'\bAppendix\s*[-–]?\s*\d+\b')
        letters=list(re.finditer(r'\(([a-z])\)\s+',appendix,re.I))
        for i,letter in enumerate(letters):
            value=appendix[letter.end():letters[i+1].start() if i+1<len(letters) else len(appendix)].strip()
            if value and len(value.split())<=80:out.append(_new(pdf,page,'plo_definition',value,row_key=f'PLO ({letter.group(1)})'))
        if re.search(r'\bNo\.?\s+PO\s+Differentiating\s+Characteristic\b',appendix,re.I):
            body=re.split(r'\bNo\.?\s+PO\s+Differentiating\s+Characteristic\b',appendix,maxsplit=1,flags=re.I)[-1]
            numbers=list(re.finditer(r'(?<!\w)(\d{1,2})\s+(?=[A-Z])',body))
            for i,number in enumerate(numbers):
                row=body[number.end():numbers[i+1].start() if i+1<len(numbers) else len(body)].strip()
                if row and len(row.split())<=80:out.append(_new(pdf,page,'po_definition',row,row_key=f'PO {number.group(1)}'))
        prepared=_section(flat,r'\bPrepared\s+by\b',r'\bAppendix|\bChecked\s+by')
        if prepared:out.append(_new(pdf,page,'prepared_by',prepared))
        # Assessment tables: first number after a row label is the weight;
        # later percentages are CO/CLO cells, never second weights.
        weight_region=_section(flat,r'\b(?:Assessment\s+Type\s*%\s*Weight|Weighting\s+COs\s+with\s+Assessment\s+methods\s*:)',r'\b(?:Grading\s+Policy|Course\s+Content\s+Outline|19\s*\.\s*Assessment\s+Strategy|Part\s+D)')
        if weight_region:
            row_pattern=re.compile(r'(?<!\w)([A-Z][A-Za-z&()+\-/ ]{2,65}?)\s+(\d{1,3}(?:\.\d+)?%)(?=\s|$)')
            rows=list(row_pattern.finditer(weight_region))
            for i,row in enumerate(rows):
                label=row.group(1).strip();weight=row.group(2)
                if re.search(r'\b(?:Attendance|Deduction|Penalty|Above|Grade)\b',label,re.I):continue
                field='weight_total' if label.lower()=='total' else 'weight'
                tail=weight_region[row.end():rows[i+1].start() if i+1<len(rows) else len(weight_region)]
                cell_values=re.findall(r'(?<!\w)(?:\d+%?|[–-])(?=\s|$)',tail)
                out.append(_new(pdf,page,field,weight,assessment=label,row_key=label,weight=weight,
                                weight_display=weight,derivation='verbatim',
                                co_clo_cells={f'outcome {j+1}':v for j,v in enumerate(cell_values)}))
    return out


def _segments(flats: list[str], head: str, stop: str):
    """Yield page-local portions of a section that may continue across pages."""
    active = False
    for page, flat in enumerate(flats, 1):
        start = re.search(head, flat, re.I)
        if start:
            active = True
            body_start = start.end()
        elif active:
            body_start = 0
        else:
            continue
        end = re.search(stop, flat[body_start:], re.I)
        body_end = body_start + end.start() if end else len(flat)
        yield page, flat[body_start:body_end].strip()
        if end:active = False


def _gap_fill(pdf: str, pages: list[str]) -> list[dict]:
    """Add short facts from continuation pages and missing table families."""
    out=[];flats=[compact(p) for p in pages]
    section_specs=[
        ('synopsis',r'\bCourse\s+Synopsis\s*:',r'\bCourse\s+Objectives\s*:'),
        ('content',r'\b(?:14\s*\.\s*)?Course\s+Content\s*:',r'\b15\s*\.\s*Alignment|\b16\s*\.\s*Class\s+Schedule|\bPart\s+C'),
        ('teaching_strategy',r'\b17\s*\.\s*Teaching\s*[-–]?\s*Learning\s+Strategies\s*:',r'\b18\s*\.\s*Assessment'),
    ]
    for field,head,stop in section_specs:
        index=0
        for page,body in _segments(flats,head,stop):
            for sentence in _sentences(body):
                for part in re.split(r'(?<=[;])\s+',sentence):
                    part=part.strip()
                    if part and len(part.split())<=60:
                        index+=1;out.append(_new(pdf,page,field,part,list_index=index))
    # Numbered objectives continue on the following page before the next
    # section heading; values do not include their numeric list marker.
    for page,body in _segments(flats,r'\b(?:11\s*\.\s*)?Course\s+Objectives(?:\s+and\s+Course\s+Summary)?\s*:',
                               r'\b(?:12\s*\.\s*)?Course\s+(?:Learning\s+Outcomes|Outcomes\s*\(CO\))'):
        items=list(re.finditer(r'(?<!\w)(\d{1,2})\s*\.\s+(?=[A-Z])',body))
        for i,item in enumerate(items):
            value=body[item.end():items[i+1].start() if i+1<len(items) else len(body)].strip()
            if value and len(value.split())<=80:out.append(_new(pdf,page,'objective',value,list_index=int(item.group(1))))
    # Section 18 can start at a page bottom with rows on the next page.
    for page,body in _segments(flats,r'\b18\s*\.\s*Assessment\s+Techniques?[^:]*:',r'\bPart\s+C'):
        if re.search(r'\bSL\.?\s*No\.?\b',body,re.I):
            body=re.split(r'\bSL\.?\s*No\.?\b[^0-9]*',body,maxsplit=1,flags=re.I)[-1]
        items=list(re.finditer(r'(?<!\w)(\d{1,2})\s+(?=[A-Z])',body))
        for i,item in enumerate(items):
            row=body[item.end():items[i+1].start() if i+1<len(items) else len(body)].strip()
            cell=re.search(r'\b(?:Lab\s+Task|Lab\s+Evaluation|Continuous\s+Project\s+Evaluation|Project\s+Report|Presentation\s+Followed\s+by\s+Viva)\b',row,re.I)
            if cell:
                out.append(_new(pdf,page,'assessment_technique',row[cell.start():].strip(),
                                row_key=f'topic {item.group(1)}',topic=row[:cell.start()].strip()))
    # Appendix items may span multiple pages. A parenthesized letter is the
    # row anchor; a numbered PO table uses its own item number.
    for page,body in _segments(flats,r'\bAppendix\s*[-–]?\s*\d*\s*:\s*Washington\s+Accord[^:]*:',
                               r'\bAppendix\s*[-–]?\s*\d+\b'):
        letters=list(re.finditer(r'\(([a-z])\)\s+',body,re.I))
        for i,item in enumerate(letters):
            value=body[item.end():letters[i+1].start() if i+1<len(letters) else len(body)].strip()
            if value and len(value.split())<=80:
                out.append(_new(pdf,page,'plo_definition',value,row_key=f'PLO ({item.group(1)})'))
    # Column headings identify course-level Ks/Ps/As cells, never a CLO row.
    for page,flat in enumerate(flats,1):
        profile=re.search(r'\bKs\s+Ps\s+As\b',flat,re.I)
        if profile:
            region=flat[profile.end():profile.end()+700]
            for marker,next_marker in [('K','P'),('P','A'),('A',None)]:
                match=re.search(r'(?<!\w)'+marker+r'\d+(?:\s*,\s*'+marker+r'\d+)+',region,re.I)
                if match:
                    out.append(_new(pdf,page,'knowledge_profile',match.group(),row_key={'K':'Ks','P':'Ps','A':'As'}[marker],course_level=True))
        # Total rows on continuation pages preserve all column totals.
        total=re.search(r'(?<!\w)Total\s+(100%|\d+(?:\.\d+)?%)\s+((?:\d+\s+){2,10}\d+)(?=\s+\d{1,2}\s*\.\s*[A-Z]|\s*$)',flat,re.I)
        if total:
            nums=re.findall(r'\d+',total.group(2))
            out.append(_new(pdf,page,'weight_total',total.group(1),row_key='Total',assessment='Total',
                            weight_display=total.group(1),clo_totals={f'CLO {i+1}':int(v) for i,v in enumerate(nums)}))
        # Grading rows are never assessment weights.
        grading=re.search(r'\b(?:UAP\s+)?Grading\s+Policy\b',flat,re.I)
        if grading or re.match(r'\s*\d+%\s+and\s+above',flat,re.I):
            for item in re.finditer(r'\b(?:\d+%\s+(?:and\s+above|to\s+less\s+than\s+\d+%)|Less\s+than\s+\d+%)\s+[A-F][+-]?\s+\d+\.\d+',flat,re.I):
                out.append(_new(pdf,page,'grading_scale',item.group(),is_weight=False))
        # A Prepared by label may be followed by blank signature columns;
        # the listed names start after the approval column header.
        prep=re.search(r'\bPrepared\s+by\s+Checked\s+by\s+Approved\s+by\s+(.+?)(?=\bAppendix|$)',flat,re.I)
        if prep and prep.group(1).strip():out.append(_new(pdf,page,'prepared_by',prep.group(1).strip(),row_key='Prepared by'))
        # Marks-split rows have a Bloom category followed by numeric cells.
        for header in re.finditer(r'\b(?:Lab\s+Exercises\s+and\s+Mini\s+Projects|Final\s+Project)\s*\(\d+\s+marks\)',flat,re.I):
            tail=flat[header.end():]
            next_header=re.search(r'\b(?:Final\s+Project\s*\(\d+\s+marks\)|20\s*\.\s*Evaluation\s+Policy)',tail,re.I)
            region=tail[:next_header.start() if next_header else len(tail)]
            cols=re.search(r'\bBloom.s\s+Category\s+(.+?)\s+\d+\s*/\s*(?:Apply|Analyze|Valuing|Create)\b',region,re.I)
            labels=re.findall(r'(?<!\w)(?:\d+\s*/\s*(?:Apply|Analyze|Valuing|Create))\s+(?:\d+|[–-])(?:\s+(?:\d+|[–-])){1,6}',region,re.I)
            for row in labels:
                bloom=re.match(r'\d+\s*/\s*\w+',row)
                values=row[bloom.end():].strip() if bloom else ''
                if values:out.append(_new(pdf,page,'marks_split',values,row_key=header.group()+' / '+bloom.group(),cells={}))
        # Numbered evaluation-policy items are distinct from the assessment
        # table and its weight cells.
        policy=_section(flat,r'\b20\s*\.\s*Evaluation\s+Policy\b',r'\bUAP\s+Grading\s+Policy|\bPart\s+D')
        for item in re.finditer(r'(?<!\w)\d+\s*\.\s*([A-Z][A-Za-z ]+\d+(?:\.\d+)?%)',policy):
            out.append(_new(pdf,page,'evaluation_policy',item.group(1).strip(),row_key=item.group(1).rsplit(' ',1)[0],is_weight=True))
    return out


def _weight_cell_hints(card: dict, layout: str, plain: str) -> dict:
    """Map percentage cells to outcome columns, verifying each in plain text."""
    if card.get('field')!='weight' or not card.get('assessment'):
        return card
    lines=layout.splitlines()
    head=next((line for line in lines if re.search(r'Assessment\s+Type',line,re.I) and re.search(r'%\s*weight|Marks',line,re.I)),None)
    if head is None:return card
    columns=[(m.group().replace(' ',''),m.start()) for m in re.finditer(r'\b(?:CLO|CO)\s*\d+\b',head,re.I)]
    if not columns:return card
    label=' '.join(str(card['assessment']).split()).casefold()
    for line in lines:
        if not line.strip().casefold().startswith(label):continue
        percents=list(re.finditer(r'\d+(?:\.\d+)?\s*%',line))
        if not percents:continue
        if card.get('weight_display') and percents[0].group().replace(' ','')!=card['weight_display']:
            continue
        cells={}
        for pct in percents[1:]:
            if pct.group() not in plain:continue
            eligible=[name for name,start in columns if start<=pct.start()]
            if eligible:cells[eligible[-1]]=pct.group().replace(' ','')
        if cells:return dict(card,co_clo_cells=cells)
    return card


def _weight_display(card: dict) -> dict:
    value=str(card.get('weight_display',''))
    if card.get('field') not in {'weight','weight_total'} or not value:return card
    match=re.fullmatch(r'(\d+)\.0+%',value)
    if match:return dict(card,weight_display=match.group(1)+'%',derivation='numeric display from verbatim percentage')
    return card


def parse_tiered_pages(pdf: str, pages: Iterable[str], layouts: Iterable[str] | None = None) -> tuple[str, list[dict], list[dict]]:
    pages=list(pages)
    if not enabled() or not pages:return 'OFF',[],[]
    template, blocks=parse_pages(pdf,pages)
    if template=='UNKNOWN':return template,[],[]
    _, atomic=parse_atomic_pages(pdf,pages)
    raw_atomic=atomic+_supplement(pdf,pages)+_gap_fill(pdf,pages)
    if layouts is not None:
        from .course_cards_mapping import mapping_cards
        column_cards,_ = mapping_cards(pdf,pages,list(layouts))
        raw_atomic += column_cards
    from .course_cards import match_text
    from .course_cards_statements import _source_region, statement_rejection
    statement_region=_source_region(template,pages)
    mapping_values={(c.get('page'),match_text(str(c.get('value','')))) for c in raw_atomic if c.get('field') in {'clo_plo_link','co_po_link'}}
    kept=[];rejected=[];seen=set()
    for card in raw_atomic:
        if card.get('field') in {'co','clo'} and re.search(r'\b(?:Part\s+[A-D]|\d{1,2}\s*\.\s*(?:Class\s+Schedule|Mapping|Assessment|Course\s+Content)|Weighting\s+COs|Appendix)\b',str(card.get('value','')),re.I):
            rejected.append(dict(card,tier='atomic',span_status='statement_rejected',reject_reason='section_or_table_heading'))
            continue
        if card.get('field') in {'co','clo'}:
            reason=statement_rejection(card,template,pages,statement_region)
            if reason is None and (card.get('page'),match_text(str(card.get('value','')))) in mapping_values:
                reason='duplicate_mapping_link'
            if reason is not None:
                rejected.append(dict(card,tier='atomic',span_status='statement_rejected',reject_reason=reason))
                continue
        if card.get('field')=='schedule' and str(card.get('row_key','')).lower().startswith('week') and any(LECTURE.search(p) for p in pages) and not any(WEEK.search(p) for p in pages):
            continue
        card=_weight_display(card)
        if layouts is not None and card.get('field')=='weight' and 1<=card.get('page',0)<=len(pages):
            card=_weight_cell_hints(card,list(layouts)[card['page']-1],pages[card['page']-1])
        anchored=_anchor(card,pages)
        if anchored is None:
            rejected.append(dict(card,span_status='not_found',tier='atomic'))
            continue
        key=(anchored['pdf'],tuple(anchored['pages']),anchored['field'],
             ' '.join(x[0] for x in _tokens(anchored['value'])))
        if key in seen:continue
        seen.add(key);kept.append(anchored)
    block_records=[dict(c.record(),tier='block') for c in blocks]
    return template,block_records+kept,rejected
