"""Opt-in atomic course facts from plain-extracted PDF pages.

Gold cards and dataset answers are never inputs to this module.  Layout extraction
is not used.  The original plain text is kept in every citation span.
"""
from __future__ import annotations
import hashlib,os,re
from pathlib import Path
from typing import Iterable
from .course_cards import enabled,match_text,parse_pages

SECTION = re.compile(r'(?<!\w)(?:Part\s+[A-D]\b|\d{1,2}\s*\.\s*(?:Course\s+Objectives|Course\s+Outcomes|Mapping|Assessment|Class\s+Schedule|Evaluation\s+Policy|Text\s+Book)|Course\s+Objectives|Course\s+Outcomes\s*\(CO\)|Weighting\s+COs|Course\s+Content\s+Outline|Special\s+Instructions?|Required\s+References?|Recommended\s+References?|Prepared\s+by|Appendix\s*[-–]?\s*\d*)\b',re.I)
TABLE_HEAD = re.compile(r'(?<!\w)(?:CO\s+No\.|CLO\s+No\.|Assessment\s+Type\s*%\s*[Ww]eight|Weeks?\s+(?:Lecture\s*#\s+)?Topics?|SL\.?\s*No\.?)\b',re.I)
TOKEN = re.compile(r'[\w%]+',re.U)
LIST_ITEM = re.compile(r'(?<!\w)(\d{1,2})\s*\.\s+(?=[A-Z“"(])')
BULLET = re.compile(r'[•●▪∙]\s*')
CO_ID = re.compile(r'(?<!\w)(CLO|CO)\s*(\d+)\b',re.I)
PO_ID = re.compile(r'(?<!\w)(PLO|PO)\s*\(?([A-Za-z]|\d+)\)?\b',re.I)
WEIGHT_HEAD = re.compile(r'\b(?:Weighting\s+COs?\s+with\s+Assessment\s+methods|Part\s+C\s*[-–]\s*Assessment\s+and\s+Evaluation|Evaluation\s+Policy|Assessment\s+Type\s*%\s*[Ww]eight)\b',re.I)
WEIGHT_STOP = re.compile(r'\b(?:Grading\s+(?:Policy|System|Scale)|Course\s+Content|Lecture\s+Schedule|Part\s+D|Text\s*Book|Appendix|Minimum\s+attendance|Special\s+Instructions?|Late\s+submission|Penalty|Deduction|Prepared\s+by)\b',re.I)
SCHEDULE_HEAD = re.compile(r'\b(?:Weeks?\s+(?:Lecture\s*#\s+)?Topics?(?:\s*/\s*Content)?|Lecture\s+(?:#\s+)?Topics?(?:\s*/\s*Content)?|Lecture\s+Schedule\s*(?:\(Tentative\))?\s+Weeks?)\b',re.I)
SCHEDULE_STOP = re.compile(r'\b(?:Part\s+C|Weighting\s+COs?|Required\s+References?|Recommended\s+References?|Special\s+Instructions?|Minimum\s+attendance|Grading\s+(?:Policy|System)|\d{1,2}\s*\.\s*Assessment\s+Techniques?)\b',re.I)


def compact(raw: str) -> str:
    return ' '.join(raw.split())


def words(s: str) -> list[str]:
    return TOKEN.findall(s)


def clean_value(s: str) -> str:
    return s.strip(' \t\n\r:;')


def boundary(s: str) -> str:
    """Trim at the next section/table heading, never through it."""
    matches=[m.start() for pattern in (SECTION,TABLE_HEAD) for m in pattern.finditer(s) if m.start()>0]
    return s[:min(matches)] if matches else s


def split_items(s: str) -> list[tuple[int|None,str]]:
    s=boundary(s)
    numbered=list(LIST_ITEM.finditer(s))
    if numbered:
        return [(int(m.group(1)),clean_value(s[m.end():numbered[i+1].start() if i+1<len(numbered) else len(s)])) for i,m in enumerate(numbered)]
    bullets=list(BULLET.finditer(s))
    if bullets:
        return [(i+1,clean_value(s[m.end():bullets[i+1].start() if i+1<len(bullets) else len(s)])) for i,m in enumerate(bullets)]
    sentences=[clean_value(x) for x in re.split(r'(?<=[.!?;])\s+(?=[A-Z“])',s) if clean_value(x)]
    return [(i+1,item) for i,item in enumerate(sentences)]


def source_span(raw: str,value: str) -> str:
    """Find the compacted value in plain text and return a local original span."""
    flat=compact(raw);at=flat.find(value)
    if at<0:return raw[:min(len(raw),300)]
    # Re-find first and last words in raw while preserving original whitespace.
    fragment=flat[max(0,at-50):min(len(flat),at+len(value)+50)]
    parts=fragment.split()
    if not parts:return ''
    first=re.search(re.escape(parts[0]),raw,re.I)
    if first is None:return raw[:min(len(raw),300)]
    last=list(re.finditer(re.escape(parts[-1]),raw[first.start():],re.I))
    return raw[first.start():first.start()+last[0].end()] if last else raw[first.start():first.start()+min(len(raw),300)]


def add(out:list[dict],pdf:str,page:int,raw:str,field:str,value:str,**extra) -> None:
    value=clean_value(value)
    if not value or not any(c.isalnum() for c in value):return
    if len(words(value))>80:
        # Keep intact table rows only; long prose is divided at sentence breaks.
        if field not in {'schedule','topic_clo','assessment_technique','marks_split'}:
            for _,part in split_items(value):
                if part and part!=value:add(out,pdf,page,raw,field,part,**extra)
            return
    course=Path(pdf).stem
    record=dict(course=course,field=field,value=value,pdf=pdf,page=page,pages=[page],
                plain_span=source_span(raw,value),**extra)
    record['card_id']=hashlib.sha256(f'{pdf}|{page}|{field}|{value}'.encode()).hexdigest()[:20]
    record['text']=f'{course} | {field} | {value} | p{page}'
    out.append(record)


def _co_cells(out:list[dict],pdf:str,page:int,raw:str,card) -> None:
    text=boundary(card.value);idmatch=CO_ID.search((card.row_key or '')+' '+text)
    if idmatch is None:return
    prefix=idmatch.group(1).lower();number=idmatch.group(2);ident=f'{prefix.upper()} {number}'
    text=re.sub(r'^\s*(?:CLO|CO)\s*\d+\s*','',text,flags=re.I)
    # Statement ends at the first explicit PLO/PO cell or Bloom-domain cell.
    link=PO_ID.search(text)
    bloom=re.search(r'\b\d+\s*/\s*(?:Apply|Understand|Analyze|Create|Evaluate|Remember|Valuing|Manipulation)|\b(?:Cognitive|Affective|Psychomotor)\s*/',text,re.I)
    end=min([m.start() for m in (link,bloom) if m is not None],default=len(text))
    statement=clean_value(text[:end])
    if statement:add(out,pdf,page,raw,prefix,statement,row_key=ident)
    if link:
        cell=link.group(2);display=f'{link.group(1).upper()} ({cell})' if cell.isalpha() else f'{link.group(1).upper()} {cell}'
        add(out,pdf,page,raw,f'{prefix}_{link.group(1).lower()}_link',link.group(0),row_key=ident,
            po_cell_verbatim=cell,**{f'{link.group(1).lower()}_display':display},header_text='Program outcome column')
    elif bloom and prefix=='co':
        # A CO table can print a bare PO number under its POs column.
        bare=re.search(r'\b(\d{1,2})\s+(?=\d+\s*/|Cognitive|Affective|Psychomotor)',text,re.I)
        if bare:add(out,pdf,page,raw,'co_po_link',bare.group(1),row_key=ident,
                    po_cell_verbatim=bare.group(1),po_display=f'PO {bare.group(1)}',header_text='Corresponding POs')
    if bloom:
        after=text[bloom.start():]
        level=re.match(r'(?:\d+\s*/\s*)?(?:Apply|Understand|Analyze|Create|Evaluate|Remember|Valuing|Manipulation|Cognitive\s*/\s*\w+)',after,re.I)
        if level:add(out,pdf,page,raw,f'{prefix}_bloom',level.group(0),row_key=ident)
        rest=after[level.end():] if level else after
        pieces=re.split(r'\b(?:Quiz|Lab\s+Evaluation|Written\s+Exam|Project\s+Report|Presentation\s+Followed\s+by\s+Viva)\b',rest,maxsplit=1,flags=re.I)
        if clean_value(pieces[0]):add(out,pdf,page,raw,f'{prefix}_delivery',pieces[0],row_key=ident)
        if len(pieces)>1:
            assessment=rest[len(pieces[0]):]
            if clean_value(assessment):add(out,pdf,page,raw,f'{prefix}_assessment_tools',assessment,row_key=ident)


def _weight_rows(out:list[dict],pdf:str,page:int,raw:str,region:str,header:str) -> None:
    # Each row begins with an assessment label and its first numeric weight cell.
    pattern=re.compile(r'(?<!\w)([A-Z][A-Za-z /&()+-]{2,65}?)\s+(\d{1,3}(?:\.\d+)?%?)(?=\s|$)')
    matches=list(pattern.finditer(region))
    for i,m in enumerate(matches):
        label=clean_value(m.group(1));weight=m.group(2)
        if re.search(r'\b(?:Total|Grade|Attendance|Deduction|Penalty|Above)\b',label,re.I):
            if re.search(r'\bTotal\b',label,re.I):
                add(out,pdf,page,raw,'weight_total',region[m.start():matches[i+1].start() if i+1<len(matches) else len(region)],
                    assessment='Total',weight=weight,weight_display=weight if weight.endswith('%') else weight+'%',derivation='column header' if not weight.endswith('%') else 'verbatim')
            continue
        tail=region[m.end():matches[i+1].start() if i+1<len(matches) else len(region)]
        tail=boundary(tail)
        value=clean_value(region[m.start():m.end()]+tail)
        display=weight if weight.endswith('%') else (weight+'%' if '% weight' in header.lower() else weight)
        add(out,pdf,page,raw,'weight',value,assessment=label,weight=weight,weight_display=display,
            derivation='column header' if display!=weight else 'verbatim',co_clo_cells={})


def _schedule(out:list[dict],pdf:str,page:int,raw:str,base:list) -> None:
    seen=set()
    for card in base:
        if card.field not in {'schedule_row','weekly_topic','lecture'}:continue
        key=card.row_key or ''
        if card.field=='lecture':key=f'Lecture {key}'
        if card.field=='weekly_topic' and not key.lower().startswith('week'):key=f'Week {key}'
        if not key or key in seen:continue
        seen.add(key)
        value=boundary(card.value)
        value=re.sub(r'^(?:Weeks?|Lecture)\s+\d+(?:\s*[-–]\s*\d+)?\s*','',value,flags=re.I)
        ref=CO_ID.search(value)
        topic=clean_value(value[:ref.start()]) if ref else clean_value(value)
        rest=value[ref.end():] if ref else ''
        if re.search(r'\b(?:Midterm|Mid-Term|Final\s+Exam)\b',topic,re.I):
            add(out,pdf,page,raw,'exam_event',topic,row_key=key)
            continue
        if topic:add(out,pdf,page,raw,'schedule',topic,row_key=key,topic=topic,
                     co_clo_refs=ref.group(0) if ref else '',outcomes=[],activities=clean_value(rest),
                     teaching_strategy='',reading='')


def parse_atomic_pages(pdf:str,pages:Iterable[str]) -> tuple[str,list[dict]]:
    pages=list(pages)
    if not enabled() or not pages:return 'OFF',[]
    template,base=parse_pages(pdf,pages)
    if template=='UNKNOWN':return template,[]
    out=[];by_page={i:[] for i in range(1,len(pages)+1)}
    for card in base:by_page[card.page].append(card)
    headers={'program','course_code','course_title','course_type','semester','level_term','credit','office','class_hours','consultation_hours','email','mobile','prerequisite','contact_hours','total_marks'}
    for page,raw in enumerate(pages,1):
        cards=by_page[page]
        for card in cards:
            field=card.field;value=boundary(card.value)
            if field in headers:
                add(out,pdf,page,raw,field,value)
            elif field=='teacher_person':
                designation=''
                if ',' in value:designation=value.rsplit(',',1)[-1].strip()
                add(out,pdf,page,raw,'teacher_person',value,designation=designation)
            elif field in {'objectives','synopsis','rationale','content'}:
                output={'objectives':'objective'}.get(field,field)
                for i,part in split_items(value):
                    if part:add(out,pdf,page,raw,output,part,list_index=i)
            elif field in {'co','clo','co_po_link','clo_plo_link'}:
                _co_cells(out,pdf,page,raw,card)
            elif field=='book_item':
                add(out,pdf,page,raw,'book',value,list_index=card.list_index)
            elif field=='books' and card.value=='BLANK':
                course=Path(pdf).stem
                out.append(dict(course=course,field='books',value='',pdf=pdf,page=page,pages=[page],
                                plain_span=card.plain_span,empty_section=True,card_id=hashlib.sha256(f'{pdf}|{page}|books|empty'.encode()).hexdigest()[:20],text=f'{course} | books | EMPTY | p{page}'))
            elif field in {'special_instructions','attendance','grading','assessment_strategy','evaluation_policy','assessment'}:
                kind={'special_instructions':'rule','attendance':'rule','grading':'grading_scale','assessment_strategy':'assessment_strategy','evaluation_policy':'evaluation_policy','assessment':'assessment_technique'}[field]
                for i,part in split_items(value):
                    if part:add(out,pdf,page,raw,kind,part,list_index=i)
        _schedule(out,pdf,page,raw,cards)
        flat=compact(raw)
        # Weight regions are bounded at their next non-weight section heading.
        for head in WEIGHT_HEAD.finditer(flat):
            endmatch=WEIGHT_STOP.search(flat,head.end())
            end=endmatch.start() if endmatch else len(flat)
            _weight_rows(out,pdf,page,raw,flat[head.end():end],flat[head.start():head.end()+30])
        # Appendix definitions: one item, not a whole appendix.
        appendix=re.search(r'\bAppendix\s*[-–]?\s*\d*\s*:\s*.*?(?:Program\s+Outcomes?|Program\s+Learning\s+Outcomes?)',flat,re.I)
        if appendix:
            body=flat[appendix.end():]
            lettered=list(re.finditer(r'\(([a-z])\)\s+',body,re.I))
            if lettered:
                for i,m in enumerate(lettered):
                    text=clean_value(body[m.end():lettered[i+1].start() if i+1<len(lettered) else len(body)])
                    if text:add(out,pdf,page,raw,'plo_definition',text,row_key=m.group(1))
            else:
                for i,part in split_items(body):
                    if i and part:add(out,pdf,page,raw,'po_definition',part,row_key=str(i))
    # Keep each fact under one field and drop punctuation-only/oversized duplicates.
    unique={};priority={'schedule':0,'weight':0,'weight_total':0,'co_po_link':0,'clo_plo_link':0,'book':0,'teacher_person':0}
    for card in out:
        value=match_text(card['value'])
        if not value and not card.get('empty_section'):continue
        key=(card['pdf'],card['page'],value)
        old=unique.get(key)
        if old is None or priority.get(card['field'],5)<priority.get(old['field'],5):unique[key]=card
    return template,list(unique.values())
