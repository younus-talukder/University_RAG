"""Flagged deterministic card answer text; source values remain verbatim."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any,Mapping

import yaml

from .card_router import CardIntent
from .card_selector import CardSelection


TEMPLATES=Path(__file__).resolve().parents[1]/'config/templates.yaml'
BN_DIGITS=str.maketrans('0123456789','০১২৩৪৫৬৭৮৯')
NUMERIC_FIELDS={'credit','contact_hours','total_marks','class_hours','consultation_hours','weight','weight_total'}
NONE_VALUES={'none','nil','n/a'}
LABELS={
 'credit':('Credit value','ক্রেডিট','credit'), 'course_title':('Course title','কোর্সের নাম','course title'),
 'level_term':('Level/Term','বর্ষ ও সেমিস্টার','level/term'),
 'semester':('Semester','সেমিস্টার','semester'), 'prerequisite':('Prerequisite','পূর্বশর্ত','prerequisite'),
 'teacher_person':('teacher','শিক্ষক','teacher'), 'objective':('objective','উদ্দেশ্য','objective'),
 'clo':('CLO','সিএলও','CLO'), 'co':('CO','সিও','CO'),
 'book':('book','বই','book'), 'books':('reference book','রেফারেন্স বই','reference book'),
 'weight':('assessment weight','মূল্যায়নের ওয়েট','assessment weight'),
 'schedule':('schedule','সময়সূচি','schedule'),
}


@dataclass(frozen=True)
class RenderedCardAnswer:
    body:str
    citation:str
    answer:str
    pdf:str
    pages:tuple[int,...]


@lru_cache(maxsize=1)
def _templates()->dict:
    return yaml.safe_load(TEMPLATES.read_text(encoding='utf-8'))


def _digits(value:Any,lang:str)->str:
    text=str(value)
    return text.translate(BN_DIGITS) if lang=='bn' else text


def _key_bn(key:str)->str:
    if key.startswith('Weeks '):return 'সপ্তাহ '+_digits(key[6:],'bn')
    if key.startswith('Week '):return 'সপ্তাহ '+_digits(key[5:],'bn')
    if key.startswith('Lecture '):return 'লেকচার '+_digits(key[8:],'bn')
    return key


def _label(field:str)->tuple[str,str,str]:
    return LABELS.get(field,(field.replace('_',' '),field.replace('_',' '),field.replace('_',' ')))


def render_card_answer(intent:CardIntent,selection:CardSelection,extra_pages:tuple[int,...]=())->RenderedCardAnswer:
    if selection.route!='card' or not selection.cards:raise ValueError('a card route is required')
    cards=selection.cards;first=cards[0];lang=intent.lang if intent.lang in {'en','bn','bl'} else 'en'
    templates=_templates()[lang]
    course=str(first['course']);field=str(first['field']);label,label_bn,label_bl=_label(field)
    key=str(first.get('row_key') or first.get('key') or first.get('assessment') or '')
    if field in {'clo','co'} and key.replace(' ','') in str(first.get('plain_span','')) and key not in str(first.get('plain_span','')):
        key=key.replace(' ','')
    value=str(first.get('value',''))
    if field in NUMERIC_FIELDS:value=_digits(value,lang)
    pdf=str(first['pdf'])
    if any(card.get('pdf')!=pdf for card in cards):raise ValueError('a card answer must cite one PDF')
    pages=tuple(sorted({int(page) for card in cards for page in card.get('pages',[card['page']])}|set(extra_pages)))
    slots=dict(course=course,label=label,label_bn=label_bn,label_bl=label_bl,label_plural=label+'s',
               key=key,key_bn=_key_bn(key),value=value,pdf=pdf,pages=', '.join(map(str,pages)),
               pages_bn=', '.join(_digits(page,'bn') for page in pages),
               attr_label=intent.attribute.replace('_',' '),attr_label_bn=intent.attribute.replace('_',' '),
               attr_label_bl=intent.attribute.replace('_',' '))
    if first.get('empty_section'):
        template='empty'
    elif selection.mode=='count':
        template='count';slots['n']=_digits(len(cards),lang)
    elif selection.mode=='list':
        values=[str(card['value']) for card in cards]
        slots['values']='; '.join(values)
        slots['items']='; '.join(f"{card.get('assessment',card.get('key',''))}: {_digits(card.get('weight_display',''),lang)}" for card in cards)
        slots['value']=slots['values']
        template='teacher_list' if field=='teacher_person' else 'weight_all' if field=='weight' else 'generic'
    elif selection.mode=='reverse':
        template='reverse';slots['asked']=value
    elif selection.mode=='attribute':
        template='schedule_attr' if field=='schedule' else 'generic'
        attribute=first.get(intent.attribute,'')
        slots['attr_value']=_digits(attribute,lang) if intent.attribute in {'weight','weight_display'} else str(attribute)
        slots['value']=slots['attr_value']
    elif value.casefold().strip() in NONE_VALUES:
        template='prerequisite_none' if field=='prerequisite' else 'generic_none'
    elif field=='credit':template='credit'
    elif field=='weight':
        template='weight_one';slots['assessment']=str(first.get('assessment') or key)
        slots['weight_display']=_digits(first.get('weight_display') or first.get('weight') or value,lang)
    elif field in {'clo','co','objective'}:template='statement'
    elif field.endswith('_link'):
        template='link';slots['po_display']=str(first.get('po_display') or first.get('plo_display') or '')
    elif field=='schedule':
        template='schedule_topic';slots['topic']=str(first.get('topic') or value)
    else:template='generic'
    body=templates[template].format(**slots)
    if body and 'a'<=body[0]<='z':body=body[0].upper()+body[1:]
    citation=templates['citation'].format(**slots)
    return RenderedCardAnswer(body,citation,body+' '+citation,pdf,pages)
