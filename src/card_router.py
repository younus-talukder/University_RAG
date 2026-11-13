"""Deterministic, opt-in query understanding for atomic course cards.

No dataset answers or course-specific facts are used here.  This module does
not answer questions or change the existing RAG path.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import Iterable,Mapping,Any

import yaml

from .fast_answer import detect_course_code
from .language_detector import detect_language
from .query_normalization import extract_course_entities


LEXICON=Path(__file__).resolve().parents[1]/'config/card_lexicon.yaml'
DIGITS=str.maketrans('০১২৩৪৫৬৭৮৯','0123456789')
DEPARTMENTS={'সিএসই':'CSE','ইইই':'EEE','ইংরেজি':'ENG','গণিত':'MTH','পদার্থ':'PHY'}
FALLBACK=re.compile(r'\b(?:why|explain|compare|comparison|difference|versus|vs)\b|কেন|ব্যাখ্যা|তুলনা|পার্থক্য|\b(?:keno|ken|bakkha|tulona)\b',re.I)
KEY_RE={
    'outcome':re.compile(r'\b(CLO|CO)\s*[-–]?\s*(\d+)\b',re.I),
    'mapping':re.compile(r'\b(PLO|PO)\s*[-–(]?\s*([a-z]|\d{1,2})\)?\b',re.I),
    'week_range':re.compile(r'\bweeks?\s*(\d+)\s*[-–]\s*(\d+)\b|সপ্তাহ\s*(\d+)\s*[-–]\s*(\d+)',re.I),
    'week':re.compile(r'\bweek\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s*week\b|সপ্তাহ\s*(\d+)\b|\b(\d+)\s*সপ্তাহ',re.I),
    'lecture':re.compile(r'\blecture\s*(\d+)\b|লেকচার\s*(\d+)',re.I),
    'objective':re.compile(r'\bobjective\s*(\d+)\b|উদ্দেশ্য\s*(\d+)',re.I),
    'book':re.compile(r'\b(?:book|textbook|reference)\s*(\d+)\b|বই\s*(\d+)',re.I),
    'teacher':re.compile(r'\b(?:teacher|instructor)\s*(\d+)\b|শিক্ষক\s*(\d+)',re.I),
}


@dataclass(frozen=True)
class CardIntent:
    lang:str
    course:str
    field:str
    key:str
    attribute:str
    mode:str
    confidence:float
    cues:tuple[str,...]


def _norm(text:str)->str:
    return unicodedata.normalize('NFKC',text).casefold().translate(DIGITS)


def _has(text:str,cue:str)->bool:
    normalized=_norm(cue)
    plural='s?' if normalized.isascii() and normalized.isalpha() and len(normalized)>2 else ''
    pattern=r'(?<!\w)'+re.escape(normalized)+plural+r'(?!\w)'
    return bool(re.search(pattern,text))


@lru_cache(maxsize=1)
def _lexicon()->dict:
    return yaml.safe_load(LEXICON.read_text(encoding='utf-8'))['cues']


def _course_text(text:str)->str:
    value=text.translate(DIGITS)
    for bn,en in DEPARTMENTS.items():
        value=value.replace(bn,en)
    return value


def _course(text:str)->tuple[str,bool]:
    converted=_course_text(text)
    entities=extract_course_entities(converted)
    canonical={item.canonical for item in entities}
    if len(canonical)>1:return '',True
    if canonical:return next(iter(canonical)),False
    return detect_course_code(converted),False


def _key(text:str)->tuple[str,str]:
    for name,pattern in KEY_RE.items():
        match=pattern.search(text)
        if not match:continue
        parts=[part for part in match.groups() if part]
        if name=='outcome':return f'{parts[0].upper()} {int(parts[1])}',name
        if name=='mapping':
            suffix=parts[1].lower() if parts[1].isalpha() else str(int(parts[1]))
            return f'{parts[0].upper()} {suffix}',name
        if name=='week_range':return f'Weeks {int(parts[0])}–{int(parts[1])}',name
        if name=='week':return f'Week {int(parts[0])}',name
        if name=='lecture':return f'Lecture {int(parts[0])}',name
        return f'{name} {int(parts[0])}',name
    return '', ''


def _score_cues(text:str,lang:str,section:str)->list[tuple[int,str,str]]:
    matches=[]
    for field,langs in _lexicon()[section].items():
        for item in langs.get(lang,[]):
            cue=item['cue']
            if _has(text,cue):matches.append((len(cue),field,cue))
    return sorted(matches,reverse=True)


def _assessment_key(text:str, cards:Iterable[Mapping[str,Any]])->str:
    assessments={str(c.get('assessment','')) for c in cards if c.get('field')=='weight'}
    query_tokens=set(re.findall(r'\w+',text))
    scored=[]
    for label in assessments:
        tokens=set(re.findall(r'\w+',_norm(label)))
        if not tokens:continue
        overlap=len(tokens & query_tokens)/len(tokens)
        if overlap>=.75:scored.append((overlap,len(tokens),label))
    return max(scored)[2] if scored else ''


def understand(question:str,cards:Iterable[Mapping[str,Any]]=())->CardIntent:
    """Return a conservative structured intent; mode none means RAG fallback."""
    try:lang={'english':'en','bangla':'bn','banglish':'bl'}[detect_language(question)]
    except (ValueError,KeyError):lang='en'
    text=_norm(question)
    course,multiple=_course(question)
    if multiple or FALLBACK.search(text):return CardIntent(lang,course,'','','','none',0.0,('multi-course-or-explanation',))
    key,key_type=_key(text)
    field_hits=_score_cues(text,lang,'fields')
    attr_hits=_score_cues(text,lang,'attributes')
    mode_hits=_score_cues(text,lang,'modes')
    field=field_hits[0][1] if field_hits else ''
    if key_type in {'week','week_range','lecture'}:field='schedule'
    elif key_type=='objective':field='objective'
    elif key_type=='book':field='book'
    elif key_type=='teacher':field='teacher_person'
    elif key_type=='outcome' and field not in {'clo','co'}:field=key.split()[0].lower()
    elif key_type=='mapping' and not field:field='co_po_link'
    if field=='weight' and not key:key=_assessment_key(text,cards)
    attribute=attr_hits[0][1] if attr_hits else ''
    mode=mode_hits[0][1] if mode_hits else 'lookup'
    if attribute and mode=='lookup' and field in {'schedule','weight','co_po_link','clo_plo_link'}:mode='attribute'
    if field=='schedule' and mode=='attribute' and attribute in {'activities','teaching_strategy','delivery','reading','co_clo_refs'}:
        mode='none'  # P1 exception: schedule attribute extras below 90%.
    if not course or not field:mode='none'
    cues=tuple(dict.fromkeys([item[2] for item in field_hits[:2]+attr_hits[:2]+mode_hits[:1]]))
    confidence=min(1.0,.35+.25*bool(course)+.2*bool(field)+.15*bool(key)+.05*bool(cues)) if mode!='none' else 0.0
    return CardIntent(lang,course,field,key,attribute,mode,confidence,cues)


def resolve_range(key:str,card_keys:Iterable[str])->str:
    """Resolve a requested single week to a range row if it lies inside it."""
    match=re.fullmatch(r'Week (\d+)',key)
    if not match:return key
    week=int(match.group(1))
    for candidate in card_keys:
        range_match=re.fullmatch(r'Weeks (\d+)[–-](\d+)',candidate)
        if range_match and int(range_match.group(1))<=week<=int(range_match.group(2)):
            return candidate
    return key
