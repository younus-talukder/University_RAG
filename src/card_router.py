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
SINGLETON_FIELDS={'program','course_code','course_title','course_type','semester','level_term','credit',
                  'prerequisite','contact_hours','total_marks','class_hours','consultation_hours','office','email','mobile'}
FALLBACK=re.compile(r'\b(?:why|explain|compare|comparison|difference|versus|vs)\b|কেন|ব্যাখ্যা|তুলনা|পার্থক্য|\b(?:keno|ken|bakkha|tulona)\b',re.I)
MAP_CUE=re.compile(r'\b(?:map|maps|mapped|mapping|alignment|corresponding|aligned)\b|ম্যাপ|সংযুক্ত|যুক্ত|\b(?:map|mape|jukto)\b',re.I)
WEIGHT_CUE=re.compile(r'\b(?:weight|percent|percentage|marks|score|distribution|worth|carries)\b|'
                      r'\b(?:counts?\s+toward|assigned\s+to|share\s+of|percentage\s+of\s+the\s+grade|koto\s+(?:percent|marks))\b|'
                      r'শতাংশ|নম্বরের\s*হার|ওয়েট|কত\s*(?:শতাংশ|নম্বর)',re.I)
YEAR_LEVEL_CUE=re.compile(r'\b(?:year|level|term)\b|বর্ষ\s*(?:ও|এবং)?\s*সেমিস্টার|লেভেল',re.I)
SESSION_CUE=re.compile(r'\b(?:academic\s+(?:semester|session)|session|semester)\b|শিক্ষাবর্ষ|সেমিস্টার',re.I)
KEY_RE={
    'outcome':re.compile(r'\b(CLO|CO)\s*[-–]?\s*(\d+)\b',re.I),
    'mapping':re.compile(r'\b(PLO|PO)\s*[-–(]?\s*([a-z]|\d{1,2})\)?\b',re.I),
    'week_range':re.compile(r'\bweeks?\s*(\d+)\s*[-–]\s*(\d+)\b|সপ্তাহ\s*(\d+)\s*[-–]\s*(\d+)',re.I),
    'week':re.compile(r'\b(?:week|soptaho)\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s*week\b|'
                      r'সপ্তাহ(?:ে|র|ের)?\s*(\d+)\b|\b(\d+)(?:তম|য়|ম|র্থ|ষ্ঠ)?\s*সপ্তাহ(?:ে|র|ের)?',re.I),
    'lecture':re.compile(r'\blecture\s*(\d+)\b|লেকচার\s*(\d+)',re.I),
    'objective':re.compile(r'\bobjective\s*(\d+)\b|উদ্দেশ্য\s*(\d+)',re.I),
    'book':re.compile(r'\b(?:book|textbook|reference)\s*(\d+)\b|বই\s*(\d+)',re.I),
    'teacher':re.compile(r'\b(?:teacher|instructor)\s*(\d+)\b|শিক্ষক\s*(\d+)',re.I),
}
EN_WEEK_WORDS={word:i for i,word in enumerate(('one','two','three','four','five','six','seven','eight',
                                                 'nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'),1)}
EN_WEEK_WORDS.update({word:i for i,word in enumerate(('first','second','third','fourth','fifth','sixth','seventh',
                                                        'eighth','ninth','tenth','eleventh','twelfth','thirteenth',
                                                        'fourteenth','fifteenth','sixteenth'),1)})
BN_WEEK_WORDS={word:i for i,word in enumerate(('প্রথম','দ্বিতীয়','তৃতীয়','চতুর্থ','পঞ্চম','ষষ্ঠ','সপ্তম','অষ্টম',
                                                 'নবম','দশম','একাদশ','দ্বাদশ','ত্রয়োদশ','চতুর্দশ','পঞ্চদশ','ষোড়শ'),1)}


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
    for word,number in BN_WEEK_WORDS.items():
        if re.search(r'(?<!\w)'+re.escape(word)+r'(?:এ|র|এর)?\s*সপ্তাহ(?:ে|র|ের)?',text):
            return f'Week {number}','week'
    en_words='|'.join(sorted(EN_WEEK_WORDS,key=len,reverse=True))
    for match in re.finditer(r'\b(?:week\s+('+en_words+r')|('+en_words+r')\s+week)\b',text):
        number=EN_WEEK_WORDS.get(match.group(1) or match.group(2))
        if number:return f'Week {number}','week'
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


def _assessment_key(text:str, cards:Iterable[Mapping[str,Any]],course:str)->str:
    assessments={str(c.get('assessment','')) for c in cards if c.get('field')=='weight' and c.get('course')==course}
    def tokens(value:str)->set[str]:
        return {token[:-1] if token.isascii() and token.endswith('s') and len(token)>3 else token
                for token in re.findall(r'\w+',_norm(value))}
    query_tokens=tokens(text)
    scored=[]
    for label in assessments:
        label_tokens=tokens(label)
        if not label_tokens:continue
        overlap=len(label_tokens & query_tokens)/len(label_tokens)
        if overlap>=.85:scored.append((overlap,len(label_tokens),label))
    if not scored:return ''
    best=max((score,length) for score,length,_ in scored)
    winners=[label for score,length,label in scored if (score,length)==best]
    return winners[0] if len(winners)==1 else ''


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
    if MAP_CUE.search(text) or (re.search(r'\b(?:PLO|PO)\s*[-–(]?\s*(?:[a-z]|\d)',text,re.I) and key_type=='outcome'):
        if re.search(r'\bPLO',text,re.I):field='clo_plo_link'
        elif re.search(r'\bPO',text,re.I):field='co_po_link'
    if WEIGHT_CUE.search(text) and field not in {'co_po_link','clo_plo_link'}:
        field='weight'
        if key_type=='outcome':key=''
    if WEIGHT_CUE.search(text) and field=='level_term':field='weight'
    elif field=='level_term' and SESSION_CUE.search(text) and not YEAR_LEVEL_CUE.search(text):field='semester'
    elif not field and SESSION_CUE.search(text):field='level_term' if YEAR_LEVEL_CUE.search(text) else 'semester'
    elif not field and YEAR_LEVEL_CUE.search(text):field='level_term'
    if field=='weight' and not key:key=_assessment_key(text,cards,course)
    if field in SINGLETON_FIELDS and not key:key=field
    attribute=attr_hits[0][1] if attr_hits else ''
    mode=mode_hits[0][1] if mode_hits else 'lookup'
    if attribute and mode=='lookup' and field in {'schedule','weight','co_po_link','clo_plo_link'}:mode='attribute'
    if field=='schedule' and mode=='attribute':
        mode='none'  # P1 exception: schedule attribute extras below 90%.
    outcome_ids=re.findall(r'\b(?:CLO|CO)\s*[-–]?\s*\d+\b',text,re.I)
    if field in {'clo','co'} and (len(outcome_ids)>1 or re.search(r'\b(?:skill|skills|divided|split)\b|দক্ষতা|ভাগ',text,re.I)):
        mode='none'  # A statement card does not answer a requested subskill or cross-outcome relation.
    if re.search(r'\bclass\s+tests?\b',text,re.I) and re.search(r'\bterms?\b',text,re.I) and re.search(r'\b(?:distributed|distribution|between|bhag)\b',text,re.I):
        mode='none'  # A term in an assessment policy is not the course level.
    if not course or not field:mode='none'
    cues=tuple(dict.fromkeys([item[2] for item in field_hits[:2]+attr_hits[:2]+mode_hits[:1]]))
    confidence=min(1.0,.35+.25*bool(course)+.2*bool(field)+.15*bool(key)+.05*bool(cues)) if mode!='none' else 0.0
    return CardIntent(lang,course,field,key,attribute,mode,confidence,cues)


def resolve_range(key:str,card_keys:Iterable[str])->str:
    """Resolve a requested single week to a range row if it lies inside it."""
    match=re.fullmatch(r'Week (\d+)',key,re.I)
    if not match:return key
    week=int(match.group(1))
    for candidate in card_keys:
        range_match=re.fullmatch(r'Weeks (\d+)\s*[–-]\s*(\d+)',candidate,re.I)
        if range_match and int(range_match.group(1))<=week<=int(range_match.group(2)):
            return candidate
    return key
