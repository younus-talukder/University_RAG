"""Schema-only normalization for opt-in atomic course cards.

This module does not parse PDFs, infer facts, change values, or affect the
answer path. It adds stable names and types to facts extracted in Step 5A.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import re

COMMON_FIELDS=(
    'card_id','course','pdf','pages','field','key','list_index','value',
    'char_start','char_end','plain_span','tier',
)

# Each atomic field has an explicit schema. The tuple lists field-specific
# members in addition to COMMON_FIELDS; empty tuples are scalar facts.
FIELD_SCHEMA:dict[str,tuple[str,...]]={
    'program':(), 'course_code':(), 'course_title':(), 'course_type':(),
    'semester':(), 'level_term':(), 'credit':(), 'prerequisite':(),
    'contact_hours':(), 'total_marks':(), 'class_hours':(),
    'consultation_hours':(), 'office':(), 'email':(), 'mobile':(),
    'teacher_person':('designation',),
    'rationale':(), 'synopsis':(), 'content':(), 'objective':(),
    'co':('row_key',), 'clo':('row_key',),
    'co_po_link':('co_id','po_cell_verbatim','po_display','header_text'),
    'clo_po_link':('co_id','po_cell_verbatim','po_display','header_text'),
    'clo_plo_link':('co_id','po_cell_verbatim','po_display','header_text'),
    'co_bloom':('row_key',), 'clo_bloom':('row_key',),
    'co_delivery':('row_key',), 'clo_delivery':('row_key',),
    'co_assessment_tools':('row_key',), 'clo_assessment_tools':('row_key',),
    'topic_clo':('row_key','co_clo_refs'),
    'schedule':('row_key','topic','outcomes','activities','teaching_strategy','delivery','reading','co_clo_refs'),
    'exam_event':('row_key',),
    'weight':('assessment','weight','weight_display','co_clo_cells'),
    'weight_total':('assessment','weight_display','co_clo_cells'),
    'assessment_technique':('row_key','topic'),
    'assessment_strategy':('row_key',),
    'marks_split':('row_key','cells'),
    'evaluation_policy':('row_key','is_weight'),
    'grading_scale':('is_weight',),
    'rule':(), 'teaching_strategy':(),
    'book':(), 'books':('empty_section',),
    'po_definition':('row_key',), 'plo_definition':('row_key',),
    'knowledge_profile':('row_key','course_level'),
    'prepared_by':(),
}


def _string(value:Any)->str:
    return '' if value is None else str(value)


def normalize_card(card:Mapping[str,Any])->dict[str,Any]:
    """Add schema names and type fixes; preserve all source facts verbatim."""
    if card.get('tier')!='atomic':
        raise ValueError('only atomic cards may be normalized')
    field=_string(card.get('field'))
    if field not in FIELD_SCHEMA:raise ValueError(f'unknown atomic field: {field}')
    result=dict(card)
    pages=result.get('pages') or [result.get('page')]
    result['pages']=[int(p) for p in pages if p is not None]
    index=result.get('list_index')
    result['list_index']=int(index) if index not in (None,'') else None
    result['key']=_string(result.get('key') or result.get('row_key') or result.get('assessment') or
                          (result['list_index'] if result['list_index'] is not None else field))
    result['value']=_string(result.get('value'))
    result['plain_span']=_string(result.get('plain_span'))
    for name in ('char_start','char_end'):
        value=result.get(name)
        result[name]=int(value) if value not in (None,'') else None
    if field=='schedule':
        outcomes=result.get('outcomes')
        result['outcomes']=outcomes if isinstance(outcomes,list) else ([] if outcomes in (None,'') else [str(outcomes)])
        for name in ('row_key','topic','activities','teaching_strategy','delivery','reading','co_clo_refs'):
            result[name]=_string(result.get(name))
        refs=result['co_clo_refs']
        if refs and re.search(r'\bCLO\s*\d+',refs,re.I):result.setdefault('clo_refs',refs)
        if refs and re.search(r'\bCO\s*\d+',refs,re.I):result.setdefault('co_refs',refs)
    elif field in {'weight','weight_total'}:
        for name in ('assessment','weight','weight_display'):
            result[name]=_string(result.get(name))
        cells=result.get('co_clo_cells')
        result['co_clo_cells']=dict(cells) if isinstance(cells,Mapping) else {}
        for prefix,alias in [('CLO','clo_cells'),('CO','co_cells')]:
            subset={k:v for k,v in result['co_clo_cells'].items() if re.fullmatch(prefix+r'\s*\d+',str(k),re.I)}
            if subset:result.setdefault(alias,subset)
    elif field.endswith('_link'):
        result['co_id']=_string(result.get('co_id') or result.get('row_key'))
        for name in ('po_cell_verbatim','po_display','plo_display','header_text'):
            if name in result:result[name]=_string(result[name])
    elif field=='books':
        result['empty_section']=bool(result.get('empty_section',False))
    for name in FIELD_SCHEMA[field]:
        if name not in result:
            result[name]=({} if name in {'co_clo_cells','cells'} else
                          [] if name=='outcomes' else
                          False if name in {'empty_section','course_level','is_weight'} else '')
    return result


def validate_card(card:Mapping[str,Any])->list[str]:
    """Report schema defects without changing or discarding a card."""
    errors=[]
    field=_string(card.get('field'))
    if field not in FIELD_SCHEMA:return ['unknown_field']
    for name in COMMON_FIELDS+FIELD_SCHEMA[field]:
        if name not in card:errors.append('missing:'+name)
    if card.get('tier')!='atomic':errors.append('non_atomic_tier')
    if not isinstance(card.get('pages'),list) or not all(isinstance(p,int) and p>0 for p in card.get('pages',[])):
        errors.append('pages_type')
    if field=='schedule' and not isinstance(card.get('outcomes'),list):errors.append('outcomes_type')
    if field in {'weight','weight_total'}:
        cells=card.get('co_clo_cells')
        if not isinstance(cells,dict):errors.append('co_clo_cells_type')
        elif any(not re.fullmatch(r'(?:CO|CLO)\s*\d+',str(key),re.I) for key in cells):
            errors.append('non_identifier_cell_key')
    if not card.get('empty_section') and (card.get('char_start') is None or card.get('char_end') is None):
        errors.append('missing_span_offsets')
    return errors


def populate_extras(card:Mapping[str,Any], page_text:str)->dict[str,Any]:
    """Populate only structural attributes from the card's plain source page.

    Values, keys, pages, spans and card identity are never changed.  Table
    headers are used to name cells; empty/dash cells are omitted.
    """
    result=dict(card)
    field=result.get('field')
    if field=='schedule':
        end=result.get('char_end')
        if isinstance(end,int) and 0<=end<len(page_text):
            tail=page_text[end:end+600]
            # The outcome column follows the topic.  Allow newlines and a
            # repeated CO/CLO prefix within the same comma-separated cell.
            match=re.search(r'\b((?:CLO|CO)\s*\d+(?:\s*,\s*(?:(?:CLO|CO)\s*)?\d+)*)\b',tail,re.I)
            if match:
                refs=re.findall(r'\b(?:CLO|CO)\s*\d+\b',match.group(1),re.I)
                current=re.findall(r'\b(?:CLO|CO)\s*\d+\b',str(result.get('co_clo_refs','')),re.I)
                if len(refs)>len(current):
                    joined=', '.join(re.sub(r'\s+',' ',x.upper()).strip() for x in refs)
                    result['co_clo_refs']=joined
                    name='clo_refs' if refs[0].upper().startswith('CLO') else 'co_refs'
                    result[name]=joined
    elif field in {'weight','weight_total'}:
        original_cells=result.get('co_clo_cells') or {}
        result.setdefault('co_cells',{})
        result.setdefault('clo_cells',{})
        label=str(result.get('assessment') or '')
        if label:
            # Match the assessment label in plain extraction even when PDF
            # line breaks split it across several lines.
            pattern=r'\s+'.join(re.escape(word) for word in label.split())
            hits=list(re.finditer(pattern,page_text,re.I))
            start=result.get('char_start')
            hit=min(hits,key=lambda item:abs(item.start()-start)) if hits and isinstance(start,int) else (hits[0] if hits else None)
            if hit:
                header=page_text[max(0,hit.start()-500):hit.start()]
                ids=re.findall(r'\b(CLO|CO)\s*(\d+)\b',header,re.I)
                ids=[f'{prefix.upper()} {int(number)}' for prefix,number in ids]
                # Use only a contiguous header sequence immediately before
                # the table's first assessment, not ids from course prose.
                ids=list(dict.fromkeys(ids))
                if ids and len(ids)<=12:
                    tail=page_text[hit.end():hit.end()+180]
                    # Cells stop at the next row's alphabetic assessment
                    # label. This prevents a neighboring percentage being
                    # mistaken for a CO/CLO column value.
                    next_label=re.search(r'[A-Za-z]{2,}',tail)
                    if next_label:tail=tail[:next_label.start()]
                    tokens=re.findall(r'\b\d+(?:\.\d+)?%?|[–—-]',tail)
                    if tokens:
                        weight=tokens[0]
                        if not result.get('weight_display') and re.fullmatch(r'\d+(?:\.\d+)?%?',weight):
                            result['weight_display']=weight if weight.endswith('%') else weight+'%'
                        cells={key:value for key,value in zip(ids,tokens[1:1+len(ids)]) if value not in {'–','—','-'}}
                        if cells:
                            result['co_clo_cells']=cells
                            result['clo_cells']={k:v for k,v in cells.items() if k.startswith('CLO ')}
                            result['co_cells']={k:v for k,v in cells.items() if k.startswith('CO ')}
        # A previously extracted, identifier-keyed cell is more precise
        # than assigning compressed plain-text columns by position.
        if original_cells and all(re.fullmatch(r'(?:CO|CLO)\s*\d+',str(k),re.I)
                                  for k in original_cells):
            result['co_clo_cells']=original_cells
            result['co_cells']={k:v for k,v in original_cells.items() if str(k).upper().startswith('CO') and not str(k).upper().startswith('CLO')}
            result['clo_cells']={k:v for k,v in original_cells.items() if str(k).upper().startswith('CLO')}
    elif field in {'co_po_link','clo_po_link','clo_plo_link'}:
        if not result.get('po_display') and result.get('plo_display'):
            result['po_display']=result['plo_display']
        elif not result.get('plo_display') and result.get('po_display') and field=='clo_plo_link':
            result['plo_display']=result['po_display']
    return result
