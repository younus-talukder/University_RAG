"""Guard atomic CO/CLO statement cards against mapping and table cells."""
from __future__ import annotations
import re
from .course_cards import match_text

IDENTIFIER = re.compile(
    r'^(?:(?:CLO|CO|PLO|PO)\s*[-(]?\s*[a-z0-9]+\s*\)?|[a-l]|\d{1,2})'
    r'(?:\s*,\s*(?:(?:CLO|CO|PLO|PO)\s*[-(]?\s*[a-z0-9]+\s*\)?|[a-l]|\d{1,2}))*$',re.I)
START_B=re.compile(r'\b(?:12\s*\.\s*)?Course\s+Learning\s+Outcomes?\b',re.I)
END_B=re.compile(r'\b(?:13\s*\.\s*Mapping|Part\s+B|14\s*\.\s*Course\s+Content)\b',re.I)
START_A=re.compile(r'\b(?:Course\s+(?:Learning\s+)?Outcomes?\b|CO\s+Statements\s*:)',re.I)
END_A=re.compile(r'\b(?:Weighting\s+COs|Course\s+Content\s+Outline|Part\s+B)\b',re.I)

def _source_region(template: str,pages: list[str]) -> str:
    flat=' '.join(' '.join(page.split()) for page in pages)
    start=(START_B if template=='B' else START_A).search(flat)
    if start is None:return ''
    end=(END_B if template=='B' else END_A).search(flat,start.end())
    return flat[start.end():end.start() if end else len(flat)]

def statement_rejection(card: dict, template: str, pages: list[str], region: str | None = None) -> str | None:
    if card.get('field') not in {'clo','co'}:return None
    value=str(card.get('value','')).strip()
    if IDENTIFIER.fullmatch(value):return 'identifier_only_cell'
    if len(re.findall(r'\b\w+\b',value))<3:return 'fewer_than_three_words'
    if region is None:region=_source_region(template,pages)
    if not region:return 'statement_source_section_not_found'
    if match_text(value) not in match_text(region):return 'outside_statement_source'
    return None
