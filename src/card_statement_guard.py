"""Generic, reversible guards for keyed cards and CLO/CO statement values."""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable, Iterable

from .course_cards import match_text
from .card_selector import is_heading_only_card

_IDENTIFIER = re.compile(r'\b(CLO|CO)\s*[-–]?\s*(\d+)\b', re.I)
_KEYED = re.compile(r'^(CLO|CO|PLO|PO|Weeks?|Lecture|objective|book)\s*[-–]?\s*([a-z]|\d+)(?:\s*[–-]\s*(\d+))?$', re.I)


def enabled() -> bool:
    value = os.environ.get('RAG_CARD_STATEMENT_GUARD', 'on').strip().casefold()
    if value not in {'on', 'off'}:
        raise ValueError('RAG_CARD_STATEMENT_GUARD must be on or off')
    return value == 'on'


def normalized_key(value: str) -> str:
    match = _KEYED.fullmatch(str(value or '').strip())
    if not match:
        return ''
    prefix = match.group(1).casefold()
    suffix = match.group(2).casefold()
    end = match.group(3)
    if suffix.isdigit():
        suffix = str(int(suffix))
    if end:
        end = str(int(end))
    return prefix + suffix + (('-' + end) if end else '')


def selected_key_matches(card: dict, requested_keys: Iterable[str]) -> bool:
    """A keyed request must match the selected card on every route."""
    wanted = {normalized_key(key) for key in requested_keys if normalized_key(key)}
    if not wanted:
        return True
    selected = normalized_key(str(card.get('row_key') or card.get('key') or ''))
    return bool(selected and wanted == {selected})


def _same_id(left: re.Match[str], right: re.Match[str]) -> bool:
    return (left.group(1).casefold(), int(left.group(2))) == (right.group(1).casefold(), int(right.group(2)))


def _audit_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['card_id', 'course', 'field', 'key', 'pdf', 'page', 'reason', 'original_value', 'new_value']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def guard_cards(cards: list[dict], plain_page: Callable[[str, int], str],
                audit_dir: Path | None = None) -> list[dict]:
    """Trim one statement after its own ID; reject mixed or mislabeled values."""
    if not enabled():
        return cards
    kept: list[dict] = []
    trimmed: list[dict] = []
    rejected: list[dict] = []
    for card in cards:
        if card.get('tier') != 'atomic' or card.get('field') not in {'clo', 'co'}:
            kept.append(card)
            continue
        original = str(card.get('value') or '')
        own = _IDENTIFIER.search(str(card.get('row_key') or card.get('key') or ''))
        ids = list(_IDENTIFIER.finditer(original))
        foreign = [m for m in ids if not own or not _same_id(m, own)]
        if not foreign:
            kept.append(card)
            continue
        record = dict(card_id=card.get('card_id'), course=card.get('course'),
                      field=card.get('field'), key=card.get('row_key') or card.get('key'),
                      pdf=card.get('pdf'), page=card.get('page'), original_value=original,
                      new_value='')
        if not own or (ids and not _same_id(ids[0], own)):
            rejected.append({**record, 'reason': 'leading_foreign_identifier'})
            continue
        first_foreign = foreign[0]
        own_before = next((m for m in ids if m.start() < first_foreign.start() and _same_id(m, own)), None)
        value = original[own_before.end():first_foreign.start()].strip(' \t\r\n:;,.–-') if own_before else ''
        if len(value.split()) < 3 or is_heading_only_card({**card, 'value': value}):
            rejected.append({**record, 'reason': 'empty_short_or_heading_trim'})
            continue
        page = plain_page(str(card['pdf']), int(card['page']))
        if not match_text(value) or match_text(value) not in match_text(page):
            rejected.append({**record, 'reason': 'trim_not_verbatim_on_page'})
            continue
        changed = {**card, 'value': value}
        kept.append(changed)
        trimmed.append({**record, 'reason': 'trimmed_before_foreign_identifier', 'new_value': value})
    if audit_dir is not None:
        _audit_rows(audit_dir / 'trimmed_cards.csv', trimmed)
        _audit_rows(audit_dir / 'rejected_cards.csv', rejected)
    return kept
