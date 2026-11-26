"""Development-only scoring for atomic card candidates.

This module is never used to choose a runtime answer or relax a trust gate.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml


POLICY_PATH = Path(__file__).resolve().parents[1] / 'config/answer_policy.yaml'
_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
_NONE = {'none', 'nil', 'nill', 'n a', 'na', 'no prerequisite',
         'কোনো পূর্বশর্ত নেই', 'পূর্বশর্ত নেই', 'kono prerequisite nei',
         'prerequisite nei'}


def canonical_value(value: str) -> str:
    text = unicodedata.normalize('NFKC', str(value)).translate(_DIGITS).casefold()
    text = re.sub(r'\s*,?\s*as printed\s*$', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09ff]', ' ', text)
    text = ' '.join(text.split())
    return '<none>' if text in _NONE else text


def none_like_value_match(reference: str, candidate: str) -> bool | None:
    """Compare printed no-prerequisite forms; return None for other facts."""
    ref, value = canonical_value(reference), canonical_value(candidate)
    return (ref == value == '<none>') if '<none>' in {ref, value} else None


def score_candidate(reference: str, candidate: str, source_relation: str = 'same',
                    policy_path: Path = POLICY_PATH) -> tuple[bool, bool, str]:
    """Return policy correctness, dataset literal correctness, and verdict."""
    policy = yaml.safe_load(policy_path.read_text(encoding='utf-8'))
    none_match = none_like_value_match(reference, candidate)
    literal = none_match if none_match is not None else canonical_value(reference) == canonical_value(candidate)
    if literal:
        return True, True, 'CORRECT'
    if source_relation not in {'alt_source', 'source_conflict'}:
        return False, False, 'WRONG'
    verdict = policy['scoring'][source_relation]
    return verdict in {'CORRECT', 'POLICY-CORRECT'}, False, verdict
