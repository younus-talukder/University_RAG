"""Banglish-only query rewrite and two-list RRF fusion.

No dataset labels or facts enter this module. Invalid rewrites are never used.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / 'config/banglish_lexicon.yaml'
PROMPT = ROOT / 'config/banglish_rewrite_prompt.txt'
_CODE = re.compile(r'(?<![\w])([A-Za-z]{2,8})\s*[- ]?\s*(\d{2,4})(?!\d)')
_NUMBER = re.compile(r'\d+(?:\.\d+)?')
_WORD = re.compile(r'[A-Za-z]+')
_BANGLA = re.compile(r'[\u0980-\u09ff]')


@lru_cache(maxsize=1)
def lexicon() -> dict[str, Any]:
    return yaml.safe_load(LEXICON.read_text(encoding='utf-8'))


def enabled() -> bool:
    flag = os.environ.get('RAG_BANGLISH_NORM', 'fuse').strip().casefold()
    if flag not in {'fuse', 'off'}:
        raise ValueError('RAG_BANGLISH_NORM must be fuse or off')
    return flag == 'fuse'


def detected(question: str) -> bool:
    if not question or _BANGLA.search(question):
        return False
    without_codes = _CODE.sub(' ', question)
    words = [w.casefold() for w in _WORD.findall(without_codes)]
    if not words:
        return False
    latin = len(re.findall(r'[A-Za-z]', without_codes))
    other = len(re.findall(r'[^\W\d_A-Za-z]', without_codes))
    return latin > other and any(w in set(lexicon()['cues']) for w in words)


def _codes(text: str) -> Counter[str]:
    return Counter(m.group(1).casefold() + m.group(2) for m in _CODE.finditer(text))


def _numbers(text: str) -> Counter[str]:
    return Counter(_NUMBER.findall(_CODE.sub(' ', text)))


def valid_rewrite(original: str, rewrite: str) -> bool:
    if not rewrite.strip() or _BANGLA.search(rewrite):
        return False
    return not (_codes(original) - _codes(rewrite)) and not (_numbers(original) - _numbers(rewrite))


def _cache_path(question: str) -> Path | None:
    folder = os.environ.get('RAG_BANGLISH_CACHE_DIR', '').strip()
    if not folder:
        return None
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return path / (hashlib.sha256(question.encode('utf-8')).hexdigest() + '.json')


def rewrite(question: str) -> tuple[str, bool, float]:
    """Use the already-loaded Qwen chat model, greedy, and cache by query hash."""
    cache = _cache_path(question)
    if cache and cache.is_file():
        item = json.loads(cache.read_text(encoding='utf-8'))
        if item.get('query_sha256') == hashlib.sha256(question.encode()).hexdigest():
            return item['text'], bool(item['valid']), float(item['latency_s'])
    from . import generator
    started = time.perf_counter()
    model = generator._get_generator(
        str(generator.GENERATOR_MODEL_PATH), generator.GENERATOR_CONTEXT_SIZE,
        generator.GENERATOR_THREADS, generator.GENERATOR_BATCH_SIZE,
        generator.GENERATOR_GPU_LAYERS, generator.GENERATOR_USE_MMAP,
        generator.GENERATOR_USE_MLOCK)
    response = model.create_chat_completion(
        messages=[{'role': 'system', 'content': PROMPT.read_text(encoding='utf-8').strip()},
                  {'role': 'user', 'content': question}],
        max_tokens=48, temperature=0, stream=False)
    raw = response['choices'][0]['message']['content']
    text = str(raw).splitlines()[0].strip().strip('"\'“”‘’')
    valid = valid_rewrite(question, text)
    latency = time.perf_counter() - started
    if cache:
        payload = {'query_sha256': hashlib.sha256(question.encode()).hexdigest(),
                   'text': text, 'valid': valid, 'latency_s': latency}
        temporary = cache.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        temporary.replace(cache)
    return text, valid, latency


def fuse(original: list[dict], rewritten: list[dict], k: int, limit: int) -> list[dict]:
    """RRF across complete ranked lists; stable original-first ties."""
    entries: dict[str, dict] = {}
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for group in (original, rewritten):
        for rank, item in enumerate(group, 1):
            key = str(item.get('chunk_id') or (item.get('source'), item.get('page'), item.get('text')))
            if key not in entries:
                entries[key] = dict(item)
                order[key] = len(order)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    keys = sorted(entries, key=lambda key: (-scores[key], order[key]))[:limit]
    return [{**entries[key], 'score': scores[key], 'fusion_score': scores[key],
             'final_rank': rank} for rank, key in enumerate(keys, 1)]


def original_anchors_safe(question: str, card: dict) -> bool:
    """Permit rewrite rescue only when original anchors are function words or in value."""
    words = re.findall(r'(?<!\w)[A-Z][a-z]{3,}(?!\w)', question)
    generic = {'Which', 'What', 'Where', 'When', 'Book', 'Books', 'Textbook',
               'Reference', 'Recommended', 'Course', 'The', 'This', 'That'}
    anchors = [word.casefold() for word in words if word not in generic]
    if not anchors and re.search(r'\b[A-Za-z]{2,8}\s*\d{2,4}-er\b', question, re.I):
        anchors.append('er')
    functions = set(lexicon()['function_words'])
    value = str(card.get('value') or '').casefold()
    return all(anchor in functions or re.search(r'(?<!\w)' + re.escape(anchor) + r'(?!\w)', value)
               for anchor in anchors)
