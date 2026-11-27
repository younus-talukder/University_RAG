"""Pinned cross-encoder page ordering after fused retrieval."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Sequence

MODEL_DIR = Path('/content/drive/MyDrive/University_RAG_Colab/models/bge-reranker-v2-m3')
MODEL_REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
WEIGHT_SHA256 = 'd9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'


def enabled() -> bool:
    flag = os.environ.get('RAG_RERANKER', 'bge_v2_m3').strip().casefold()
    if flag not in {'bge_v2_m3', 'off'}:
        raise ValueError('RAG_RERANKER must be bge_v2_m3 or off')
    return flag == 'bge_v2_m3'


def device() -> str:
    value = os.environ.get('RAG_RERANKER_DEVICE', 'cpu').strip().casefold()
    if value not in {'cpu', 'cuda'}:
        raise ValueError('RAG_RERANKER_DEVICE must be cpu or cuda')
    return value


class PageReranker:
    def __init__(self, model_dir: Path = MODEL_DIR, selected_device: str | None = None):
        selected_device = selected_device or device()
        weight = model_dir / 'model.safetensors'
        if not weight.is_file():
            raise RuntimeError(f'Pinned reranker missing: {weight}')
        digest = hashlib.file_digest(weight.open('rb'), 'sha256').hexdigest()
        if digest != WEIGHT_SHA256:
            raise RuntimeError('Pinned reranker weight SHA-256 differs from recorded value')
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if selected_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('RAG_RERANKER_DEVICE=cuda but CUDA is unavailable')
        self.device = selected_device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32).to(selected_device).eval()

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), 8):
            batch = list(passages[start:start + 8])
            tokens = self.tokenizer([query] * len(batch), batch, padding=True,
                                    truncation=True, max_length=512, return_tensors='pt')
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with self.torch.inference_mode():
                logits = self.model(**tokens).logits
            scores.extend(float(value) for value in logits[:, 0].cpu().tolist())
        return scores


@lru_cache(maxsize=1)
def scorer() -> PageReranker:
    return PageReranker()


def page_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get('document_id') or item.get('source') or ''), str(item.get('page') or ''))


def pages(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = page_key(item)
        if key not in seen:
            seen.add(key)
            result.append({'document_id': key[0], 'source': item.get('source'), 'page': item.get('page')})
            if len(result) >= limit:
                break
    return result


def rerank_pages(query: str, candidates: Sequence[dict[str, Any]], model: Any | None = None,
                 candidate_k: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score top chunks, sort their pages by max logit, retain all chunks."""
    original = list(candidates)
    pre = pages(original, 5)
    top10 = pages(original, 10)
    if not enabled():
        return original, {'pre_rerank_pages': pre, 'post_rerank_pages': pre,
                          'top10_fused_pages': top10, 'rerank_latency_s': 0.0,
                          'rerank_candidate_depth': 0}
    started = time.perf_counter()
    model = model or scorer()
    pool = original[:candidate_k]
    scores = model.score_pairs(query, [str(item.get('text') or '') for item in pool])
    if len(scores) != len(pool):
        raise RuntimeError('Reranker score count differs from candidate count')
    import math
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError('Reranker returned nonfinite scores')
    page_scores: dict[tuple[str, str], float] = {}
    page_order: dict[tuple[str, str], int] = {}
    for index, (item, score) in enumerate(zip(pool, scores)):
        key = page_key(item)
        page_order.setdefault(key, index)
        page_scores[key] = max(score, page_scores.get(key, -math.inf))
    sorted_pages = sorted(page_scores, key=lambda key: (-page_scores[key], page_order[key]))
    position = {key: rank for rank, key in enumerate(sorted_pages)}
    ranked = [dict(item) for _, item in sorted(enumerate(original),
              key=lambda pair: (position.get(page_key(pair[1]), len(position) + pair[0]), pair[0]))]
    return ranked, {'pre_rerank_pages': pre, 'post_rerank_pages': pages(ranked, 5),
                    'top10_fused_pages': top10, 'rerank_latency_s': time.perf_counter() - started,
                    'rerank_candidate_depth': len(pool)}
