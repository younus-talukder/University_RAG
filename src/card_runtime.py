"""Opt-in atomic card route with the existing trust validators."""
from __future__ import annotations
from functools import lru_cache
import json,os,time
from pathlib import Path
from typing import Any
import yaml
from pypdf import PdfReader

from .answer_safety import answer_safety_failures
from .card_answer import render_card_answer
from .card_router import understand
from .card_selector import CardEmbeddingIndex,infer_complete_fields,select_cards
from .course_cards import match_text
from .embeddings import get_embedding_model
from .grounding_validator import validate_grounding
from .language_validator import validate_language


DEFAULT_BUNDLE=Path('/content/drive/MyDrive/University_RAG_Colab/Roadmap_2026-11/Step05B_Card_Answers')
DEFAULT_PDFS=Path('/content/drive/MyDrive/University_RAG_Colab/corpus/source_pdfs_44_v1_2026-10-25')


def enabled()->bool:
    card_answers=os.environ.get('RAG_REPAIR_CARD_ANSWERS','0').casefold() in {'1','true','yes','on'}
    course_cards=os.environ.get('RAG_REPAIR_COURSE_CARDS','0').casefold() in {'1','true','yes','on'}
    return card_answers and course_cards


@lru_cache(maxsize=1)
def _bundle():
    root=Path(os.environ.get('RAG_CARD_BUNDLE_DIR',DEFAULT_BUNDLE))
    source=root/'cards_normalized_p1b.jsonl'
    cards=[json.loads(line) for line in source.open(encoding='utf-8')]
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/card_routing.yaml').read_text(encoding='utf-8'))
    index=CardEmbeddingIndex(root/'card_embeddings_v1',cards)
    import hashlib
    if config['card_source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('card bundle differs from calibrated source')
    return cards,index,config


@lru_cache(maxsize=256)
def _plain_page(pdf:str,page:int)->str:
    root=Path(os.environ.get('RAG_CARD_PDF_DIR',DEFAULT_PDFS))
    return PdfReader(root/pdf).pages[page-1].extract_text()


def _span_verified(card:dict[str,Any])->bool:
    pages=card.get('pages') or [card.get('page')]
    if not pages or card.get('span_status')=='not_found':return False
    if card.get('empty_section'):
        return bool(card.get('plain_span','').strip())
    start,end=card.get('char_start'),card.get('char_end')
    if not isinstance(start,int) or not isinstance(end,int) or start<0 or end<=start:return False
    raw=_plain_page(card['pdf'],int(pages[0]))
    if end>len(raw):return False
    value=match_text(str(card.get('value','')))
    source=match_text(raw[start:end])
    return bool(value and value in source)


def attempt_card_answer(question:str)->tuple[dict[str,Any]|None,list[dict[str,Any]]]:
    """Return a validated card answer, or extra plain spans for RAG fallback."""
    if not enabled():return None,[]
    started=time.perf_counter()
    cards,index,config=_bundle()
    intent=understand(question,cards)
    if not intent.course:return None,[]
    course_cards=[c for c in cards if c['course']==intent.course]
    vector=get_embedding_model().embed(question)
    scores=index.scores(vector,intent.course)
    enabled_fields={c['field'] for c in course_cards}-set(config['disabled_fields'])
    selected=select_cards(intent,course_cards,scores,float(config['tau']),float(config['tau_rev']),
                          infer_complete_fields(course_cards),enabled_fields)
    ranked=sorted(course_cards,key=lambda c:scores.get(c['card_id'],-1),reverse=True)
    extras=[dict(text=c['plain_span'],source=c['pdf'],relative_path=c['pdf'],page=c['page'],
                 chunk_id=c['card_id'],score=scores.get(c['card_id'],0),card_tier='atomic')
            for c in ranked[:3] if c.get('plain_span') and _span_verified(c)]
    if selected.route!='card':return None,extras
    chosen=selected.cards
    if not all(_span_verified(card) for card in chosen):return None,extras
    rendered=render_card_answer(intent,selected)
    evidence=[dict(excerpt=card['plain_span'],source=card['pdf'],page=card['page']) for card in chosen]
    language=validate_language(rendered.body,{'en':'english','bn':'bangla','bl':'banglish'}[intent.lang])
    grounding=validate_grounding(rendered.body,evidence)
    safety=answer_safety_failures(question,rendered.body,[card['plain_span'] for card in chosen],
                                  cited_excerpt=chosen[0]['plain_span'])
    if not language.get('validation_passed') or not grounding.get('grounding_validation_passed') or safety:
        return None,extras
    latency=time.perf_counter()-started
    first=chosen[0]
    result=dict(question=question,answer=rendered.answer,answer_body=rendered.body,citation=rendered.citation,
                detected_language={'en':'english','bn':'bangla','bl':'banglish'}[intent.lang],
                response_language=language['response_language'],language_validation_passed=True,
                language_consistency=True,language_validation_reason=language['validation_reason'],
                grounding_validation_passed=True,grounding_validation_reason=grounding['grounding_validation_reason'],
                support_status='SUPPORTED',final_status='ANSWER_RETURNED',final_answer_allowed=True,
                answer_mode='card',answer_strategy='card',generation_mode='card',generation_used=False,
                generation_attempts=0,retry_used=False,answer_bank_enabled=False,
                source=rendered.pdf,relative_path=rendered.pdf,page=rendered.pages[0],
                chunk_id=first['card_id'],card_id=first['card_id'],card_ids=[card['card_id'] for card in chosen],
                card_mode=selected.mode,card_field=first['field'],supporting_excerpt=first['plain_span'],
                sources=[dict(source=card['pdf'],relative_path=card['pdf'],page=card['page'],
                              chunk_id=card['card_id'],supporting_excerpt=card['plain_span'],score=scores.get(card['card_id'],0)) for card in chosen],
                evidence=evidence,retrieved_context=extras,
                latency_seconds=dict(retrieval=latency,answering=0.0,generation=0.0,retry=0.0,total=latency),
                answerability_seconds=latency,requested_entity=intent.course,requested_field=intent.field,
                card_selection_reason=selected.reason,card_score=selected.top_score,card_margin=selected.margin)
    return result,extras
