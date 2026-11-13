"""Conservative card selection over frozen BGE-M3 embeddings."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,re
from pathlib import Path
from typing import Any,Iterable,Mapping
import numpy as np

from .card_router import CardIntent,resolve_range


@dataclass(frozen=True)
class CardSelection:
    route:str
    mode:str
    cards:tuple[dict[str,Any],...]
    top_score:float
    margin:float
    reason:str


def _key_equal(requested:str,card:Mapping[str,Any])->bool:
    key=str(card.get('row_key') or card.get('key') or card.get('assessment') or '')
    if requested==key:return True
    if requested.startswith('Week '):
        return resolve_range(requested,[key])==key
    # PLO-a, PLO (a), PLO a, and bare a are equivalent under a PLO header.
    def canonical(value:str)->str:
        return re.sub(r'[^a-z0-9]','',value.casefold())
    if re.match(r'^(?:PLO|PO)[\s(-]',requested,re.I) and card.get('header_text'):
        return canonical(requested)==canonical(key) or canonical(requested)==canonical(str(card.get('po_display','')))
    return False


def select_cards(intent:CardIntent,cards:Iterable[Mapping[str,Any]],scores:Mapping[str,float],
                 tau:float,tau_rev:float,complete_fields:Iterable[str]=(),
                 enabled_fields:Iterable[str]|None=None)->CardSelection:
    if enabled_fields is not None and intent.field not in set(enabled_fields):
        return CardSelection('fallback',intent.mode,(),0,0,'field_not_precision_certified')
    course_cards=[dict(c) for c in cards if c.get('tier')=='atomic' and c.get('course')==intent.course]
    if intent.mode=='none' or not intent.course:return CardSelection('fallback',intent.mode,(),0,0,'no_intent')
    if not course_cards:return CardSelection('fallback',intent.mode,(),0,0,'course_not_in_cards')
    same_field=[c for c in course_cards if c.get('field')==intent.field]
    if intent.mode in {'list','count'}:
        if intent.field not in set(complete_fields) or not same_field:
            return CardSelection('fallback',intent.mode,(),0,0,'field_incomplete')
        ordered=tuple(sorted(same_field,key=lambda c:(c.get('list_index') is None,c.get('list_index') or 0,str(c.get('row_key') or c.get('key')))))
        return CardSelection('card',intent.mode,ordered,1,1,'complete_field')
    ranked=sorted(course_cards,key=lambda c:scores.get(c['card_id'],-1),reverse=True)
    first=ranked[0];top=float(scores.get(first['card_id'],-1))
    second=float(scores.get(ranked[1]['card_id'],-1)) if len(ranked)>1 else -1
    margin=top-second
    if intent.mode=='reverse':
        if top>=tau_rev and first.get('field')==intent.field:
            return CardSelection('card',intent.mode,(first,),top,margin,'reverse_threshold')
        return CardSelection('fallback',intent.mode,(),top,margin,'reverse_low_score')
    field_and_key=bool(intent.field==first.get('field') and intent.key and _key_equal(intent.key,first))
    if field_and_key or (margin>=tau and first.get('field')==intent.field and (not intent.key or _key_equal(intent.key,first))):
        return CardSelection('card',intent.mode,(first,),top,margin,'field_key' if field_and_key else 'margin')
    return CardSelection('fallback',intent.mode,(),top,margin,'ambiguous_or_wrong_field')


class CardEmbeddingIndex:
    """Read a completed, checkpointed card embedding cache; never write it."""
    def __init__(self,root:Path,cards:list[dict[str,Any]])->None:
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        item_bytes=(root/'items.jsonl').read_bytes()
        if hashlib.sha256(item_bytes).hexdigest()!=manifest.get('items_sha256'):
            raise ValueError('card embedding item manifest mismatch')
        self.items=[json.loads(line) for line in item_bytes.decode('utf-8').splitlines()]
        if len(self.items)!=manifest.get('item_count'):
            raise ValueError('card embedding item count mismatch')
        parts=sorted(root.glob('part_*.npy'))
        if not parts:raise ValueError('card embeddings are absent')
        self.vectors=np.concatenate([np.load(part) for part in parts])
        if len(self.items)!=len(self.vectors):raise ValueError('card embedding cache incomplete')
        self.cards=cards

    def scores(self,query_vector:np.ndarray,course:str)->dict[str,float]:
        selected=[i for i,item in enumerate(self.items) if self.cards[item['card_index']].get('course')==course]
        if not selected:return {}
        values=self.vectors[selected]@query_vector.astype(np.float32)
        result={}
        for index,value in zip(selected,values):
            card_id=self.items[index]['card_id']
            result[card_id]=max(result.get(card_id,-1),float(value))
        return result


def infer_complete_fields(cards:Iterable[Mapping[str,Any]])->set[str]:
    """Certify only contiguous, numbered list fields within one outline.

    Other fields fall back until their section completeness is independently
    established.  This avoids answering a count from a partial extraction.
    """
    grouped={}
    for card in cards:
        if card.get('tier')!='atomic':continue
        if card.get('field') not in {'objective','book'}:continue
        grouped.setdefault(card['field'],[]).append(card)
    complete=set()
    for field,items in grouped.items():
        indices=[item.get('list_index') for item in items]
        if all(isinstance(index,int) and index>0 for index in indices) and sorted(indices)==list(range(1,len(indices)+1)):
            complete.add(field)
    return complete
