"""Conservative card selection over frozen BGE-M3 embeddings."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math,re
from pathlib import Path
from typing import Any,Iterable,Mapping
import numpy as np

from .card_router import CardIntent,SINGLETON_FIELDS,resolve_range
from .sparse_index import SparseIndex,sparse_tokenize

SELECT_FIELDS={'objective','clo','co','book','books','rule','synopsis','content','topic_clo','schedule'}
HEADING_ONLY=re.compile(r'^(?:\d{1,2}\s*[.)]\s*)?(?:part\s+[a-d]|course\s+(?:objectives?|outcomes?|synopsis|content\s+outline)|'
                        r'reference\s+books?(?:\s*(?:and|&)\s*materials?)?|text\s*books?|'
                        r'learning\s+resources?|(?:required|recommended)\s+references?|'
                        r'special\s+instructions?|weighting\s+cos?(?:\s+with\s+assessment\s+methods)?|'
                        r'evaluation\s+policy|assessment\s+(?:strategy|techniques?)|'
                        r'prepared\s+by|appendix(?:\s*\d+)?)$',re.I)


def is_heading_only_card(card:Mapping[str,Any])->bool:
    """Reject section labels accidentally emitted as atomic values."""
    if card.get('tier')!='atomic':return False
    value=re.sub(r'[^\w&]+',' ',str(card.get('value') or '').casefold()).strip()
    value=re.sub(r'\s+',' ',value)
    return bool(HEADING_ONLY.fullmatch(value))


@dataclass(frozen=True)
class CardSelection:
    route:str
    mode:str
    cards:tuple[dict[str,Any],...]
    top_score:float
    margin:float
    reason:str


def _key_equal(requested:str,card:Mapping[str,Any])->bool:
    key=str(card.get('row_key') or card.get('key') or card.get('assessment') or '').strip()
    wanted=requested.strip()
    if wanted.casefold()==key.casefold():return True
    if wanted.startswith(('Week ','Weeks ')) and card.get('field')=='schedule':
        if key.casefold().startswith('lecture '):return False
        if resolve_range(wanted,[key]).casefold()==key.casefold():return True
        header=str(card.get('key_column_header') or card.get('table_header') or card.get('header_text') or '')
        if re.search(r'\bweeks?\b',header,re.I):
            bare=re.fullmatch(r'(\d+)(?:\s*[-–]\s*(\d+))?',key)
            if bare:
                normalized=f'Weeks {bare.group(1)}–{bare.group(2)}' if bare.group(2) else f'Week {bare.group(1)}'
                return resolve_range(wanted,[normalized])==normalized
    # PLO-a, PLO (a), PLO a, and bare a are equivalent under a PLO header.
    def canonical(value:str)->str:
        return re.sub(r'[^a-z0-9]','',value.casefold())
    if re.match(r'^(?:PLO|PO)[\s(-]',wanted,re.I) and card.get('header_text'):
        return canonical(wanted)==canonical(key) or canonical(wanted)==canonical(str(card.get('po_display','')))
    return False


def select_cards(intent:CardIntent,cards:Iterable[Mapping[str,Any]],scores:Mapping[str,float],
                 tau:float,tau_rev:float,complete_fields:Iterable[str]=(),
                 enabled_fields:Iterable[str]|None=None,
                 bm25_scores:Mapping[str,float]|None=None)->CardSelection:
    if enabled_fields is not None and intent.field not in set(enabled_fields):
        return CardSelection('fallback',intent.mode,(),0,0,'field_not_precision_certified')
    course_cards=[dict(c) for c in cards if c.get('tier')=='atomic' and c.get('course')==intent.course]
    if intent.mode=='none' or not intent.course:return CardSelection('fallback',intent.mode,(),0,0,'no_intent')
    if not course_cards:return CardSelection('fallback',intent.mode,(),0,0,'course_not_in_cards')
    same_field=[c for c in course_cards if c.get('field')==intent.field]
    if bm25_scores is not None and intent.mode in {'lookup','attribute','select'} and same_field:
        keyed=[c for c in same_field if intent.key and _key_equal(intent.key,c)]
        if len(keyed)==1:
            card=keyed[0]
            return CardSelection('card',intent.mode,(card,),float(scores.get(card['card_id'],-1)),1.0,'exact_field_key')
        if intent.field in SINGLETON_FIELDS and not intent.key and len(same_field)==1:
            card=same_field[0]
            return CardSelection('card',intent.mode,(card,),float(scores.get(card['card_id'],-1)),1.0,'unique_field')
        ranked_field=sorted(same_field,key=lambda c:scores.get(c['card_id'],-1),reverse=True)
        ranked_sparse=sorted(same_field,key=lambda c:bm25_scores.get(c['card_id'],0),reverse=True)
        if ranked_field and ranked_sparse and ranked_field[0]['card_id']==ranked_sparse[0]['card_id']:
            first=ranked_field[0];top=float(scores.get(first['card_id'],-1))
            second=float(scores.get(ranked_field[1]['card_id'],-1)) if len(ranked_field)>1 else -1
            if top-second>=tau and bm25_scores.get(first['card_id'],0)>0:
                return CardSelection('card',intent.mode,(first,),top,top-second,'field_bm25_bge_agreement')
    if intent.mode=='lookup' and intent.field in SINGLETON_FIELDS and intent.key==intent.field and len(same_field)==1:
        card=same_field[0]
        return CardSelection('card',intent.mode,(card,),float(scores.get(card['card_id'],-1)),1.0,'unique_field_key')
    if intent.mode in {'list','count'}:
        if intent.field not in set(complete_fields) or not same_field:
            return CardSelection('fallback',intent.mode,(),0,0,'field_incomplete')
        ordered=tuple(sorted(same_field,key=lambda c:(c.get('list_index') is None,c.get('list_index') or 0,str(c.get('row_key') or c.get('key')))))
        return CardSelection('card',intent.mode,ordered,1,1,'complete_field')
    if bm25_scores is not None and not intent.key:
        return CardSelection('fallback',intent.mode,(),0,0,'no_bm25_bge_agreement')
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

    def all_scores(self,query_vector:np.ndarray)->dict[str,float]:
        values=self.vectors@query_vector.astype(np.float32)
        result={}
        for item,value in zip(self.items,values):
            card_id=item['card_id']
            result[card_id]=max(result.get(card_id,-1),float(value))
        return result


class CardValueIndex:
    """BM25 over atomic values only; no persistent index or corpus chunk changes."""
    def __init__(self,cards:Iterable[Mapping[str,Any]])->None:
        self.cards=[dict(card) for card in cards if card.get('tier')=='atomic']
        self.metadata=[{'chunk_id':card['card_id'],'text':str(card.get('value') or '')} for card in self.cards]
        self.index=SparseIndex.build(self.metadata)
        self.token_sets=[set(sparse_tokenize(item['text'])) for item in self.metadata]
        self.document_frequency={token:len(entries) for token,entries in self.index.postings.items()}

    def scores(self,question:str,course:str|None=None,field:str|None=None)->dict[str,float]:
        hits=self.index.search(question,self.metadata,len(self.metadata))
        allowed={card['card_id'] for card in self.cards if (not course or card.get('course')==course)
                 and (not field or card.get('field')==field)}
        return {item['chunk_id']:item['sparse_score'] for item in hits if item['chunk_id'] in allowed}

    def distinctive_overlap(self,question:str,card:Mapping[str,Any])->set[str]:
        """Shared value tokens whose IDF exceeds the median in this field."""
        field_cards=[item for item in self.cards if item.get('field')==card.get('field')]
        field_tokens={token for item in field_cards for token in sparse_tokenize(str(item.get('value') or ''))}
        if not field_tokens:return set()
        count=max(1,len(self.cards))
        idfs=sorted(math.log(1+(count-self.document_frequency.get(token,0)+.5)/
                             (self.document_frequency.get(token,0)+.5)) for token in field_tokens)
        median=idfs[len(idfs)//2]
        shared=set(sparse_tokenize(question))&set(sparse_tokenize(str(card.get('value') or '')))
        return {token for token in shared if math.log(1+(count-self.document_frequency.get(token,0)+.5)/
                 (self.document_frequency.get(token,0)+.5))>median}


def resolve_course_by_value(question:str,cards:Iterable[Mapping[str,Any]],
                            bm25:CardValueIndex,bge_scores:Mapping[str,float],
                            margin:float)->str:
    """Require cross-model course agreement, two distinctive terms, and a course margin."""
    atomic=[dict(c) for c in cards if c.get('tier')=='atomic']
    sparse=bm25.scores(question)
    if not sparse:return ''
    ranked_sparse=sorted(atomic,key=lambda c:sparse.get(c['card_id'],0),reverse=True)
    winner=ranked_sparse[0]
    if sparse.get(winner['card_id'],0)<=0 or len(bm25.distinctive_overlap(question,winner))<2:return ''
    best_other=max((sparse.get(c['card_id'],0) for c in atomic if c['course']!=winner['course']),default=0)
    if sparse[winner['card_id']]-best_other<margin:return ''
    ranked_bge=sorted(atomic,key=lambda c:bge_scores.get(c['card_id'],-1),reverse=True)
    if not ranked_bge or ranked_bge[0]['course']!=winner['course']:return ''
    first=bge_scores.get(ranked_bge[0]['card_id'],-1)
    next_other=max((bge_scores.get(c['card_id'],-1) for c in atomic if c['course']!=winner['course']),default=-1)
    return winner['course'] if first-next_other>=margin else ''


def select_item_by_value(question:str,intent:CardIntent,cards:Iterable[Mapping[str,Any]],
                         bm25:CardValueIndex,bge_scores:Mapping[str,float],tau:float)->CardSelection:
    """Choose one list item only when a distinctive lexical match and BGE agree."""
    if intent.mode!='lookup' or intent.key or intent.field not in SELECT_FIELDS:
        return CardSelection('fallback',intent.mode,(),0,0,'not_select_intent')
    candidates=[dict(c) for c in cards if c.get('tier')=='atomic' and c.get('course')==intent.course
                and c.get('field')==intent.field]
    if not candidates:return CardSelection('fallback',intent.mode,(),0,0,'no_items')
    sparse=bm25.scores(question,intent.course,intent.field)
    ranked=sorted(candidates,key=lambda c:sparse.get(c['card_id'],0),reverse=True)
    first=ranked[0];best=sparse.get(first['card_id'],0)
    second=sparse.get(ranked[1]['card_id'],0) if len(ranked)>1 else 0
    if best<=0 or best-second<tau:return CardSelection('fallback',intent.mode,(),best,best-second,'bm25_margin')
    shared=set(sparse_tokenize(question))&set(sparse_tokenize(str(first.get('value') or '')))
    other_tokens={token for other in candidates if other['card_id']!=first['card_id']
                  for token in sparse_tokenize(str(other.get('value') or ''))}
    distinctive=shared-other_tokens
    if not distinctive:return CardSelection('fallback',intent.mode,(),best,best-second,'no_unique_value_token')
    bge_first=max(candidates,key=lambda c:bge_scores.get(c['card_id'],-1))
    if bge_first['card_id']!=first['card_id']:
        return CardSelection('fallback',intent.mode,(),best,best-second,'bge_disagrees')
    return CardSelection('card','select',(first,),float(bge_scores.get(first['card_id'],-1)),best-second,'value_anchor')


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
