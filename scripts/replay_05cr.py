"""Checkpointed 05C-R card replay using the saved 05C queries and rewrite cache."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.card_runtime import attempt_card_answer
from src.pipeline import answer_question

ROADMAP=Path('/content/drive/MyDrive/University_RAG_Colab/Roadmap_2026-11')
STEP=ROADMAP/'Step05C_Banglish_Norm'
OUT=STEP/'repair_05CR_20261127T114958Z'
BASE=STEP/'verification_runs/20261127T090815Z/stage-a-05c-banglish-fuse-20261127T090824456579Z-283443'
CACHE=STEP/'verification_runs/20261127T090815Z/rewrite_cache'
GATE=ROADMAP/'Step03_Stage_A_v10/SAFETY_GATE_PASSED.json'


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def cache_sha()->str:
    h=hashlib.sha256()
    for path in sorted(CACHE.glob('*.json')):
        h.update(path.name.encode());h.update(bytes.fromhex(sha(path)))
    return h.hexdigest()


def setup()->dict:
    launch=json.loads((BASE.parent/'launch_manifest.json').read_text())
    os.environ.update(launch['rag_env'])
    os.environ.update({'RAG_CARD_STATEMENT_GUARD':'on','RAG_CARD_GUARD_AUDIT_DIR':str(OUT),
                       'GENERATOR_GPU_LAYERS':'0','RERANKER_ENABLED':'false',
                       'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','EMBEDDING_LOCAL_ONLY':'true'})
    gate=json.loads(GATE.read_text())
    for name,digest in gate['index_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'index mismatch {name}')
    return {'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'rag_env':{key:value for key,value in sorted(os.environ.items()) if key.startswith('RAG_')},
            'cache_sha256':cache_sha(),'base_results_sha256':sha(BASE/'results.csv'),
            'card_source_sha256':sha(ROADMAP/'Step05B_Card_Answers/cards_normalized_p1b.jsonl'),
            'command_line':' '.join(sys.argv),'started_at_utc':datetime.now(timezone.utc).isoformat()}


def atomic_json(path:Path,value)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def cached_rewrite(row:dict)->str:
    if row.get('rewrite_valid')!='True':return ''
    query=row['question'];path=CACHE/(hashlib.sha256(query.encode()).hexdigest()+'.json')
    item=json.loads(path.read_text())
    if item.get('query_sha256')!=hashlib.sha256(query.encode()).hexdigest() or item.get('text')!=row['rewrite_text'] or not item.get('valid'):
        raise ValueError(f'cached rewrite mismatch {row["evaluation_id"]}')
    return item['text']


def write_csv(path:Path,records:list[dict],fields:list[str])->None:
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['routes','fallback'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    global OUT
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/f'replay_{args.mode}_manifest.json'
    current=setup()
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        for key in ('commit','rag_env','cache_sha256','base_results_sha256','card_source_sha256'):
            if old[key]!=current[key]:raise ValueError(f'unsafe resume {key}')
    else:atomic_json(manifest_path,current)
    base=list(csv.DictReader((BASE/'results.csv').open(encoding='utf-8-sig')))
    if len(base)!=825:raise ValueError('base run is not 825 rows')
    if args.mode=='routes':
        folder=OUT/'replay_route_checkpoint';folder.mkdir(exist_ok=True)
        completed={r['evaluation_id']:r for path in sorted(folder.glob('batch-*.json')) for r in json.loads(path.read_text())}
        batch=[];batch_number=len(list(folder.glob('batch-*.json')))
        for row in base:
            if row['evaluation_id'] in completed:continue
            result,_=attempt_card_answer(row['question'],cached_rewrite(row))
            record={'evaluation_id':row['evaluation_id'],'language':row['language'],'old_route':row['route'],
                    'new_route':'card' if result else 'fallback','old_card_id':row['card_id'],
                    'new_card_id':result.get('card_id','') if result else '',
                    'old_status':row['answerability_status'],'new_status':result.get('answerability_status','') if result else '',
                    'old_answer':row['final_answer'],'new_answer':result.get('answer','') if result else '',
                    'source':result.get('source','') if result else '', 'page':result.get('page','') if result else '',
                    'anchor_source':result.get('anchor_source','none') if result else 'none'}
            batch.append(record)
            if len(batch)>=10:
                batch_number+=1;atomic_json(folder/f'batch-{batch_number:06d}.json',batch);completed.update((x['evaluation_id'],x) for x in batch);batch=[]
            print(f'{len(completed)+len(batch)}/825 {row["evaluation_id"]}',flush=True)
        if batch:
            batch_number+=1;atomic_json(folder/f'batch-{batch_number:06d}.json',batch);completed.update((x['evaluation_id'],x) for x in batch)
        records=[completed[row['evaluation_id']] for row in base]
        write_csv(OUT/'replay_routes.csv',records,list(records[0]))
        write_csv(OUT/'replay_cards.csv',[r for r in records if r['new_route']=='card'],list(records[0]))
    else:
        route_rows={x['evaluation_id']:x for x in csv.DictReader((OUT/'replay_routes.csv').open(encoding='utf-8'))}
        changed=[x for x in base if x['route']=='card' and route_rows[x['evaluation_id']]['new_route']=='fallback']
        folder=OUT/'replay_fallback_checkpoint';folder.mkdir(exist_ok=True)
        completed={r['evaluation_id']:r for path in sorted(folder.glob('batch-*.json')) for r in json.loads(path.read_text())}
        for row in changed:
            if row['evaluation_id'] in completed:continue
            result=answer_question(row['question'],use_generation=True,use_answer_bank=False)
            record={'evaluation_id':row['evaluation_id'],'language':row['language'],
                    'status':result.get('answerability_status','SYSTEM_ERROR'),
                    'answer':result.get('answer',''),'source':result.get('source',''),
                    'page':result.get('page',''),'card_id':result.get('card_id',''),
                    'error':result.get('runtime_error','')}
            atomic_json(folder/f'batch-{len(completed)+1:06d}.json',[record]);completed[row['evaluation_id']]=record
            print(f'{len(completed)}/{len(changed)} {row["evaluation_id"]}',flush=True)
        records=[completed[x['evaluation_id']] for x in changed]
        write_csv(OUT/'replay_fallbacks.csv',records,list(records[0]) if records else ['evaluation_id','language','status','answer','source','page','card_id','error'])

if __name__=='__main__':main()
