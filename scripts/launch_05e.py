"""Single checkpointed Step05E Stage A run, with immutable rewrite-cache provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = Path('/content/drive/MyDrive/University_RAG_Colab/Roadmap_2026-11')
STEP = ROADMAP / 'Step05E_Reranker'
CACHE = ROADMAP / 'Step05C_Banglish_Norm/verification_runs/20261127T090815Z/rewrite_cache'
DATASET = Path('/content/drive/MyDrive/University_RAG_Colab/Stage_A_5PDF_Pilot_Preparation_2026-10-09/stage_a_5pdf_questions.csv')
GATE = ROADMAP / 'Step03_Stage_A_v10/SAFETY_GATE_PASSED.json'
RERANKER = Path('/content/drive/MyDrive/University_RAG_Colab/models/bge-reranker-v2-m3')
CACHE_SHA = '1a521d411745985665cd0f9ce566993a2f0b9483e22069ddcc5958d5b9650730'
WEIGHT_SHA = 'd9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
FLAGS = {
    'RAG_BANGLISH_NORM': 'fuse', 'RAG_CARD_STATEMENT_GUARD': 'on',
    'RAG_RERANKER': 'bge_v2_m3', 'RAG_RERANKER_DEVICE': 'cpu',
    'RAG_REPAIR_CARD_ANSWERS': '1', 'RAG_REPAIR_COURSE_CARDS': '1',
    'RAG_REPAIR_COURSE_ANCHORED_RETRIEVAL': '1',
    'RAG_REPAIR_DEEPER_GATE_CHECKED_SELECTION': '1',
    'RAG_REPAIR_FIX_COURSE_CODE_EXTRACTOR': '1',
    'RAG_REPAIR_SAME_PAGE_LABELED_FIELD_JOIN': '1',
    'RAG_REPAIR_SINGLE_COURSE_DOCUMENT_OWNERSHIP': '0',
    'RAG_REPAIR_TABLE_HEADER_CONTEXT': '1',
}
OTHER = {'GENERATOR_GPU_LAYERS': '0', 'HF_HUB_OFFLINE': '1',
         'TRANSFORMERS_OFFLINE': '1', 'RERANKER_ENABLED': 'false',
         'EMBEDDING_LOCAL_ONLY': 'true'}


def sha(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def cache_sha(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.glob('*.json')):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def inputs() -> dict:
    gate = json.loads(GATE.read_text())
    if sha(DATASET) != '70c968d623167cef2720cec59bf789ce9433401fe68d278097b3bcdb708662eb':
        raise ValueError('Stage A dataset hash differs')
    for name, digest in gate['index_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Index mismatch: {name}')
    if cache_sha(CACHE) != CACHE_SHA:
        raise ValueError('Immutable Banglish rewrite cache differs')
    if sha(RERANKER / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Pinned reranker weight differs')
    configs = [*sorted((ROOT / 'config').glob('*.yaml')),
               ROOT / 'config/banglish_rewrite_prompt.txt',
               ROOT / 'config/answer_policy.yaml']
    return {
        'dataset_sha256': sha(DATASET), 'index_sha256': gate['index_sha256'],
        'config_sha256': {str(path.relative_to(ROOT)): sha(path) for path in configs},
        'lexicon_sha256': sha(ROOT / 'config/banglish_lexicon.yaml'),
        'rewrite_prompt_sha256': sha(ROOT / 'config/banglish_rewrite_prompt.txt'),
        'model_hashes': {**gate['model_hashes'], 'bge-reranker-v2-m3/model.safetensors': WEIGHT_SHA},
        'reranker_model': 'BAAI/bge-reranker-v2-m3', 'reranker_revision': REVISION,
        'reranker_device': 'cpu', 'reranker_candidate_depth': 20,
        'immutable_cache_sha256': CACHE_SHA,
        'rag_env': {key: os.environ[key] for key in sorted(os.environ) if key.startswith('RAG_')},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['start', 'resume'])
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    allowed = set(FLAGS) | {'RAG_BANGLISH_CACHE_DIR', 'RAG_CARD_GUARD_AUDIT_DIR'}
    unknown = {key for key in os.environ if key.startswith('RAG_')} - allowed
    if unknown:
        raise ValueError(f'Unexpected RAG_* variables: {sorted(unknown)}')
    os.environ.update(FLAGS)
    os.environ.update(OTHER)
    if args.mode == 'start':
        if args.run_dir:
            raise ValueError('start creates its own run directory')
        run_dir = STEP / 'verification_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir.mkdir(parents=True, exist_ok=False)
        cache_dest = run_dir / 'rewrite_cache'
        shutil.copytree(CACHE, cache_dest)
        if cache_sha(cache_dest) != CACHE_SHA:
            raise ValueError('Copied rewrite cache differs')
    else:
        if not args.run_dir:
            raise ValueError('resume requires --run-dir')
        run_dir = args.run_dir.resolve()
        cache_dest = run_dir / 'rewrite_cache'
        if not cache_dest.is_dir():
            raise ValueError('Run rewrite cache missing')
    os.environ['RAG_BANGLISH_CACHE_DIR'] = str(cache_dest)
    os.environ['RAG_CARD_GUARD_AUDIT_DIR'] = str(run_dir)
    actual = inputs()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluate_benchmark.py'),
               '--dataset', str(DATASET), '--output-dir', str(run_dir),
               '--run-name', 'stage-a-05e-bge-v2-m3', '--classification', 'PILOT',
               '--require-references', '--checkpoint-every', '10']
    manifest_path = run_dir / 'launch_manifest.json'
    if args.mode == 'start':
        manifest = {'code_commit': commit, **actual, 'command_argv': command,
                    'command_line': subprocess.list2cmdline(command),
                    'started_at_utc': datetime.now(timezone.utc).isoformat(),
                    'initial_cache_entries': len(list(cache_dest.glob('*.json')))}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads(manifest_path.read_text())
        for key, value in {'code_commit': commit, **actual}.items():
            if manifest.get(key) != value:
                raise ValueError(f'Unsafe resume: {key} differs')
        children = [path for path in run_dir.iterdir() if path.is_dir() and (path / 'run_manifest.json').exists()]
        if len(children) != 1:
            raise ValueError('Resume needs exactly one checkpointed child run')
        command += ['--resume', str(children[0])]
    subprocess.run(command, check=True, env=os.environ.copy())


if __name__ == '__main__':
    main()
