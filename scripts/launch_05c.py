"""Single checkpointed Step05C Stage A verification run; start or safe resume."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = Path('/content/drive/MyDrive/University_RAG_Colab/Roadmap_2026-11')
STEP = ROADMAP / 'Step05C_Banglish_Norm'
DATASET = Path('/content/drive/MyDrive/University_RAG_Colab/Stage_A_5PDF_Pilot_Preparation_2026-10-09/stage_a_5pdf_questions.csv')
GATE = ROADMAP / 'Step03_Stage_A_v10/SAFETY_GATE_PASSED.json'
FLAGS = {
    'RAG_BANGLISH_NORM': 'fuse',
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
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs() -> dict:
    gate = json.loads(GATE.read_text(encoding='utf-8'))
    if sha(DATASET) != '70c968d623167cef2720cec59bf789ce9433401fe68d278097b3bcdb708662eb':
        raise ValueError('Stage A dataset SHA differs from safety gate')
    for name, digest in gate['index_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'index mismatch: {name}')
    config_files = [*sorted((ROOT / 'config').glob('*.yaml')),
                    ROOT / 'config/banglish_rewrite_prompt.txt']
    model_files = [ROOT / 'models' / name for name in gate['model_hashes']]
    recorded = STEP / 'integrity/model_hashes.txt'
    hashes = {}
    if recorded.exists():
        for line in recorded.read_text(encoding='utf-8').splitlines():
            digest, _, name = line.partition('  ')
            hashes[Path(name).name] = digest
    if hashes != gate['model_hashes']:
        raise ValueError('model integrity hashes missing or different from safety gate')
    model_stats = {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                            'sha256': hashes[p.name]} for p in model_files}
    return {'dataset_sha256': sha(DATASET), 'index_sha256': gate['index_sha256'],
            'config_sha256': {str(p.relative_to(ROOT)): sha(p) for p in config_files},
            'lexicon_sha256': sha(ROOT / 'config/banglish_lexicon.yaml'),
            'rewrite_prompt_sha256': sha(ROOT / 'config/banglish_rewrite_prompt.txt'),
            'model_stats_and_recorded_sha256': model_stats,
            'rag_env': {key: os.environ[key] for key in sorted(os.environ) if key.startswith('RAG_')}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['start', 'resume'])
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    unknown = {name for name in os.environ if name.startswith('RAG_')} - set(FLAGS) - {'RAG_BANGLISH_CACHE_DIR'}
    if unknown:
        raise ValueError(f'unexpected RAG_* variables: {sorted(unknown)}')
    os.environ.update(FLAGS)
    os.environ.update(OTHER)
    if args.mode == 'start':
        if args.run_dir:
            raise ValueError('start creates its own run directory')
        run_dir = STEP / 'verification_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not args.run_dir:
            raise ValueError('resume requires --run-dir')
        run_dir = args.run_dir.resolve()
    os.environ['RAG_BANGLISH_CACHE_DIR'] = str(run_dir / 'rewrite_cache')
    actual = inputs()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluate_benchmark.py'),
               '--dataset', str(DATASET), '--output-dir', str(run_dir),
               '--run-name', 'stage-a-05c-banglish-fuse', '--classification', 'PILOT',
               '--require-references', '--checkpoint-every', '10']
    manifest_path = run_dir / 'launch_manifest.json'
    if args.mode == 'start':
        manifest = {'code_commit': commit, **actual, 'command_argv': command,
                    'command_line': subprocess.list2cmdline(command),
                    'started_at_utc': datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key, value in {'code_commit': commit, **actual}.items():
            if manifest.get(key) != value:
                raise ValueError(f'unsafe resume: {key} differs')
        children = [p for p in run_dir.iterdir() if p.is_dir() and (p / 'run_manifest.json').exists()]
        if len(children) != 1:
            raise ValueError('resume needs exactly one checkpointed child run')
        command += ['--resume', str(children[0])]
    subprocess.run(command, check=True, env=os.environ.copy())


if __name__ == '__main__':
    main()
