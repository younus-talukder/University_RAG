"""Checkpointed Step 5B pilot launcher with frozen input verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = Path('/content/drive/MyDrive/University_RAG_Colab/Roadmap_2026-11')
STEP = ROADMAP / 'Step05B_Card_Answers'
PILOT_ROOT = STEP / 'pilot_runs_stage-a-5pdf-pilot-v10-cards-final-r2'
REFERENCE_LAUNCH = PILOT_ROOT / '20261114T065740263235Z' / 'launch_manifest.json'
DATASET = Path('/content/drive/MyDrive/University_RAG_Colab/Stage_A_5PDF_Pilot_Preparation_2026-10-09/stage_a_5pdf_questions.csv')
SAFETY = ROADMAP / 'Step03_Stage_A_v10/SAFETY_GATE_PASSED.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_commit() -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def frozen_inputs() -> dict:
    expected = json.loads(REFERENCE_LAUNCH.read_text(encoding='utf-8'))
    gate = json.loads(SAFETY.read_text(encoding='utf-8'))
    checks = {
        'frozen_env_sha256': STEP / 'scripts/frozen_env.sh',
        'dataset_sha256': DATASET,
        'index_manifest_sha256': ROOT / 'vector_db/index_manifest.json',
        'field_policy_sha256': STEP / 'field_policy.csv',
        'answer_policy_sha256': ROOT / 'config/answer_policy.yaml',
    }
    actual = {name: sha(path) for name, path in checks.items()}
    for name, value in actual.items():
        if value != expected[name]:
            raise ValueError(f'frozen input mismatch: {name}')
    for relative, digest in gate['index_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Stage A index mismatch: {relative}')
    flags = {k: v for k, v in os.environ.items() if k.startswith('RAG_')}
    if flags != expected['rag_env']:
        raise ValueError('frozen RAG_* flags mismatch')
    if os.environ.get('GENERATOR_GPU_LAYERS') != '0' or os.environ.get('RERANKER_ENABLED') != 'false':
        raise ValueError('frozen generator/reranker settings mismatch')
    # Hash text inputs directly. Large model hashes are the recorded Stage 3
    # safety-gate values; model size and mtime are recorded before a background
    # integrity check by the run operator.
    text_inputs = [*sorted((ROOT / 'config').glob('*.yaml')), ROOT / 'src/generator.py']
    return {**actual, 'index_sha256': gate['index_sha256'],
            'text_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in text_inputs},
            'model_sha256': gate['model_hashes'],
            'model_size_mtime': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                 for p in (ROOT / 'models').glob('*.gguf')},
            'rag_env': flags}


def command(run_dir: Path, resume: Path | None = None) -> list[str]:
    args = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluate_benchmark.py'),
            '--dataset', str(DATASET), '--output-dir', str(run_dir),
            '--run-name', 'stage-a-5pdf-pilot-v10-cards-final-r2',
            '--classification', 'PILOT', '--require-references', '--checkpoint-every', '10']
    if resume:
        args += ['--resume', str(resume)]
    return args


def launch(mode: str, run_dir: Path | None = None) -> Path:
    inputs = frozen_inputs()
    if mode == 'start':
        if run_dir is not None:
            raise ValueError('start does not accept an existing run directory')
        run_dir = PILOT_ROOT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir.mkdir(parents=True, exist_ok=False)
        argv = command(run_dir)
        manifest = {'code_commit': code_commit(), **inputs, 'command_argv': argv,
                    'command_line': subprocess.list2cmdline(argv),
                    'started_at_utc': datetime.now(timezone.utc).isoformat()}
        (run_dir / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        if run_dir is None:
            raise ValueError('resume requires --run-dir')
        manifest = json.loads((run_dir / 'launch_manifest.json').read_text(encoding='utf-8'))
        for name, value in {'code_commit': code_commit(), **inputs}.items():
            if manifest.get(name) != value:
                raise ValueError(f'unsafe resume: {name} changed')
        children = [p for p in run_dir.iterdir() if p.is_dir() and (p / 'run_manifest.json').exists()]
        if len(children) != 1:
            raise ValueError('resume requires exactly one checkpointed child run')
        argv = command(run_dir, children[0])
    subprocess.run(argv, check=True)
    return run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'resume'))
    parser.add_argument('--run-dir', type=Path)
    opts = parser.parse_args()
    print(launch(opts.mode, opts.run_dir))
