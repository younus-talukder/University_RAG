import csv
import json
from pathlib import Path

from scripts.launch_5b_final import REFERENCE_LAUNCH, command, frozen_inputs
from src.evaluation.runner import RESULT_FIELDS, infer_one


def test_frozen_manifest_inputs_match_saved_r2(monkeypatch):
    saved = json.loads(REFERENCE_LAUNCH.read_text(encoding='utf-8'))
    for key, value in saved['rag_env'].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('GENERATOR_GPU_LAYERS', '0')
    monkeypatch.setenv('RERANKER_ENABLED', 'false')
    actual = frozen_inputs()
    for key in ('answer_policy_sha256', 'field_policy_sha256', 'frozen_env_sha256',
                'dataset_sha256', 'index_manifest_sha256'):
        assert actual[key] == saved[key]
    assert len(actual['index_sha256']) == 4


def test_launcher_command_and_exported_card_fields(tmp_path):
    argv = command(tmp_path)
    assert argv[-2:] == ['--checkpoint-every', '10']
    assert '--require-references' in argv
    assert {'route', 'card_id'} <= set(RESULT_FIELDS)


def test_card_route_and_id_survive_benchmark_inference():
    descriptor = dict(evaluation_id='EX:english', base_question_id='EX', language='english',
                      question='What is EX 101 credit?', reference_answer='3', expected_sources=[],
                      expected_pages=[], expected_evidence=[], annotation_status='verified',
                      intent='', difficulty='', course_code='EX 101')
    response = dict(answerability_status='SUPPORTED', final_answer_allowed=True,
                    answer='EX 101 has 3 credits.', answer_strategy='card', card_id='synthetic',
                    source='outline.pdf', page=2, evidence=[])
    row = infer_one(descriptor, 'test', infer=lambda *a, **k: response)
    assert row['route'] == 'card' and row['card_id'] == 'synthetic'
    assert row['answerability_status'] == 'SUPPORTED'
