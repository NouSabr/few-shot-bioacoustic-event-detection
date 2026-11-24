#!/usr/bin/env python3
"""Check the shipped run plan and its committed evidence references."""

import argparse
import csv
import json
from pathlib import Path
import sys

from reproduce import ROOT, read_presets, require, sha

REPOSITORY = ROOT.parents[1]
CONFIGURATIONS = ('baseline', 'cife', 'phase1_random', 'slot_uniform', 'phase2_energy',
                  'phase3_activation', 'afri_direct', 'afri_inverse')
SEEDS = (1234, 2026, 3009)


def reference(base, value):
    path = (base / value).resolve()
    inside_checkout = True
    try:
        path.relative_to(REPOSITORY)
    except ValueError:
        inside_checkout = False
    require(inside_checkout and path.is_file(), 'Missing checkout reference: ' + str(path))
    return path


def check_plan(plan, config):
    manifest = json.loads(config.read_text(encoding='utf-8'))
    require(manifest['configurations'] == list(CONFIGURATIONS), 'Example configuration cohort changed')
    require(manifest['seeds'] == list(SEEDS), 'Example seed cohort changed')
    require(set(read_presets()) == set(CONFIGURATIONS), 'Ablation preset cohort changed')
    linked = manifest['logged_evidence']
    evidence_path = reference(config.parent, linked['path'])
    require(sha(evidence_path) == linked['sha256'], 'Evidence hash mismatch')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    require(evidence['configurations'] == list(CONFIGURATIONS) and evidence['seeds'] == list(SEEDS),
            'Evidence cohort labels differ')
    for name in ('train.py', 'test.py', 'data_preparation.py', 'requirements.txt',
                 'configs/train.yaml', 'configs/test.yaml', 'tools/reproduce.py',
                 'tools/preparation_execution.py'):
        reference(ROOT, name)
    # Producer/helper hashes identify the extraction snapshot.
    for source in (evidence['producer'], evidence['producer']['helpers'], evidence['preset_source']):
        reference(ROOT, source['source'])
    prediction_source = evidence['prediction_manifest']
    prediction_path = reference(ROOT, prediction_source['source'])
    require(sha(prediction_path) == prediction_source['sha256'], 'Prediction-manifest hash mismatch')
    with prediction_path.open(newline='', encoding='utf-8') as handle:
        predictions = list(csv.DictReader(handle))
    selected = {(r['configuration'], int(r['seed'])): r['sha256'] for r in predictions}
    expected = {(name, seed) for seed in SEEDS for name in CONFIGURATIONS}
    require(set(selected) == expected and len(predictions) == len(expected), 'Prediction cohort differs')
    recorded = {(r['configuration'], r['seed']): r for r in evidence['runs']}
    require(set(recorded) == expected and len(evidence['runs']) == len(expected), 'Evidence cohort differs')
    presets = read_presets()
    require(evidence['preset_overrides'] == presets, 'Presets differ from recorded settings')
    require(plan['stage'] == 'all' and plan['status'] == 'planned', 'Expected an all-stage dry-run plan')
    require(plan['execution_enabled'] is False and plan['workflow_commands_executed'] == 0,
            'Dry-run plan indicates workflow execution')
    require(plan['configurations'] == list(CONFIGURATIONS) and plan['seeds'] == list(SEEDS),
            'Plan cohort differs from the example')
    require(plan['run_count'] == len(expected) and len(plan['runs']) == len(expected), 'Expected 24 runs')
    require(plan['sources']['manifest_sha256'] == sha(config), 'Plan was generated from a different manifest')
    require(plan['sources']['logged_evidence']['sha256'] == linked['sha256'], 'Plan evidence binding differs')
    observed = [(r['configuration'], r['seed']) for r in plan['runs']]
    require(observed == [(name, seed) for seed in SEEDS for name in CONFIGURATIONS],
            'Runs must be unique and seed-major')
    require(plan['preparation'] is not None, 'Preparation is missing from the plan')
    for run in plan['runs']:
        name, seed = run['configuration'], run['seed']
        run_id = f'{name}_seed{seed}'
        require(run['run_id'] == run_id and set(run['stages']) == {'train', 'evaluate'},
                'Run identity/stages differ: ' + run_id)
        row = recorded[name, seed]
        linked_run = run['logged_evidence']
        require(linked_run['record_run_id'] == run_id and
                linked_run['printed_fields_sha256'] == row['printed_fields_sha256'] and
                linked_run['selected_prediction_sha256'] == row['selected_prediction_sha256'] == selected[name, seed],
                'Run evidence binding differs: ' + run_id)
        require(linked_run['selected_operating_point'] == row['selected_operating_point'],
                'Operating-point metadata differs: ' + run_id)
        require(linked_run['operating_point_applied'] is False, 'Archived operating point applied: ' + run_id)
        for stage, script in (('train', 'train.py'), ('evaluate', 'test.py')):
            command = run['stages'][stage]['argv']
            if stage == 'evaluate' and run_id not in manifest['checkpoints']:
                require(command is None and run['stages'][stage].get('blocked_by'),
                        'Unregistered checkpoint must stay blocked: ' + run_id)
                continue
            require(isinstance(command, list) and len(command) > 2 and
                    Path(command[1]).resolve() == ROOT / script, 'Unexpected command: ' + run_id)
            require(all(o in command for o in presets[name]) and f'seed={seed}' in command and
                    f'train_param.freq_mask_seed={seed}' in command, 'Preset or seed overrides differ: ' + run_id)
            overrides = {a.split('=', 1)[0].lstrip('+'): a.split('=', 1)[1] for a in command if '=' in a}
            heads = json.loads(overrides['features.cife_nhead'])
            pool = json.loads(overrides['features.cife_pool_size'])
            require(type(heads) is int and heads == 4 and isinstance(pool, list) and
                    pool == [2, 16] and all(type(n) is int for n in pool),
                    'Logged CIFE settings differ: ' + run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reproduction.example.json')
    args = parser.parse_args(argv)
    try:
        check_plan(json.loads(args.plan.read_text(encoding='utf-8')), args.config.resolve())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    print('Verified 8 configurations, 3 seeds, 24 planned runs and committed evidence references.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
