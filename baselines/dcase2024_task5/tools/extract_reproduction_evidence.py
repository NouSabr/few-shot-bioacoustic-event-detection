#!/usr/bin/env python3
"""Record selected logged settings from bounded cached-run headers."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from reproduce import LOGGED_SHARED_FIELDS, ROOT, exists, read_presets, require, sha

MAX_LINES = 390
MAX_BYTES = 512 * 1024
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
ARCHIVES = {1234: 'sweep_archive_seeds1234_2026',
            2026: 'sweep_archive_seeds1234_2026', 3009: 'sweep_archive_seed_3'}


def source_path(path):
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def scalar(value):
    try:
        return json.loads(value)
    except ValueError:
        return value


def prefix(path):
    chunks = []
    size = 0
    with path.open('rb') as handle:
        for _ in range(MAX_LINES):
            line = handle.readline(MAX_BYTES - size + 1)
            if not line:
                break
            size += len(line)
            require(size <= MAX_BYTES, 'Header exceeds byte limit: ' + str(path))
            chunks.append(line)
    require(len(chunks) == MAX_LINES, 'Incomplete cached header: ' + str(path))
    raw = b''.join(chunks)
    lines = [ANSI.sub('', line.decode('utf-8')).rstrip() for line in chunks]
    return lines, dict(first_line=1, line_count=len(lines), bytes_read=size,
                       sha256=hashlib.sha256(raw).hexdigest())


def observed(lines, key):
    leaf = key.rsplit('.', 1)[-1]
    values, locations = [], []
    for index, line in enumerate(lines):
        if key == 'seed':
            if not re.fullmatch(r'[├└]── seed', line):
                continue
            match = re.search(r'(\d+)\s*$', lines[index + 1])
            require(match is not None, 'Unparsed root seed')
            value = int(match.group(1))
            location = index + 2
        else:
            match = re.search(r'\b' + re.escape(leaf) + r':\s*(.*?)\s*$', line)
            if not match:
                continue
            if leaf == 'cife_pool_size':
                items = [re.search(r'-\s*(\d+)\s*$', item)
                         for item in lines[index + 1:index + 3]]
                require(len(items) == 2 and all(items), 'Unparsed CIFE pool size')
                value = [int(item.group(1)) for item in items]
            else:
                value = scalar(match.group(1))
            location = index + 1
        values.append(value)
        locations.append(location)
    require(values, 'Missing printed field: ' + key)
    require(all(value == values[0] for value in values), 'Conflicting printed copies of ' + key)
    return values[0], locations


def extract(archive_root, prediction_manifest):
    presets = read_presets()
    with prediction_manifest.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    predictions = {(row['configuration'], int(row['seed'])): row['sha256'] for row in rows}
    expected = {(name, seed) for seed in ARCHIVES for name in presets}
    require(len(rows) == len(predictions) and set(predictions) == expected,
            'Prediction manifest must identify exactly the selected 8 x 3 cohort')
    common = None
    runs = []
    for seed, archive in ARCHIVES.items():
        for name, overrides in presets.items():
            relative = Path(archive) / f'seed_{seed}' / name
            folder = archive_root / relative
            selection_path = folder / 'selection.json'
            selection_bytes = selection_path.read_bytes()
            selection = json.loads(selection_bytes)
            run_id = f'{name}_seed{seed}'
            require(selection['configuration'] == name and selection['seed'] == seed,
                    'Selection identity differs: ' + run_id)
            require(selection['checksums']['final_csv_sha256'] == predictions[name, seed],
                    'Selected-output hash differs from audit manifest: ' + run_id)
            lines, header = prefix(folder / 'stdout.log')
            logged_name, _ = observed(lines, 'exp_name')
            require(isinstance(logged_name, str) and logged_name.startswith(run_id + '_'),
                    'Printed run name differs: ' + run_id)
            fields = {key: observed(lines, key) for key in LOGGED_SHARED_FIELDS}
            shared = {key: value for key, (value, _) in fields.items()}
            if common is None:
                common = shared
            require(shared == common, 'Shared CIFE fields disagree across runs: ' + run_id)
            for key in ('seed', 'train_param.freq_mask_seed'):
                fields[key] = observed(lines, key)
                require(type(fields[key][0]) is int and fields[key][0] == seed,
                        'Printed seed differs: ' + run_id + ':' + key)
            for override in overrides:
                key, expected_value = override.split('=', 1)
                fields[key] = observed(lines, key)
                require(fields[key][0] == scalar(expected_value),
                        'Preset disagrees with printed field: ' + run_id + ':' + key)
            selected_fields = {key: value for key, (value, _) in fields.items()}
            fields_bytes = json.dumps(selected_fields, sort_keys=True, separators=(',', ':')).encode()
            runs.append(dict(
                run_id=run_id, configuration=name, seed=seed, logged_exp_name=logged_name,
                log_prefix=dict(source=source_path(folder / 'stdout.log'), **header),
                printed_field_lines={key: locations for key, (_, locations) in fields.items()},
                printed_fields_sha256=hashlib.sha256(fields_bytes).hexdigest(),
                selection_metadata=dict(source=source_path(selection_path),
                                        sha256=hashlib.sha256(selection_bytes).hexdigest()),
                declared_full_log_sha256=selection['checksums']['log_sha256'],
                selected_prediction_sha256=predictions[name, seed],
                selected_operating_point=selection['selection'],
                historical_source_commit=selection.get('provenance', {}).get('source_git_commit'),
                checkpoint_binding=None,
            ))
    return dict(
        schema_version=1,
        scope='Selected printed CIFE/preset fields and global/masking seeds. Full historical configs and checkpoint/source-code identity are outside the extraction scope.',
        source_path_base='Implementation directory. Cache paths record provenance; planning requires no cache access.',
        producer=dict(source='tools/extract_reproduction_evidence.py', sha256=sha(Path(__file__)),
                      helpers=dict(source='tools/reproduce.py', sha256=sha(ROOT / 'tools/reproduce.py'))),
        preset_source=dict(source='run_ablation_experiments.py', sha256=sha(ROOT / 'run_ablation_experiments.py')),
        prediction_manifest=dict(source=source_path(prediction_manifest),
                                 sha256=sha(prediction_manifest)),
        configurations=list(presets), seeds=list(ARCHIVES),
        common_overrides=[key + '=' + json.dumps(value, separators=(',', ':')) for key, value in common.items()],
        preset_overrides=presets,
        read_limits=dict(max_physical_lines_per_log=MAX_LINES, max_bytes_per_log=MAX_BYTES),
        verification=dict(all_selected_fields_consistent=True, all_preset_overrides_match=True,
                          both_seeds_match_labels=True, selection_hashes_match_prediction_manifest=True,
                          full_log_hashes_recomputed=False, prediction_files_rehashed=False,
                          full_historical_configuration_recovered=False, checkpoint_binding_verified=False,
                          historical_source_identity_established=False),
        operating_point_note='Archived label-selected output choices. They are not training overrides or a deployable selection rule. The planner does not apply them.',
        runs=runs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=ROOT / 'analysis')
    parser.add_argument('--prediction-manifest', type=Path,
                        default=ROOT.parents[1] / 'analysis/silence_audit/inputs/prediction_manifest.csv')
    parser.add_argument('--output', type=Path, required=True, help='NEW JSON file; never overwritten')
    args = parser.parse_args(argv)
    try:
        require(not exists(args.output), 'Refusing to overwrite evidence path: ' + str(args.output))
        evidence = extract(args.archive_root, args.prediction_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
        print(f'Recorded {len(evidence["runs"])} runs; at most {MAX_LINES} lines per log; no workflows executed.')
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, SyntaxError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
