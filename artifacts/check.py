#!/usr/bin/env python3
"""Check the local artifact inventory against its manifest and archived predictions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
AUDIT_MANIFEST = ROOT.parent / 'analysis/silence_audit/inputs/prediction_manifest.csv'
CONFIGURATIONS = ('baseline', 'cife', 'phase1_random', 'slot_uniform', 'phase2_energy',
                  'phase3_activation', 'afri_direct', 'afri_inverse')
SEEDS = (1234, 2026, 3009)
REQUIRED_ROLES = {'checkpoint', 'config', 'overrides', 'training_log', 'evaluation_log',
                  'selected_raw', 'selected_final', 'selection'}
SHARED_ROLES = {'environment', 'normalization', 'afri_profiles'}
OPTIONAL_ROLES = SHARED_ROLES | {'source_snapshot'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Invalid JSON number: ' + value)


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _reference(root, name, prefix):
    _require(isinstance(name, str) and name and '\\' not in name and ':' not in name
             and '\0' not in name, 'Invalid artifact path: ' + repr(name))
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and '..' not in relative.parts
             and relative.as_posix() == name and name.startswith(prefix),
             'Invalid artifact path: ' + name)
    path = root
    for part in relative.parts:
        path = path / part
        _require(not path.is_symlink(), 'Artifact symlink: ' + name)
    return path


def _local_files(root):
    def fail(error):
        raise error

    files = set()
    for folder in ('runs', 'shared'):
        start = root / folder
        _require(not start.is_symlink(), 'Artifact symlink: ' + folder)
        if not start.exists():
            continue
        _require(start.is_dir(), 'Expected artifact directory: ' + folder)
        for directory, directories, names in os.walk(str(start), onerror=fail):
            for name in directories + names:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                _require(not path.is_symlink(), 'Artifact symlink: ' + relative)
                mode = path.stat().st_mode
                _require(stat.S_ISREG(mode) or stat.S_ISDIR(mode),
                         'Non-regular artifact: ' + relative)
                if stat.S_ISREG(mode):
                    files.add(relative)
    return files


def _audit_predictions(path, expected_digest):
    _require(path.is_file() and not path.is_symlink(), 'Missing regular audit manifest')
    _require(_digest(path) == expected_digest, 'Audit prediction manifest hash mismatch')
    expected = {(name, seed) for name in CONFIGURATIONS for seed in SEEDS}
    predictions = {}
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ['configuration', 'seed', 'bundle_path', 'sha256'],
                 'Unexpected audit prediction manifest columns')
        for row in reader:
            _require(set(row) == set(reader.fieldnames) and None not in row.values(),
                     'Malformed audit prediction row')
            _require(row['configuration'] in CONFIGURATIONS and row['seed'] in tuple(map(str, SEEDS)),
                     'Unknown audit configuration/seed')
            name, seed = row['configuration'], int(row['seed'])
            key = (name, seed)
            _require(key not in predictions, 'Duplicate audit configuration/seed')
            _require(row['bundle_path'] == 'inputs/predictions/{}/seed_{}.csv'.format(name, seed)
                     and _valid_digest(row['sha256']), 'Invalid audit prediction binding')
            predictions[key] = row['sha256']
    _require(set(predictions) == expected, 'Audit prediction cohort must cover all 24 runs')
    return predictions


def check_artifacts(root, audit_manifest=AUDIT_MANIFEST):
    root, audit_manifest = Path(root).absolute(), Path(audit_manifest)
    _require(root.is_dir() and not root.is_symlink(), 'Expected regular artifact directory')
    path = root / 'manifest.json'
    _require(path.is_file() and not path.is_symlink(), 'Missing regular artifact manifest')
    manifest = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    fields = {'schema_version', 'cohort_id', 'python_version',
              'audit_prediction_manifest_sha256', 'shared_files', 'runs'}
    _require(isinstance(manifest, dict) and set(manifest) == fields, 'Unexpected manifest fields')
    _require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1,
             'Unsupported manifest schema')
    _require(isinstance(manifest['cohort_id'], str)
             and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', manifest['cohort_id']),
             'Invalid cohort identity')
    _require(manifest['python_version'] == '3.7.17', 'Expected research Python 3.7.17')
    _require(_valid_digest(manifest['audit_prediction_manifest_sha256']), 'Invalid audit manifest hash')
    predictions = _audit_predictions(audit_manifest, manifest['audit_prediction_manifest_sha256'])
    _require(isinstance(manifest['shared_files'], dict)
             and set(manifest['shared_files']) <= SHARED_ROLES, 'Unknown shared file roles')
    _require(isinstance(manifest['runs'], list), 'Runs must be a list')
    report = dict(cohort_id=manifest['cohort_id'], runs=len(predictions), registered_files=0,
                  verified_files=0, audit_predictions_verified=0, missing=[],
                  unregistered_files=[], source_commits_missing=[],
                  model_compatibility_verified=False, checkpoint_prediction_binding_verified=False)
    registered = set()

    def inspect(record, run_id, role, prefix, prediction_hash=None):
        if record is None:
            report['missing'].append(dict(run_id=run_id, role=role, path=None))
            return
        _require(isinstance(record, dict) and {'path', 'sha256', 'size_bytes'} <= set(record)
                 <= {'path', 'sha256', 'size_bytes', 'url'}, 'Invalid file record: ' + run_id + '/' + role)
        _require(_valid_digest(record['sha256']), 'Invalid file hash: ' + run_id + '/' + role)
        _require(type(record['size_bytes']) is int and record['size_bytes'] > 0,
                 'Expected positive file size: ' + run_id + '/' + role)
        local = _reference(root, record['path'], prefix)
        _require(record['path'] not in registered, 'Duplicate artifact path: ' + record['path'])
        registered.add(record['path'])
        report['registered_files'] += 1
        if 'url' in record:
            _require(isinstance(record['url'], str), 'Invalid artifact URL')
            url = urlsplit(record['url'])
            _require(url.scheme == 'https' and url.hostname and not url.username
                     and not url.password and not url.fragment, 'Expected HTTPS artifact URL without credentials/fragment')
        if prediction_hash is not None:
            _require(record['sha256'] == prediction_hash, 'Selected prediction differs from audit: ' + run_id)
        if not local.exists():
            report['missing'].append(dict(run_id=run_id, role=role, path=record['path']))
            return
        _require(local.is_file(), 'Expected regular file: ' + record['path'])
        _require(local.stat().st_size == record['size_bytes'], 'Artifact size mismatch: ' + record['path'])
        _require(_digest(local) == record['sha256'], 'Artifact hash mismatch: ' + record['path'])
        report['verified_files'] += 1
        if prediction_hash is not None:
            report['audit_predictions_verified'] += 1

    for role, record in sorted(manifest['shared_files'].items()):
        inspect(record, 'shared', role, 'shared/')
    seen = set()
    for run in manifest['runs']:
        _require(isinstance(run, dict) and set(run) == {'configuration', 'seed', 'source_commit',
                 'audit_prediction_sha256', 'files'}, 'Unexpected run fields')
        _require(run['configuration'] in CONFIGURATIONS and type(run['seed']) is int
                 and run['seed'] in SEEDS, 'Unknown configuration/seed')
        key = (run['configuration'], run['seed'])
        _require(key not in seen, 'Duplicate configuration/seed')
        seen.add(key)
        run_id = '{}_seed{}'.format(*key)
        commit = run['source_commit']
        _require(commit is None or (isinstance(commit, str)
                 and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit)), 'Invalid source commit: ' + run_id)
        if commit is None:
            report['source_commits_missing'].append(run_id)
        _require(run['audit_prediction_sha256'] == predictions[key], 'Audit binding differs: ' + run_id)
        files = run['files']
        _require(isinstance(files, dict) and REQUIRED_ROLES <= set(files) <= REQUIRED_ROLES | OPTIONAL_ROLES,
                 'Unexpected file roles: ' + run_id)
        for role, record in sorted(files.items()):
            inspect(record, run_id, role, 'runs/{}/seed_{}/'.format(*key),
                    predictions[key] if role == 'selected_final' else None)
    _require(seen == set(predictions), 'Artifact cohort must cover all 24 runs')
    report['unregistered_files'] = sorted(_local_files(root) - registered)
    report['inventory_complete'] = not report['missing'] and not report['unregistered_files']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=ROOT)
    parser.add_argument('--audit-manifest', type=Path, default=AUDIT_MANIFEST)
    parser.add_argument('--require-complete', action='store_true', help='Fail for missing/unregistered payloads')
    parser.add_argument('--json', action='store_true', help='Print the full inventory report')
    args = parser.parse_args(argv)
    try:
        report = check_artifacts(args.artifacts_root, args.audit_manifest)
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print('{}: {} runs; {} registered files, {} verified.'.format(
            report['cohort_id'], report['runs'], report['registered_files'], report['verified_files']))
        print('Missing: {} slots/files; unregistered local files: {}; audit predictions verified: {}/24.'.format(
            len(report['missing']), len(report['unregistered_files']), report['audit_predictions_verified']))
        print('Inventory {}. Model compatibility and historical checkpoint bindings remain unverified.'.format(
            'complete' if report['inventory_complete'] else 'incomplete'))
    return 2 if args.require_complete and not report['inventory_complete'] else 0


if __name__ == '__main__':
    sys.exit(main())
