#!/usr/bin/env python3
"""Plan reproduction runs and optionally execute data preparation."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from preparation_execution import execute_preparation, inspect_preparation

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'train', 'evaluate')
CHECKPOINT_FIELDS = {'path', 'sha256', 'config_path', 'config_sha256'}
LOGGED_SHARED_FIELDS = ('features.cife_nhead', 'features.cife_pool_size')
LIMITS = [
    'Only preparation supports explicit execution; training and evaluation remain planning-only.',
    'Hydra composition is pending; the report contains source hashes and override arguments.',
    'Environment checks cover package metadata. Imports, CUDA and native libraries require separate validation.',
    'Checkpoint hashes identify files. Historical-run binding and model compatibility require separate validation.',
    'Pinned log settings provide selected-field evidence. Full checkpoint/config compatibility and historical source identity remain unverified.',
]
ENVIRONMENT_PROBE = '''\
import json, platform, re
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
versions, direct_urls = {}, {}
for dist in metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        key = re.sub(r'[-_.]+', '-', name.lower())
        versions[key] = dist.version
        direct = dist.read_text('direct_url.json')
        if direct:
            direct_urls[key] = json.loads(direct)
print(json.dumps(dict(python=platform.python_version(), packages=versions, direct_urls=direct_urls)))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def path_from(value, base):
    require(isinstance(value, str) and value.strip(), 'Paths must be nonempty strings')
    path = Path(value).expanduser()
    # Resolving this symlink would bypass the selected virtualenv.
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def exists(path):
    return path.exists() or path.is_symlink()


def new_target(path):
    return not exists(path) and all(not exists(parent) or parent.is_dir() for parent in path.parents)


def read_presets():
    tree = ast.parse((ROOT / 'run_ablation_experiments.py').read_text(encoding='utf-8'))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS' for t in node.targets)]
    require(len(values) == 1, 'Expected one literal EXPERIMENTS definition')
    presets = ast.literal_eval(values[0])
    presets = {preset['name']: preset['overrides'] for preset in presets}
    return presets


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    required = {'schema_version', 'python', 'dataset_root', 'output_root', 'seeds',
                'configurations', 'overrides', 'checkpoints'}
    require(isinstance(manifest, dict) and required <= set(manifest) <= required | {'logged_evidence'},
            'Manifest requires: ' + ', '.join(sorted(required)) + '; optional: logged_evidence')
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1,
            'Unsupported schema_version')
    for name in ('seeds', 'configurations', 'overrides'):
        require(isinstance(manifest[name], list), name + ' must be a list')
    seeds = manifest['seeds']
    require(seeds and all(type(seed) is int and 0 < seed < 2 ** 32 for seed in seeds),
            'seeds must be explicit positive integers below 2**32 (entry points skip seed 0)')
    require(len(seeds) == len(set(seeds)), 'Duplicate seeds are not independent replicates')
    names = manifest['configurations']
    require(names and all(isinstance(name, str) for name in names),
            'configurations must be a nonempty list of preset names')
    require(len(names) == len(set(names)), 'Duplicate configurations')
    presets = read_presets()
    require(set(names) <= set(presets), 'Unknown configurations: ' + ', '.join(sorted(set(names) - set(presets))))
    protected = {o.split('=', 1)[0] for values in presets.values() for o in values}
    protected |= {'seed', 'name', 'exp_name', 'ckpt_path', 'test', 'train', 'local',
                  'path', 'features', 'train_param', 'model', 'datamodule', 'experiment',
                  'debug', 'hparams_search', 'set', 'logger.tensorboard.save_dir'}
    seen_overrides = set()
    for override in manifest['overrides']:
        require(isinstance(override, str) and '=' in override, 'overrides must contain KEY=VALUE strings')
        key = override.split('=', 1)[0].lstrip('+')
        require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', key) is not None,
                'Only ordinary KEY=VALUE overrides are supported')
        require(key not in seen_overrides, 'Duplicate override key: ' + key)
        seen_overrides.add(key)
        require(key not in protected and not key.startswith(('path.', 'hydra.', 'set.', 'model.', 'logger.', 'datamodule.path',
                           'datamodule.features', 'datamodule.train_param', 'train_param.freq_mask_seed')),
                'Override would replace a managed path, seed or experiment setting: ' + key)
    require(manifest['overrides'].count('logger=tensorboard') == 1,
            'This planner requires exactly one logger=tensorboard override')
    require(all(not o.startswith(('logger=', '+logger=', '++logger=')) or o == 'logger=tensorboard'
                for o in manifest['overrides']), 'Only the local tensorboard logger is supported')
    checkpoints = manifest['checkpoints']
    require(isinstance(checkpoints, dict), 'checkpoints must be an object keyed by CONFIG_seedSEED')
    run_ids = {f'{name}_seed{seed}' for name in names for seed in seeds}
    require(set(checkpoints) <= run_ids, 'Checkpoint registered for a run outside this plan')
    for run_id, record in checkpoints.items():
        require(isinstance(record, dict) and set(record) == CHECKPOINT_FIELDS,
                run_id + ': checkpoint needs path, sha256, config_path, config_sha256')
        for key in ('sha256', 'config_sha256'):
            require(isinstance(record[key], str) and re.fullmatch(r'[0-9a-f]{64}', record[key]) is not None,
                    run_id + ': ' + key + ' must be a lowercase SHA-256 digest')
        for key in ('path', 'config_path'):
            path_from(record[key], path.parent)
    for key in ('python', 'dataset_root', 'output_root'):
        path_from(manifest[key], path.parent)
    return manifest, presets


def read_logged_evidence(manifest, base, presets):
    reference = manifest.get('logged_evidence')
    if reference is None:
        return None, {}
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'},
            'logged_evidence needs path and sha256')
    require(isinstance(reference['sha256'], str) and
            re.fullmatch(r'[0-9a-f]{64}', reference['sha256']) is not None,
            'logged_evidence.sha256 must be a lowercase SHA-256 digest')
    path = path_from(reference['path'], base)
    require(path.stat().st_size <= 1024 * 1024, 'Logged evidence must be a compact JSON record')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Logged evidence hash differs')
    evidence = json.loads(raw)
    require(evidence['schema_version'] == 1, 'Unsupported logged-evidence schema')
    keys = [o.split('=', 1)[0] for o in evidence['common_overrides']]
    require(len(keys) == len(LOGGED_SHARED_FIELDS) and set(keys) == set(LOGGED_SHARED_FIELDS),
            'Logged evidence must identify both shared CIFE settings exactly once')
    require(all(evidence['verification'][key] is True for key in (
        'all_selected_fields_consistent', 'all_preset_overrides_match',
        'both_seeds_match_labels', 'selection_hashes_match_prediction_manifest')),
        'Logged-evidence verification is incomplete')
    overrides = {o.split('=', 1)[0].lstrip('+'): o.split('=', 1)[1]
                 for o in manifest['overrides']}
    for override in evidence['common_overrides']:
        key, value = override.split('=', 1)
        require(key in overrides and json.dumps(json.loads(overrides[key]), sort_keys=True) ==
                json.dumps(json.loads(value), sort_keys=True),
                'Manifest must explicitly preserve logged override: ' + override)
    for name in manifest['configurations']:
        require(evidence['preset_overrides'].get(name) == presets[name],
                'Current preset differs from logged evidence: ' + name)
    runs = {(r['configuration'], r['seed']): r for r in evidence['runs']}
    require(len(runs) == len(evidence['runs']), 'Duplicate run in logged evidence')
    require(all(r['run_id'] == f"{r['configuration']}_seed{r['seed']}" for r in evidence['runs']),
            'Logged-evidence run identity differs')
    expected = {(name, seed) for name in evidence['configurations'] for seed in evidence['seeds']}
    require(set(runs) == expected, 'Logged-evidence cohort is incomplete')
    requested = {(name, seed) for name in manifest['configurations'] for seed in manifest['seeds']}
    require(requested <= set(runs), 'Requested run has no logged evidence; use an unlinked manifest for a new cohort')
    summary = dict(path=str(path), sha256=reference['sha256'], scope=evidence['scope'],
                   common_overrides=evidence['common_overrides'],
                   full_historical_configuration_recovered=False, checkpoint_binding_verified=False)
    return summary, runs


def source_identity(config):
    paths = {ROOT / name for name in ('train.py', 'test.py', 'data_preparation.py',
                                      'run_ablation_experiments.py', 'requirements.txt', 'tools/reproduce.py',
                                      'tools/preparation_execution.py')}
    paths.update((ROOT / 'configs').rglob('*.yaml'))
    paths.update((ROOT / 'src').rglob('*.py'))
    paths = {p for p in paths if '__pycache__' not in p.parts and 'local' not in p.relative_to(ROOT).parts}
    identity = {'implementation_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
                'manifest_sha256': sha(config), 'git_commit': None, 'git_tracked_changes': None}
    try:
        options = dict(cwd=ROOT, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
                       text=True, stderr=subprocess.DEVNULL, timeout=10)
        identity['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], **options).strip()
        identity['git_tracked_changes'] = bool(subprocess.check_output(['git', 'diff', 'HEAD', '--', '.'], **options))
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def build_plan(config, stage):
    manifest, presets = load_manifest(config)
    base = config.parent
    evidence, evidence_runs = read_logged_evidence(manifest, base, presets)
    python = path_from(manifest['python'], base)
    data = path_from(manifest['dataset_root'], base).resolve()
    output = path_from(manifest['output_root'], base).resolve()
    require(output != data and data not in output.parents and output not in data.parents,
            'output_root must not overlap dataset_root')
    stages = STAGES if stage == 'all' else (stage,)
    common = [f'path.root_dir={json.dumps(str(data))}',
              f'path.train_dir={json.dumps(str(data / "Development_Set/Training_Set"))}',
              f'path.eval_dir={json.dumps(str(data / "Development_Set/Validation_Set"))}',
              'path.test_dir=null', 'path.extra_train_dir=null', 'path.mask_dir=null']
    plan = dict(schema_version=1, status='planned', execution_enabled=False,
                supported_execution_stages=['prepare'],
                runtime_verified=False, resolved_config_compatibility_verified=False,
                workflow_commands_executed=0, stage=stage, cwd=str(ROOT), python=str(python),
                dataset_root=str(data), output_root=str(output),
                configurations=manifest['configurations'], seeds=manifest['seeds'],
                run_count=len(manifest['configurations']) * len(manifest['seeds']),
                sources=source_identity(config), limits=LIMITS, preparation=None, runs=[])
    if evidence is not None:
        plan['sources']['logged_evidence'] = evidence
    if 'prepare' in stages:
        plan['preparation'] = dict(arguments=dict(data_dir=str(data), train_dir='Training_Set',
             val_dir='Validation_Set', test_dir='Test_Set', output_format='hdf5'), argv=[str(python), str(ROOT / 'data_preparation.py'),
             '--data-dir', str(data), '--train-dir', 'Training_Set', '--val-dir', 'Validation_Set',
             '--test-dir', 'Test_Set', '--output-format', 'hdf5'],
             note='Writes features/normalization beside development data; also processes Development_Set/Test_Set if present. Runtime evaluation is validation-only.')
    for seed in manifest['seeds']:
        for name in manifest['configurations']:
            run_id = f'{name}_seed{seed}'
            overrides = [*manifest['overrides'], *presets[name], f'seed={seed}',
                         f'train_param.freq_mask_seed={seed}', f'name={run_id}', f'exp_name={run_id}', *common]
            run = dict(run_id=run_id, configuration=name, seed=seed, stages={})
            if evidence is not None:
                record = evidence_runs[name, seed]
                run['logged_evidence'] = dict(record_run_id=record['run_id'],
                    printed_fields_sha256=record['printed_fields_sha256'],
                    selected_prediction_sha256=record['selected_prediction_sha256'],
                    selected_operating_point=record['selected_operating_point'],
                    operating_point_applied=False)
            for kind in ('train', 'evaluate'):
                if kind not in stages:
                    continue
                directory = output / kind / run_id
                command = [str(python), str(ROOT / ('train.py' if kind == 'train' else 'test.py')),
                           *overrides, f'hydra.run.dir={json.dumps(str(directory))}',
                           f'logger.tensorboard.save_dir={json.dumps(str(directory / "tensorboard"))}']
                item = dict(output_dir=str(directory), argv=command, checkpoint=None)
                if kind == 'train':
                    command.extend(['local=null', 'test=false'])
                else:
                    record = manifest['checkpoints'].get(run_id)
                    if record is None:
                        item.update(argv=None, blocked_by='Register checkpoint and saved-config paths/hashes for ' + run_id)
                    else:
                        item['checkpoint'] = {**record, **{key: str(path_from(record[key], base)) for key in ('path', 'config_path')}}
                        command.append('ckpt_path=' + json.dumps(item['checkpoint']['path']))
                run['stages'][kind] = item
            plan['runs'].append(run)
    return plan


def check_environment(python):
    require(python.is_file() and os.access(python, os.X_OK), 'Interpreter is missing or not executable: ' + str(python))
    completed = subprocess.run([str(python), '-I', '-B', '-c', ENVIRONMENT_PROBE],
                               cwd=ROOT, capture_output=True, text=True, timeout=30)
    require(completed.returncode == 0, 'Could not inspect interpreter package metadata: ' + completed.stderr.strip())
    observed = json.loads(completed.stdout)
    missing, mismatched, unverified = [], [], []
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line:
            vcs = re.fullmatch(r'([\w.-]+) @ git\+(.+)@([0-9a-f]{40})', line)
            if vcs:
                name, url, commit = vcs.groups()
                key = re.sub(r'[-_.]+', '-', name.lower())
                direct = observed['direct_urls'].get(key, {})
                matches = (direct.get('url') == url and
                           direct.get('vcs_info', {}).get('commit_id') == commit)
                if key not in observed['packages']:
                    missing.append(name)
                elif not matches:
                    unverified.append(line)
            else:
                unverified.append(line)
            continue
        name, expected = line.split('==', 1)
        actual = observed['packages'].get(re.sub(r'[-_.]+', '-', name.lower()))
        if actual is None:
            missing.append(name)
        elif actual != expected:
            mismatched.append(dict(package=name, expected=expected, observed=actual))
    return dict(python=observed['python'], missing=missing, mismatched=mismatched,
                unverified_requirements=unverified, metadata_matches=not (missing or mismatched or unverified))


def preflight(plan):
    checks = []

    def check(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    try:
        environment = check_environment(Path(plan['python']))
        check('environment_metadata', environment['metadata_matches'], environment)
    except (OSError, ValueError, subprocess.SubprocessError, KeyError) as error:
        check('environment_metadata', False, str(error))
    data = Path(plan['dataset_root'])
    stages = STAGES if plan['stage'] == 'all' else (plan['stage'],)
    prepared = 'prepare' not in stages
    readiness = inspect_preparation(data) if not prepared else None
    if readiness is not None and not readiness['checks'][0]['passed']:
        plan['checks'] = checks + readiness['checks']
        plan['status'] = 'preflight_failed'
        return plan
    splits = ['Training_Set', 'Validation_Set']
    if not prepared and (data / 'Development_Set/Test_Set').is_dir():
        splits.append('Test_Set')
    for split in splits:
        directory = data / 'Development_Set' / split
        audio = sorted(directory.rglob('*.wav')) if directory.is_dir() else []
        missing_csv = [str(p.with_suffix('.csv')) for p in audio if not p.with_suffix('.csv').is_file()]
        check(split, bool(audio) and not missing_csv,
              dict(directory=str(directory), wav_files=len(audio), missing_csv=missing_csv))
        if prepared:
            missing_h5 = [str(p.with_name(p.stem + '_logmel.h5')) for p in audio
                          if not p.with_name(p.stem + '_logmel.h5').is_file()]
            check(split + '_features', bool(audio) and not missing_h5,
                  dict(missing_hdf5=missing_h5, contents_validated=False))
        else:
            existing = [str(p) for p in directory.rglob('*') if (p.is_file() or p.is_symlink()) and
                        (p.name.endswith(('_logmel.h5', '_logmel.npy', '_logmel_unnormalized.h5',
                                          '_logmel_unnormalized.npy', '_logmel_norm_comparison.png')))] if directory.is_dir() else []
            check(split + '_preparation_targets_new', not existing, existing)
    stats = data / 'Development_Set/norm_stats.npz'
    check('normalization_statistics', stats.is_file() if prepared else not exists(stats),
          dict(path=str(stats), expected='existing' if prepared else 'absent', contents_validated=False))
    for run in plan['runs']:
        for kind, item in run['stages'].items():
            check(kind + ':' + run['run_id'] + ':output_new', new_target(Path(item['output_dir'])), item['output_dir'])
            if kind == 'evaluate':
                record = item['checkpoint']
                check(run['run_id'] + ':checkpoint_registered', record is not None,
                      item.get('blocked_by', 'Checkpoint/config hash binding supplied; compatibility not validated'))
                if record:
                    for key, digest_key in (('path', 'sha256'), ('config_path', 'config_sha256')):
                        path = Path(record[key])
                        check(run['run_id'] + ':' + key, path.is_file() and sha(path) == record[digest_key], str(path))
    if readiness is not None:
        checks.extend(readiness['checks'])
    plan['checks'] = checks
    plan['status'] = 'preflight_passed' if all(c['passed'] for c in checks) else 'preflight_failed'
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit reproduction JSON manifest')
    parser.add_argument('--stage', choices=(*STAGES, 'all'), help='Execution requires explicit --stage prepare')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Print command plan without checking the research environment')
    mode.add_argument('--preflight', action='store_true', help='Check file/layout/package metadata; never launch a workflow')
    mode.add_argument('--execute', action='store_true', help='Execute preparation only, after automatic preflight')
    parser.add_argument('--report-dir', type=Path, help='Optional NEW directory for report.json; never overwritten')
    args = parser.parse_args(argv)
    if args.execute and (args.stage != 'prepare' or args.report_dir is None):
        parser.error('--execute requires explicit --stage prepare and a NEW --report-dir; other stages remain disabled')
    try:
        if args.report_dir is not None:
            require(not exists(args.report_dir), 'Refusing to overwrite report directory: ' + str(args.report_dir))
        config = args.config.resolve()
        plan = build_plan(config, args.stage or 'all')
        if args.preflight or args.execute:
            plan = preflight(plan)
        if args.execute:
            plan, exit_code = execute_preparation(plan, args.report_dir, config)
            sys.stdout.write(json.dumps(plan, indent=2, sort_keys=True) + '\n')
            return exit_code
        text = json.dumps(plan, indent=2, sort_keys=True) + '\n'
        if args.report_dir is not None:
            args.report_dir.mkdir(parents=True, exist_ok=False)
            (args.report_dir / 'report.json').write_text(text, encoding='utf-8')
        sys.stdout.write(text)
        return 1 if plan['status'] == 'preflight_failed' else 0
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
