"""Guard and record the existing in-place HDF5 preparation stage."""

from datetime import datetime, timezone
from glob import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

CLAIM_NAME = '.reproduction-prepare.json'


def _exists(path):
    return path.exists() or path.is_symlink()


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def inspect_preparation(data):
    development = data / 'Development_Set'
    splits = ['Training_Set', 'Validation_Set']
    if _exists(development / 'Test_Set'):
        splits.append('Test_Set')
    unsafe, inputs, targets = [], [], []
    if development.is_symlink():
        unsafe.append(str(development))
    if not development.is_symlink():
        for parent, directories, files in os.walk(development, followlinks=False):
            unsafe.extend(str(Path(parent) / name) for name in directories + files
                          if (Path(parent) / name).is_symlink())
    for split in ([] if unsafe else splits):
        directory = development / split
        # Match data_preparation.recursive_glob without importing its dependencies.
        audio = [Path(p) for depth in range(6)
                 for p in glob(str(directory.joinpath(*(['*'] * depth), '*.wav')))]
        discovered = set(audio)
        unsafe.extend(str(p) for p in directory.rglob('*.wav') if p not in discovered)
        for wav in audio:
            if not wav.is_file() or str(wav).count('.wav') != 1:
                unsafe.append(str(wav))
            for path in (wav, wav.with_suffix('.csv')):
                if path.is_file() and not path.is_symlink():
                    stat = path.stat()
                    inputs.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
            targets.append(dict(path=str(wav.with_name(wav.stem + '_logmel.h5')),
                                role='normalized_features', required=True))
            if split == 'Training_Set':
                targets.extend((
                    dict(path=str(wav.with_name(wav.stem + '_logmel_unnormalized.h5')),
                         role='unnormalized_sample', required=False),
                    dict(path=str(wav.parent / 'visualizations' / (wav.stem + '_logmel_norm_comparison.png')),
                         role='comparison_plot', required=False),
                ))
    targets.append(dict(path=str(development / 'norm_stats.npz'), role='training_statistics', required=True))
    collisions = []
    for target in targets:
        path = Path(target['path'])
        if _exists(path) or any(_exists(p) and not p.is_dir() for p in path.parents):
            collisions.append(str(path))
    claim = development / CLAIM_NAME
    checks = [dict(name='preparation_safe_input_layout', passed=not unsafe, detail=sorted(set(unsafe))),
              dict(name='preparation_output_targets_new', passed=not collisions, detail=collisions),
              dict(name='preparation_claim_absent', passed=not _exists(claim), detail=str(claim))]
    return dict(inputs=sorted(inputs, key=lambda r: r['path']), targets=targets,
                claim=str(claim), checks=checks)


def _stop(process):
    if process is None:
        return
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _artifacts(state, hash_contents=True):
    artifacts, errors = [], []
    for target in state['targets']:
        path = Path(target['path'])
        if not _exists(path) and not target['required']:
            continue
        try:
            if (not path.is_file() or path.is_symlink() or
                    any(p.is_symlink() for p in path.parents) or not path.stat().st_size):
                errors.append(str(path))
            else:
                artifacts.append(dict(**target, bytes=path.stat().st_size,
                                      sha256=_sha(path) if hash_contents else None))
        except OSError as error:
            errors.append(str(path) + ': ' + str(error))
    return artifacts, errors


def execute_preparation(plan, report_dir, config):
    if plan['stage'] != 'prepare':
        raise ValueError('Only preparation supports execution')
    data = Path(plan['dataset_root'])
    report_dir = report_dir.resolve()
    if report_dir == data or data in report_dir.parents or report_dir in data.parents:
        raise ValueError('Execution report directory must not overlap dataset_root')
    report_dir.mkdir(parents=True, exist_ok=False)
    plan['execute_requested'] = True
    if plan['status'] != 'preflight_passed':
        _json(report_dir / 'report.json', plan)
        return plan, 1

    execution = dict(stage='prepare', argv=plan['preparation']['argv'], cwd=str(report_dir / 'work'),
                     started_at=datetime.now(timezone.utc).isoformat(), returncode=None,
                     claim_created=False, artifacts=[], raw_input_contents_hashed=False,
                     artifact_validation='Checks cover regular, nonempty files and SHA-256. HDF5/NPZ contents and numerical correctness remain unchecked.',
                     environment_changes={'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'})
    plan['execution'] = execution
    plan['status'] = 'starting'
    process = None
    started = time.monotonic()
    owned_files = []
    state = None
    exit_code = 1
    try:
        state = inspect_preparation(data)
        if not all(c['passed'] for c in state['checks']):
            raise ValueError('Preparation paths changed after preflight; refusing execution')
        claim = Path(state['claim'])
        with claim.open('x', encoding='utf-8') as handle:
            json.dump(dict(report_dir=str(report_dir), created_at=execution['started_at'],
                           note='Retained after success, failure or interruption. Review this claim and any partial artifacts before manual cleanup.'), handle, indent=2)
            handle.write('\n')
        execution['claim_created'] = True
        execution['claim_path'] = str(claim)
        # Check for new outputs after acquiring the dataset claim.
        state = inspect_preparation(data)
        if not all(c['passed'] for c in state['checks'] if c['name'] != 'preparation_claim_absent'):
            raise ValueError('Preparation outputs changed while claiming dataset')
        script = Path(execution['argv'][1])
        if _sha(config) != plan['sources']['manifest_sha256'] or _sha(script) != plan['sources']['implementation_files_sha256']['data_preparation.py']:
            raise ValueError('Manifest or preparation source changed after planning')
        (report_dir / 'work').mkdir()
        (report_dir / 'sources').mkdir()
        for source, destination in ((config, report_dir / 'manifest.json'),
                                    (script, report_dir / 'sources/data_preparation.py')):
            with destination.open('xb') as handle:
                handle.write(source.read_bytes())
            owned_files.append(destination)
        _json(report_dir / 'preparation-arguments.json', plan['preparation']['arguments'])
        _json(report_dir / 'inputs.json', dict(files=state['inputs'], contents_hashed=False))
        owned_files.extend((report_dir / 'preparation-arguments.json', report_dir / 'inputs.json'))
        plan['execution_enabled'] = True
        plan['status'] = 'running'
        _json(report_dir / 'report.json', plan)
        stdout, stderr = report_dir / 'stdout.log', report_dir / 'stderr.log'
        owned_files.extend((stdout, stderr))
        with stdout.open('xb') as out, stderr.open('xb') as err:
            process = subprocess.Popen(execution['argv'], cwd=execution['cwd'],
                         env={**os.environ, **execution['environment_changes']},
                         stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                         start_new_session=(os.name == 'posix'))
            plan['workflow_commands_executed'] = 1
            execution['returncode'] = process.wait()
            _stop(process)
        if execution['returncode']:
            plan['status'] = 'execution_failed'
        else:
            execution['artifacts'], errors = _artifacts(state)
            after = inspect_preparation(data)
            execution['inputs_unchanged'] = after['inputs'] == state['inputs']
            execution['source_unchanged'] = _sha(script) == plan['sources']['implementation_files_sha256']['data_preparation.py']
            execution['safe_input_layout_after'] = after['checks'][0]['passed']
            execution['missing_or_invalid_artifacts'] = errors
            passed = (not errors and execution['inputs_unchanged'] and execution['source_unchanged']
                      and execution['safe_input_layout_after'])
            plan['status'] = 'completed' if passed else 'postcheck_failed'
            exit_code = 0 if passed else 1
    except KeyboardInterrupt:
        _stop(process)
        plan['status'] = 'interrupted'
        execution['error'] = 'Interrupted; partial outputs and dataset claim retained.'
        exit_code = 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _stop(process)
        plan['status'] = 'execution_failed'
        execution['error'] = str(error)
    finally:
        if process is not None:
            execution['returncode'] = process.returncode
        if state is not None and not execution['artifacts']:
            execution['artifacts'], errors = _artifacts(state, hash_contents=False)
            execution['missing_or_invalid_artifacts'] = errors
        execution['elapsed_seconds'] = round(time.monotonic() - started, 3)
        execution['finished_at'] = datetime.now(timezone.utc).isoformat()
        execution['files_sha256'] = {}
        for path in owned_files:
            try:
                if path.is_file():
                    execution['files_sha256'][str(path.relative_to(report_dir))] = _sha(path)
            except OSError as error:
                execution.setdefault('file_hash_errors', []).append(str(error))
                if exit_code == 0:
                    plan['status'], exit_code = 'postcheck_failed', 1
        _json(report_dir / 'report.json', plan)
    return plan, exit_code
