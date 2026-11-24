"""Preparation wrapper tests using temporary dummy programs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
DUMMY_PREPARATION = r'''
import argparse, json, os, subprocess, sys, time
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ('data-dir', 'train-dir', 'val-dir', 'test-dir', 'output-format'):
    parser.add_argument('--' + name, required=True)
args = parser.parse_args()
Path('launch.json').write_text(json.dumps(dict(argv=sys.argv, cwd=str(Path.cwd()))))
mode = os.environ.get('PREP_FIXTURE_MODE', 'success')
print('dummy stdout', flush=True)
print('dummy stderr', file=sys.stderr, flush=True)
if mode == 'interrupt':
    subprocess.Popen([sys.executable, '-c', "import time; from pathlib import Path; time.sleep(1); Path('orphan.txt').write_text('bad')"])
    Path('ready').write_text('ready')
    while True:
        time.sleep(0.05)
development = Path(args.data_dir) / 'Development_Set'
if mode != 'no-outputs':
    for wav in development.rglob('*.wav'):
        wav.with_name(wav.stem + '_logmel.h5').write_bytes(b'' if mode == 'empty-feature' else b'dummy feature, not HDF5')
        if mode == 'failure':
            sys.exit(7)
    (development / 'norm_stats.npz').write_bytes(b'dummy statistics, not NPZ')
if mode == 'change-input':
    next(development.rglob('*.wav')).write_bytes(b'changed input')
if mode == 'change-source':
    Path(__file__).write_text('# source changed in dummy only\n')
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreparationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reproduction-dummy-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'implementation with spaces'
        (self.root / 'tools').mkdir(parents=True)
        (self.root / 'configs').mkdir()
        for name in ('reproduce.py', 'preparation_execution.py'):
            shutil.copyfile(ROOT / 'tools' / name, self.root / 'tools' / name)
        shutil.copyfile(ROOT / 'run_ablation_experiments.py', self.root / 'run_ablation_experiments.py')
        shutil.copyfile(ROOT / 'configs/reproduction.evidence.json', self.root / 'configs/reproduction.evidence.json')
        for name in ('train.py', 'test.py'):
            (self.root / name).write_text("raise SystemExit('Training/evaluation must never run')\n")
        (self.root / 'data_preparation.py').write_text(DUMMY_PREPARATION)
        (self.root / 'requirements.txt').write_text('fixture-dependency==1.0\n')
        self.data = self.base / 'dummy data'
        for split in ('Training_Set', 'Validation_Set'):
            directory = self.data / 'Development_Set' / split
            directory.mkdir(parents=True)
            (directory / 'sample.wav').write_bytes(b'not real audio')
            (directory / 'sample.csv').write_text('not real annotations\n')
        self.development = self.data / 'Development_Set'
        self.probes = self.base / 'metadata-probes'
        self.python = self.base / 'dummy-python'
        self.python.write_text('#!' + sys.executable + '\n' +
            'import json,os,sys\nfrom pathlib import Path\n' +
            "if '-c' in sys.argv:\n" +
            f'    p=Path({str(self.probes)!r}); p.write_text(p.read_text()+"probe\\n" if p.exists() else "probe\\n")\n' +
            "    if os.environ.get('PREP_FIXTURE_MODE') == 'cannot-launch': os.chmod(__file__,0o644)\n" +
            "    print(json.dumps(dict(python='fixture-only',packages={'fixture-dependency':'1.0'},direct_urls={})))\n" +
            'else:\n    os.execv(sys.executable,[sys.executable,"-B",*sys.argv[1:]])\n')
        self.python.chmod(0o755)
        self.manifest = json.loads((ROOT / 'configs/reproduction.example.json').read_text())
        self.manifest.update(python=str(self.python), dataset_root=str(self.data),
                             output_root=str(self.base / 'future-runs'))
        self.config = self.root / 'configs/reproduction.local.json'
        self.save_config()
        self.report = self.base / 'receipt'
        self.env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}

    def save_config(self):
        self.config.write_text(json.dumps(self.manifest))

    def command(self, *arguments):
        return [sys.executable, '-E', '-S', '-B', str(self.root / 'tools/reproduce.py'),
                '--config', str(self.config), *map(str, arguments)]

    def run_cli(self, *arguments, mode='success'):
        return subprocess.run(self.command(*arguments), cwd=self.base,
            env={**self.env, 'PREP_FIXTURE_MODE': mode}, capture_output=True, text=True, timeout=15)

    def execute(self, mode='success'):
        return self.run_cli('--execute', '--stage', 'prepare', '--report-dir', self.report, mode=mode)

    def receipt(self):
        return json.loads((self.report / 'report.json').read_text())

    def assert_not_launched(self):
        self.assertFalse((self.report / 'work/launch.json').exists())

    def test_dry_run_keeps_24_runs_without_starting_interpreter(self):
        result = self.run_cli('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['run_count'], 24)
        self.assertEqual(len(plan['configurations']), 8)
        self.assertFalse(plan['execution_enabled'])
        self.assertEqual(plan['supported_execution_stages'], ['prepare'])
        self.assertFalse(self.probes.exists())
        self.assertFalse(self.report.exists())

    def test_explicit_mode_stage_and_report_are_required(self):
        for arguments in ((), ('--execute',), ('--execute', '--stage', 'prepare'),
                          ('--execute', '--report-dir', self.report),
                          ('--execute', '--stage', 'train', '--report-dir', self.report),
                          ('--execute', '--stage', 'evaluate', '--report-dir', self.report),
                          ('--execute', '--stage', 'all', '--report-dir', self.report),
                          ('--execute', '--dry-run', '--stage', 'prepare', '--report-dir', self.report)):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_cli(*arguments).returncode, 2)
        self.assertFalse(self.probes.exists())
        self.assert_not_launched()

    def test_preflight_never_launches_preparation(self):
        result = self.run_cli('--preflight', '--stage', 'prepare')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'preflight_passed')
        self.assert_not_launched()
        self.assertFalse((self.development / '.reproduction-prepare.json').exists())

    def test_success_captures_command_arguments_logs_hashes_and_claim(self):
        raw = {str(p): sha(p) for p in self.development.rglob('*') if p.is_file()}
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = self.receipt()
        execution = receipt['execution']
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['workflow_commands_executed'], 1)
        self.assertEqual(execution['returncode'], 0)
        self.assertEqual(execution['cwd'], str(self.report / 'work'))
        self.assertEqual(Path(execution['argv'][0]), self.python)
        self.assertEqual((self.report / 'stdout.log').read_text(), 'dummy stdout\n')
        self.assertEqual((self.report / 'stderr.log').read_text(), 'dummy stderr\n')
        self.assertEqual(len(execution['artifacts']), 3)
        for row in execution['artifacts']:
            self.assertEqual(row['sha256'], sha(Path(row['path'])))
        for name, digest in execution['files_sha256'].items():
            self.assertEqual(digest, sha(self.report / name))
        self.assertEqual(json.loads((self.report / 'preparation-arguments.json').read_text())['output_format'], 'hdf5')
        self.assertEqual(sha(self.report / 'sources/data_preparation.py'), sha(self.root / 'data_preparation.py'))
        self.assertTrue((self.development / '.reproduction-prepare.json').is_file())
        self.assertTrue(execution['inputs_unchanged'])
        self.assertFalse(execution['raw_input_contents_hashed'])
        self.assertFalse(receipt['resolved_config_compatibility_verified'])
        self.assertTrue(all(sha(Path(p)) == digest for p, digest in raw.items()))

    def test_existing_report_paths_refused_without_metadata_probe(self):
        for kind in ('file', 'empty', 'populated', 'symlink'):
            target = self.base / kind
            if kind == 'file':
                target.write_text('preserve')
            elif kind == 'symlink':
                target.symlink_to(self.base / 'missing')
            else:
                target.mkdir()
                if kind == 'populated':
                    (target / 'preserve').write_text('preserve')
            result = self.run_cli('--execute', '--stage', 'prepare', '--report-dir', target)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.probes.exists())

    def test_failed_preflight_has_receipt_but_no_dataset_claim(self):
        self.manifest['python'] = str(self.base / 'missing-python')
        self.save_config()
        result = self.execute()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.receipt()['status'], 'preflight_failed')
        self.assertEqual(self.receipt()['workflow_commands_executed'], 0)
        self.assertFalse((self.development / '.reproduction-prepare.json').exists())
        self.assert_not_launched()

    def test_existing_features_or_statistics_block_launch(self):
        cases = [self.development / 'norm_stats.npz',
                 self.development / 'Training_Set/sample_logmel.h5',
                 self.development / 'Training_Set/sample_logmel.npy',
                 self.development / 'Training_Set/sample_logmel_unnormalized.h5',
                 self.development / 'Training_Set/visualizations/sample_logmel_norm_comparison.png']
        for index, path in enumerate(cases):
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'preserve')
            report = self.base / f'blocked-{index}'
            result = self.run_cli('--execute', '--stage', 'prepare', '--report-dir', report)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(path.read_bytes(), b'preserve')
            self.assertFalse((report / 'work/launch.json').exists())
            path.unlink()
        self.assertFalse((self.development / '.reproduction-prepare.json').exists())

    def test_existing_empty_target_directory_blocks_launch(self):
        path = self.development / 'Training_Set/sample_logmel.h5'
        path.mkdir()
        result = self.execute()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assert_not_launched()
        self.assertTrue(path.is_dir())

    def test_existing_claim_blocks_second_report_directory(self):
        self.assertEqual(self.execute().returncode, 0)
        claim = self.development / '.reproduction-prepare.json'
        digest = sha(claim)
        result = self.run_cli('--execute', '--stage', 'prepare', '--report-dir', self.base / 'second')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(sha(claim), digest)
        self.assertFalse((self.base / 'second/work/launch.json').exists())

    def test_claim_alone_blocks_even_without_existing_features(self):
        claim = self.development / '.reproduction-prepare.json'
        claim.write_text('preserve prior attempt')
        digest = sha(claim)
        result = self.execute()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.receipt()['status'], 'preflight_failed')
        self.assertEqual(sha(claim), digest)
        self.assert_not_launched()

    def test_report_must_be_outside_dataset(self):
        result = self.run_cli('--execute', '--stage', 'prepare', '--report-dir', self.data / 'receipt')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.data / 'receipt').exists())
        self.assertFalse((self.development / '.reproduction-prepare.json').exists())

    def test_symlink_directory_blocks_launch(self):
        link = self.development / 'Training_Set/linked'
        link.symlink_to(self.base, target_is_directory=True)
        result = self.execute()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assert_not_launched()

    def test_child_failure_retains_logs_partial_outputs_and_claim(self):
        result = self.execute(mode='failure')
        self.assertEqual(result.returncode, 1, result.stderr)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'execution_failed')
        self.assertEqual(receipt['execution']['returncode'], 7)
        self.assertTrue(receipt['execution']['claim_created'])
        self.assertTrue(receipt['execution']['artifacts'])
        self.assertTrue(all(row['sha256'] is None for row in receipt['execution']['artifacts']))
        self.assertTrue((self.report / 'stderr.log').exists())

    def test_zero_exit_without_required_artifacts_is_not_success(self):
        result = self.execute(mode='no-outputs')
        self.assertEqual(result.returncode, 1, result.stderr)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'postcheck_failed')
        self.assertEqual(receipt['execution']['returncode'], 0)
        self.assertEqual(len(receipt['execution']['missing_or_invalid_artifacts']), 3)

    def test_changed_raw_input_fails_postcheck(self):
        result = self.execute(mode='change-input')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.receipt()['status'], 'postcheck_failed')
        self.assertFalse(self.receipt()['execution']['inputs_unchanged'])

    def test_deep_hidden_and_ambiguous_wav_paths_block_launch(self):
        for index, relative in enumerate(('a/b/c/d/e/f/deep.wav', '.hidden/hidden.wav', 'parent.wav/name.wav')):
            wav = self.development / 'Training_Set' / relative
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b'not audio')
            wav.with_suffix('.csv').write_text('not annotations')
            report = self.base / f'unsafe-{index}'
            result = self.run_cli('--execute', '--stage', 'prepare', '--report-dir', report)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse((report / 'work/launch.json').exists())
            wav.unlink()
            wav.with_suffix('.csv').unlink()
        self.assertFalse((self.development / '.reproduction-prepare.json').exists())

    def test_optional_development_test_split_is_inventoried(self):
        split = self.development / 'Test_Set'
        split.mkdir()
        (split / 'optional.wav').write_bytes(b'not audio')
        (split / 'optional.csv').write_text('not annotations')
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.receipt()['execution']['artifacts']), 4)

    def test_empty_required_artifacts_fail_postcheck(self):
        result = self.execute(mode='empty-feature')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.receipt()['status'], 'postcheck_failed')
        self.assertEqual(len(self.receipt()['execution']['missing_or_invalid_artifacts']), 2)

    def test_changed_preparation_source_is_reported(self):
        result = self.execute(mode='change-source')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.receipt()['status'], 'postcheck_failed')
        self.assertFalse(self.receipt()['execution']['source_unchanged'])
        self.assertEqual((self.report / 'sources/data_preparation.py').read_text(), DUMMY_PREPARATION)

    def test_launch_failure_keeps_receipt_without_claiming_child_execution(self):
        result = self.execute(mode='cannot-launch')
        self.assertEqual(result.returncode, 1, result.stderr)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'execution_failed')
        self.assertEqual(receipt['workflow_commands_executed'], 0)
        self.assertIsNone(receipt['execution']['returncode'])
        self.assertIn('error', receipt['execution'])
        self.assertTrue(receipt['execution']['claim_created'])
        self.assert_not_launched()

    @unittest.skipUnless(os.name == 'posix', 'Process-group interruption is POSIX-specific')
    def test_interrupt_stops_child_group_and_preserves_receipt(self):
        process = subprocess.Popen(self.command('--execute', '--stage', 'prepare', '--report-dir', self.report),
            cwd=self.base, env={**self.env, 'PREP_FIXTURE_MODE': 'interrupt'},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.report / 'work/ready').exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.report / 'work/ready').exists())
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertEqual(json.loads(stdout)['status'], 'interrupted')
            self.assertEqual(self.receipt()['status'], 'interrupted')
            time.sleep(1.1)
            self.assertFalse((self.report / 'work/orphan.txt').exists())
            self.assertTrue((self.development / '.reproduction-prepare.json').exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == '__main__':
    unittest.main()
