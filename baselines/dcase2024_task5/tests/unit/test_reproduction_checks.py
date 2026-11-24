"""Plan-check regression tests using JSON fixtures."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from check_reproduction_plan import check_plan, reference

CONFIG = ROOT / 'configs/reproduction.example.json'


class ReproductionCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([sys.executable, '-E', '-S', '-B', str(ROOT / 'tools/reproduce.py'),
                                 '--config', str(CONFIG), '--stage', 'all', '--dry-run'],
                                capture_output=True, text=True, check=True, timeout=30,
                                env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
        cls.plan = json.loads(result.stdout)

    def setUp(self):
        self.candidate = copy.deepcopy(self.plan)

    def test_shipped_plan(self):
        check_plan(self.candidate, CONFIG)

    def test_execution_is_rejected(self):
        for key, value in (('execution_enabled', True), ('workflow_commands_executed', 1)):
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.plan)
                candidate[key] = value
                with self.assertRaises(ValueError):
                    check_plan(candidate, CONFIG)

    def test_missing_run_is_rejected(self):
        self.candidate['runs'].pop()
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_duplicate_run_is_rejected(self):
        self.candidate['runs'][-1] = self.candidate['runs'][0]
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_changed_seed_is_rejected(self):
        self.candidate['runs'][0]['seed'] = 5678
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_changed_manifest_binding_is_rejected(self):
        self.candidate['sources']['manifest_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_changed_evidence_binding_is_rejected(self):
        self.candidate['sources']['logged_evidence']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_changed_prediction_binding_is_rejected(self):
        self.candidate['runs'][0]['logged_evidence']['selected_prediction_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_changed_cife_settings_are_rejected(self):
        command = self.candidate['runs'][0]['stages']['train']['argv']
        command[command.index('features.cife_nhead=4')] = 'features.cife_nhead=8'
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_archived_operating_point_application_is_rejected(self):
        self.candidate['runs'][0]['logged_evidence']['operating_point_applied'] = True
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_missing_checkpoint_cannot_gain_an_evaluation_command(self):
        self.candidate['runs'][0]['stages']['evaluate']['argv'] = ['unexpected', 'test.py']
        with self.assertRaises(ValueError):
            check_plan(self.candidate, CONFIG)

    def test_preset_and_seed_overrides_are_required(self):
        for override in ('features.use_cife=false', 'seed=1234', 'train_param.freq_mask_seed=1234'):
            with self.subTest(override=override):
                candidate = copy.deepcopy(self.plan)
                candidate['runs'][0]['stages']['train']['argv'].remove(override)
                with self.assertRaises(ValueError):
                    check_plan(candidate, CONFIG)

    def test_required_references_stay_inside_checkout(self):
        with self.assertRaises(ValueError):
            reference(ROOT, 'missing-reference.json')
        with tempfile.TemporaryDirectory() as folder:
            outside = Path(folder) / 'outside.json'
            outside.write_text('{}')
            with self.assertRaises(ValueError):
                reference(ROOT, str(outside))

    def test_evidence_hash_mismatch_is_rejected(self):
        manifest = json.loads(CONFIG.read_text())
        manifest['logged_evidence']['path'] = str(CONFIG.parent / manifest['logged_evidence']['path'])
        manifest['logged_evidence']['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'manifest.json'
            config.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Evidence hash mismatch'):
                check_plan(self.candidate, config)


if __name__ == '__main__':
    unittest.main()
