import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fixtures import DOWNLOAD, make_dataset
from scripts.data.file_io import write_json_new
from scripts.data.loader import TrajectoryDataset
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('scripts/data/smoke.py').exists(), 'Parameterized smoke CLI not implemented')
        self.smoke = importlib.import_module('scripts.data.smoke')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hdf5 = make_dataset(self.root / 'data.h5', (16,) * 20)
        self.manifest = build_manifest(inspect_file(self.hdf5), DOWNLOAD, 'synthetic-reviewed-revision')
        self.manifest_path = self.root / 'manifest.json'
        write_json_new(self.manifest_path, self.manifest)
        self.output = self.root / 'smoke.json'

    def cli(self, *extra):
        return subprocess.run([
            sys.executable, '-m', 'scripts.data.smoke', '--hdf5', str(self.hdf5),
            '--manifest', str(self.manifest_path), '--output', str(self.output), *extra,
        ], capture_output=True, text=True)

    def test_cli_checks_all_splits_and_publishes_provenance_and_measurement_shapes(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['source'], self.manifest['source'])
        self.assertEqual(report['preparation_code_revision'], 'synthetic-reviewed-revision')
        self.assertEqual(report['seed'], 42)
        for split, windows, frames in [('train', 36, 288), ('validation', 2, 16), ('test', 2, 16)]:
            summary = report['splits'][split]
            self.assertEqual(summary['windows'], windows)
            self.assertEqual(summary['raw_frames'], frames)
            self.assertEqual([s['index'] for s in summary['samples']], [0, windows // 2, windows - 1])
            for sample in summary['samples']:
                self.assertEqual(sample['shapes'], {'frames': [3, 3, 96, 96], 'actions': [2, 10]})
                self.assertEqual(sample['model_fields'], ['actions', 'frames', 'metadata'])
                self.assertIn(sample['metadata']['episode_id'], self.manifest['splits'][split])
                self.assertEqual(sample['metadata']['split'], split)
            self.assertEqual(summary['measurement_shapes'], {'state': [3, 7], 'proprio': [3, 4]})
        before = self.output.read_bytes()
        self.assertNotEqual(self.cli().returncode, 0)
        self.assertEqual(self.output.read_bytes(), before)

    def test_cli_wrong_identity_cannot_publish_a_success_report(self):
        self.manifest['source']['hdf5_sha256'] = '0' * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA-256', result.stderr)
        self.assertFalse(self.output.exists())

    def test_no_k2_windows_is_explicit_failure_without_report(self):
        short = make_dataset(self.root / 'short.h5', (6,) * 20)
        manifest_path = self.root / 'short-manifest.json'
        write_json_new(manifest_path, build_manifest(inspect_file(short), DOWNLOAD, 'candidate'))
        with self.assertRaisesRegex(ValueError, 'K2 windows'):
            self.smoke.run_smoke(short, manifest_path, self.output)
        self.assertFalse(self.output.exists())

    def test_state_leak_from_model_reader_is_rejected_without_report(self):
        original = TrajectoryDataset.__getitem__
        def leaking_sample(dataset, index):
            sample = original(dataset, index)
            sample['state'] = 'simulator data must not be in model samples'
            return sample
        with patch.object(TrajectoryDataset, '__getitem__', leaking_sample):
            with self.assertRaisesRegex(ValueError, 'model fields'):
                self.smoke.run_smoke(self.hdf5, self.manifest_path, self.output)
        self.assertFalse(self.output.exists())

    def test_diagnostic_cli_opt_in_is_explicitly_carried_in_report(self):
        manifest = build_manifest(inspect_file(self.hdf5), DOWNLOAD, 'candidate', split_unit='episode')
        self.manifest_path.write_text(json.dumps(manifest))
        self.assertNotEqual(self.cli().returncode, 0)
        self.assertFalse(self.output.exists())
        result = self.cli('--allow-diagnostic-episode-split')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertTrue(report['diagnostic_episode_split'])
        self.assertEqual(report['split_unit'], 'episode')
        self.assertEqual(report['splits']['train']['samples'][0]['metadata']['split_unit'], 'episode')
