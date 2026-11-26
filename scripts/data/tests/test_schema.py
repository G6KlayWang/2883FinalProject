import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import h5py
from fixtures import make_dataset


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('scripts/data/schema.py').exists(), 'Schema inspection not implemented')
        self.schema = importlib.import_module('scripts.data.schema')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = make_dataset(Path(self.temp.name) / 'data.h5')

    def test_reports_actual_layout_hash_ranges_and_episode_boundaries(self):
        report = self.schema.inspect_file(self.path)
        self.assertEqual(report['episodes'], [
            {'episode_id': 0, 'offset': 0, 'length': 12},
            {'episode_id': 1, 'offset': 12, 'length': 6},
        ])
        self.assertEqual(report['arrays']['pixels']['shape'], [18, 224, 224, 3])
        self.assertEqual(report['arrays']['pixels']['dtype'], 'uint8')
        self.assertEqual(report['arrays']['action']['min'], [0.0, -17.0])
        self.assertEqual(report['hdf5_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_rejects_overlapping_offsets(self):
        with h5py.File(self.path, 'a') as f:
            f['ep_offset'][1] = 11
        with self.assertRaisesRegex(ValueError, 'offset'):
            self.schema.inspect_file(self.path)

    def test_rejects_fractional_episode_lengths(self):
        with h5py.File(self.path, 'a') as f:
            del f['ep_len']
            f['ep_len'] = [12.0, 6.0]
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.schema.inspect_file(self.path)

    def test_rejects_wrong_row_count_or_action_dimension(self):
        with h5py.File(self.path, 'a') as f:
            del f['action']
            f['action'] = [[1., 2., 3.]] * 17
        with self.assertRaisesRegex(ValueError, 'action'):
            self.schema.inspect_file(self.path)

    def test_rejects_step_index_crossing_an_episode_boundary(self):
        with h5py.File(self.path, 'a') as f:
            f['step_idx'][12] = 12
        with self.assertRaisesRegex(ValueError, 'step_idx'):
            self.schema.inspect_file(self.path)

    def test_rejects_wrong_episode_id(self):
        with h5py.File(self.path, 'a') as f:
            f['episode_idx'][12] = 0
        with self.assertRaisesRegex(ValueError, 'episode_idx'):
            self.schema.inspect_file(self.path)

    def test_rejects_nonfinite_numeric_fields(self):
        with h5py.File(self.path, 'a') as f:
            f['action'][3, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.schema.inspect_file(self.path)

    def test_detects_duplicate_episode_content(self):
        path = make_dataset(Path(self.temp.name) / 'duplicates.h5', (6, 6))
        with h5py.File(path, 'a') as f:
            f['action'][6:12] = f['action'][:6]
            f['pixels'][6] = f['pixels'][0]
        self.assertEqual(self.schema.inspect_file(path)['duplicate_episode_groups'], [[0, 1]])

    def test_json_publication_preserves_existing_report(self):
        output = Path(self.temp.name) / 'schema.json'
        self.schema.write_json_new(output, {'complete': True})
        with self.assertRaises(FileExistsError):
            self.schema.write_json_new(output, {'complete': False})
        self.assertEqual(output.read_text().strip(), '{\n  "complete": true\n}')

    def test_cli_publishes_full_machine_readable_report(self):
        output = Path(self.temp.name) / 'report.json'
        result = subprocess.run([
            sys.executable, '-m', 'scripts.data.schema', '--hdf5', str(self.path), '--output', str(output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())['episodes'][1]['offset'], 12)
