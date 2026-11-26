import copy
import importlib
from pathlib import Path
import unittest
from fixtures import DOWNLOAD


def schema_fixture(count=41):
    return {
        'schema_version': 1,
        'hdf5_sha256': 'a' * 64, 'hdf5_size_bytes': 100,
        'episodes': [{'episode_id': i, 'offset': i * 6, 'length': 6} for i in range(count)],
        'duplicate_episode_groups': [],
    }



class SplitTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('scripts/data/splits.py').exists(), 'Episode split not implemented')
        self.splits = importlib.import_module('scripts.data.splits')

    def build(self, schema=None):
        return self.splits.build_manifest(schema or schema_fixture(), DOWNLOAD, 'candidate-revision', split_unit='episode')

    def test_split_disjoint_exhaustive_and_floor_rounding_before_filtering(self):
        manifest = self.build()
        groups = manifest['splits']
        self.assertEqual({k: len(v) for k, v in groups.items()}, {'train': 37, 'validation': 2, 'test': 2})
        all_ids = [ep for group in groups.values() for ep in group]
        self.assertEqual(sorted(all_ids), list(range(41)))
        self.assertEqual(len(set(all_ids)), 41)

    def test_assignment_independent_of_input_order_and_global_randomness(self):
        import random
        random.seed(999)
        first = self.splits.split_episodes(list(range(41)), seed=42)
        random.seed(0)
        second = self.splits.split_episodes(list(reversed(range(41))), seed=42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, self.splits.split_episodes(list(range(41)), seed=43))

    def test_short_episodes_still_get_an_assignment(self):
        schema = schema_fixture()
        schema['episodes'][0]['length'] = 1
        for episode in schema['episodes'][1:]:
            episode['offset'] -= 5
        manifest = self.build(schema)
        self.assertIn(0, sum(manifest['splits'].values(), []))
        tiny = self.splits.split_episodes([9])
        self.assertEqual(tiny, {'train': [9], 'validation': [], 'test': []})

    def test_manifest_retains_source_identity_and_regenerates(self):
        manifest = self.build()
        self.assertEqual(manifest['source']['hdf5_sha256'], 'a' * 64)
        self.assertEqual(manifest['source']['archive_sha256'], 'b' * 64)
        self.assertEqual(manifest['seed'], 42)
        self.splits.validate_manifest(manifest)
        self.assertEqual(manifest, self.build())

    def test_duplicate_missing_or_changed_split_assignment_rejected(self):
        for edit in ('overlap', 'omission', 'swap'):
            with self.subTest(edit=edit):
                manifest = self.build()
                if edit == 'overlap':
                    manifest['splits']['test'][0] = manifest['splits']['train'][0]
                elif edit == 'omission':
                    manifest['splits']['train'].pop()
                else:
                    a, b = manifest['splits']['train'][0], manifest['splits']['test'][0]
                    manifest['splits']['train'][0], manifest['splits']['test'][0] = b, a
                with self.assertRaisesRegex(ValueError, 'split'):
                    self.splits.validate_manifest(manifest)

    def test_invalid_episode_ids_and_seed_rejected(self):
        for ids in ([], [1, 1], [-1], ['1'], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.splits.split_episodes(ids)
        with self.assertRaises(ValueError):
            self.splits.split_episodes([1], seed=True)

    def test_malformed_boundaries_and_provenance_rejected(self):
        for key in ('offset', 'length'):
            schema = schema_fixture()
            schema['episodes'][1][key] = -1
            with self.assertRaises(ValueError):
                self.build(schema)
        schema = schema_fixture()
        schema['hdf5_sha256'] = 'unknown'
        with self.assertRaises(ValueError):
            self.build(schema)

    def test_duplicate_content_across_splits_blocks_manifest(self):
        schema = schema_fixture()
        schema['duplicate_episode_groups'] = [list(range(41))]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.build(schema)

    def test_missing_manifest_fields_have_actionable_value_errors(self):
        for field in ('source', 'episodes', 'duplicate_episode_groups', 'splits', 'code_revision'):
            with self.subTest(field=field):
                manifest = self.build()
                del manifest[field]
                with self.assertRaisesRegex(ValueError, field):
                    self.splits.validate_manifest(manifest)

    def test_build_missing_source_fields_are_value_errors(self):
        download = copy.deepcopy(DOWNLOAD)
        del download['sha256']
        with self.assertRaisesRegex(ValueError, 'sha256'):
            self.splits.build_manifest(schema_fixture(), download, 'candidate', split_unit='episode')
