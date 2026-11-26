import copy
import importlib
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from fixtures import DOWNLOAD, make_dataset
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest, validate_manifest


class FamilyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('scripts/data/families.py').exists(), 'Family grouping not implemented')
        self.module = importlib.import_module('scripts.data.families')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = make_dataset(Path(self.temp.name) / 'families.h5', (6,) * 40)
        with h5py.File(self.path, 'a') as f:
            # Twenty families, two interleaved episodes apiece. Grouping cannot
            # be inferred from adjacent IDs, lengths, or integer division.
            for family in range(20):
                f['state'][(family + 20) * 6] = f['state'][family * 6]
        self.schema = inspect_file(self.path)

    def test_grouping_uses_exact_initial_state_and_retains_stable_hashes(self):
        family = self.schema['families'][0]
        self.assertEqual(family['episode_ids'], [0, 20])
        self.assertEqual(len(family['initial_state_sha256']), 64)
        self.assertEqual(len(self.schema['families']), 20)
        self.assertFalse(self.schema['family_structure']['contiguous_members'])

    def test_family_split_disjointness_and_all_episode_coverage(self):
        manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
        self.assertEqual(manifest['split_unit'], 'family')
        self.assertEqual({k: len(v) for k, v in manifest['family_splits'].items()}, {'train': 18, 'validation': 1, 'test': 1})
        self.assertEqual({k: len(v) for k, v in manifest['splits'].items()}, {'train': 36, 'validation': 2, 'test': 2})
        membership = {i: split for split, ids in manifest['splits'].items() for i in ids}
        self.assertEqual(sorted(membership), list(range(40)))
        for i in range(20):
            self.assertEqual(membership[i], membership[i + 20])
        self.assertEqual(manifest, build_manifest(self.schema, DOWNLOAD, 'candidate'))

    def test_swapping_members_between_families_rejected(self):
        manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
        manifest['families'][0]['episode_ids'][1], manifest['families'][1]['episode_ids'][1] = 21, 20
        with self.assertRaisesRegex(ValueError, 'family'):
            validate_manifest(manifest)

    def test_missing_or_duplicate_family_members_rejected(self):
        for replacement in ([0], [0, 0, 20]):
            with self.subTest(replacement=replacement):
                manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
                manifest['families'][0]['episode_ids'] = replacement
                with self.assertRaises(ValueError):
                    validate_manifest(manifest)

    def test_tampered_family_split_rejected(self):
        manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
        manifest['family_splits']['test'] = manifest['family_splits']['validation']
        with self.assertRaisesRegex(ValueError, 'family'):
            validate_manifest(manifest)

    def test_known_real_source_requires_observed_185_by_101_structure(self):
        schema = copy.deepcopy(self.schema)
        schema['hdf5_sha256'] = 'b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb'
        with self.assertRaisesRegex(ValueError, '185|101'):
            build_manifest(schema, DOWNLOAD, 'candidate')

    def test_duplicate_trajectory_cannot_cross_family_split(self):
        manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
        schema = copy.deepcopy(self.schema)
        schema['duplicate_episode_groups'] = [[manifest['splits']['train'][0], manifest['splits']['test'][0]]]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_manifest(schema, DOWNLOAD, 'candidate')

    def test_missing_family_evidence_fields_have_actionable_value_errors(self):
        for field in ('grouping_method', 'families', 'episode_initial_state_sha256', 'family_structure'):
            with self.subTest(field=field):
                manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
                del manifest[field]
                with self.assertRaisesRegex(ValueError, field):
                    validate_manifest(manifest)

    def test_missing_family_record_fields_have_actionable_value_errors(self):
        for field in ('family_id', 'initial_state_sha256', 'episode_ids'):
            with self.subTest(field=field):
                manifest = build_manifest(self.schema, DOWNLOAD, 'candidate')
                del manifest['families'][0][field]
                with self.assertRaisesRegex(ValueError, field):
                    validate_manifest(manifest)

    def test_build_reports_missing_family_record_field_without_key_error(self):
        schema = copy.deepcopy(self.schema)
        del schema['families'][0]['family_id']
        with self.assertRaisesRegex(ValueError, 'family_id'):
            build_manifest(schema, DOWNLOAD, 'candidate')
