import copy
import importlib
import multiprocessing
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from fixtures import make_dataset
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest
from test_splits import DOWNLOAD


def worker_read(dataset, index, queue):
    sample = dataset[index]
    queue.put((sample['metadata']['episode_id'], sample['frames'][0, 0, 0, 0].item()))


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('scripts/data/loader.py').exists(), 'Trajectory loader not implemented')
        self.loader = importlib.import_module('scripts.data.loader')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = make_dataset(Path(self.temp.name) / 'data.h5')
        self.manifest = build_manifest(inspect_file(self.path), DOWNLOAD, 'candidate')
        self.prepared = self.loader.PreparedData(self.path, self.manifest)

    def test_five_actions_pair_with_following_frame_and_keep_initial_condition(self):
        dataset = self.loader.TrajectoryDataset(self.prepared, 'train')
        sample = dataset[0]
        self.assertEqual(sample['actions'].shape, (1, 10))
        np.testing.assert_array_equal(sample['actions'][0], [0, 0, 1, -1, 2, -2, 3, -3, 4, -4])
        self.assertEqual(sample['frames'].shape, (2, 3, 96, 96))
        np.testing.assert_allclose(sample['frames'][:, 0, 0, 0], [0, 5 / 255])
        self.assertEqual(sample['frames'].dtype, np.float32)
        self.assertEqual(sample['metadata']['frame_indices'], (0, 5))

    def test_phase_zero_tails_and_episode_boundaries_by_value(self):
        dataset = self.loader.TrajectoryDataset(self.prepared, 'train')
        self.assertEqual(len(dataset), 3)
        last_first_episode = dataset[1]
        self.assertEqual(last_first_episode['metadata']['frame_indices'], (5, 10))
        self.assertEqual(last_first_episode['actions'][0, -2:].tolist(), [9, -9])
        second_episode = dataset[2]
        self.assertEqual(second_episode['metadata']['episode_id'], 1)
        self.assertEqual(second_episode['actions'][0, :2].tolist(), [12, -12])
        np.testing.assert_allclose(second_episode['frames'][:, 0, 0, 0], [12 / 255, 17 / 255])
        self.assertEqual(dataset.statistics['dropped_tail_transitions'], 1)
        self.assertEqual(dataset.statistics['terminal_action_rows_ignored'], 2)

    def test_horizon_two_window_never_joins_two_episodes(self):
        dataset = self.loader.TrajectoryDataset(self.prepared, 'train', horizon=2)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample['frames'].shape, (3, 3, 96, 96))
        self.assertEqual(sample['actions'].shape, (2, 10))
        self.assertEqual(sample['metadata']['frame_indices'], (0, 5, 10))
        self.assertEqual(dataset.statistics['episodes_without_windows'], 1)
        with self.assertRaises(IndexError):
            dataset[1]

    def test_empty_split_is_explicit_and_negative_indices_work(self):
        empty = self.loader.TrajectoryDataset(self.prepared, 'test')
        self.assertEqual(len(empty), 0)
        with self.assertRaises(IndexError):
            empty[0]
        full = self.loader.TrajectoryDataset(self.prepared, 'train')
        self.assertEqual(full[-1]['metadata']['episode_id'], 1)
        with self.assertRaises(IndexError):
            full[-4]

    def test_invalid_horizon_stride_and_split_rejected(self):
        for kwargs in ({'horizon': 0}, {'horizon': True}, {'stride': -1}, {'stride': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.loader.TrajectoryDataset(self.prepared, 'train', **kwargs)
        with self.assertRaises(ValueError):
            self.loader.TrajectoryDataset(self.prepared, 'validation-typo')

    def test_stride_in_macro_steps_and_shortest_valid_episode(self):
        self.assertEqual(self.loader.window_count(5, 1, 1), 0)
        self.assertEqual(self.loader.window_count(6, 1, 1), 1)
        self.assertEqual(self.loader.window_count(21, 2, 2), 2)

    def test_raw_frames_cover_only_assigned_episodes_without_exposing_state(self):
        dataset = self.loader.RawFrameDataset(self.prepared, 'train')
        self.assertEqual(len(dataset), 18)
        sample = dataset[12]
        self.assertEqual(set(sample), {'image', 'metadata'})
        self.assertEqual(sample['metadata']['episode_id'], 1)
        self.assertEqual(sample['metadata']['frame_index'], 0)
        np.testing.assert_allclose(sample['image'], 12 / 255)

    def test_measurements_require_explicit_purpose_and_stay_out_of_model_samples(self):
        sample = self.loader.TrajectoryDataset(self.prepared, 'train')[0]
        self.assertEqual(set(sample), {'frames', 'actions', 'metadata'})
        reader = self.loader.MeasurementReader(self.prepared, purpose='probe')
        measurement = reader.read(episode_id=1, frame_indices=[0, 5])
        self.assertEqual(measurement['state'].shape, (2, 7))
        np.testing.assert_array_equal(measurement['state'][:, 0], [84, 119])
        np.testing.assert_array_equal(measurement['proprio'], measurement['state'][:, [0, 1, 5, 6]])
        with self.assertRaises(ValueError):
            self.loader.MeasurementReader(self.prepared, purpose='training')
        with self.assertRaises(IndexError):
            reader.read(1, [6])

    def test_resize_preserves_rgb_channels_and_full_field_of_view(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[:, :, 0], image[:, :, 1], image[:, :, 2] = 255, 128, 0
        result = self.loader.resize_rgb(image)
        self.assertEqual(result.shape, (3, 96, 96))
        np.testing.assert_allclose(result[:, 20, 20], [1, 128 / 255, 0])
        with self.assertRaises(ValueError):
            self.loader.resize_rgb(image.astype(np.float32))

    def test_same_size_corrupt_hdf5_rejected_by_identity(self):
        with h5py.File(self.path, 'a') as f:
            f['action'][0] = [999., 999.]
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.loader.PreparedData(self.path, self.manifest)

    def test_altered_manifest_boundaries_rejected_even_with_valid_file_hash(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['episodes'][0]['length'] = 11
        manifest['episodes'][1]['offset'] = 11
        with self.assertRaisesRegex(ValueError, 'layout'):
            self.loader.PreparedData(self.path, manifest)

    def test_process_workers_read_independently_without_inherited_hdf5_handles(self):
        dataset = self.loader.TrajectoryDataset(self.prepared, 'train')
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        workers = [context.Process(target=worker_read, args=(dataset, i, queue)) for i in (0, 2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted([queue.get(timeout=1), queue.get(timeout=1)]), [(0, 0.0), (1, np.float32(12 / 255).item())])
        queue.close()

    def test_all_three_splits_load_only_their_assigned_episodes(self):
        path = make_dataset(Path(self.temp.name) / 'twenty.h5', (6,) * 20)
        manifest = build_manifest(inspect_file(path), DOWNLOAD, 'candidate')
        prepared = self.loader.PreparedData(path, manifest)
        observed = {}
        for split in ('train', 'validation', 'test'):
            dataset = self.loader.TrajectoryDataset(prepared, split)
            observed[split] = {dataset[i]['metadata']['episode_id'] for i in range(len(dataset))}
            self.assertEqual(observed[split], set(manifest['splits'][split]))
        self.assertEqual(sorted(map(len, observed.values())), [1, 1, 18])
        self.assertFalse(observed['train'] & observed['validation'])
        self.assertFalse(observed['train'] & observed['test'])
        self.assertFalse(observed['validation'] & observed['test'])

    def test_model_paths_never_read_state_or_proprio(self):
        from unittest.mock import patch
        original = h5py.Dataset.__getitem__
        def reject_measurements(dataset, index):
            if dataset.name in ('/state', '/proprio'):
                raise AssertionError('Model path read a simulator measurement')
            return original(dataset, index)
        with patch.object(h5py.Dataset, '__getitem__', reject_measurements):
            self.loader.TrajectoryDataset(self.prepared, 'train')[0]
            self.loader.RawFrameDataset(self.prepared, 'train')[0]

    def test_prepared_reader_rejects_file_changed_after_verification(self):
        with h5py.File(self.path, 'a') as f:
            f['action'][0] = [12., 12.]
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.loader.TrajectoryDataset(self.prepared, 'train')[0]
