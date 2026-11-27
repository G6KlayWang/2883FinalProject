import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from scripts.data.tests.fixtures import make_dataset, DOWNLOAD
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest
from scripts.data.loader import PreparedData, RawFrameDataset
from scripts.encoders.tests.test_losses import implementation


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.frames = implementation("frames")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = make_dataset(Path(self.temp.name) / "frames.h5", (12,) * 20)
        self.manifest = build_manifest(inspect_file(self.path), DOWNLOAD, "synthetic")
        self.prepared = PreparedData(self.path, self.manifest)

    def test_selected_frames_are_uniform_positions_and_never_window_duplicates(self):
        data = self.frames.ImageFrames(self.prepared, "train", "phase0")
        self.assertEqual(len(data), 54)
        np.testing.assert_array_equal(data.ids[:3], [[0, 0], [0, 5], [0, 10]])
        self.assertEqual(
            len(self.frames.ImageFrames(self.prepared, "train", "raw")), 216
        )
        original = h5py.Dataset.__getitem__

        def no_measurements(dataset, index):
            if dataset.name in ("/state", "/proprio", "/action"):
                raise AssertionError("Encoder frame read simulator/action data")
            return original(dataset, index)

        with patch.object(h5py.Dataset, "__getitem__", no_measurements):
            np.testing.assert_array_equal(
                data[1], RawFrameDataset(self.prepared, "train")[5]["image"]
            )
        with self.assertRaisesRegex(ValueError, "test"):
            self.frames.ImageFrames(self.prepared, "test", "phase0")

    def test_full_batches_and_smoke_subset_are_deterministic_and_distinct(self):
        sampler = self.frames.BatchOrder(10, 4, 7)
        self.assertEqual(len(sampler.indices(0)), 4)
        self.assertEqual(len(set(sampler.indices(0))), 4)
        self.assertFalse(set(sampler.indices(0)) & set(sampler.indices(1)))
        self.assertEqual(
            sampler.indices(2), self.frames.BatchOrder(10, 4, 7).indices(2)
        )
        with self.assertRaises(ValueError):
            self.frames.BatchOrder(3, 4, 0)
        data = self.frames.ImageFrames(self.prepared, "train", "phase0")
        indices = self.frames.smoke_indices(data.ids, self.manifest, 16, 0)
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(len(set(data.ids[indices, 0])), 16)

    def test_uint8_cache_parity_identity_and_corruption(self):
        cache = implementation("image_cache")
        frames = self.frames.ImageFrames(self.prepared, "train", "phase0")
        output = Path(self.temp.name) / "cache"
        cache.build_image_cache(frames, output)
        cached = cache.CachedImages(output, frames.identity)
        np.testing.assert_array_equal(cached.ids, frames.ids)
        for index in (0, 1, len(frames) - 1):
            np.testing.assert_array_equal(cached[index], frames[index])
        bad = dict(frames.identity, frame_policy="raw")
        with self.assertRaisesRegex(ValueError, "identity"):
            cache.CachedImages(output, bad)
        with (output / "images.npy").open("r+b") as target:
            target.seek(-1, 2)
            target.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "hash"):
            cache.CachedImages(output, frames.identity)

    def test_cache_rejects_reordered_ids_even_if_file_hash_was_updated(self):
        from scripts.data.file_io import sha256_file

        cache = implementation("image_cache")
        frames = self.frames.ImageFrames(self.prepared, "train", "phase0")
        output = Path(self.temp.name) / "reordered"
        cache.build_image_cache(frames, output)
        ids = np.load(output / "ids.npy", allow_pickle=False)
        ids[[0, 1]] = ids[[1, 0]]
        np.save(output / "ids.npy", ids, allow_pickle=False)
        report = json.loads((output / "manifest.json").read_text())
        report["files"]["ids.npy"] = sha256_file(output / "ids.npy")
        (output / "manifest.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "order|row"):
            cache.CachedImages(output, frames.identity)

    def test_cache_failure_removes_partial_and_never_publishes(self):
        cache = implementation("image_cache")
        frames = self.frames.ImageFrames(self.prepared, "train", "phase0")
        output = Path(self.temp.name) / "failed"
        original = cache.iter_resized_images

        def fail_after_first(dataset):
            iterator = iter(original(dataset))
            yield next(iterator)
            raise ValueError("fixture interrupted cache build")

        with patch.object(cache, "iter_resized_images", fail_after_first):
            with self.assertRaisesRegex(ValueError, "interrupted"):
                cache.build_image_cache(frames, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(Path(self.temp.name).glob("*.incomplete")), [])
