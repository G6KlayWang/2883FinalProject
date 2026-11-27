"""Sequential cache preparation preserves reviewed per-frame pixels and row order."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from scripts.data.tests.fixtures import make_dataset, DOWNLOAD
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest
from scripts.data.loader import PreparedData
from scripts.encoders.frames import ImageFrames
from scripts.encoders.tests.test_losses import implementation


class PreparationTests(unittest.TestCase):
    def test_chunk_iterator_matches_reviewed_reader_across_episodes_and_chunk_edges(
        self,
    ):
        cache = implementation("image_cache")
        self.assertTrue(
            hasattr(cache, "iter_resized_images"), "Chunk-aware image iterator missing"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = make_dataset(root / "source.h5", (17,) * 20)
            # Give each source row a distinct asymmetric image and chunk shape.
            with h5py.File(path, "a") as handle:
                values = handle["pixels"][:]
                del handle["pixels"]
                for i in range(len(values)):
                    values[i, :, :, 0] = i % 256
                    values[i, ::2, :, 1] = (i * 7) % 256
                handle.create_dataset(
                    "pixels", data=values, chunks=(7, *values.shape[1:])
                )
            prepared = PreparedData(
                path, build_manifest(inspect_file(path), DOWNLOAD, "fixture")
            )
            for policy in ("phase0", "raw"):
                frames = ImageFrames(prepared, "train", policy)
                expected = [frames[i] for i in range(len(frames))]
                original = h5py.Dataset.__getitem__
                reads = []

                def image_only(dataset, key):
                    if dataset.name != "/pixels":
                        raise AssertionError("Cache read measurements")
                    reads.append(key)
                    return original(dataset, key)

                with patch.object(h5py.Dataset, "__getitem__", image_only):
                    actual = list(cache.iter_resized_images(frames))
                self.assertEqual(len(actual), len(expected))
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                self.assertLess(len(reads), len(frames))
                self.assertTrue(
                    all(
                        isinstance(key, slice) and key.stop - key.start <= 7
                        for key in reads
                    )
                )
                seen = []
                cache.build_image_cache(
                    frames,
                    root / policy,
                    progress=lambda done, total: seen.append((done, total)),
                )
                self.assertEqual(seen[-1], (len(frames), len(frames)))
                loaded = cache.CachedImages(root / policy, frames.identity)
                for index in (0, 3, len(frames) - 1):
                    np.testing.assert_array_equal(loaded[index], expected[index])
