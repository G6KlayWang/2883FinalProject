"""Small real-HDF5 preparation fixture; only real-source identity gates are substituted."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fcntl
from scripts.data.tests.fixtures import make_dataset, DOWNLOAD
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest
from scripts.encoders.config import load_config
from scripts.encoders.tests.test_losses import implementation


class PreparationCliTests(unittest.TestCase):
    def test_serial_train_validation_publication_read_only_reuse_and_collision(self):
        prep = implementation("preparation")
        config = load_config(Path("scripts/encoders/configs/production-approved.json"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make_dataset(root / "source.h5", (12,) * 20)
            manifest = build_manifest(inspect_file(source), DOWNLOAD, "synthetic")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            cache = root / "cache"
            with patch.object(
                prep, "PUSHT_HDF5_SHA256", manifest["source"]["hdf5_sha256"]
            ), patch.object(prep, "validate_population_count"):
                stream = io.StringIO()
                report = prep.prepare_caches(
                    source, path, cache, config, ("train", "validation"), stream=stream
                )
                self.assertEqual(report["splits"]["train"]["count"], 54)
                self.assertEqual(report["splits"]["validation"]["count"], 3)
                self.assertIn("Preparing train", stream.getvalue())
                self.assertIn("Preparing validation", stream.getvalue())
                self.assertEqual(
                    (cache / "train-phase0/images.npy").stat().st_mode & 0o222, 0
                )
                with (cache / ".preparation.lock").open("ab") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        prep.prepare_caches(source, path, cache, config, ("train",))
                reused = prep.prepare_caches(source, path, cache, config, ("train",))
                self.assertIn(
                    "full file verification remains mandatory",
                    reused["splits"]["train"]["status"],
                )
                metadata = cache / "train-phase0/manifest.json"
                value = json.loads(metadata.read_text())
                value["identity"]["frame_policy"] = "raw"
                metadata.chmod(0o644)
                metadata.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "identity"):
                    prep.prepare_caches(source, path, cache, config, ("train",))
