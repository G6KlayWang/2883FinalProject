from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from torch import nn
from scripts.data.tests.fixtures import make_dataset, DOWNLOAD
from scripts.data.schema import inspect_file
from scripts.data.splits import build_manifest
from scripts.data.loader import PreparedData
from scripts.encoders.config import RunConfig
from scripts.encoders.frames import ImageFrames
from scripts.encoders.extract import extract_features, FeatureCache
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("frequency", torch.arange(1, 129, dtype=torch.float32))

    def forward(self, images):
        return torch.sin(images.mean((1, 2, 3))[:, None] * self.frequency)


class EvaluationPipelineTests(unittest.TestCase):
    def test_final_candidate_evaluation_binds_sources_and_refuses_test_metadata(self):
        evaluation = implementation("evaluate")
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = make_dataset(root / "source.h5", (16,) * 20)
            with h5py.File(path, "a") as handle:
                row = np.arange(320, dtype=np.float32)
                states = np.column_stack(
                    (
                        50 + row * 0.2,
                        100 + row * 0.15,
                        200 + 20 * np.sin(row * 0.05),
                        250 + 30 * np.cos(row * 0.03),
                        row * 0.1 % (2 * np.pi),
                        row * 0,
                        row * 0,
                    )
                )
                handle["state"][:] = states
                handle["proprio"][:] = states[:, [0, 1, 5, 6]]
            manifest = build_manifest(inspect_file(path), DOWNLOAD, "synthetic")
            prepared = PreparedData(path, manifest)
            value = run_config_dict()
            value.update(profile="production", recipe_status="approved")
            for key in ("grids", "gate", "probe"):
                value[key]["status"] = "approved"
            value["gate"]["enabled"] = False
            value["probe"].update(epochs=1, hidden=16, batch_size=16)
            config = RunConfig.from_dict(value)
            caches = {}
            for split in ("train", "validation"):
                frames = ImageFrames(prepared, split, "phase0")
                extract_features(
                    SyntheticEncoder(),
                    frames,
                    root / split,
                    {
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_role": "final",
                        "arm": "A",
                        "config_hash": config.digest(),
                        "effective_transform_sha256": None,
                    },
                    device="cpu",
                    batch_size=16,
                )
                caches[split] = FeatureCache(root / split)
            report = evaluation.evaluate_candidate(
                config,
                caches["train"],
                caches["validation"],
                prepared,
                root / "evaluation",
                run_id="A-rho10-seed0",
                device="cpu",
            )
            self.assertEqual(report["score_split"], "validation")
            self.assertEqual(report["checkpoint_role"], "final")
            self.assertTrue(np.isfinite(report["score"]))
            self.assertEqual(len(report["probe_train_curve"]), 1)
            self.assertTrue(
                (root / "evaluation" / "linear-probe" / "manifest.json").exists()
            )
            self.assertTrue(
                (root / "evaluation" / "mlp-probe" / "manifest.json").exists()
            )
            caches["validation"].identity["frame_identity"]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "test|validation"):
                evaluation.evaluate_candidate(
                    config,
                    caches["train"],
                    caches["validation"],
                    prepared,
                    root / "blocked",
                    run_id="invalid",
                    device="cpu",
                )
            self.assertFalse((root / "blocked").exists())
