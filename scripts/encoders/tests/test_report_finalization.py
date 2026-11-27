"""Real CPU retention regression; external GPU/input metadata alone is substituted."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from scripts.data.file_io import sha256_file
from scripts.encoders import train
from scripts.encoders.config import RunConfig
from scripts.encoders.runtime import configure_precision
from scripts.encoders.training import Trainer
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_training_resume import SyntheticImages, tiny_model


class ReportFinalizationTests(unittest.TestCase):
    def test_retained_verified_resume_hash_survives_pruning_and_finalization_has_zero_updates(
        self,
    ):
        torch.set_num_threads(2)
        values = run_config_dict()
        values.update(profile="production", recipe_status="approved")
        for key in ("grids", "gate", "c_transform", "probe"):
            values[key]["status"] = "approved"
        values["optimizer"].update(max_updates=4, warmup_updates=2)
        values["runtime"].update(workers=0, checkpoint_every=1, keep_checkpoints=1)
        config = RunConfig.from_dict(values)
        frames = SyntheticImages()
        frames.identity["diagnostic_subset"] = False

        def cpu_trainer(model, config, output, identity, device, *, resume=None):
            return Trainer(model, config, output, identity, "cpu", resume=resume)

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            cfg = root / "config.json"
            cfg.write_text(json.dumps(config.to_dict()))
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            output = root / "run"
            for name, value in [
                ("configure_process", configure_precision),
                ("require_single_gpu", lambda: "GPU-fixture"),
                ("open_images", lambda *args: (None, frames)),
                ("build_encoder", lambda *args: tiny_model()),
                ("environment_record", lambda: {"fixture": "same"}),
                ("Trainer", cpu_trainer),
            ]:
                stack.enter_context(patch.object(train, name, value))
            for name, value in [
                ("reset_peak_memory_stats", None),
                ("max_memory_allocated", 0),
                ("get_device_properties", SimpleNamespace(total_memory=1024)),
            ]:
                stack.enter_context(
                    patch.object(train.torch.cuda, name, return_value=value)
                )
            argv = [
                "train",
                "--config",
                str(cfg),
                "--hdf5",
                "not-read.h5",
                "--manifest",
                str(manifest),
                "--cache",
                "not-read-cache",
                "--output",
                str(output),
                "--code-revision",
                "1" * 40,
                "--candidate",
                "A-rho10-seed0",
                "--progress",
                "off",
            ]
            with patch.object(sys, "argv", argv + ["--stop-at", "2"]):
                train.main()
            resume = output / "step-000002.pt"
            verified = sha256_file(resume)
            with patch.object(
                sys, "argv", argv + ["--resume", str(resume), "--stop-at", "4"]
            ):
                train.main()
            self.assertFalse(resume.exists())
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(
                (report["completed_updates"], report["resumed_from_sha256"]),
                (4, verified),
            )
            before = {
                name: (output / name).read_bytes()
                for name in ("run_state.json", "step-000004.pt", "training_report.json")
            }
            with patch.object(
                Trainer,
                "run",
                side_effect=AssertionError("Finalization must not train"),
            ):
                with patch.object(
                    sys,
                    "argv",
                    argv
                    + [
                        "--resume",
                        str(output / "step-000004.pt"),
                        "--stop-at",
                        "4",
                        "--finalize-only",
                    ],
                ):
                    train.main()
            for name, data in before.items():
                self.assertEqual((output / name).read_bytes(), data)
            finalized = list(output.glob("report-finalization-*.json"))
            self.assertEqual(len(finalized), 1)
            recovered = json.loads(finalized[0].read_text())
            self.assertEqual(recovered["completed_updates_this_attempt"], 0)
            self.assertIsNone(recovered["peak_gpu_memory_bytes_this_attempt"])
            self.assertFalse((output / "source_succession.json").exists())
