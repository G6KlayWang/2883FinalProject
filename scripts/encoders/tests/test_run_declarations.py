"""Production execution must match the declared matrix or a bound selection receipt."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.encoders.config import RunConfig, shared_training_controls
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


def approved_config():
    value = run_config_dict()
    value.update(profile="production", recipe_status="approved")
    for block in ("grids", "gate", "c_transform", "probe"):
        value[block]["status"] = "approved"
    return RunConfig.from_dict(value)


class RunDeclarationTests(unittest.TestCase):
    def test_only_explicit_declared_first_seed_candidate_can_execute(self):
        runs = implementation("runs")
        config = approved_config()
        with self.assertRaisesRegex(ValueError, "candidate"):
            runs.resolve_run(config, candidate=None, selection_report=None)
        resolved, declaration = runs.resolve_run(
            config, candidate="B-rho2.5-seed0", selection_report=None
        )
        self.assertEqual((resolved.arm, resolved.rho, resolved.seed), ("B", 2.5, 0))
        self.assertEqual(declaration["run_id"], "B-rho2.5-seed0")
        with self.assertRaisesRegex(ValueError, "candidate"):
            runs.resolve_run(config, candidate="A-rho7-seed0", selection_report=None)
        with self.assertRaisesRegex(ValueError, "approval"):
            proposed = run_config_dict()
            proposed.update(profile="production", recipe_status="proposed")
            runs.resolve_run(
                RunConfig.from_dict(proposed),
                candidate="A-rho10-seed0",
                selection_report=None,
            )

    def test_repeat_requires_selection_generated_matching_config_receipt(self):
        from dataclasses import replace

        selection, runs = implementation("selection"), implementation("runs")
        config = approved_config()
        report = {
            "status": "approved-policy",
            "selected": {
                arm: {
                    "arm": arm,
                    "rho": rho,
                    "seed": 0,
                    "checkpoint_role": "final",
                    "score_split": "validation",
                    "protocol": {
                        "shared_training_controls": shared_training_controls(config)
                    },
                }
                for arm, rho in (("A", 10.0), ("B", 2.5), ("C", 1.0))
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            report = selection.write_repeat_configs(
                report, config, Path(folder) / "repeats"
            )
            repeat = RunConfig.from_dict(
                json.loads((Path(folder) / "repeats/A-selected-seed1.json").read_text())
            )
            resolved, declaration = runs.resolve_run(
                repeat, candidate=None, selection_report=report
            )
            self.assertEqual(
                (resolved.arm, resolved.rho, resolved.seed), ("A", 10.0, 1)
            )
            self.assertEqual(declaration["stage"], "repeat")
            for bad in (None, {**report, "status": "provisional-policy"}):
                with self.subTest(receipt=bad is None), self.assertRaisesRegex(
                    ValueError, "selection|approved"
                ):
                    runs.resolve_run(repeat, candidate=None, selection_report=bad)
            with self.assertRaisesRegex(ValueError, "hash|config"):
                runs.resolve_run(
                    replace(repeat, runtime=replace(repeat.runtime, prefetch=3)),
                    candidate=None,
                    selection_report=report,
                )
            bad = copy.deepcopy(report)
            bad["selected"]["A"]["rho"] = 40.0
            with self.assertRaisesRegex(ValueError, "selected|match"):
                runs.resolve_run(repeat, candidate=None, selection_report=bad)
            with self.assertRaisesRegex(ValueError, "repeat|candidate"):
                runs.resolve_run(
                    repeat, candidate="A-rho10-seed0", selection_report=report
                )

    def test_direct_production_cli_is_refused_before_gpu_or_source_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for config in (
                approved_config(),
                RunConfig.from_dict({**approved_config().to_dict(), "seed": 1}),
            ):
                path = root / "config.json"
                path.write_text(json.dumps(config.to_dict()))
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.encoders.train",
                        "--config",
                        str(path),
                        "--hdf5",
                        "/not-read.h5",
                        "--manifest",
                        "/not-read.json",
                        "--output",
                        str(root / "absent"),
                        "--code-revision",
                        "synthetic",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, "candidate|selection")
                self.assertFalse((root / "absent").exists())
