"""Runtime metadata, pinned population assertions and conservative cgroup guards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.encoders.config import RunConfig
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation
from scripts.encoders.tests.test_training_resume import tiny_model


class RuntimeControlTests(unittest.TestCase):
    def test_record_actual_determinism_flags_in_run_record(self):
        training = implementation("training")
        with tempfile.TemporaryDirectory() as folder:
            training.Trainer(
                tiny_model(),
                RunConfig.from_dict(run_config_dict()),
                Path(folder) / "run",
                {"fixture": "runtime"},
                "cpu",
            )
            record = json.loads((Path(folder) / "run/run_record.json").read_text())
            controls = record["runtime_controls"]
            self.assertTrue(controls["deterministic_algorithms"])
            self.assertFalse(controls["deterministic_warn_only"])
            self.assertTrue(controls["cudnn_deterministic"])
            self.assertFalse(controls["cudnn_benchmark"])
            self.assertFalse(controls["matmul_tf32"])
            self.assertFalse(controls["cudnn_tf32"])

    def test_pinned_population_counts_reject_truncated_or_wrong_policy_rows(self):
        inputs = implementation("inputs")
        for split, policy, count in (
            ("train", "phase0", 433896),
            ("validation", "phase0", 21715),
            ("train", "raw", 2136352),
            ("validation", "raw", 107464),
        ):
            with self.subTest(split=split, policy=policy):
                inputs.validate_population_count(split, policy, count)
                with self.assertRaisesRegex(ValueError, "population|count"):
                    inputs.validate_population_count(split, policy, count - 1)

    def test_cgroup_working_set_excludes_inactive_file_and_retains_limit_guard(self):
        runtime = implementation("runtime")
        gib = 1024**3
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "memory.current").write_text(str(230 * gib))
            (root / "memory.stat").write_text(
                f"anon {80*gib}\ninactive_file {150*gib}\n"
            )
            (root / "memory.max").write_text(str(464 * gib))
            usage = runtime.memory_usage(root)
            self.assertEqual(usage["working_set_bytes"], 80 * gib)
            self.assertEqual(usage["accounting"], "current-minus-inactive_file")
            with patch.object(runtime, "memory_usage", return_value=usage):
                runtime.check_resources(
                    root, RunConfig.from_dict(run_config_dict()).runtime
                )
            (root / "memory.max").write_text(str(240 * gib))
            usage = runtime.memory_usage(root)
            with patch.object(runtime, "memory_usage", return_value=usage):
                with self.assertRaisesRegex(MemoryError, "actual|limit"):
                    runtime.check_resources(
                        root, RunConfig.from_dict(run_config_dict()).runtime
                    )
            (root / "memory.stat").unlink()
            fallback = runtime.memory_usage(root)
            self.assertEqual(fallback["working_set_bytes"], 230 * gib)
            self.assertEqual(fallback["accounting"], "current-fallback")
            with patch.object(runtime, "memory_usage", return_value=fallback):
                with self.assertRaisesRegex(MemoryError, "working.set|memory"):
                    runtime.check_resources(
                        root, RunConfig.from_dict(run_config_dict()).runtime
                    )
            (root / "memory.stat").write_text("anon 1\n")
            self.assertEqual(
                runtime.memory_usage(root)["accounting"], "current-fallback"
            )
