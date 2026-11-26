"""Focused recovery findings: reconcile, park, explicit archive/restart and guard stop."""

from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.encoders.devices import GpuLeases
from scripts.encoders.queue import QueueRunner
from scripts.encoders.queue_state import QueueStore
from scripts.encoders.tests.test_losses import implementation
from scripts.encoders.tests.test_training_resume import (
    SyntheticImages,
    short_config,
    tiny_model,
)
from scripts.encoders.training import Trainer


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_clean_orphan_reconciles_checkpoint_counts_but_unresolved_stays_failed(
        self,
    ):
        recovery = implementation("recovery")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = short_config()
            identity = {
                "code_revision": "fixture",
                "environment": {},
                "manifest_sha256": "manifest",
            }
            spec = {
                "run_id": "orphan",
                "arm": "A",
                "rho": 10,
                "seed": 0,
                "max_updates": 4,
                "config_hash": config.digest(),
                "code_revision": "fixture",
            }
            store = QueueStore(root / "queue", identity, [spec], resume=False)
            store.start("orphan", "GPU-one", 4, {"pid": 0})
            trainer = Trainer(
                tiny_model(),
                config,
                store.root / "runs/orphan",
                {**identity, "gpu_uuid": "GPU-one"},
                "cpu",
            )
            trainer.run(SyntheticImages(), stop_at=2)
            with GpuLeases(root / "leases", ["GPU-one"]) as leases:
                recovery.reconcile_queue(store, leases)
                self.assertEqual(
                    (
                        store.run("orphan")["state"],
                        store.run("orphan")["completed_updates"],
                    ),
                    ("paused", 2),
                )
                trainer.run(SyntheticImages(), stop_at=4)
                store.run("orphan")["state"] = "running"
                recovery.reconcile_queue(store, leases)
                self.assertEqual(
                    (
                        store.run("orphan")["state"],
                        store.run("orphan")["completed_updates"],
                    ),
                    ("completed", 4),
                )
                path = store.root / "runs/orphan/run_state.json"
                ledger = json.loads(path.read_text())
                ledger["attempted_updates"] = 5
                path.write_text(json.dumps(ledger))
                store.run("orphan")["state"] = "running"
                recovery.reconcile_queue(store, leases)
                self.assertEqual(store.run("orphan")["state"], "failed")

    def test_failed_run_stays_parked_while_other_s3_run_completes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specs = [
                {
                    "run_id": name,
                    "arm": "A",
                    "rho": 2.5,
                    "seed": 0,
                    "max_updates": 4,
                    "max_seconds": 86400,
                    "config_hash": name,
                    "code_revision": "fixture",
                    "memory_limit_gib": 200,
                }
                for name in ("failed", "healthy")
            ]
            store = QueueStore(root / "queue", {"fixture": "park"}, specs, resume=False)
            store.run("failed")["state"] = "failed"
            store.state["phase"] = "S3"
            store.save()
            store.resume_failed(set())

            def command(run, gpu, target, resume, ready, gate):
                return [
                    sys.executable,
                    "scripts/encoders/tests/queue_child.py",
                    "--output",
                    str(store.root / "runs" / run["run_id"]),
                    "--run-id",
                    run["run_id"],
                    "--stop-at",
                    str(target),
                    "--ready-file",
                    str(ready),
                    "--start-gate",
                    "none",
                    "--config-hash",
                    run["config_hash"],
                    "--code-revision",
                    "fixture",
                ]

            with GpuLeases(root / "leases", ["GPU-one"]) as leases:
                runner = QueueRunner(
                    store, leases, command, stream=io.StringIO(), poll_seconds=0.01
                )
                self.assertEqual(runner.run(), 2)
            self.assertEqual(store.run("healthy")["state"], "completed")
            self.assertEqual(store.run("failed")["state"], "failed")
            self.assertEqual(store.state["incomplete_runs"], ["failed"])
            self.assertNotEqual(store.state["phase"], "completed")

    def test_explicit_restart_preserves_failed_artifacts_and_records_discarded_work(
        self,
    ):
        recovery = implementation("recovery")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = {"run_id": "failed", "max_updates": 30000}
            store = QueueStore(
                root / "queue", {"fixture": "restart"}, [spec], resume=False
            )
            run = store.run("failed")
            run.update(state="failed", gpu_uuid="GPU-one", completed_updates=7)
            run["attempts"] = [{"exit_code": 1}]
            old = store.root / "runs/failed"
            old.mkdir(parents=True)
            (old / "run_state.json").write_text(
                json.dumps(
                    {
                        "completed_updates": 7,
                        "attempted_updates": 8,
                        "model_seconds": 12.5,
                    }
                )
            )
            (old / "metrics.jsonl").write_text("preserved history\n")
            with GpuLeases(root / "leases", ["GPU-one"]) as leases:
                with self.assertRaisesRegex(ValueError, "decision"):
                    recovery.restart_from_zero(store, "failed", leases, decision="")
                recovery.restart_from_zero(
                    store,
                    "failed",
                    leases,
                    decision="Later explicit team decision TEST-ONLY",
                )
            self.assertFalse(old.exists())
            record = run["restarts"][-1]
            self.assertEqual(
                (
                    record["discarded_completed_updates"],
                    record["discarded_attempted_updates"],
                    record["discarded_model_seconds"],
                ),
                (7, 8, 12.5),
            )
            self.assertEqual(
                (Path(record["archive"]) / "metrics.jsonl").read_text(),
                "preserved history\n",
            )
            self.assertEqual(
                (run["state"], run["completed_updates"], run["gpu_uuid"]),
                ("pending", 0, "GPU-one"),
            )
            self.assertEqual(run["attempts"], [{"exit_code": 1}])

    def test_between_update_resource_guard_publishes_current_checkpoint_before_raising(
        self,
    ):
        training = implementation("training")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            trainer = Trainer(
                tiny_model(), short_config(), output, {"fixture": "guard"}, "cpu"
            )

            def guard(*args):
                if trainer.completed_updates == 1:
                    raise MemoryError("synthetic working-set guard")

            with patch.object(training, "check_resources", guard):
                with self.assertRaisesRegex(MemoryError, "working-set"):
                    trainer.run(SyntheticImages(), stop_at=4)
            ledger = json.loads((output / "run_state.json").read_text())
            self.assertEqual(
                (ledger["completed_updates"], ledger["attempted_updates"]), (1, 1)
            )
            self.assertIsNotNone(
                ledger["last_checkpoint_name"],
                "Guard must publish clean current checkpoint",
            )
            checkpoint = output / ledger["last_checkpoint_name"]
            self.assertTrue(checkpoint.exists())
            resumed = Trainer(
                tiny_model(),
                short_config(),
                output,
                {"fixture": "guard"},
                "cpu",
                resume=checkpoint,
            )
            resumed.run(SyntheticImages(), stop_at=4)
            self.assertEqual(resumed.completed_updates, 4)
