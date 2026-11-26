"""Real subprocess/lock tests of queue scheduling, failure and explicit continuation."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.encoders.tests.test_losses import implementation


class QueueTests(unittest.TestCase):
    def test_queue_identity_collisions_gpu_pins_and_failed_retry_are_explicit(self):
        state = implementation("queue_state")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "queue"
            runs = [
                {
                    "run_id": "A-low",
                    "arm": "A",
                    "rho": 2.5,
                    "seed": 0,
                    "max_updates": 30000,
                }
            ]
            store = state.QueueStore(
                root, {"code": "one", "config": "same"}, runs, resume=False
            )
            with self.assertRaises(FileExistsError):
                state.QueueStore(
                    root, {"code": "one", "config": "same"}, runs, resume=False
                )
            store.start(
                "A-low", "GPU-one", 200, {"ready_file": "ready.json", "log": "log.txt"}
            )
            with self.assertRaisesRegex(ValueError, "GPU|UUID"):
                store.check_gpu("A-low", "GPU-two")
            store.finish(
                "A-low",
                7,
                {"completed_updates": 0, "attempted_updates": 1},
                "fixture failure",
            )
            with self.assertRaisesRegex(ValueError, "failed|retry"):
                store.resume_failed(set())
            store.resume_failed({"A-low"})
            self.assertEqual(store.run("A-low")["state"], "paused")
            self.assertEqual(store.run("A-low")["gpu_uuid"], "GPU-one")
            with self.assertRaisesRegex(ValueError, "identity"):
                state.QueueStore(
                    root, {"code": "two", "config": "same"}, runs, resume=True
                )

    def test_gpu_lease_is_exclusive_across_real_processes(self):
        devices = implementation("devices")
        with tempfile.TemporaryDirectory() as folder:
            with devices.GpuLeases(Path(folder), ["GPU-one"]):
                code = "from pathlib import Path; from scripts.encoders.devices import GpuLeases; GpuLeases(Path(__import__('sys').argv[1]),['GPU-one']).__enter__()"
                other = subprocess.run(
                    [sys.executable, "-c", code, folder], capture_output=True, text=True
                )
                self.assertNotEqual(other.returncode, 0)
                self.assertIn("leased", other.stderr)
            with devices.GpuLeases(Path(folder), ["GPU-one"]):
                pass

    def test_capacity_gates_require_real_overlap_timing_memory_and_finite_history(self):
        capacity = implementation("capacity")
        rows = [
            {
                "update": i,
                "update_seconds": 0.2,
                "data_wait_seconds": 0.01,
                "total": 1.0,
                "invariance": 0.1,
                "regularizer": 0.2,
            }
            for i in range(1, 201)
        ]
        report = {
            "completed_updates": 200,
            "attempted_updates": 200,
            "model_seconds": 50,
            "peak_gpu_memory_bytes_this_attempt": 1e9,
            "gpu_total_memory_bytes": 96e9,
            "training_started_at": 10,
            "training_finished_at": 60,
        }
        observed = capacity.assess_run(
            rows,
            report,
            start=0,
            target=200,
            total=30000,
            max_seconds=86400,
            phase="S1",
        )
        self.assertTrue(observed["passed"])
        slow = [dict(row, update_seconds=4) for row in rows]
        self.assertFalse(
            capacity.assess_run(
                slow,
                report,
                start=0,
                target=200,
                total=30000,
                max_seconds=86400,
                phase="S1",
            )["passed"]
        )
        self.assertFalse(
            capacity.assess_run(
                rows,
                dict(report, peak_gpu_memory_bytes_this_attempt=90e9),
                start=0,
                target=200,
                total=30000,
                max_seconds=86400,
                phase="S1",
            )["passed"]
        )
        self.assertGreater(
            capacity.overlap_seconds([report, dict(report, training_started_at=20)]), 0
        )
        self.assertEqual(
            capacity.overlap_seconds(
                [report, dict(report, training_started_at=61, training_finished_at=90)]
            ),
            0,
        )

    def test_saved_shell_launcher_preserves_arguments_and_nonzero_exit(self):
        path = Path("scripts/encoders/launch.sh")
        self.assertTrue(path.exists(), "Saved shell launcher missing")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake = root / "python with spaces"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARG_FILE"\nexit 7\n')
            fake.chmod(0o755)
            output = root / "args"
            result = subprocess.run(
                ["bash", str(path), "--output", "a path with spaces", "--status"],
                env={
                    **os.environ,
                    "ENCODER_PYTHON": str(fake),
                    "ARG_FILE": str(output),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7)
            self.assertEqual(
                output.read_text().splitlines(),
                [
                    "-m",
                    "scripts.encoders.queue",
                    "--output",
                    "a path with spaces",
                    "--status",
                ],
            )

    def test_real_subprocess_queue_stages_and_continues_without_duplicate_steps(self):
        queue, state, devices = (
            implementation("queue"),
            implementation("queue_state"),
            implementation("devices"),
        )
        from scripts.encoders.config import run_matrix

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specs = [
                {
                    **r,
                    "max_updates": 500,
                    "max_seconds": 86400,
                    "config_hash": r["run_id"],
                    "code_revision": "fixture",
                    "memory_limit_gib": 200,
                }
                for r in run_matrix()[:10]
            ]
            store = state.QueueStore(
                root / "queue", {"fixture": "staging"}, specs, resume=False
            )

            def command(run, gpu, target, resume, ready, gate):
                args = [
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
                    str(gate) if gate else "none",
                    "--config-hash",
                    run["config_hash"],
                    "--code-revision",
                    "fixture",
                ]
                if resume:
                    args.extend(["--resume", str(resume)])
                return args

            resources = lambda: {
                "memory": {"working_set_bytes": 1000},
                "memory_events": {"oom_kill": 0},
                "cpu_stat": {"nr_throttled": 0},
            }
            with devices.GpuLeases(
                root / "leases", ["GPU-one", "GPU-two", "GPU-three", "GPU-four"]
            ) as leases:

                class DeadStream(io.StringIO):
                    def write(self, value):
                        raise BrokenPipeError("terminal closed")

                runner = queue.QueueRunner(
                    store,
                    leases,
                    command,
                    stream=DeadStream(),
                    resource_reader=resources,
                    poll_seconds=0.01,
                )
                self.assertEqual(runner.run(), 0)
            self.assertTrue(all(r["state"] == "completed" for r in store.state["runs"]))
            first = store.state["runs"][0]
            self.assertEqual([a["target"] for a in first["attempts"]], [200, 400, 500])
            self.assertEqual(store.state["runs"][1]["attempts"][0]["target"], 200)
            self.assertTrue(store.state["capacity"]["S2"]["overlap_seconds"] > 0)
            for run in store.state["runs"]:
                rows = [
                    json.loads(line)["update"]
                    for line in (store.root / "runs" / run["run_id"] / "metrics.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(rows, list(range(1, 501)))
            events = [
                json.loads(line)
                for line in (store.root / "queue_events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(e["event"] == "display-disabled" for e in events), 1)
            inflight = 0
            for event in events:
                if event["event"] == "started":
                    inflight += 1
                    self.assertEqual(inflight, 1)
                if event["event"] == "ready":
                    inflight -= 1

    def test_child_failure_stops_other_owned_jobs_and_preserves_failed_queue(self):
        queue, state, devices = (
            implementation("queue"),
            implementation("queue_state"),
            implementation("devices"),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specs = [
                {
                    "run_id": name,
                    "arm": "A",
                    "rho": 2.5,
                    "seed": 0,
                    "max_updates": 500,
                    "max_seconds": 86400,
                    "config_hash": name,
                    "code_revision": "fixture",
                    "memory_limit_gib": 200,
                }
                for name in ("first", "failure", "pending")
            ]
            store = state.QueueStore(
                root / "queue", {"fixture": "failure"}, specs, resume=False
            )
            store.state["phase"] = "S3"
            store.save()

            def command(run, gpu, target, resume, ready, gate):
                args = [
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
                    "--linger",
                    "1",
                ]
                if run["run_id"] == "failure":
                    args.append("--fail")
                return args

            with devices.GpuLeases(root / "leases", ["GPU-one", "GPU-two"]) as leases:
                runner = queue.QueueRunner(
                    store, leases, command, stream=io.StringIO(), poll_seconds=0.01
                )
                self.assertEqual(runner.run(), 2)
            self.assertEqual(store.run("failure")["state"], "failed")
            self.assertEqual(store.run("failure")["attempts"][-1]["exit_code"], 7)
            self.assertEqual(store.run("pending")["state"], "pending")
            self.assertFalse((store.root / "runs/pending").exists())
            self.assertEqual(store.run("first")["attempts"][-1]["exit_code"], 130)
            with self.assertRaisesRegex(ValueError, "retry"):
                store.resume_failed(set())

    def test_watchdog_begins_at_ready_and_only_requests_graceful_stop(self):
        queue, state, devices = (
            implementation("queue"),
            implementation("queue_state"),
            implementation("devices"),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specs = [
                {
                    "run_id": "capacity",
                    "arm": "A",
                    "rho": 2.5,
                    "seed": 0,
                    "max_updates": 30000,
                    "max_seconds": 86400,
                    "config_hash": "one",
                    "code_revision": "fixture",
                    "memory_limit_gib": 200,
                }
            ]
            store = state.QueueStore(
                root / "queue", {"fixture": "watchdog"}, specs, resume=False
            )

            def command(run, gpu, target, resume, ready, gate):
                return [
                    sys.executable,
                    "scripts/encoders/tests/queue_child.py",
                    "--output",
                    str(store.root / "runs/capacity"),
                    "--run-id",
                    "capacity",
                    "--stop-at",
                    str(target),
                    "--ready-file",
                    str(ready),
                    "--start-gate",
                    str(gate),
                    "--config-hash",
                    "one",
                    "--code-revision",
                    "fixture",
                    "--linger",
                    "1",
                ]

            with devices.GpuLeases(root / "leases", ["GPU-one"]) as leases:
                runner = queue.QueueRunner(
                    store,
                    leases,
                    command,
                    stream=io.StringIO(),
                    poll_seconds=0.01,
                    watchdog_seconds=0.04,
                )
                self.assertEqual(runner.run(), 2)
            run = store.run("capacity")
            self.assertEqual(run["state"], "paused")
            self.assertEqual(run["attempts"][-1]["exit_code"], 130)
            ledger = json.loads(
                (store.root / "runs/capacity/run_state.json").read_text()
            )
            self.assertEqual(ledger["attempted_updates"], ledger["completed_updates"])
            self.assertTrue(queue.safe_checkpoint(store.root, run).exists())
            self.assertGreater(
                run["attempts"][-1]["ready_at"], run["attempts"][-1]["started_at"]
            )
