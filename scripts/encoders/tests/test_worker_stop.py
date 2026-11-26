"""Regression: graceful supervisor stop must leave DataLoader workers alive."""

import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from scripts.encoders.devices import GpuLeases
from scripts.encoders.queue import QueueRunner
from scripts.encoders.queue_state import QueueStore


class RealWorkerStopTests(unittest.TestCase):
    def test_trainer_only_signal_checkpoints_and_resumes_with_two_real_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / "worker-alive.json"
            output = root / "queue/runs/cpu"
            spec = {
                "run_id": "cpu",
                "arm": "A",
                "rho": 10,
                "seed": 0,
                "max_updates": 4,
                "config_hash": "fixture",
                "code_revision": "fixture",
            }
            store = QueueStore(
                root / "queue", {"fixture": "worker-stop"}, [spec], resume=False
            )
            argv = [
                sys.executable,
                "-m",
                "scripts.encoders.tests.worker_stop_child",
                "--output",
                str(output),
                "--marker",
                str(marker),
            ]
            with GpuLeases(root / "leases", ["GPU-fixture"]) as leases:
                log = (root / "first.log").open("wb")
                child = subprocess.Popen(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(leases.fd("GPU-fixture"),),
                )
                try:
                    deadline = time.monotonic() + 20
                    while (
                        not marker.exists()
                        and child.poll() is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                    self.assertTrue(
                        marker.exists(),
                        "Real-worker fixture did not reach first update",
                    )
                    store.start(
                        "cpu",
                        "GPU-fixture",
                        4,
                        {"ready_file": str(marker), "log": str(root / "first.log")},
                    )
                    runner = QueueRunner(
                        store,
                        leases,
                        lambda *args: [],
                        stream=io.StringIO(),
                        poll_seconds=0.02,
                        grace_seconds=15,
                    )
                    runner.active["cpu"] = {
                        "child": child,
                        "log": log,
                        "ready": marker,
                        "ready_at": time.monotonic(),
                        "gpu": "GPU-fixture",
                    }
                    runner._stop_owned("real-worker regression")
                    self.assertEqual(
                        child.returncode, 130, (root / "first.log").read_text()
                    )
                    self.assertEqual(store.run("cpu")["state"], "paused")
                    ledger = json.loads((output / "run_state.json").read_text())
                    self.assertEqual(
                        (ledger["attempted_updates"], ledger["completed_updates"]),
                        (1, 1),
                    )
                    checkpoint = output / ledger["last_checkpoint_name"]
                    self.assertTrue(checkpoint.exists())
                finally:
                    if child.poll() is None:
                        os.kill(child.pid, signal.SIGTERM)
                        child.wait(timeout=20)
                    log.close()
                resumed = subprocess.run(
                    argv + ["--resume", str(checkpoint)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                self.assertEqual(
                    json.loads((output / "run_state.json").read_text())[
                        "completed_updates"
                    ],
                    4,
                )
