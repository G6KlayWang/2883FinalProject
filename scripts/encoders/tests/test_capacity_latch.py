"""Metadata-only regression: idle resume cannot rewrite observed capacity evidence."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from scripts.encoders.devices import GpuLeases
from scripts.encoders.interruptions import StopRequest
from scripts.encoders.queue import QueueRunner
from scripts.encoders.queue_state import QueueStore


class CapacityLatchTests(unittest.TestCase):
    def test_failed_and_passed_capacity_evidence_survives_lower_idle_resume(self):
        gib = 1024**3
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uuids = ["GPU-one", "GPU-two", "GPU-three", "GPU-four"]
            specs = [
                {
                    "run_id": name,
                    "arm": arm,
                    "rho": rho,
                    "seed": 0,
                    "max_updates": 30000,
                    "max_seconds": 86400,
                    "memory_limit_gib": 200,
                    "config_hash": name,
                    "code_revision": "fixture",
                }
                for name, arm, rho in (
                    ("A-low", "A", 2.5),
                    ("B-low", "B", 2.5),
                    ("C-low", "C", 0.25),
                    ("Aprime", "Aprime", 0.21),
                )
            ]

            def snapshot(peak):
                return {
                    "memory": {"working_set_bytes": peak * gib},
                    "memory_events": {"oom_kill": 0},
                    "cpu_stat": {},
                }

            def forbidden(*args):
                raise AssertionError(
                    "Recorded capacity resume must not start a model or validation"
                )

            with GpuLeases(root / "leases", uuids) as leases:
                for phase in ("S1", "S2"):
                    for peak, passed in ((201, False), (150, True)):
                        with self.subTest(phase=phase, passed=passed):
                            store = QueueStore(
                                root / f"{phase}-{peak}",
                                {"code_revision": "fixture"},
                                specs,
                                resume=False,
                            )
                            store.state["phase"] = phase
                            targets = (
                                {"A-low": 200}
                                if phase == "S1"
                                else {
                                    "A-low": 400,
                                    "B-low": 200,
                                    "C-low": 200,
                                    "Aprime": 200,
                                }
                            )
                            for run, gpu in zip(store.state["runs"], uuids):
                                target = targets.get(run["run_id"], 0)
                                run.update(
                                    state="paused",
                                    gpu_uuid=gpu,
                                    completed_updates=target,
                                )
                                if not target:
                                    continue
                                report = {
                                    "completed_updates": target,
                                    "attempted_updates": target,
                                    "model_seconds": 100,
                                    "peak_gpu_memory_bytes_this_attempt": gib,
                                    "gpu_total_memory_bytes": 96 * gib,
                                    "training_started_at": 10,
                                    "training_finished_at": 100,
                                }
                                run["attempts"] = [
                                    {"target": target, "exit_code": 0, "report": report}
                                ]
                                output = store.root / "runs" / run["run_id"]
                                output.mkdir(parents=True)
                                rows = [
                                    {
                                        "update": i,
                                        "total": 1.0,
                                        "invariance": 0.1,
                                        "regularizer": 0.2,
                                        "update_seconds": 0.1,
                                        "data_wait_seconds": 0.01,
                                    }
                                    for i in range(1, target + 1)
                                ]
                                (output / "metrics.jsonl").write_text(
                                    "".join(json.dumps(row) + "\n" for row in rows)
                                )
                            store.save()
                            initial = QueueRunner(
                                store,
                                leases,
                                forbidden,
                                stream=io.StringIO(),
                                resource_reader=lambda: snapshot(peak),
                            )
                            self.assertEqual(
                                initial._phase(phase, StopRequest()), 0 if passed else 2
                            )
                            recorded = copy.deepcopy(store.state["capacity"][phase])
                            reason = store.state.get("blocked_reason")
                            resumed = QueueStore(
                                store.root,
                                {"code_revision": "fixture"},
                                specs,
                                resume=True,
                            )
                            idle_reads = []

                            def idle():
                                idle_reads.append(True)
                                return snapshot(1)

                            runner = QueueRunner(
                                resumed,
                                leases,
                                forbidden,
                                stream=io.StringIO(),
                                resource_reader=idle,
                                prepare_validation=forbidden,
                            )
                            result = (
                                runner._phase(phase, StopRequest())
                                if passed
                                else runner.run()
                            )
                            self.assertEqual(result, 0 if passed else 2)
                            self.assertEqual(resumed.state["capacity"][phase], recorded)
                            self.assertEqual(
                                resumed.state.get("blocked_reason"), reason
                            )
                            self.assertEqual(idle_reads, [])
                            if passed:
                                resumed.state["runs"][0][
                                    "config_hash"
                                ] = "changed-run-identity"
                                self.assertEqual(runner.run(), 2)
                                self.assertEqual(
                                    resumed.state["capacity"][phase], recorded
                                )
                                self.assertEqual(idle_reads, [])
