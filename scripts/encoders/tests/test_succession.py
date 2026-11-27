"""Metadata-only pin/equality boundaries for the single declared predecessor."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.queue_state import QueueStore
from scripts.encoders.tests.test_losses import implementation


class SuccessionTests(unittest.TestCase):
    def test_exact_legacy_pin_only_and_queue_identity_stays_historical(self):
        succession = implementation("succession")
        old = succession.PREDECESSOR_REVISION
        new = "1" * 40
        pin = "2" * 64
        origin = {
            "config_hash": "3" * 64,
            "run_identity": {
                "code_revision": old,
                "gpu_uuid": "GPU-one",
                "manifest_sha256": "4" * 64,
                "environment": {},
                "images": {},
            },
        }
        record = {
            "predecessor_revision": old,
            "successor_revision": new,
            "receipt_sha256": "5" * 64,
            "checkpoint_name": "step-029307.pt",
            "checkpoint_sha256": pin,
            "completed_updates": 29307,
            "model_seconds": 100.0,
        }
        expected = copy.deepcopy(origin)
        expected["run_identity"].update(code_revision=new, source_succession=record)
        self.assertTrue(succession.identity_matches(origin, expected, pin))
        self.assertFalse(succession.identity_matches(origin, expected, "6" * 64))
        self.assertTrue(succession.identity_matches(expected, expected, "6" * 64))
        for key, value in [
            ("gpu_uuid", "GPU-two"),
            ("manifest_sha256", "7" * 64),
            ("code_revision", "8" * 40),
        ]:
            changed = copy.deepcopy(origin)
            changed["run_identity"][key] = value
            self.assertFalse(succession.identity_matches(changed, expected, pin))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "queue"
            stored = {"code_revision": old, "environment": {}}
            fresh = {**stored, "code_revision": new}
            store = QueueStore(
                root, stored, [{"run_id": "A", "max_updates": 30000}], resume=False
            )
            store.state["capacity"] = {
                "S1": {"passed": True, "peak": 123},
                "S2": {"passed": False, "reason": "preserved"},
            }
            store.save()
            before = copy.deepcopy(store.state)
            receipt = {
                "predecessor_revision": old,
                "successor_revision": new,
                "queue_identity_sha256": identity_digest(stored),
                "queue_state_sha256": succession.sha256_file(root / "queue_state.json"),
                "queue_record_sha256": succession.sha256_file(
                    root / "queue_record.json"
                ),
                "capacity_sha256": identity_digest(before["capacity"]),
                "runs": {},
                "rollback_receipts": [],
                "_sha256": "9" * 64,
            }
            accepted = QueueStore(
                root,
                fresh,
                [{"run_id": "A", "max_updates": 30000}],
                resume=True,
                succession=receipt,
            )
            self.assertEqual(accepted.state["identity"], before["identity"])
            self.assertEqual(accepted.state["capacity"], before["capacity"])
            with self.assertRaisesRegex(ValueError, "identity"):
                QueueStore(
                    root,
                    {**fresh, "environment": {"changed": True}},
                    [{"run_id": "A", "max_updates": 30000}],
                    resume=True,
                    succession=receipt,
                )
