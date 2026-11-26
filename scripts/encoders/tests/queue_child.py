"""Synthetic subprocess fixture: records argument/handshake/history, no model/GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

p = argparse.ArgumentParser()
for name in (
    "output",
    "ready-file",
    "start-gate",
    "config-hash",
    "code-revision",
    "run-id",
):
    p.add_argument("--" + name, required=True)
p.add_argument("--stop-at", type=int, required=True)
p.add_argument("--resume")
p.add_argument("--fail", action="store_true")
p.add_argument("--linger", type=float, default=0.15)
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
stop = False


def request(*args):
    global stop
    stop = True


signal.signal(signal.SIGINT, request)
signal.signal(signal.SIGTERM, request)
start = (
    json.loads((root / "run_state.json").read_text())["completed_updates"]
    if a.resume
    else 0
)
Path(a.ready_file).parent.mkdir(parents=True, exist_ok=True)
time.sleep(0.03)
ready_tmp = Path(a.ready_file).with_suffix(".tmp")
ready_tmp.write_text(
    json.dumps(
        {
            "status": "ready",
            "config_hash": a.config_hash,
            "code_revision": a.code_revision,
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "completed_updates": start,
        }
    )
)
ready_tmp.replace(a.ready_file)
while a.start_gate != "none" and not Path(a.start_gate).exists() and not stop:
    time.sleep(0.01)
began = time.time()
if a.fail:
    raise SystemExit(7)
count = start
with (root / "metrics.jsonl").open("a") as f:
    for i in range(start + 1, a.stop_at + 1):
        if stop:
            break
        count = i
        f.write(
            json.dumps(
                {
                    "update": i,
                    "total": 1.0,
                    "invariance": 0.1,
                    "regularizer": 0.2,
                    "update_seconds": 0.01,
                    "data_wait_seconds": 0.001,
                    "model_seconds": i * 0.011,
                    "lr": 0.0005,
                    "source_frames_per_second": 512 / 0.011,
                }
            )
            + "\n"
        )
    f.flush()
deadline = time.monotonic() + a.linger
while time.monotonic() < deadline and not stop:
    time.sleep(0.01)
checkpoint = root / f"step-{count:06d}.pt"
checkpoint.write_text(str(count))
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
ledger = {
    "completed_updates": count,
    "attempted_updates": count,
    "last_checkpoint_name": checkpoint.name,
    "last_checkpoint_sha256": digest,
}
(root / "run_state.json").write_text(json.dumps(ledger))
report = {
    **ledger,
    "status": "stopped" if stop else "candidate",
    "model_seconds": count * 0.011,
    "peak_gpu_memory_bytes_this_attempt": 1000000000,
    "gpu_total_memory_bytes": 96000000000,
    "training_started_at": began,
    "training_finished_at": time.time(),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
}
(root / "training_report.json").write_text(json.dumps(report))
raise SystemExit(130 if stop else 0)
