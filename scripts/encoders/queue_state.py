"""Atomic queue transitions and append-only attempt/exit evidence."""

import json
import os
from pathlib import Path
import time
from scripts.data.file_io import write_json_new
from scripts.encoders.checkpoints import replace_json


class QueueStore:
    def __init__(
        self,
        root: Path,
        identity: dict,
        runs: list[dict],
        *,
        resume: bool,
        succession: dict | None = None,
    ):
        self.root = Path(root)
        if resume:
            self.state = json.loads((self.root / "queue_state.json").read_text())
            if self.state["identity"] != identity:
                if succession is None:
                    raise ValueError("Queue identity changed; refuse resume")
                from scripts.encoders.succession import validate_queue_succession

                validate_queue_succession(self.root, self.state, identity, succession)
            if [r["run_id"] for r in self.state["runs"]] != [r["run_id"] for r in runs]:
                raise ValueError("Queue run declarations changed")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.state = {
                "version": 1,
                "identity": identity,
                "phase": "S1",
                "capacity": {},
                "runs": [
                    {
                        **r,
                        "state": "pending",
                        "gpu_uuid": None,
                        "completed_updates": 0,
                        "attempts": [],
                    }
                    for r in runs
                ],
            }
            write_json_new(
                self.root / "queue_record.json",
                {"identity": identity, "runs": runs, "created_at": time.time()},
            )
            self.save()

    def save(self) -> None:
        replace_json(self.root / "queue_state.json", self.state)

    def event(self, kind: str, **values) -> None:
        with (self.root / "queue_events.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {"time": time.time(), "event": kind, **values}, allow_nan=False
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def run(self, run_id: str) -> dict:
        return next(r for r in self.state["runs"] if r["run_id"] == run_id)

    def check_gpu(self, run_id: str, gpu: str) -> None:
        assigned = self.run(run_id)["gpu_uuid"]
        if assigned is not None and assigned != gpu:
            raise ValueError(
                f"Resume must retain original GPU UUID {assigned} for {run_id}"
            )

    def start(self, run_id: str, gpu: str, target: int, attempt: dict) -> None:
        run = self.run(run_id)
        self.check_gpu(run_id, gpu)
        if run["state"] not in ("pending", "paused"):
            raise ValueError(f"Cannot launch {run_id} from state {run['state']}")
        run.update(state="running", gpu_uuid=gpu)
        run["attempts"].append(
            {**attempt, "target": target, "started_at": time.time(), "exit_code": None}
        )
        self.save()
        self.event("started", run_id=run_id, gpu_uuid=gpu, target=target)

    def finish(
        self, run_id: str, code: int, report: dict, reason: str | None = None
    ) -> None:
        run = self.run(run_id)
        count = report.get("completed_updates", run["completed_updates"])
        attempt = run["attempts"][-1]
        attempt.update(
            exit_code=code, finished_at=time.time(), reason=reason, report=report
        )
        safe = report.get("attempted_updates") == count
        expected = count == attempt["target"]
        stopped = code == 130 and report.get("status") == "stopped"
        passed = safe and ((code == 0 and expected) or stopped)
        run.update(
            completed_updates=count,
            state=(
                ("completed" if count == run["max_updates"] else "paused")
                if passed
                else "failed"
            ),
        )
        self.save()
        self.event(
            "exited", run_id=run_id, exit_code=code, state=run["state"], reason=reason
        )

    def resume_failed(self, retry_ids: set[str]) -> None:
        known = {r["run_id"] for r in self.state["runs"]}
        if retry_ids - known:
            raise ValueError("Unknown explicit retry run IDs")
        for run in self.state["runs"]:
            if run["state"] != "failed":
                continue
            # Reconciliation already promotes demonstrably clean checkpoints.
            # A retry request cannot bypass an unresolved/corrupt checkpoint.
            run["parked"] = True
            self.event(
                (
                    "retry-unavailable-parked"
                    if run["run_id"] in retry_ids
                    else "failed-run-parked"
                ),
                run_id=run["run_id"],
                reason=run.get("recovery_error"),
            )
        self.save()
