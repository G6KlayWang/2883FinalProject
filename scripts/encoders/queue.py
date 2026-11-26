"""Resumable ten-candidate supervisor; no automatic retries or scientific changes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from typing import TextIO
from scripts.data.file_io import write_json_new
from scripts.encoders.capacity import assess_run, overlap_seconds, resource_snapshot
from scripts.encoders.history import read_history, export_csv
from scripts.encoders.interruptions import StopRequest
from scripts.encoders.queue_state import QueueStore
from scripts.encoders.devices import GpuLeases


def latest_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        lines = handle.read().split(b"\n")
    for line in reversed(lines[:-1]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue  # Ignore an incomplete first tail line only for display.
    return {}


def render_status(store: QueueStore, stream: TextIO, *, redraw_lines: int = 0) -> int:
    if redraw_lines and stream.isatty():
        stream.write(f"\x1b[{redraw_lines}F\x1b[J")
    lines = [f"Encoder queue {store.state['phase']} — {store.root}"]
    for run in store.state["runs"]:
        row = latest_metrics(store.root / "runs" / run["run_id"] / "metrics.jsonl")
        step = row.get("update", run["completed_updates"])
        total = run["max_updates"]
        fraction = step / total
        bar = "#" * int(16 * fraction) + "-" * (16 - int(16 * fraction))
        text = f"{run['state']:9} {run['run_id']} [{bar}] {step}/{total} {100*fraction:.1f}%"
        if row:
            text += f" loss={row['total']:.5g} inv={row['invariance']:.5g} reg={row['regularizer']:.5g} lr={row['lr']:.4g} {row['source_frames_per_second']:.1f}frames/s"
            text += f" ETA estimate {max(0,total-step)*row['model_seconds']/max(step,1)/60:.1f}min"
        lines.append(text)
    stream.write("\n".join(lines) + "\n")
    stream.flush()
    return len(lines)


def safe_checkpoint(root: Path, run: dict) -> Path | None:
    path = root / "runs" / run["run_id"]
    if not path.exists():
        if run["completed_updates"]:
            raise ValueError("Resume run artifacts disappeared")
        return None
    ledger_path = path / "run_state.json"
    if not ledger_path.exists():
        raise ValueError(
            "Existing run has no safe checkpoint ledger; preserve and report"
        )
    ledger = json.loads(ledger_path.read_text())
    count = ledger["completed_updates"]
    if ledger["attempted_updates"] != count or count != run["completed_updates"]:
        raise ValueError(
            "Unresolved/stale attempted update: no automatic rollback or fresh restart"
        )
    name = ledger.get("last_checkpoint_name")
    if not name or Path(name).name != name or not (path / name).is_file():
        raise ValueError(
            "No clean checkpoint for explicit resume; preserve failed directory"
        )
    return (
        path / name
    )  # Trainer performs its mandatory checkpoint hash/identity verification.


class QueueRunner:
    def __init__(
        self,
        store: QueueStore,
        leases: GpuLeases,
        command: Callable,
        *,
        stream: TextIO | None = None,
        resource_reader: Callable = resource_snapshot,
        poll_seconds: float = 0.5,
        watchdog_seconds: float = 1200,
        grace_seconds: float = 120,
        prepare_validation: Callable | None = None,
    ):
        self.store, self.leases, self.command = store, leases, command
        self.stream = stream if stream is not None else sys.stderr
        self.resource_reader = resource_reader
        self.poll = poll_seconds
        self.watchdog = watchdog_seconds
        self.grace = grace_seconds
        self.prepare_validation = prepare_validation
        self.active = {}
        self.last_display = 0.0
        self.drawn_lines = 0
        self.display_enabled = True

    def _targets(self, phase: str) -> dict[str, int]:
        runs = self.store.state["runs"]
        if phase == "S1":
            return {runs[0]["run_id"]: 200}
        if phase == "S2":
            if len(self.leases.uuids) != 4:
                raise ValueError(
                    "S2 requires four independent leased GPUs for observed concurrency"
                )
            return {
                r["run_id"]: (400 if i == 0 else 200) for i, r in enumerate(runs[:4])
            }
        return {r["run_id"]: r["max_updates"] for r in runs}

    def _launch(self, run: dict, gpu: str, target: int, gate: Path | None) -> None:
        resume = safe_checkpoint(self.store.root, run)
        number = len(run["attempts"])
        ready = self.store.root / "ready" / f"{run['run_id']}-{number}.json"
        ready.parent.mkdir(exist_ok=True)
        log = self.store.root / "logs" / f"{run['run_id']}-{number}.log"
        log.parent.mkdir(exist_ok=True)
        argv = self.command(run, gpu, target, resume, ready, gate)
        self.store.start(
            run["run_id"],
            gpu,
            target,
            {
                "ready_file": str(ready),
                "log": str(log),
                "argv": argv,
                "resume": str(resume) if resume else None,
            },
        )
        handle = log.open("xb")
        try:
            child = subprocess.Popen(
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONUNBUFFERED": "1",
                },
                start_new_session=True,
                pass_fds=(self.leases.fd(gpu),),
            )
        except OSError as error:
            handle.close()
            self.store.finish(run["run_id"], 127, {}, str(error))
            raise
        self.active[run["run_id"]] = {
            "child": child,
            "log": handle,
            "ready": ready,
            "ready_at": None,
            "gpu": gpu,
        }
        run["attempts"][-1]["pid"] = child.pid
        self.store.save()

    def _poll(self) -> bool:
        failed = False
        for run_id, active in list(self.active.items()):
            run = self.store.run(run_id)
            if active["ready_at"] is None and active["ready"].exists():
                ready = json.loads(active["ready"].read_text())
                expected = {
                    "config_hash": run["config_hash"],
                    "code_revision": run["code_revision"],
                    "gpu_uuid": active["gpu"],
                    "completed_updates": run["completed_updates"],
                }
                if any(ready.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Startup identity mismatch for {run_id}")
                active["ready_at"] = time.monotonic()
                run["attempts"][-1]["ready_at"] = time.time()
                self.store.save()
                self.store.event("ready", run_id=run_id)
            code = active["child"].poll()
            if code is None:
                continue
            active["log"].close()
            output = self.store.root / "runs" / run_id
            report = (
                json.loads((output / "training_report.json").read_text())
                if (output / "training_report.json").exists()
                else {}
            )
            if code not in (0, 130) and (output / "run_state.json").exists():
                report.update(json.loads((output / "run_state.json").read_text()))
            self.store.finish(
                run_id,
                code,
                report,
                None if code == 0 else f"Trainer exit {code}; inspect saved log",
            )
            failed |= run["state"] == "failed"
            del self.active[run_id]
        return failed

    def _show(self, force: bool = False) -> None:
        if not self.display_enabled:
            return
        now = time.monotonic()
        try:
            if force or now - self.last_display >= (1 if self.stream.isatty() else 30):
                self.drawn_lines = render_status(
                    self.store, self.stream, redraw_lines=self.drawn_lines
                )
                self.last_display = now
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.display_enabled = False
            self.store.event("display-disabled", reason=str(error))

    def _stop_owned(self, reason: str) -> None:
        for run_id, active in self.active.items():
            self.store.event(
                "graceful-stop-request",
                run_id=run_id,
                reason=reason,
                pid=active["child"].pid,
            )
            if active["child"].poll() is None:
                try:
                    os.killpg(active["child"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.grace
        while self.active and time.monotonic() < deadline:
            self._poll()
            self._show()
            time.sleep(self.poll)
        if self.active:
            owned = {key: entry["child"].pid for key, entry in self.active.items()}
            self.store.state["operator_intervention_required"] = {
                "reason": reason,
                "owned_pids": owned,
            }
            self.store.save()
            self.store.event("operator-intervention-required", owned_pids=owned)
            if self.display_enabled:
                try:
                    self.stream.write(
                        f"Graceful exit not observed; no SIGKILL/retry. Coordinator must inspect owned processes: {owned}\n"
                    )
                    self.stream.flush()
                except (OSError, ValueError):
                    self.display_enabled = False
            for entry in self.active.values():
                entry["log"].close()

    def _capacity(self, phase: str, targets: dict, resources: dict) -> bool:
        reports = []
        checks = {}
        for run_id, target in targets.items():
            run = self.store.run(run_id)
            attempt = run["attempts"][-1]
            report = attempt["report"]
            reports.append(report)
            rows = read_history(self.store.root / "runs" / run_id / "metrics.jsonl")
            start = (
                200
                if phase == "S2" and run_id == self.store.state["runs"][0]["run_id"]
                else 0
            )
            checks[run_id] = assess_run(
                rows,
                report,
                start=start,
                target=target,
                total=run["max_updates"],
                max_seconds=run["max_seconds"],
                phase=phase,
            )
        overlap = overlap_seconds(reports)
        memory = resources["peak_working_set_bytes"]
        memory_limit = (
            min(r["memory_limit_gib"] for r in self.store.state["runs"]) * 1024**3
        )
        end = self.resource_reader()
        before = resources["before"]
        oom_delta = end["memory_events"].get("oom_kill", 0) - before[
            "memory_events"
        ].get("oom_kill", 0)
        passed = (
            all(c["passed"] for c in checks.values())
            and (phase != "S2" or overlap > 0)
            and memory is not None
            and memory <= memory_limit
            and oom_delta == 0
        )
        self.store.state["capacity"][phase] = {
            "passed": bool(passed),
            "runs": checks,
            "overlap_seconds": overlap,
            "resources": {**resources, "after": end, "oom_kill_delta": oom_delta},
        }
        self.store.save()
        self.store.event(
            "capacity-assessed",
            phase=phase,
            passed=bool(passed),
            overlap_seconds=overlap,
        )
        return passed

    def _phase(self, phase: str, stop: StopRequest) -> int:
        targets = self._targets(phase)
        gate = (
            self.store.root / f"{phase}-start-{time.time_ns()}.json"
            if phase in ("S1", "S2")
            else None
        )
        snapshot = self.resource_reader()
        resources = {
            "before": snapshot,
            "peak_working_set_bytes": (
                snapshot["memory"]["working_set_bytes"] if snapshot["memory"] else None
            ),
        }
        while True:
            if self._poll():
                self._stop_owned("trainer failure")
                return 2
            if stop():
                self._stop_owned("supervisor interrupted")
                return 130
            if phase in ("S1", "S2") and any(
                a["ready_at"] is not None
                and time.monotonic() - a["ready_at"] > self.watchdog
                for a in self.active.values()
            ):
                self.store.event(
                    "capacity-watchdog", phase=phase, seconds=self.watchdog
                )
                self._stop_owned("capacity20min watchdog after startup-ready")
                return 2
            current = self.resource_reader()
            if current["memory"] is not None:
                resources["peak_working_set_bytes"] = max(
                    resources["peak_working_set_bytes"] or 0,
                    current["memory"]["working_set_bytes"],
                )
            unfinished = [
                self.store.run(k)
                for k, target in targets.items()
                if self.store.run(k)["completed_updates"] < target
            ]
            if not unfinished and not self.active:
                break
            # One mandatory source/cache verification in flight; ready means the
            # next startup may proceed without serializing whole training jobs.
            starting = any(a["ready_at"] is None for a in self.active.values())
            if not starting:
                used = {a["gpu"] for a in self.active.values()}
                for run in unfinished:
                    if run["state"] in ("running", "failed"):
                        continue
                    gpu = run["gpu_uuid"] or next(
                        (g for g in self.leases.uuids if g not in used), None
                    )
                    if gpu is not None and gpu not in used:
                        if gpu not in self.leases.uuids:
                            raise ValueError(
                                "Original GPU UUID missing from resume leases"
                            )
                        self._launch(run, gpu, targets[run["run_id"]], gate)
                        break
            if (
                gate
                and not gate.exists()
                and len(self.active) == len(unfinished)
                and all(a["ready_at"] is not None for a in self.active.values())
            ):
                write_json_new(gate, {"phase": phase, "released_at": time.time()})
                self.store.event("capacity-barrier-released", phase=phase)
            self._show()
            time.sleep(self.poll)
        if phase in ("S1", "S2") and not self._capacity(phase, targets, resources):
            self.store.state["blocked_reason"] = (
                "Capacity gate failed; preserve checkpoints and report without changing recipe"
            )
            self.store.save()
            return 2
        return 0

    def run(self, stop: StopRequest | None = None) -> int:
        stop = stop if stop is not None else StopRequest()
        try:
            while self.store.state["phase"] != "completed":
                phase = self.store.state["phase"]
                code = self._phase(phase, stop)
                if code:
                    return code
                if phase == "S2" and self.prepare_validation:
                    self.prepare_validation()
                self.store.state["phase"] = {"S1": "S2", "S2": "S3", "S3": "completed"}[
                    phase
                ]
                self.store.save()
            for run in self.store.state["runs"]:
                path = self.store.root / "runs" / run["run_id"]
                if not (path / "history.csv").exists():
                    export_csv(path / "metrics.jsonl", path / "history.csv")
            self._show(force=True)
            return 0
        except BaseException:
            self._stop_owned("supervisor exception; inspect durable evidence")
            raise


def main() -> None:
    from scripts.encoders.launcher import main as launch_main

    launch_main()


if __name__ == "__main__":
    main()
