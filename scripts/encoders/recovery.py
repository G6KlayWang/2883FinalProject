"""Explicit, audited recovery after exclusive leases prove no live trainer owner."""

import json
from pathlib import Path
import time
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.checkpoints import load_checkpoint
from scripts.encoders.devices import GpuLeases
from scripts.encoders.queue_state import QueueStore


def _require_owned_slot(run: dict, leases: GpuLeases) -> None:
    gpu = run.get("gpu_uuid")
    if gpu is not None:
        try:
            leases.fd(gpu)
        except KeyError:
            raise ValueError(
                f"Original GPU UUID is not exclusively leased: {gpu}"
            ) from None


def clean_checkpoint(store: QueueStore, run: dict) -> tuple[dict, Path]:
    """Validate actual checkpoint contents, not a possibly stale supervisor count."""
    directory = store.root / "runs" / run["run_id"]
    ledger = json.loads((directory / "run_state.json").read_text())
    count = ledger["completed_updates"]
    if (
        type(count) is not int
        or not 0 <= count <= run["max_updates"]
        or ledger["attempted_updates"] != count
    ):
        raise ValueError("Unresolved attempted update; preserved as failed")
    name = ledger.get("last_checkpoint_name")
    if not name or Path(name).name != name:
        raise ValueError("No safe checkpoint recorded")
    path = directory / name
    if sha256_file(path) != ledger.get("last_checkpoint_sha256"):
        raise ValueError("Recovery checkpoint hash mismatch")
    checkpoint = load_checkpoint(path)
    if (
        checkpoint["completed_updates"] != count
        or checkpoint["attempted_updates"] != count
    ):
        raise ValueError("Recovery ledger and checkpoint counts differ")
    identity = checkpoint["identity"]
    if identity.get("config_hash") != run["config_hash"]:
        raise ValueError("Recovery checkpoint config differs from declared run")
    saved = identity["run_identity"]
    expected = {
        "code_revision": run["code_revision"],
        "gpu_uuid": run["gpu_uuid"],
        "environment": store.state["identity"]["environment"],
        "manifest_sha256": store.state["identity"]["manifest_sha256"],
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Recovery checkpoint source/code/environment/GPU identity mismatch"
        )
    return ledger, path


def reconcile_queue(store: QueueStore, leases: GpuLeases) -> None:
    """Caller holds all original leases and has checked hardware idle ownership."""
    for run in store.state["runs"]:
        if run["state"] == "pending":
            continue
        _require_owned_slot(run, leases)
        try:
            ledger, path = clean_checkpoint(store, run)
        except (OSError, ValueError, KeyError, EOFError) as error:
            run.update(state="failed", recovery_error=str(error), parked=True)
            store.event("recovery-parked", run_id=run["run_id"], reason=str(error))
            continue
        previous = {
            "state": run["state"],
            "completed_updates": run["completed_updates"],
        }
        count = ledger["completed_updates"]
        run.update(
            state="completed" if count == run["max_updates"] else "paused",
            completed_updates=count,
            parked=False,
            recovery_error=None,
        )
        # Reattach a matching trainer report when the supervisor missed exit;
        # an orphan's exit code remains unknown, never fabricated as zero.
        report_path = store.root / "runs" / run["run_id"] / "training_report.json"
        if report_path.exists() and run["attempts"]:
            report = json.loads(report_path.read_text())
            if (
                report.get("completed_updates") == count
                and report.get("attempted_updates") == count
                and report.get("checkpoint_sha256") == ledger["last_checkpoint_sha256"]
                and report.get("config_hash") == run["config_hash"]
            ):
                run["attempts"][-1].update(
                    report=report, recovered_from_checkpoint=True
                )
        # Preserve attempts and record exactly what the checkpoint establishes.
        store.event(
            "reconciled-checkpoint",
            run_id=run["run_id"],
            previous=previous,
            completed_updates=count,
            checkpoint=str(path),
            gpu_uuid=run["gpu_uuid"],
        )
    store.save()


def restart_from_zero(
    store: QueueStore, run_id: str, leases: GpuLeases, *, decision: str
) -> None:
    if not decision.strip():
        raise ValueError("A later explicit restart decision must be recorded")
    run = store.run(run_id)
    if run["state"] != "failed":
        raise ValueError("Restart-from-zero is limited to failed runs")
    _require_owned_slot(run, leases)
    directory = store.root / "runs" / run_id
    ledger_path = directory / "run_state.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    archive = store.root / "discarded" / f"{run_id}-{time.time_ns()}"
    record = {
        "decision": decision,
        "original_directory": str(directory),
        "declared_at": time.time(),
        "archive": str(archive) if directory.exists() else None,
        "discarded_completed_updates": ledger.get(
            "completed_updates", run["completed_updates"]
        ),
        "discarded_attempted_updates": ledger.get("attempted_updates"),
        "discarded_model_seconds": ledger.get("model_seconds"),
        "runtime_scope": "last persisted ledger; unobserved hard-crash tail unknown",
        "gpu_uuid": run["gpu_uuid"],
        "config_hash": run.get("config_hash"),
    }
    store.event("restart-from-zero-declared", run_id=run_id, **record)
    if directory.exists():
        archive.parent.mkdir(exist_ok=True)
        directory.rename(archive)
        write_json_new(archive / "restart_record.json", record)
    run.setdefault("restarts", []).append(record)
    run.update(state="pending", completed_updates=0, parked=False, recovery_error=None)
    # Original UUID, config and all queue attempts remain unchanged.
    store.save()
