"""One reviewed reporting-only predecessor; never a general identity bypass."""

import copy
import json
import subprocess
from pathlib import Path
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.checkpoints import load_checkpoint

PREDECESSOR_REVISION = "ba914f6bb20fb2052f8e63fbc53a8e8b38c73582"
HANDOVER_RUNS = {"A-rho2.5-seed0", "C-rho0.25-seed0", "Aprime-rho0.210526-seed0"}
FINALIZE_RUN = "B-rho2.5-seed0"


def load_succession(path: Path | None, revision: str) -> dict | None:
    if path is None:
        return None
    receipt = json.loads(path.read_text())
    if (
        receipt.get("version") != 1
        or receipt.get("predecessor_revision") != PREDECESSOR_REVISION
        or receipt.get("successor_revision") != revision
        or revision == PREDECESSOR_REVISION
    ):
        raise ValueError(
            "Succession receipt must name the one predecessor and actual successor"
        )
    if set(receipt.get("runs", {})) != HANDOVER_RUNS | {FINALIZE_RUN}:
        raise ValueError(
            "Succession receipt must pin exactly the four interrupted runs"
        )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("Succession successor differs from checked-out HEAD")
    receipt["_sha256"] = sha256_file(path)
    return receipt


def source_record(receipt: dict, run_id: str) -> dict:
    pin = receipt["runs"][run_id]
    return {
        key: receipt[key] for key in ("predecessor_revision", "successor_revision")
    } | {
        "receipt_sha256": receipt["_sha256"],
        **{
            key: pin[key]
            for key in (
                "checkpoint_name",
                "checkpoint_sha256",
                "completed_updates",
                "model_seconds",
            )
        },
    }


def identity_matches(saved: dict, expected: dict, checkpoint_sha256: str) -> bool:
    if saved == expected:
        return True
    record = expected.get("run_identity", {}).get("source_succession")
    if not record or (
        record.get("predecessor_revision") != PREDECESSOR_REVISION
        or record.get("successor_revision")
        != expected["run_identity"].get("code_revision")
        or record.get("checkpoint_sha256") != checkpoint_sha256
    ):
        return False
    legacy = copy.deepcopy(expected)
    legacy["run_identity"].pop("source_succession")
    legacy["run_identity"]["code_revision"] = PREDECESSOR_REVISION
    return saved == legacy


def prepare_handover(
    directory: Path, receipt: dict | None, identity: dict
) -> dict | None:
    path = directory / "source_succession.json"
    if path.exists():
        if receipt is None or directory.name not in HANDOVER_RUNS:
            raise ValueError("Existing source succession requires its original receipt")
        record = source_record(receipt, directory.name)
        if json.loads(path.read_text()) != record:
            raise ValueError("Source succession record differs from receipt")
    elif receipt and directory.name in HANDOVER_RUNS:
        record = source_record(receipt, directory.name)
        pin = receipt["runs"][directory.name]
        ledger = json.loads((directory / "run_state.json").read_text())
        if (
            sha256_file(directory / "run_state.json") != pin["ledger_sha256"]
            or sha256_file(directory / "run_record.json") != pin["run_record_sha256"]
            or ledger["last_checkpoint_sha256"] != record["checkpoint_sha256"]
            or ledger["last_checkpoint_name"] != record["checkpoint_name"]
            or ledger["completed_updates"] != record["completed_updates"]
            or ledger["model_seconds"] != record["model_seconds"]
        ):
            raise ValueError(
                "Initial handover ledger/record differs from pinned receipt"
            )
    else:
        return None
    identity["source_succession"] = record
    return record


def publish_handover(directory: Path, record: dict | None) -> None:
    # Called only after Trainer._restore has verified and loaded the exact pin.
    if record is not None and not (directory / "source_succession.json").exists():
        write_json_new(directory / "source_succession.json", record)


def validate_queue_succession(
    root: Path, state: dict, fresh: dict, receipt: dict
) -> None:
    historical = {**fresh, "code_revision": PREDECESSOR_REVISION}
    if (
        state["identity"] != historical
        or fresh["code_revision"] != receipt["successor_revision"]
        or identity_digest(historical) != receipt["queue_identity_sha256"]
    ):
        raise ValueError(
            "Queue identity differs beyond the declared reporting revision"
        )
    if (
        sha256_file(root / "queue_record.json") != receipt["queue_record_sha256"]
        or identity_digest(state["capacity"]) != receipt["capacity_sha256"]
    ):
        raise ValueError("Historical queue record or capacity evidence changed")
    registered = any(
        entry["receipt_sha256"] == receipt["_sha256"]
        for entry in state.get("source_successions", [])
    )
    for run_id, pin in receipt["runs"].items():
        if (
            sha256_file(root / "runs" / run_id / "run_record.json")
            != pin["run_record_sha256"]
        ):
            raise ValueError(f"Historical run record changed: {run_id}")
    for artifact in receipt["rollback_receipts"]:
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Historical rollback receipt changed")
    if registered:
        return
    if sha256_file(root / "queue_state.json") != receipt["queue_state_sha256"]:
        raise ValueError("Initial queue state differs from receipt")
    for run_id, pin in receipt["runs"].items():
        directory = root / "runs" / run_id
        for name, field in (
            ("run_state.json", "ledger_sha256"),
            ("run_record.json", "run_record_sha256"),
            ("metrics.jsonl", "history_sha256"),
        ):
            if sha256_file(directory / name) != pin[field]:
                raise ValueError(
                    f"Initial succession metadata changed: {run_id}/{name}"
                )


def finalize_report(
    directory: Path,
    checkpoint: Path,
    config,
    identity: dict,
    frames,
    receipt: dict | None,
) -> Path:
    """Validate a completed state and write only a new immutable reporting receipt."""
    ledger = json.loads((directory / "run_state.json").read_text())
    digest = sha256_file(checkpoint)
    state = load_checkpoint(checkpoint)
    expected = {"config_hash": config.digest(), "run_identity": identity}
    revision = identity["code_revision"]
    checkpoint_revision = state["identity"]["run_identity"]["code_revision"]
    if checkpoint_revision != revision:
        if receipt is None or directory.name != FINALIZE_RUN:
            raise ValueError("Finalization predecessor requires the pinned B receipt")
        pin = receipt["runs"][FINALIZE_RUN]
        if (
            digest != pin["checkpoint_sha256"]
            or checkpoint.name != pin["checkpoint_name"]
            or sha256_file(directory / "run_state.json") != pin["ledger_sha256"]
            or sha256_file(directory / "run_record.json") != pin["run_record_sha256"]
        ):
            raise ValueError(
                "Finalization checkpoint/ledger/record differs from receipt"
            )
        expected["run_identity"] = {
            **identity,
            "source_succession": source_record(receipt, FINALIZE_RUN),
        }
    count = config.optimizer.max_updates
    ledger_identity = {key: ledger[key] for key in ("config_hash", "run_identity")}
    if not (
        identity_matches(state["identity"], expected, digest)
        and identity_matches(ledger_identity, expected, digest)
    ):
        raise ValueError("Finalization identity mismatch")
    if (
        state["checkpoint_role"] != "final"
        or state["config"] != config.to_dict()
        or any(
            item[key] != count
            for item in (state, ledger)
            for key in ("completed_updates", "attempted_updates")
        )
        or state["scheduler"]["completed_updates"] != count
        or state["sampler"]["next_update"] != count
        or state["data_identity"] != frames.identity
        or ledger["last_checkpoint_name"] != checkpoint.name
        or ledger["last_checkpoint_sha256"] != digest
    ):
        raise ValueError(
            "Finalization requires the current complete final checkpoint and population"
        )
    report = {
        "status": "recovered-final-report",
        "reporting_revision": revision,
        "checkpoint_revision": checkpoint_revision,
        "succession_receipt_sha256": receipt["_sha256"] if receipt else None,
        "completed_updates": count,
        "attempted_updates": count,
        "completed_updates_this_attempt": 0,
        "model_seconds": ledger["model_seconds"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "config_hash": config.digest(),
        "source": frames.identity,
        "peak_gpu_memory_bytes_this_attempt": None,
        "parameters_changed": None,
        "training_started_at": None,
        "training_finished_at": None,
        "source_frame_count": len(frames),
        "gpu_uuid": identity["gpu_uuid"],
        "failed_attempt_log": (
            receipt["runs"][FINALIZE_RUN]["failed_attempt_log"] if receipt else None
        ),
        "caveat": "Reporting repair only; historical failed exit and unavailable telemetry are preserved.",
    }
    path = (
        directory
        / f"report-finalization-{receipt['_sha256'] if receipt else digest}.json"
    )
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError("Existing finalization report differs")
    else:
        write_json_new(path, report)
    return path
