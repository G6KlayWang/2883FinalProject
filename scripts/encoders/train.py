"""Single-GPU training/resume entrypoint; smoke and proposed production stay separate."""

import argparse
import fcntl
import json
import logging
from pathlib import Path
import time
import torch
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.checkpoints import parameter_digest, replace_json
from scripts.encoders.config import load_config
from scripts.encoders.runs import resolve_run
from scripts.encoders.environment import environment_record
from scripts.encoders.inputs import open_images
from scripts.encoders.model import build_encoder
from scripts.encoders.runtime import (
    configure_process,
    require_single_gpu,
    runtime_record,
)
from scripts.encoders.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument(
        "--candidate", help="Declared first-seed run_id from the fixed run matrix"
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Approved selection receipt for a generated seed1 config",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    selection_report = (
        json.loads(args.selection_report.read_text()) if args.selection_report else None
    )
    config, declaration = resolve_run(
        config, candidate=args.candidate, selection_report=selection_report
    )
    stop_at = config.stop_update(args.stop_at)
    config.require_execution_approval()
    if config.profile == "smoke" and not args.output.name.startswith("diagnostic-"):
        raise ValueError("Smoke output directory must start with diagnostic-")
    if config.profile == "production" and args.cache is None:
        raise ValueError("Production requires the reviewed shared uint8 image cache")
    configure_process(config)
    uuid = require_single_gpu()  # Coordinator owns the physical device lease.
    preparation_started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / ("." + args.output.name + ".training.lock")).open(
        "ab"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared, frames = open_images(
            args.hdf5, args.manifest, config, "train", args.cache
        )
        identity = {
            "code_revision": args.code_revision,
            "run_declaration": declaration,
            "runtime_controls": runtime_record(),
            "selection_report_sha256": (
                sha256_file(args.selection_report) if args.selection_report else None
            ),
            "manifest_sha256": sha256_file(args.manifest),
            "images": frames.identity,
            "environment": environment_record(),
            "gpu_uuid": uuid,
        }
        model = build_encoder(config.seed, config.projector_hidden)
        torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(
            model, config, args.output, identity, "cuda:0", resume=args.resume
        )
        preparation_seconds = time.monotonic() - preparation_started
        if config.profile == "smoke":
            selected_ids = frames.ids.tolist()
            selection_path = args.output / "diagnostic-frame-ids.json"
            if selection_path.exists():
                if json.loads(selection_path.read_text()) != selected_ids:
                    raise ValueError("Resume diagnostic frame IDs changed")
            else:
                write_json_new(selection_path, selected_ids)
        attempt_start = trainer.completed_updates
        write_json_new(
            args.output / f"attempt-{attempt_start:06d}-started.json",
            {
                "completed_before_attempt": attempt_start,
                "stop_at": stop_at,
                "config_hash": config.digest(),
                "status": "started-not-yet-passed",
            },
        )
        checkpoint = trainer.run(frames, stop_at=stop_at)
        report = {
            "status": "diagnostic" if config.profile == "smoke" else "candidate",
            "attempt_start_update": attempt_start,
            "completed_updates_this_attempt": trainer.completed_updates - attempt_start,
            "completed_updates": trainer.completed_updates,
            "attempted_updates": trainer.attempted_updates,
            "model_seconds": trainer.model_seconds,
            "preparation_seconds_this_attempt": preparation_seconds,
            "peak_gpu_memory_bytes_this_attempt": torch.cuda.max_memory_allocated(),
            "gpu_uuid": uuid,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config_hash": config.digest(),
            "source": frames.identity,
            "source_frame_count": len(frames),
            "resumed_from_sha256": sha256_file(args.resume) if args.resume else None,
            "restored_update_count": attempt_start,
            "restored_state_sections": (
                [
                    "model/buffers",
                    "optimizer",
                    "scheduler",
                    "sampler",
                    "named-RNGs",
                    "global-RNGs",
                ]
                if args.resume
                else []
            ),
            "parameters_changed": parameter_digest(trainer.model)
            != trainer.initial_parameter_digest,
            "precision": {
                "dtype": "float32",
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "caveat": (
                "Diagnostic runtime/quality does not establish batch512 feasibility or select rho."
                if config.profile == "smoke"
                else "Final checkpoint only; validation selection is separate."
            ),
        }
        write_json_new(args.output / f"attempt-{attempt_start:06d}-report.json", report)
        replace_json(args.output / "training_report.json", report)
        logging.info("TRAINING REPORT %s", report)


if __name__ == "__main__":
    main()
