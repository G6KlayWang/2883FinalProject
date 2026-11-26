"""Reviewed-checkpoint extraction entrypoint; diagnostic forwards share the600s ledger."""

import argparse
import fcntl
import json
import hashlib
from pathlib import Path
import time
import torch
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.checkpoints import load_checkpoint, replace_json
from scripts.encoders.config import RunConfig
from scripts.encoders.extract import (
    FeatureCache,
    extract_features,
    train_mode_comparison,
)
from scripts.encoders.inputs import open_images
from scripts.encoders.frames import population_ids
from scripts.encoders.model import build_encoder
from scripts.encoders.runtime import (
    configure_process,
    model_time_limit,
    require_single_gpu,
)
from scripts.encoders.transforms import EffectiveEncoder, ZcaTransform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    state = load_checkpoint(args.checkpoint)
    config = RunConfig.from_dict(state["config"])
    role = state["checkpoint_role"]
    if role == "diagnostic" and not args.diagnostic:
        raise ValueError(
            "Smoke checkpoint requires explicit --diagnostic and cannot become a candidate"
        )
    if args.diagnostic and (role != "diagnostic" or args.split != "train"):
        raise ValueError("Diagnostic extraction is smoke/train-only")
    if (
        role not in ("final", "diagnostic")
        or state["completed_updates"] != config.optimizer.max_updates
    ):
        raise ValueError(
            "Extract only a completed final candidate or completed bounded diagnostic"
        )
    run_identity = state["identity"]["run_identity"]
    if args.code_revision != run_identity["code_revision"]:
        raise ValueError(
            "Extraction code revision differs from the training checkpoint"
        )
    if sha256_file(args.manifest) != run_identity["manifest_sha256"]:
        raise ValueError("Extraction manifest differs from checkpoint")
    configure_process(config)
    uuid = require_single_gpu()
    prepared, frames = open_images(
        args.hdf5, args.manifest, config, args.split, args.cache
    )
    if args.split == "train" and frames.identity != state["data_identity"]:
        raise ValueError("Extraction training rows differ from checkpoint population")
    model = build_encoder(config.seed, config.projector_hidden)
    model.load_state_dict(state["model"])
    checkpoint_sha = sha256_file(args.checkpoint)
    identity = {
        "checkpoint_sha256": checkpoint_sha,
        "arm": config.arm,
        "checkpoint_role": role,
        "config_hash": config.digest(),
        "code_revision": args.code_revision,
        "effective_transform_sha256": None,
    }
    if args.transform:
        if config.arm != "C" or role != "final":
            raise ValueError(
                "Only final C checkpoints permit a train-fitted ZCA transform"
            )
        manifest = json.loads((args.transform / "manifest.json").read_text())
        training_ids, _ = population_ids(
            prepared.manifest, "train", config.frame_policy
        )
        expected = {
            "checkpoint_sha256": checkpoint_sha,
            "source": prepared.manifest["source"],
            "frame_policy": config.frame_policy,
            "training_row_digest": hashlib.sha256(
                training_ids.astype("<i8").tobytes()
            ).hexdigest(),
        }
        if manifest["identity"] != expected:
            raise ValueError(
                "C-transform was fitted on a different source/training population"
            )
        if (
            expected["checkpoint_sha256"] != checkpoint_sha
            or expected["source"] != frames.identity["source"]
        ):
            raise ValueError("C-transform checkpoint/source mismatch")
        if expected["frame_policy"] != config.frame_policy:
            raise ValueError("C-transform frame-policy mismatch")
        model = EffectiveEncoder(model, ZcaTransform.load(args.transform, expected))
        identity["effective_transform_sha256"] = sha256_file(
            args.transform / "manifest.json"
        )
    run_dir = args.checkpoint.parent
    lock_path = run_dir.parent / ("." + run_dir.name + ".training.lock")
    with lock_path.open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads((run_dir / "run_state.json").read_text())
        if (
            ledger["completed_updates"] != state["completed_updates"]
            or ledger["attempted_updates"] != state["completed_updates"]
        ):
            raise ValueError("Checkpoint is stale or a model update is unresolved")
        if ledger.get("last_checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Checkpoint file hash differs from run ledger")
        initial = ledger["model_seconds"]
        started = time.monotonic()
        seconds = (
            config.runtime.max_seconds - initial
            if args.diagnostic
            else config.runtime.max_seconds
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            with model_time_limit(seconds):
                extract_features(
                    model,
                    frames,
                    args.output,
                    identity,
                    device="cuda:0",
                    batch_size=config.optimizer.batch_size,
                )
                comparison = train_mode_comparison(
                    model,
                    frames,
                    batch_size=config.optimizer.batch_size,
                    device="cuda:0",
                )
        finally:
            if args.diagnostic:
                ledger["model_seconds"] = initial + time.monotonic() - started
                replace_json(run_dir / "run_state.json", ledger)
        cache = FeatureCache(args.output)
        write_json_new(
            args.output / "extraction_report.json",
            {
                "status": (
                    "diagnostic-passed" if args.diagnostic else "clean-final-features"
                ),
                "shape": list(cache.features.shape),
                "finite": True,
                "gpu_uuid": uuid,
                "feature_cache_sha256": cache.digest,
                "checkpoint_sha256": checkpoint_sha,
                "model_seconds_this_phase": time.monotonic() - started,
                "cumulative_smoke_model_seconds": (
                    ledger["model_seconds"] if args.diagnostic else None
                ),
                "peak_gpu_memory_bytes_this_phase": torch.cuda.max_memory_allocated(),
                "train_eval_comparison": comparison,
                "candidate_eligible": (
                    False if args.diagnostic else "requires-separate-gate-and-probes"
                ),
            },
        )


if __name__ == "__main__":
    main()
