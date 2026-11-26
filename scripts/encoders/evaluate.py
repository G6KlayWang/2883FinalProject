"""Final-candidate gate, training-only transforms/probes and validation diagnostics."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
from scripts.data.file_io import sha256_file, write_json_new
from scripts.data.loader import MeasurementReader, PreparedData
from scripts.encoders.artifacts import identity_digest, new_directory
from scripts.encoders.checkpoints import load_checkpoint
from scripts.encoders.config import RunConfig, shared_training_controls
from scripts.encoders.feature_cache import FeatureCache
from scripts.encoders.frames import population_ids
from scripts.encoders.gate import assess_training_features
from scripts.encoders.goal_metrics import goal_correlations
from scripts.encoders.measure import feature_statistics
from scripts.encoders.pose import pose_targets, visibility_mask, group_scores
from scripts.encoders.probes import fit_linear_probe, fit_mlp_probe
from scripts.encoders.runtime import (
    check_resources,
    configure_process,
    require_single_gpu,
)


def _check_features(
    cache: FeatureCache, prepared: PreparedData, config: RunConfig, split: str
) -> None:
    cache.require_candidate(split)
    identity = cache.identity
    if identity["arm"] != config.arm or identity.get("config_hash") != config.digest():
        raise ValueError("Feature encoder/config identity mismatch")
    frame = identity["frame_identity"]
    if frame["source"] != prepared.manifest["source"] or frame[
        "manifest_digest"
    ] != identity_digest(prepared.manifest):
        raise ValueError("Feature source/family-manifest identity mismatch")
    if frame["frame_policy"] != config.frame_policy:
        raise ValueError("Feature frame policy differs from candidate recipe")
    expected, _ = population_ids(prepared.manifest, split, config.frame_policy)
    if not np.array_equal(cache.ids, expected):
        raise ValueError(
            "Candidate feature rows differ from complete declared split population"
        )
    if identity.get("effective_transform_sha256") is not None:
        raise ValueError(
            "Gate/probe evaluation requires raw features before any C-transform"
        )


def _labels(prepared: PreparedData, ids: np.ndarray, split: str) -> np.ndarray:
    reader = MeasurementReader(prepared, purpose="probe")
    states = np.empty((len(ids), 7), dtype=np.float32)
    boundaries = np.r_[0, np.flatnonzero(ids[1:, 0] != ids[:-1, 0]) + 1, len(ids)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        episode_id = int(ids[start, 0])
        labels = reader.read(episode_id, ids[start:end, 1].tolist())
        if labels["metadata"]["split"] != split:
            raise ValueError("Probe measurement rows crossed their declared split")
        states[start:end] = labels["state"]
    return states


def evaluate_candidate(
    config: RunConfig,
    train: FeatureCache,
    validation: FeatureCache,
    prepared: PreparedData,
    output: Path,
    *,
    run_id: str,
    device: str
) -> dict:
    if config.profile != "production":
        raise ValueError("Diagnostic smoke cannot be evaluated as an encoder candidate")
    config.require_execution_approval()
    _check_features(train, prepared, config, "train")
    _check_features(validation, prepared, config, "validation")
    if train.identity["checkpoint_sha256"] != validation.identity["checkpoint_sha256"]:
        raise ValueError("Train and validation features use different checkpoints")
    gate, transform = assess_training_features(
        train.features,
        config.arm,
        config.gate,
        c_policy=config.c_transform,
        split="train",
        checkpoint_role=train.identity["checkpoint_role"],
    )
    train_values = (
        transform.apply(train.features) if transform is not None else train.features
    )
    validation_values = (
        transform.apply(validation.features)
        if transform is not None
        else validation.features
    )
    train_state = _labels(prepared, train.ids, "train")
    validation_state = _labels(prepared, validation.ids, "validation")
    targets = pose_targets(train_state)
    truth = pose_targets(validation_state)
    train_visible = visibility_mask(
        train_state, config.probe["visibility_margin"], config.probe["visibility"]
    )
    validation_visible = visibility_mask(
        validation_state, config.probe["visibility_margin"], config.probe["visibility"]
    )
    protocol = {
        "source": prepared.manifest["source"],
        "manifest_digest": identity_digest(prepared.manifest),
        "frame_policy": config.frame_policy,
        "shared_training_controls": shared_training_controls(config),
        "train_rows": train.identity["row_digest"],
        "validation_rows": validation.identity["row_digest"],
        "probe_policy": config.probe,
        "gate_policy": config.gate,
        "c_transform_policy": config.c_transform,
        "selection": "equal-three-normalized-group-validation-MLP-final-epoch-v1",
    }
    with new_directory(output) as temporary:
        check_resources(temporary, config.runtime)
        transform_sha = None
        if transform is not None:
            transform.save(
                temporary / "C-transform",
                {
                    "checkpoint_sha256": train.identity["checkpoint_sha256"],
                    "source": prepared.manifest["source"],
                    "frame_policy": config.frame_policy,
                    "training_row_digest": train.identity["row_digest"],
                },
            )
            transform_sha = sha256_file(temporary / "C-transform" / "manifest.json")
        probe_identity = {
            "checkpoint_sha256": train.identity["checkpoint_sha256"],
            "effective_transform_sha256": transform_sha,
            "train_feature_cache_sha256": train.digest,
            "validation_feature_cache_sha256": validation.digest,
            "protocol_digest": identity_digest(protocol),
        }
        linear = fit_linear_probe(
            train_values, targets, train_visible, split="train", policy=config.probe
        )
        mlp = fit_mlp_probe(
            train_values,
            targets,
            train_visible,
            split="train",
            policy=config.probe,
            device=device,
        )
        linear.save(temporary / "linear-probe", probe_identity)
        mlp.save(temporary / "mlp-probe", probe_identity)
        linear_scores = group_scores(
            linear.predict(validation_values),
            truth,
            linear.scaling.target_variance,
            validation_visible,
        )
        mlp_scores = group_scores(
            mlp.predict(validation_values),
            truth,
            mlp.scaling.target_variance,
            validation_visible,
        )
        report = {
            "run_id": run_id,
            "arm": config.arm,
            "rho": config.rho,
            "seed": config.seed,
            "checkpoint_sha256": train.identity["checkpoint_sha256"],
            "checkpoint_role": "final",
            "config_hash": config.digest(),
            "score": mlp_scores["score"],
            "score_split": "validation",
            "eligible": gate["eligible"],
            "gate_policy_status": config.gate["status"],
            "c_transform_policy_status": config.c_transform["status"],
            "gate": gate,
            "protocol_digest": identity_digest(protocol),
            "protocol": protocol,
            "linear_scores": linear_scores,
            "mlp_scores": mlp_scores,
            "probe_train_curve": mlp.curve,
            "probe_checkpoint_policy": "fixed-final-epoch;no-validation-early-stopping",
            "train_target_mean": mlp.scaling.target_mean.tolist(),
            "train_target_variance": mlp.scaling.target_variance.tolist(),
            "effective_transform_sha256": transform_sha,
            "validation_statistics": feature_statistics(validation_values),
            "train_goal_metrics": goal_correlations(
                train_values, targets, train_visible, split="train"
            ),
            "validation_goal_metrics": goal_correlations(
                validation_values, truth, validation_visible, split="validation"
            ),
            "source_feature_caches": {
                "train": train.digest,
                "validation": validation.digest,
            },
            "note": "Linear and goal metrics are diagnostics, not selection scores; test rows unused",
        }
        write_json_new(temporary / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    state = load_checkpoint(args.checkpoint)
    config = RunConfig.from_dict(state["config"])
    if (
        state["checkpoint_role"] != "final"
        or state["completed_updates"] != config.optimizer.max_updates
    ):
        raise ValueError(
            "Only completed final production checkpoints can fit candidate probes"
        )
    config.require_execution_approval()
    configure_process(config)
    require_single_gpu()
    train, validation = FeatureCache(args.train_features), FeatureCache(
        args.validation_features
    )
    if train.identity["checkpoint_sha256"] != sha256_file(args.checkpoint):
        raise ValueError("Evaluation checkpoint differs from feature caches")
    if (
        sha256_file(args.manifest)
        != state["identity"]["run_identity"]["manifest_sha256"]
    ):
        raise ValueError("Probe label manifest differs from training")
    prepared = PreparedData(args.hdf5, json.loads(args.manifest.read_text()))
    result = evaluate_candidate(
        config,
        train,
        validation,
        prepared,
        args.output,
        run_id=args.run_id,
        device="cuda:0",
    )
    logging.info(
        "CANDIDATE %s eligible=%s validation_score=%s",
        args.run_id,
        result["eligible"],
        result["score"],
    )


if __name__ == "__main__":
    main()
