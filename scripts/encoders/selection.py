"""Validation-only, eligibility-aware selection; no test/planning-driven fallback."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scripts.data.file_io import write_json_new
from scripts.encoders.config import load_config, shared_training_controls
from scripts.data.families import require_fields
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.probes import ProbePolicy


def select_weight(records: list[dict], arm: str, policy: dict) -> dict:
    settings = ProbePolicy.from_dict(policy)
    if arm not in ("A", "B", "C"):
        raise ValueError("Select weights only for A/B/C; A-prime is a fixed reference")
    candidates = [record for record in records if record.get("arm") == arm]
    for record in candidates:
        require_fields(
            record,
            {
                "run_id",
                "rho",
                "seed",
                "score",
                "eligible",
                "checkpoint_role",
                "score_split",
                "protocol_digest",
                "checkpoint_sha256",
                "gate_policy_status",
            },
            "candidate",
        )
        if record["checkpoint_role"] != "final":
            raise ValueError(
                "Only final candidates, never intermediate/diagnostic smoke artifacts, enter selection"
            )
        if record["score_split"] != "validation" or record["seed"] != 0:
            raise ValueError(
                "Weight selection requires first-seed validation scores, never test/repeat outcomes"
            )
        if (
            not math.isfinite(record["score"])
            or record["score"] < 0
            or record["rho"] <= 0
        ):
            raise ValueError(
                "Candidate score/rho must be finite and nonnegative/positive"
            )
        if type(record["eligible"]) is not bool:
            raise ValueError("Candidate eligibility must be an explicit boolean")
    if len({r["protocol_digest"] for r in candidates}) > 1:
        raise ValueError(
            "Candidates do not share source/frame/probe/selection protocol"
        )
    if len({r["run_id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate run IDs")
    eligible = [record for record in candidates if record["eligible"]]
    if not eligible:
        raise ValueError(
            f"No eligible candidate for arm{arm}; stop without relaxing policy"
        )
    best = min(record["score"] for record in eligible)
    denominator = max(abs(best), settings.tie_denominator_floor)
    ties = [
        record
        for record in eligible
        if record["score"] - best <= settings.tie_fraction * denominator
    ]
    anchor = 1.0 if arm == "C" else 10.0
    chosen = min(
        ties, key=lambda r: (abs(math.log2(r["rho"] / anchor)), r["rho"], r["run_id"])
    )
    approved = (
        settings.status == "approved" and chosen["gate_policy_status"] == "approved"
    )
    return {
        **chosen,
        "selection_status": "approved-policy" if approved else "provisional-policy",
        "tie_rule": {
            "fraction": settings.tie_fraction,
            "denominator": "max(abs(best),floor)",
            "floor": settings.tie_denominator_floor,
            "anchor": anchor,
            "distance": "absolute-log2-ratio",
            "residual": "smaller-rho-then-run-id",
        },
    }


def aprime_reference_summary(records: list[dict]) -> dict:
    a = [r for r in records if r["arm"] == "A"]
    reference = [r for r in records if r["arm"] == "Aprime"]
    if len(a) != 3 or len(reference) != 1:
        raise ValueError(
            "A-prime check requires all three A weights and one fixed reference"
        )

    def summaries(record):
        stats = record["gate"]["raw_statistics"]
        coordinate = [v for v in stats["coordinate_excess_kurtosis"] if v is not None]
        projection = [v for v in stats["projection_excess_kurtosis"] if v is not None]
        return {
            "max_abs_mean": max(abs(v) for v in stats["mean"]),
            "covariance_min": min(stats["covariance_eigenvalues"]),
            "covariance_max": max(stats["covariance_eigenvalues"]),
            "median_coordinate_kurtosis": (
                float(np.median(coordinate)) if coordinate else None
            ),
            "median_projection_kurtosis": (
                float(np.median(projection)) if projection else None
            ),
            "mlp_pose_score": record["score"],
        }

    observed = summaries(reference[0])
    baseline = [summaries(r) for r in a]
    report = {}
    for name, value in observed.items():
        values = [r[name] for r in baseline if r[name] is not None]
        interval = [min(values), max(values)] if values else None
        report[name] = {
            "Aprime": value,
            "A_all_weight_range": interval,
            "within_range": (
                interval[0] <= value <= interval[1]
                if interval is not None and value is not None
                else None
            ),
        }
    return report


def select_all(records: list[dict], config) -> dict:
    if config.profile != "production":
        raise ValueError("Diagnostic profiles cannot select production candidates")
    if len(records) != 10:
        raise ValueError("Selection requires all ten predeclared first-seed candidates")
    for record in records:
        require_fields(
            record,
            {
                "arm",
                "seed",
                "checkpoint_role",
                "score_split",
                "protocol",
                "protocol_digest",
            },
            "candidate",
        )
        if record["checkpoint_role"] != "final" or record["seed"] != 0:
            raise ValueError(
                "Every selection/reference record must be a final first-seed candidate, never diagnostic"
            )
        if record["score_split"] != "validation":
            raise ValueError(
                "Every candidate score must come from validation, never test"
            )
        protocol = record["protocol"]
        if identity_digest(protocol) != record["protocol_digest"]:
            raise ValueError("Candidate protocol digest mismatch")
        if protocol.get("shared_training_controls") != shared_training_controls(config):
            raise ValueError(
                "Candidate shared training controls differ from the frozen selection protocol"
            )
        if (
            protocol.get("probe_policy") != config.probe
            or protocol.get("gate_policy") != config.gate
        ):
            raise ValueError(
                "Candidate probe/gate protocol differs from selection configuration"
            )
    if len({r["protocol_digest"] for r in records}) != 1:
        raise ValueError(
            "All ten candidates must share one source/frame/training/probe protocol"
        )
    for arm in ("A", "B", "C"):
        values = [r["rho"] for r in records if r["arm"] == arm]
        if sorted(values) != config.grids[arm]:
            raise ValueError(
                f"Arm{arm} does not match its complete declared three-weight grid"
            )
    selected = {
        arm: select_weight(records, arm, config.probe) for arm in ("A", "B", "C")
    }
    a_kurt = selected["A"]["gate"]["median_coordinate_kurtosis"]
    b_kurt = selected["B"]["gate"]["median_coordinate_kurtosis"]
    heavier = a_kurt is not None and b_kurt is not None and b_kurt > a_kurt
    approved = config.recipe_status == "approved" and all(
        r["selection_status"] == "approved-policy" for r in selected.values()
    )
    return {
        "selected": selected,
        "Aprime_vs_all_A": aprime_reference_summary(records),
        "B_heavier_than_A": bool(heavier),
        "ready_for_separate_DF_decision": bool(heavier and approved),
        "status": "approved-policy" if approved else "provisional-policy",
        "scope": "No DF launch; no grid extension; test/planning outcomes excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-config-directory", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    records = [json.loads(path.read_text()) for path in args.records]
    report = select_all(records, config)
    if args.repeat_config_directory:
        if report["status"] != "approved-policy":
            raise ValueError(
                "Provisional recommendations cannot materialize production repeat configs"
            )
        from dataclasses import replace
        from scripts.encoders.artifacts import new_directory

        with new_directory(args.repeat_config_directory) as temporary:
            for arm, record in report["selected"].items():
                repeat = replace(config, arm=arm, rho=record["rho"], seed=1)
                repeat.validate()
                repeat.require_execution_approval()
                write_json_new(
                    temporary / f"{arm}-selected-seed1.json", repeat.to_dict()
                )
    write_json_new(args.output, report)


if __name__ == "__main__":
    main()
