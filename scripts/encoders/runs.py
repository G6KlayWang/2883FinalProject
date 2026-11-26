"""Bind production execution to a declared candidate or selection-generated repeat."""

from dataclasses import replace
from scripts.encoders.config import RunConfig, run_matrix, shared_training_controls


def resolve_run(
    config: RunConfig, *, candidate: str | None, selection_report: dict | None
) -> tuple[RunConfig, dict]:
    config.validate()
    config.require_execution_approval()
    if config.profile == "smoke":
        if candidate is not None or selection_report is not None:
            raise ValueError(
                "Diagnostic smoke cannot declare a production candidate/repeat"
            )
        return config, {"run_id": "diagnostic-armA-seed0", "stage": "diagnostic"}
    if candidate is not None:
        if selection_report is not None or config.seed != 0:
            raise ValueError(
                "First-seed candidate cannot use a repeat selection receipt"
            )
        rows = [
            r
            for r in run_matrix(config.grids, config.native_lambda)
            if r["stage"] == "candidate" and r["run_id"] == candidate
        ]
        if len(rows) != 1:
            raise ValueError("Unknown candidate: use a declared first-seed matrix ID")
        row = rows[0]
        resolved = replace(config, arm=row["arm"], rho=row["rho"], seed=0)
        resolved.validate()
        return resolved, {"run_id": candidate, "stage": "candidate"}
    if config.seed == 0:
        raise ValueError("Production seed0 requires an explicit --candidate matrix ID")
    if (
        not isinstance(selection_report, dict)
        or selection_report.get("status") != "approved-policy"
    ):
        raise ValueError(
            "Production repeat requires an approved --selection-report receipt"
        )
    selected = selection_report.get("selected", {}).get(config.arm)
    if not isinstance(selected, dict):
        raise ValueError("Selection report lacks the selected repeat arm")
    expected = {
        "arm": config.arm,
        "rho": config.rho,
        "seed": 0,
        "checkpoint_role": "final",
        "score_split": "validation",
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Repeat does not match the selected final first-seed candidate"
        )
    controls = shared_training_controls(replace(config, seed=0))
    if selected.get("protocol", {}).get("shared_training_controls") != controls:
        raise ValueError("Repeat controls differ from the selected candidate protocol")
    hashes = selection_report.get("repeat_config_hashes", {})
    if hashes.get(config.arm) != config.digest():
        raise ValueError(
            "Repeat config hash is not bound to this selection-generated receipt"
        )
    return config, {"run_id": f"{config.arm}-selected-seed1", "stage": "repeat"}
