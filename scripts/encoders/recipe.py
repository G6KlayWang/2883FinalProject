"""Bind the executable recipe to the actual user-approved reviewed recommendation."""

import json
from pathlib import Path
from scripts.data.file_io import sha256_file
from scripts.encoders.config import RunConfig, load_config


def _without_statuses(value: dict) -> dict:
    result = json.loads(json.dumps(value))
    result.pop("recipe_status", None)
    for key in ("grids", "gate", "c_transform", "probe"):
        result[key].pop("status", None)
    return result


def load_approved_recipe(config_path: Path, approval_path: Path) -> RunConfig:
    reference = Path(__file__).parent / "configs/production-proposed.json"
    original = load_config(reference)
    approval = json.loads(approval_path.read_text())
    if (
        approval.get("decision") != "Use the recommendation recipe"
        or approval.get("reference_raw_sha256") != sha256_file(reference)
        or approval.get("reference_resolved_sha256") != original.digest()
    ):
        raise ValueError("Recipe approval does not bind the reviewed recommendation")
    config = load_config(config_path)
    config.require_execution_approval()
    if _without_statuses(config.to_dict()) != _without_statuses(original.to_dict()):
        raise ValueError(
            "Executable recipe values differ from the approved recommendation"
        )
    return config
