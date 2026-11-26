"""Atomic trusted-local checkpoints and the durable attempted-update ledger."""

import json
import hashlib
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from scripts.data.file_io import new_output


def replace_json(path: Path, payload: dict) -> None:
    path = Path(path)
    fd, name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".incomplete", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as target:
            json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, state: dict) -> None:
    with new_output(path) as temporary:
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict:
    # Artifacts are created by this reviewed pipeline, never fetched pickle files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("checkpoint_version") != 1:
        raise ValueError("Unsupported encoder checkpoint format")
    return state


def capture_rng(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode() + b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
