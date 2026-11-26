"""Serialized shared-cache preparation; trainer startup retains full verification."""

import fcntl
import hashlib
import json
from pathlib import Path
import time
from typing import TextIO
import numpy as np
from scripts.data.file_io import sha256_file
from scripts.data.loader import PreparedData
from scripts.data.families import PUSHT_HDF5_SHA256
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.config import RunConfig
from scripts.encoders.frames import ImageFrames, PREPROCESSING, population_ids
from scripts.encoders.image_cache import build_image_cache
from scripts.encoders.inputs import validate_population_count
from scripts.encoders.progress import ProgressDisplay


def expected_identity(manifest: dict, split: str, policy: str) -> dict:
    ids, _ = population_ids(manifest, split, policy)
    return {
        "source": manifest["source"],
        "manifest_digest": identity_digest(manifest),
        "split": split,
        "frame_policy": policy,
        "preprocessing": PREPROCESSING,
        "row_digest": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "diagnostic_subset": False,
    }


def prepare_caches(
    hdf5: Path,
    manifest_path: Path,
    root: Path,
    config: RunConfig,
    splits: tuple[str, ...],
    *,
    stream: TextIO | None = None,
) -> dict:
    if not splits or any(s not in ("train", "validation") for s in splits):
        raise ValueError("Prepare only declared train/validation caches")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"]["hdf5_sha256"] != PUSHT_HDF5_SHA256:
        raise ValueError("Production preparation requires pinned DATA-001 source")
    root.mkdir(parents=True, exist_ok=True)
    report = {"manifest_sha256": sha256_file(manifest_path), "splits": {}}
    with (root / ".preparation.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = None
        for split in splits:
            target = root / f"{split}-{config.frame_policy}"
            ids, _ = population_ids(manifest, split, config.frame_policy)
            validate_population_count(split, config.frame_policy, len(ids))
            identity = expected_identity(manifest, split, config.frame_policy)
            if target.exists():
                metadata = json.loads((target / "manifest.json").read_text())
                if metadata.get("identity") != identity or metadata.get("count") != len(
                    ids
                ):
                    raise ValueError(
                        f"Existing cache identity/count mismatch: {target}"
                    )
                if not all(
                    (target / name).is_file() for name in ("images.npy", "ids.npy")
                ):
                    raise ValueError(f"Existing cache is incomplete: {target}")
                status = "reused; full file verification remains mandatory at trainer/extraction startup"
            else:
                if prepared is None:
                    prepared = PreparedData(hdf5, manifest)
                frames = ImageFrames(prepared, split, config.frame_policy)
                display = ProgressDisplay(
                    f"Preparing {split} resized images", len(frames), stream=stream
                )
                started = time.monotonic()
                try:
                    build_image_cache(
                        frames,
                        target,
                        reserve_gib=config.runtime.disk_reserve_gib,
                        progress=lambda done, total: display.update(
                            {
                                "update": done,
                                "model_seconds": time.monotonic() - started,
                                "source_frames_per_second": done
                                / max(time.monotonic() - started, 1e-12),
                            }
                        ),
                    )
                finally:
                    display.close()
                status = "built; hashes recorded by reviewed cache writer"
            for name in ("images.npy", "ids.npy", "manifest.json"):
                (target / name).chmod(0o444)
            report["splits"][split] = {
                "path": str(target.resolve()),
                "status": status,
                "count": len(ids),
                "identity": identity,
            }
    return report
