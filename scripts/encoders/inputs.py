"""Parent-side source verification and explicit image/cache population assembly."""

import json
from pathlib import Path
from scripts.data.loader import PreparedData
from scripts.data.families import PUSHT_HDF5_SHA256
from scripts.encoders.config import RunConfig
from scripts.encoders.frames import ImageFrames, SubsetImages, smoke_indices
from scripts.encoders.image_cache import CachedImages


def validate_population_count(split: str, policy: str, count: int) -> None:
    """Assert known metadata counts after the CLI verifies the pinned PushT source."""
    expected = {
        ("train", "phase0"): 433896,
        ("validation", "phase0"): 21715,
        ("train", "raw"): 2136352,
        ("validation", "raw"): 107464,
    }
    if (split, policy) not in expected:
        raise ValueError("Encoder population must be train/validation and phase0/raw")
    if count != expected[(split, policy)]:
        raise ValueError(
            f"Pinned PushT population count mismatch for {split}/{policy}: expected {expected[(split, policy)]}, got {count}"
        )


def open_images(
    hdf5: Path, manifest: Path, config: RunConfig, split: str, cache: Path | None = None
):
    prepared = PreparedData(hdf5, json.loads(Path(manifest).read_text()))
    if prepared.manifest["source"]["hdf5_sha256"] != PUSHT_HDF5_SHA256:
        raise ValueError("Encoder CLI requires the verified DATA-001 PushT source")
    frames = ImageFrames(prepared, split, config.frame_policy)
    validate_population_count(split, config.frame_policy, len(frames))
    images = CachedImages(cache, frames.identity) if cache is not None else frames
    if config.profile == "smoke":
        if split != "train":
            raise ValueError("Diagnostic smoke must not access validation/test rows")
        indices = smoke_indices(
            frames.ids,
            prepared.manifest,
            min(config.runtime.max_frames, len(frames)),
            config.seed,
        )
        images = SubsetImages(images, indices)
    return prepared, images
