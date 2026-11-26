"""Parent-side source verification and explicit image/cache population assembly."""

import json
from pathlib import Path
from scripts.data.loader import PreparedData
from scripts.data.families import PUSHT_HDF5_SHA256
from scripts.encoders.config import RunConfig
from scripts.encoders.frames import ImageFrames, SubsetImages, smoke_indices
from scripts.encoders.image_cache import CachedImages


def open_images(
    hdf5: Path, manifest: Path, config: RunConfig, split: str, cache: Path | None = None
):
    prepared = PreparedData(hdf5, json.loads(Path(manifest).read_text()))
    if prepared.manifest["source"]["hdf5_sha256"] != PUSHT_HDF5_SHA256:
        raise ValueError("Encoder CLI requires the verified DATA-001 PushT source")
    frames = ImageFrames(prepared, split, config.frame_policy)
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
