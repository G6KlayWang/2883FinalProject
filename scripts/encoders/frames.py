"""Explicit encoder populations and frame-uniform, full-batch epoch order."""

import hashlib
from itertools import accumulate
import numpy as np
import PIL
import torch
from scripts.data.loader import PreparedData, RawFrameDataset
from scripts.data.splits import validate_manifest
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.rng import keyed_seed

PREPROCESSING = {
    "resize": "full-image-Pillow-bilinear-RGB96-v1",
    "pillow": PIL.__version__,
    "storage": "uint8-HWC",
    "model_input": "float32-CHW-div255",
}


def population_ids(
    manifest: dict, split: str, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return episode/local-frame IDs and matching indices into RawFrameDataset."""
    validate_manifest(manifest)
    if manifest["split_unit"] != "family":
        raise ValueError("Encoder populations require whole-family splits")
    if split not in ("train", "validation", "test") or policy not in ("phase0", "raw"):
        raise ValueError("Unknown split or frame policy")
    stride = 5 if policy == "phase0" else 1
    ids, raw_indices, cursor = [], [], 0
    for episode_id in manifest["splits"][split]:
        length = manifest["episodes"][episode_id]["length"]
        local = np.arange(0, length, stride, dtype=np.int64)
        ids.append(
            np.column_stack((np.full(len(local), episode_id, dtype=np.int64), local))
        )
        raw_indices.append(cursor + local)
        cursor += length
    if not ids:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(ids), np.concatenate(raw_indices)


class ImageFrames:
    """Image-only access through the reviewed resize/loader, with IDs kept separately."""

    def __init__(
        self,
        prepared: PreparedData,
        split: str,
        policy: str,
        *,
        diagnostic_indices: list[int] | None = None
    ):
        if split == "test":
            raise ValueError("Encoder development does not read test rows")
        self.ids, self.raw_indices = population_ids(prepared.manifest, split, policy)
        if diagnostic_indices is not None:
            if split != "train" or policy != "phase0":
                raise ValueError("Diagnostic frame subset must be train phase0")
            if len(diagnostic_indices) != len(set(diagnostic_indices)):
                raise ValueError("Diagnostic frames must be distinct")
            self.ids, self.raw_indices = (
                self.ids[diagnostic_indices],
                self.raw_indices[diagnostic_indices],
            )
        self.raw = RawFrameDataset(prepared, split)
        self.identity = {
            "source": prepared.manifest["source"],
            "manifest_digest": identity_digest(prepared.manifest),
            "split": split,
            "frame_policy": policy,
            "preprocessing": PREPROCESSING,
            "row_digest": hashlib.sha256(self.ids.astype("<i8").tobytes()).hexdigest(),
            "diagnostic_subset": diagnostic_indices is not None,
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.raw[int(self.raw_indices[index])]["image"]


class BatchOrder:
    """Fresh seeded permutation per epoch; incomplete tail dropped, no duplicates."""

    def __init__(self, size: int, batch_size: int, seed: int):
        if size < batch_size or batch_size < 2:
            raise ValueError(
                "Population must contain a complete batch of at least two distinct frames"
            )
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.batches_per_epoch = size // batch_size
        self._epoch, self._permutation = None, None

    def indices(self, completed_updates: int) -> list[int]:
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("Completed updates must be a nonnegative integer")
        epoch, batch = divmod(completed_updates, self.batches_per_epoch)
        if epoch != self._epoch:
            generator = torch.Generator().manual_seed(
                keyed_seed(self.seed, "data", epoch)
            )
            self._permutation = torch.randperm(self.size, generator=generator).tolist()
            self._epoch = epoch
        start = batch * self.batch_size
        return self._permutation[start : start + self.batch_size]


def smoke_indices(ids: np.ndarray, manifest: dict, limit: int, seed: int) -> list[int]:
    """Diagnostic-only round-robin over training families, with seeded row ranks."""
    if not 16 <= limit <= 256 or limit > len(ids):
        raise ValueError("Smoke requires16..256 available distinct frames")
    family_for_episode = {
        i: f["family_id"] for f in manifest["families"] for i in f["episode_ids"]
    }
    grouped = {}
    training_ids = set(manifest["splits"]["train"])
    for index, (episode_id, frame) in enumerate(ids.tolist()):
        if episode_id not in training_ids:
            raise ValueError("Smoke cannot select nontraining frames")
        grouped.setdefault(family_for_episode[episode_id], []).append(
            (index, episode_id, frame)
        )
    families = sorted(grouped, key=lambda f: keyed_seed(seed, "smoke-family", f))
    rows = [
        sorted(
            grouped[f], key=lambda row: keyed_seed(seed, "smoke-row", row[1], row[2])
        )
        for f in families
    ]
    selected, position = [], 0
    while len(selected) < limit:
        for family in rows:
            if position < len(family):
                selected.append(family[position][0])
                if len(selected) == limit:
                    break
        position += 1
    return selected


class KeyedBatches:
    def __init__(self, order: BatchOrder, start: int, stop: int):
        if start < 0 or stop < start:
            raise ValueError("Invalid update range")
        self.order, self.start, self.stop = order, start, stop

    def __iter__(self):
        for update in range(self.start, self.stop):
            yield [(update, index) for index in self.order.indices(update)]

    def __len__(self) -> int:
        return self.stop - self.start


class SubsetImages:
    """Explicit diagnostic-only subset; base images can remain shared mmap data."""

    def __init__(self, frames, indices: list[int]):
        if len(indices) != len(set(indices)) or any(
            i < 0 or i >= len(frames) for i in indices
        ):
            raise ValueError(
                "Subset indices must be distinct and within the population"
            )
        self.frames, self.indices = frames, indices
        self.ids = frames.ids[indices]
        self.identity = {
            **frames.identity,
            "diagnostic_subset": True,
            "row_digest": hashlib.sha256(self.ids.astype("<i8").tobytes()).hexdigest(),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.frames[self.indices[index]]
