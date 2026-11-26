"""Verified frozen-feature artifacts with explicit checkpoint, transform and row identity."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.data.file_io import sha256_file


class FeatureCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "manifest.json").read_text())
        if self.metadata.get("version") != 1:
            raise ValueError("Unsupported feature-cache version")
        self.identity = self.metadata["identity"]
        if set(self.metadata.get("files", {})) != {"features.npy", "ids.npy"}:
            raise ValueError("Feature cache must bind features.npy and ids.npy hashes")
        for name, digest in self.metadata["files"].items():
            if sha256_file(self.path / name) != digest:
                raise ValueError(f"Feature cache hash mismatch: {name}")
        self.features = np.load(
            self.path / "features.npy", mmap_mode="r", allow_pickle=False
        )
        self.ids = np.load(self.path / "ids.npy", mmap_mode="r", allow_pickle=False)
        if (
            self.features.shape != (self.metadata["count"], 128)
            or self.features.dtype != np.float32
        ):
            raise ValueError(
                "Feature cache must contain float32(N,128) projector outputs"
            )
        if self.ids.shape != (len(self.features), 2) or self.ids.dtype != np.int64:
            raise ValueError("Feature cache must retain episode/frame IDs in order")
        digest = hashlib.sha256(self.ids.astype("<i8").tobytes()).hexdigest()
        if digest != self.identity["row_digest"]:
            raise ValueError("Feature cache row-order identity mismatch")
        if not np.isfinite(self.features).all():
            raise ValueError("Nonfinite cached features")

    @property
    def digest(self) -> str:
        return sha256_file(self.path / "manifest.json")

    def require_candidate(self, split: str) -> None:
        if (
            split not in ("train", "validation")
            or self.identity["frame_identity"]["split"] != split
        ):
            raise ValueError(
                "Candidate development requires the explicit train/validation split, never test"
            )
        if self.identity["checkpoint_role"] != "final" or self.identity[
            "frame_identity"
        ].get("diagnostic_subset"):
            raise ValueError(
                "Intermediate/diagnostic smoke features cannot be candidate artifacts"
            )
