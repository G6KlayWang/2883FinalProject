"""Train-fitted C-only symmetric ZCA, persisted as part of the effective encoder."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.artifacts import new_directory
from scripts.encoders.measure import feature_matrix


@dataclass
class ZcaTransform:
    mean: np.ndarray
    matrix: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        values = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
        if values.shape[-1] != len(self.mean):
            raise ValueError("C-transform feature dimension mismatch")
        return (
            (values - torch.from_numpy(self.mean)) @ torch.from_numpy(self.matrix)
        ).numpy()

    def save(self, output: Path, identity: dict) -> None:
        with new_directory(output) as temporary:
            np.savez(temporary / "transform.npz", mean=self.mean, matrix=self.matrix)
            write_json_new(
                temporary / "manifest.json",
                {
                    "version": 1,
                    "fit_split": "train",
                    "dtype": "float32",
                    "identity": identity,
                    "transform_sha256": sha256_file(temporary / "transform.npz"),
                },
            )

    @classmethod
    def load(cls, path: Path, identity: dict) -> "ZcaTransform":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("fit_split") != "train" or manifest.get("identity") != identity:
            raise ValueError("C-transform training/checkpoint identity mismatch")
        if sha256_file(path / "transform.npz") != manifest["transform_sha256"]:
            raise ValueError("C-transform hash mismatch")
        with np.load(path / "transform.npz", allow_pickle=False) as values:
            mean, matrix = values["mean"].copy(), values["matrix"].copy()
        if mean.ndim != 1 or matrix.shape != (len(mean), len(mean)):
            raise ValueError("Invalid C-transform shape")
        if (
            mean.dtype != np.float32
            or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
            or not np.isfinite(mean).all()
        ):
            raise ValueError("C-transform must contain finite float32 values")
        return cls(mean, matrix)


def fit_zca(
    features: np.ndarray, *, split: str, eigenvalue_floor: float
) -> ZcaTransform:
    if split != "train":
        raise ValueError("C-transform must be fitted on training features only")
    values = feature_matrix(features)
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if eigenvalues.min() < eigenvalue_floor:
        raise ValueError(
            "C-transform raw covariance eigenvalue below eligibility floor"
        )
    matrix = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
    return ZcaTransform(mean.float().numpy(), matrix.float().numpy())


class EffectiveEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, transform: ZcaTransform):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("feature_mean", torch.from_numpy(transform.mean.copy()))
        self.register_buffer("zca_matrix", torch.from_numpy(transform.matrix.copy()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (self.encoder(images) - self.feature_mean) @ self.zca_matrix
