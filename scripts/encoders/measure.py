"""Exhaustive FP64 moments/kurtosis plus fixed sampled characteristic-function checks."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.data.file_io import write_json_new


def feature_matrix(features: np.ndarray) -> torch.Tensor:
    if features.ndim != 2 or len(features) < 2 or features.shape[1] < 1:
        raise ValueError(
            "Measurements require at least two rows of finite feature vectors"
        )
    values = torch.from_numpy(np.array(features, dtype=np.float64, copy=True))
    if not torch.isfinite(values).all():
        raise ValueError("Feature measurements must be finite")
    return values


def excess_kurtosis(values: torch.Tensor) -> list[float | None]:
    centered = values - values.mean(0)
    variance = centered.square().mean(0)
    fourth = centered.pow(4).mean(0)
    return [
        float(f / v.square() - 3) if v > 0 else None for f, v in zip(fourth, variance)
    ]


def _epps_pulley(values: torch.Tensor) -> list[float]:
    knots = torch.linspace(0, 3, 17, dtype=torch.float64)
    phi = torch.exp(-knots.square() / 2)
    weights = torch.full_like(knots, 6 / 16)
    weights[[0, -1]] = 3 / 16
    phase = values.unsqueeze(-1) * knots
    error = (phase.cos().mean(0) - phi).square() + phase.sin().mean(0).square()
    return ((error @ (weights * phi)) * len(values)).tolist()


def feature_statistics(
    features: np.ndarray,
    *,
    projection_count: int = 128,
    ep_sample_size: int = 4096,
    seed: int = 20261126
) -> dict:
    values = feature_matrix(features)
    if projection_count < 1 or ep_sample_size < 2:
        raise ValueError(
            "Positive projection count and at least two EP samples required"
        )
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(
        values.shape[1], projection_count, generator=generator, dtype=torch.float64
    )
    directions /= directions.norm(dim=0)
    centered = values - values.mean(0)
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    projected = values @ directions
    sampled = torch.randperm(len(values), generator=generator)[:ep_sample_size]
    return {
        "count": len(values),
        "dimension": values.shape[1],
        "mean": values.mean(0).tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "coordinate_excess_kurtosis": excess_kurtosis(values),
        "projection_excess_kurtosis": excess_kurtosis(projected),
        "moment_scope": "exhaustive",
        "projection_moment_scope": "exhaustive",
        "covariance_estimator": "unbiased-N-1",
        "kurtosis_estimator": "population-m4/m2^2-minus3;null-if-constant",
        "projection_count": projection_count,
        "seed": seed,
        "epps_pulley_rows": len(sampled),
        "epps_pulley_scope": "sampled" if len(sampled) < len(values) else "exhaustive",
        "epps_pulley_coordinate": _epps_pulley(values[sampled]),
        "epps_pulley_projection": _epps_pulley(projected[sampled]),
        "ep_row_index_sha256": hashlib.sha256(
            sampled.numpy().astype("<i8").tobytes()
        ).hexdigest(),
        "ep_protocol": {
            "range": [0, 3],
            "knots": 17,
            "symmetric_weights": True,
            "batch_multiplier": True,
            "dtype": "float64",
        },
    }


def main() -> None:
    from scripts.encoders.extract import FeatureCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--ep-samples", type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache = FeatureCache(args.features)
    if cache.identity["frame_identity"]["split"] not in ("train", "validation"):
        raise ValueError("Measurement development cannot use test features")
    write_json_new(
        args.output,
        {
            "feature_identity": cache.identity,
            "statistics": feature_statistics(
                cache.features,
                projection_count=args.projections,
                ep_sample_size=args.ep_samples,
            ),
        },
    )


if __name__ == "__main__":
    main()
