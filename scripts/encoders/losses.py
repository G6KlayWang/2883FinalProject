"""Four regularizers, with explicit batch/view reductions and isolated draws.

SIGReg quadrature follows LeJEPA c293d291, univariate/epps_pulley.py:
17 knots on [0,3], symmetric trapezoidal weights times exp(-t²/2), statistic × B.
No distributed reduction is implemented: each run owns one full-batch GPU.
"""

import math
import torch
from scripts.encoders.rng import RandomStreams


def _validate_views(z1: torch.Tensor, z2: torch.Tensor) -> None:
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("Views must have identical (batch, feature) shapes")
    if z1.shape[0] < 2 or z1.shape[1] < 1:
        raise ValueError(
            "Regularizers require batch >= 2 and positive feature dimension"
        )


def draw_directions(
    dim: int, count: int, generator: torch.Generator, *, include_axes: bool = True
) -> torch.Tensor:
    if dim < 1 or count < 1:
        raise ValueError("Direction dimension and count must be positive")
    directions = torch.randn(dim, count, generator=generator)
    directions = directions / directions.norm(dim=0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (
        torch.cat((directions, torch.eye(dim)), dim=1) if include_axes else directions
    )


def sample_target(
    shape: tuple[int, int], arm: str, generator: torch.Generator
) -> torch.Tensor:
    if arm == "A":
        return torch.randn(shape, generator=generator)
    if arm == "B":
        # Product Laplace is sampled before projection. Guard finite-precision
        # uniform endpoints; torch.distributions.sample has no generator argument.
        uniform = torch.rand(shape, generator=generator).mul(2).sub(1)
        magnitude = uniform.abs().clamp_max(1 - torch.finfo(uniform.dtype).eps)
        return -math.sqrt(0.5) * uniform.sign() * torch.log1p(-magnitude)
    raise ValueError(f"Target sampler is defined only for A/B, got {arm!r}")


def sliced_wasserstein(
    z: torch.Tensor, target: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    _validate_views(z, target)
    projected = (z @ directions).sort(dim=0).values
    reference = (target @ directions).sort(dim=0).values
    return (projected - reference).square().mean()


def _vicreg_view(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = z - z.mean(dim=0)
    variance = torch.relu(1 - torch.sqrt(z.var(dim=0, unbiased=True) + 1e-4)).mean()
    covariance = centered.T @ centered / (len(z) - 1)
    mask = ~torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
    off_diagonal = covariance[mask].square().sum()
    return variance, off_diagonal / z.shape[1]


def vicreg(z1: torch.Tensor, z2: torch.Tensor) -> dict[str, torch.Tensor]:
    _validate_views(z1, z2)
    var1, cov1 = _vicreg_view(z1)
    var2, cov2 = _vicreg_view(z2)
    variance, covariance = (var1 + var2) / 2, cov1 + cov2
    return {
        "variance": variance,
        "covariance": covariance,
        "regularizer": variance + covariance / 25,
    }


def sigreg(
    z1: torch.Tensor, z2: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    _validate_views(z1, z2)
    knots = torch.linspace(0, 3, 17, dtype=z1.dtype, device=z1.device)
    phi = torch.exp(-knots.square() / 2)
    weights = torch.full_like(knots, 2 * 3 / 16)
    weights[[0, -1]] = 3 / 16
    weights = weights * phi
    # Shape (views,B,slices,knots); batch mean precedes view/slice mean.
    projected = torch.stack((z1 @ directions, z2 @ directions)).unsqueeze(-1) * knots
    discrepancy = (projected.cos().mean(dim=1) - phi).square() + projected.sin().mean(
        dim=1
    ).square()
    return ((discrepancy @ weights) * len(z1)).mean()


def encoder_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    arm: str,
    rho: float,
    streams: RandomStreams,
    aprime_convention: str,
    *,
    projections: int = 1024,
    native_lambda: float = 0.05,
) -> dict[str, torch.Tensor]:
    _validate_views(z1, z2)
    invariance = (z1 - z2).square().mean()
    terms = {"invariance": invariance}
    if arm in ("A", "B"):
        directions = draw_directions(
            z1.shape[1], projections, streams["directions"]
        ).to(z1)
        targets = [
            sample_target(tuple(z1.shape), arm, streams["targets"]).to(z1)
            for _ in range(2)
        ]
        regularizer = (
            sliced_wasserstein(z1, targets[0], directions)
            + sliced_wasserstein(z2, targets[1], directions)
        ) / 2
    elif arm == "C":
        control = vicreg(z1, z2)
        terms.update(variance=control["variance"], covariance=control["covariance"])
        regularizer = control["regularizer"]
    elif arm == "Aprime":
        if not 0 < native_lambda < 1:
            raise ValueError("A-prime native_lambda must lie in (0,1)")
        if not math.isclose(rho, 4 * native_lambda / (1 - native_lambda)):
            raise ValueError("A-prime rho must equal4*lambda/(1-lambda)")
        if aprime_convention not in ("common", "native"):
            raise ValueError("A-prime convention must be explicit common/native")
        directions = draw_directions(
            z1.shape[1], projections, streams["directions"], include_axes=False
        ).to(z1)
        regularizer = sigreg(z1, z2, directions)
    else:
        raise ValueError(f"Unknown encoder arm {arm!r}")
    total = invariance + rho * regularizer
    if arm == "Aprime":
        if aprime_convention == "native":
            total = total * (1 - native_lambda) / 4
    return {**terms, "regularizer": regularizer, "total": total}
