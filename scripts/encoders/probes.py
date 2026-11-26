"""Frozen-feature OLS/MLP probes with training-only scales and final-epoch policy."""

from dataclasses import dataclass, fields
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.artifacts import new_directory
from scripts.encoders.config import validate_fields
from scripts.encoders.pose import (
    pose_targets,
    visibility_mask,
    group_error_tensors,
    group_scores,
)
from scripts.encoders.rng import keyed_seed

POSITIONS = [0, 1, 4, 5]


@dataclass(frozen=True)
class ProbePolicy:
    status: str
    hidden: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    variance_floor: float
    visibility: str
    visibility_margin: float
    tie_fraction: float
    tie_denominator_floor: float
    weight_decay: float
    betas: list[float]
    epsilon: float
    linear_rcond: float
    feature_std_epsilon: float

    @classmethod
    def from_dict(cls, value: dict) -> "ProbePolicy":
        validate_fields(value, {f.name for f in fields(cls)}, "probe policy")
        result = cls(**value)
        if result.status not in ("proposed", "approved"):
            raise ValueError("Probe policy status must be explicit")
        for name in ("hidden", "epochs", "batch_size"):
            if type(getattr(result, name)) is not int or getattr(result, name) < 1:
                raise ValueError(f"Probe {name} must be a positive integer")
        for name in (
            "learning_rate",
            "variance_floor",
            "tie_denominator_floor",
            "epsilon",
            "linear_rcond",
            "feature_std_epsilon",
        ):
            value = getattr(result, name)
            if (
                not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Probe {name} must be finite and positive")
        if result.visibility not in ("expanded_box", "circle_intersection"):
            raise ValueError("Probe visibility convention must be explicit")
        if (
            not 0 <= result.tie_fraction < 1
            or result.visibility_margin < 0
            or result.weight_decay < 0
        ):
            raise ValueError("Invalid probe tie/mask/weight-decay policy")
        if len(result.betas) != 2 or any(not 0 <= v < 1 for v in result.betas):
            raise ValueError("Probe AdamW betas must be in[0,1)")
        return result


@dataclass
class Scaling:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_variance: np.ndarray

    def features(self, values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            (np.asarray(values, dtype=np.float64) - self.feature_mean)
            / self.feature_std
        )

    def targets(self, values: np.ndarray) -> torch.Tensor:
        result = np.array(values, dtype=np.float64, copy=True)
        result[:, POSITIONS] = (
            result[:, POSITIONS] - self.target_mean[POSITIONS]
        ) / np.sqrt(self.target_variance[POSITIONS])
        return torch.from_numpy(result)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        mean = torch.as_tensor(
            self.target_mean, dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(
            np.sqrt(self.target_variance), dtype=values.dtype, device=values.device
        )
        result[:, POSITIONS] = result[:, POSITIONS] * std[POSITIONS] + mean[POSITIONS]
        return result


def _fit_scaling(
    x: np.ndarray, y: np.ndarray, visible: np.ndarray, split: str, policy: ProbePolicy
) -> Scaling:
    if split != "train":
        raise ValueError("Probe fitting and normalization require training rows only")
    if (
        x.ndim != 2
        or y.shape != (len(x), 6)
        or visible.shape != (len(x),)
        or len(x) < 2
    ):
        raise ValueError("Probe requires features(N,D), targets(N,6), and visible(N)")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or visible.sum() < 2:
        raise ValueError(
            "Probe needs finite inputs and at least two visible training rows"
        )
    feature_mean = np.mean(x, dtype=np.float64, axis=0)
    std = np.std(x, dtype=np.float64, axis=0, ddof=1)
    std = np.where(std >= policy.feature_std_epsilon, std, 1.0)
    target_mean = np.mean(y, dtype=np.float64, axis=0)
    variance = np.var(y, dtype=np.float64, axis=0, ddof=1)
    target_mean[4:] = np.mean(y[visible, 4:], axis=0, dtype=np.float64)
    variance[4:] = np.var(y[visible, 4:], axis=0, dtype=np.float64, ddof=1)
    if (
        np.any(variance[POSITIONS] < policy.variance_floor)
        or variance[2:4].sum() < policy.variance_floor
    ):
        raise ValueError(
            "Degenerate training target variance; policy stops rather than changing normalization"
        )
    return Scaling(feature_mean, std, target_mean, variance)


class FittedProbe:
    def __init__(
        self, kind: str, scaling: Scaling, parameters, policy: dict, curve: list[float]
    ):
        self.kind, self.scaling, self.parameters, self.policy, self.curve = (
            kind,
            scaling,
            parameters,
            policy,
            curve,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = self.scaling.features(features)
        if self.kind == "linear":
            design = torch.cat(
                (values, torch.ones(len(values), 1, dtype=torch.float64)), dim=1
            )
            return self.scaling.decode(design @ self.parameters).numpy()
        model = self.parameters.eval()
        device = next(model.parameters()).device
        output = []
        with torch.inference_mode():
            for start in range(0, len(values), self.policy["batch_size"]):
                prediction = model(
                    values[start : start + self.policy["batch_size"]].float().to(device)
                )
                output.append(self.scaling.decode(prediction).cpu().numpy())
        return np.concatenate(output)

    def save(self, output: Path, identity: dict) -> None:
        with new_directory(output) as temporary:
            torch.save(
                {
                    "kind": self.kind,
                    "scaling": self.scaling.__dict__,
                    "parameters": (
                        self.parameters
                        if self.kind == "linear"
                        else self.parameters.state_dict()
                    ),
                    "policy": self.policy,
                    "curve": self.curve,
                },
                temporary / "probe.pt",
            )
            write_json_new(
                temporary / "manifest.json",
                {
                    "version": 1,
                    "identity": identity,
                    "fit_split": "train",
                    "probe_sha256": sha256_file(temporary / "probe.pt"),
                },
            )


def _network(dim: int, policy: ProbePolicy) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(keyed_seed(policy.seed, "probes", "initialization"))
        return nn.Sequential(
            nn.Linear(dim, policy.hidden), nn.ReLU(), nn.Linear(policy.hidden, 6)
        )


def fit_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    *,
    split: str,
    policy: dict,
) -> FittedProbe:
    settings = ProbePolicy.from_dict(policy)
    scaling = _fit_scaling(features, targets, visible, split, settings)
    values = scaling.features(features)
    design = torch.cat((values, torch.ones(len(values), 1, dtype=torch.float64)), dim=1)
    encoded = scaling.targets(targets)
    coefficients = torch.empty(design.shape[1], 6, dtype=torch.float64)
    coefficients[:, :4] = torch.linalg.lstsq(
        design, encoded[:, :4], rcond=settings.linear_rcond, driver="gelsd"
    ).solution
    coefficients[:, 4:] = torch.linalg.lstsq(
        design[visible],
        encoded[visible, 4:],
        rcond=settings.linear_rcond,
        driver="gelsd",
    ).solution
    return FittedProbe("linear", scaling, coefficients, policy, [])


def fit_mlp_probe(
    features: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    *,
    split: str,
    policy: dict,
    device: str,
) -> FittedProbe:
    settings = ProbePolicy.from_dict(policy)
    scaling = _fit_scaling(features, targets, visible, split, settings)
    values = scaling.features(features).float().to(device)
    raw_targets = torch.from_numpy(np.array(targets, dtype=np.float32, copy=True)).to(
        device
    )
    mask = torch.from_numpy(visible.copy()).to(device)
    variance = torch.as_tensor(
        scaling.target_variance, dtype=torch.float32, device=device
    )
    model = _network(features.shape[1], settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=tuple(settings.betas),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
        foreach=False,
    )
    curve = []
    for epoch in range(settings.epochs):
        generator = torch.Generator().manual_seed(
            keyed_seed(settings.seed, "probes", "order", epoch)
        )
        order = torch.randperm(len(values), generator=generator)
        total = 0.0
        for start in range(0, len(values), settings.batch_size):
            indices = order[start : start + settings.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = scaling.decode(model(values[indices]))
            terms = group_error_tensors(
                prediction, raw_targets[indices], variance, mask[indices]
            )
            loss = sum(terms) / 3
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite probe gradient")
            optimizer.step()
            total += loss.item() * len(indices)
        curve.append(total / len(values))
    model.eval().requires_grad_(False)
    return FittedProbe("mlp", scaling, model, policy, curve)


def load_probe(path: Path, identity: dict) -> FittedProbe:
    path = Path(path)
    metadata = json.loads((path / "manifest.json").read_text())
    if metadata.get("identity") != identity or metadata.get("fit_split") != "train":
        raise ValueError("Probe checkpoint/training identity mismatch")
    if sha256_file(path / "probe.pt") != metadata["probe_sha256"]:
        raise ValueError("Probe hash mismatch")
    state = torch.load(path / "probe.pt", map_location="cpu", weights_only=False)
    scaling = Scaling(**state["scaling"])
    parameters = state["parameters"]
    if state["kind"] == "mlp":
        parameters = _network(
            len(scaling.feature_mean), ProbePolicy.from_dict(state["policy"])
        )
        parameters.load_state_dict(state["parameters"])
        parameters.eval().requires_grad_(False)
    return FittedProbe(
        state["kind"], scaling, parameters, state["policy"], state["curve"]
    )
