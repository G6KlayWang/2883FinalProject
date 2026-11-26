"""Explicit resolved configuration; proposed recipes cannot launch production."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from scripts.encoders.augment import validate_augmentation


def validate_fields(value: dict, expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if missing := expected - value.keys():
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if extra := value.keys() - expected:
        raise ValueError(f"{name} unknown fields: {sorted(extra)}")


def _positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class OptimizerConfig:
    batch_size: int
    max_updates: int
    warmup_updates: int
    peak_lr: float
    final_lr: float
    weight_decay: float
    betas: list[float]
    epsilon: float

    def validate(self) -> None:
        for name in ("batch_size", "max_updates", "warmup_updates"):
            _integer(getattr(self, name), "optimizer." + name)
        if self.batch_size < 2 or self.warmup_updates >= self.max_updates:
            raise ValueError(
                "Batch must be >=2; warmup must end before total update budget"
            )
        for name in ("peak_lr", "final_lr", "epsilon"):
            _positive(getattr(self, name), "optimizer." + name)
        if not isinstance(self.weight_decay, (int, float)) or not math.isfinite(
            self.weight_decay
        ):
            raise ValueError("weight_decay must be finite")
        if self.final_lr > self.peak_lr or self.weight_decay < 0:
            raise ValueError("Require final_lr<=peak_lr and nonnegative weight_decay")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("AdamW betas must contain two values in [0,1)")


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int
    prefetch: int
    intraop_threads: int
    interop_threads: int
    max_seconds: float
    max_frames: int | None
    checkpoint_every: int
    gradient_terms_every: int
    keep_checkpoints: int
    disk_reserve_gib: float
    memory_limit_gib: float

    def validate(self) -> None:
        _integer(self.workers, "runtime.workers", 0)
        for name in (
            "prefetch",
            "intraop_threads",
            "interop_threads",
            "checkpoint_every",
            "gradient_terms_every",
            "keep_checkpoints",
        ):
            _integer(getattr(self, name), "runtime." + name)
        for name in ("max_seconds", "disk_reserve_gib", "memory_limit_gib"):
            _positive(getattr(self, name), "runtime." + name)
        if self.max_frames is not None:
            _integer(self.max_frames, "runtime.max_frames")


@dataclass(frozen=True)
class RunConfig:
    profile: str
    arm: str
    seed: int
    rho: float
    recipe_status: str
    frame_policy: str
    aprime_convention: str
    native_lambda: float
    stem: str
    projector_hidden: int
    feature_dim: int
    projections: int
    view_forward: str
    input_normalization: str
    tf32_matmul: bool
    tf32_cudnn: bool
    augment: dict[str, Any]
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    grids: dict[str, Any]
    gate: dict[str, Any]
    probe: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict) -> "RunConfig":
        validate_fields(value, {f.name for f in fields(cls)}, "run config")
        nested = {}
        for key, kind in (("optimizer", OptimizerConfig), ("runtime", RuntimeConfig)):
            validate_fields(value[key], {f.name for f in fields(kind)}, key)
            nested[key] = kind(**value[key])
            nested[key].validate()
        result = cls(**{**value, **nested})
        result.validate()
        return result

    def validate(self) -> None:
        if self.profile not in ("smoke", "production") or self.arm not in (
            "A",
            "B",
            "C",
            "Aprime",
        ):
            raise ValueError("Unknown profile or arm")
        if self.recipe_status not in ("diagnostic", "proposed", "approved"):
            raise ValueError(
                "recipe_status must be explicit diagnostic/proposed/approved"
            )
        _integer(self.seed, "seed", 0)
        _positive(self.rho, "rho")
        if self.frame_policy not in ("phase0", "raw") or self.aprime_convention not in (
            "common",
            "native",
        ):
            raise ValueError(
                "Explicit phase0/raw frame policy and common/native A-prime convention required"
            )
        if (
            self.stem != "standard"
            or self.feature_dim != 128
            or self.projector_hidden != 2048
        ):
            raise ValueError(
                "Shared model requires standard ResNet18 stem, hidden2048, output128"
            )
        if self.projections != 1024 or self.view_forward != "separate":
            raise ValueError(
                "Require1024 random projections and separate shared view forwards"
            )
        if self.input_normalization != "identity_0_1":
            raise ValueError("Current shared input normalization is identity_0_1")
        if type(self.tf32_matmul) is not bool or type(self.tf32_cudnn) is not bool:
            raise ValueError("Both TF32 policies must be explicit booleans")
        if not 0 < self.native_lambda < 1:
            raise ValueError("native_lambda must lie in (0,1)")
        if self.arm == "Aprime" and not math.isclose(
            self.rho, 4 * self.native_lambda / (1 - self.native_lambda)
        ):
            raise ValueError("A-prime rho must equal4*lambda/(1-lambda)")
        for name in ("grids", "gate", "probe"):
            block = getattr(self, name)
            if not isinstance(block, dict) or block.get("status") not in (
                "proposed",
                "approved",
            ):
                raise ValueError(f"{name}.status must be explicit proposed/approved")
        from scripts.encoders.gate import GatePolicy
        from scripts.encoders.probes import ProbePolicy

        GatePolicy.from_dict(self.gate)
        ProbePolicy.from_dict(self.probe)
        validate_fields(self.grids, {"status", "A", "B", "C"}, "grids")
        for arm in ("A", "B", "C"):
            weights = self.grids[arm]
            if not isinstance(weights, list) or len(weights) != 3:
                raise ValueError(f"grids.{arm} must list exactly three weights")
            for weight in weights:
                _positive(weight, f"grids.{arm}")
            if weights != sorted(set(weights)):
                raise ValueError(f"grids.{arm} must be increasing and unique")
        validate_augmentation(self.augment)
        if self.profile == "smoke":
            self._validate_smoke()

    def _validate_smoke(self) -> None:
        if (self.arm, self.seed, self.rho, self.frame_policy) != (
            "A",
            0,
            10.0,
            "phase0",
        ):
            raise ValueError("Smoke is ArmA seed0 rho10 train-phase0 only")
        if self.recipe_status != "diagnostic" or self.optimizer.batch_size != 16:
            raise ValueError("Smoke requires diagnostic status and batch16")
        if self.optimizer.max_updates > 20 or self.runtime.max_seconds > 600:
            raise ValueError(
                "Smoke cannot exceed20 lifetime updates or600 model seconds"
            )
        if self.runtime.max_frames is None or not 16 <= self.runtime.max_frames <= 256:
            raise ValueError("Smoke frame cap must be16..256")
        if self.tf32_matmul or self.tf32_cudnn:
            raise ValueError("Smoke requires both TF32 flags disabled")

    def require_execution_approval(self) -> None:
        if self.profile == "production" and self.recipe_status != "approved":
            raise ValueError(
                "Production requires recorded recipe approval; proposed configuration cannot execute"
            )
        if self.profile == "production" and any(
            getattr(self, name)["status"] != "approved"
            for name in ("grids", "gate", "probe")
        ):
            raise ValueError("Production scientific policies require recorded approval")

    def stop_update(self, requested: int | None) -> int:
        value = self.optimizer.max_updates if requested is None else requested
        if type(value) is not int or not 1 <= value <= self.optimizer.max_updates:
            raise ValueError(f"--stop-at must be1..{self.optimizer.max_updates}")
        return value

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


def shared_training_controls(config: RunConfig) -> dict:
    controls = config.to_dict()
    # Arms/weights are the experimental variables; runtime worker/IO settings
    # do not change logical samples/views and may differ by available resources.
    for key in ("arm", "rho", "runtime"):
        controls.pop(key)
    return controls


def load_config(path: Path) -> RunConfig:
    return RunConfig.from_dict(json.loads(Path(path).read_text()))


def run_matrix(grids: dict | None = None, native_lambda: float = 0.05) -> list[dict]:
    """Nominal four-GPU queue: exactly ten candidates and three selected repeats."""
    grids = grids or {
        "A": [2.5, 10.0, 40.0],
        "B": [2.5, 10.0, 40.0],
        "C": [0.25, 1.0, 4.0],
    }
    a, b, c = (grids[arm] for arm in ("A", "B", "C"))
    order = [
        ("A", a[0]),
        ("B", b[0]),
        ("C", c[0]),
        ("Aprime", 4 * native_lambda / (1 - native_lambda)),
        ("B", b[1]),
        ("C", c[1]),
        ("A", a[1]),
        ("B", b[2]),
        ("C", c[2]),
        ("A", a[2]),
    ]
    rows = [
        {
            "run_id": f"{arm}-rho{rho:g}-seed0",
            "arm": arm,
            "rho": rho,
            "seed": 0,
            "stage": "candidate",
            "status": "proposed",
            "queue_position": i,
        }
        for i, (arm, rho) in enumerate(order)
    ]
    return rows + [
        {
            "run_id": f"{arm}-selected-seed1",
            "arm": arm,
            "rho": None,
            "seed": 1,
            "stage": "repeat",
            "status": "awaits-validation-selection",
        }
        for arm in ("A", "B", "C")
    ]
