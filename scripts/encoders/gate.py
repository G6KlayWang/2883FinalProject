"""Explicit proposed eligibility policy; only C may receive train-fitted ZCA."""

from dataclasses import dataclass, fields
import numpy as np
from scripts.encoders.config import validate_fields
from scripts.encoders.measure import feature_statistics
from scripts.encoders.transforms import ZcaTransform, fit_zca


@dataclass(frozen=True)
class GatePolicy:
    status: str
    enabled: bool
    mean_limit: float
    eigenvalue_min: float
    eigenvalue_max: float
    gaussian_kurtosis_limit: float
    laplace_kurtosis_min: float

    @classmethod
    def from_dict(cls, value: dict) -> "GatePolicy":
        validate_fields(value, {f.name for f in fields(cls)}, "gate policy")
        result = cls(**value)
        if (
            result.status not in ("proposed", "approved")
            or type(result.enabled) is not bool
        ):
            raise ValueError("Gate status and enabled policy must be explicit")
        thresholds = [
            result.mean_limit,
            result.eigenvalue_min,
            result.eigenvalue_max,
            result.gaussian_kurtosis_limit,
            result.laplace_kurtosis_min,
        ]
        if not all(np.isfinite(v) and v > 0 for v in thresholds):
            raise ValueError("Gate thresholds must be finite and positive")
        if result.eigenvalue_min > result.eigenvalue_max:
            raise ValueError("Gate covariance interval is reversed")
        return result


@dataclass(frozen=True)
class CTransformPolicy:
    """Standing C moment matching, independent of the optional eligibility gate."""

    status: str
    mode: str
    mean_limit: float
    eigenvalue_min: float
    eigenvalue_max: float
    eigenvalue_floor: float

    @classmethod
    def from_dict(cls, value: dict) -> "CTransformPolicy":
        validate_fields(value, {f.name for f in fields(cls)}, "C transform policy")
        result = cls(**value)
        if result.status not in ("proposed", "approved"):
            raise ValueError("C transform status must be explicit proposed/approved")
        if result.mode != "if_moments_fail":
            raise ValueError("C transform mode must preserve if_moments_fail matching")
        thresholds = (
            result.mean_limit,
            result.eigenvalue_min,
            result.eigenvalue_max,
            result.eigenvalue_floor,
        )
        if not all(np.isfinite(v) and v > 0 for v in thresholds):
            raise ValueError("C transform thresholds must be finite and positive")
        if result.eigenvalue_min > result.eigenvalue_max:
            raise ValueError("C transform covariance interval is reversed")
        return result


def _moments_pass(statistics: dict, policy: GatePolicy | CTransformPolicy) -> bool:
    return (
        max(abs(v) for v in statistics["mean"]) <= policy.mean_limit
        and min(statistics["covariance_eigenvalues"]) >= policy.eigenvalue_min
        and max(statistics["covariance_eigenvalues"]) <= policy.eigenvalue_max
    )


def assess_training_features(
    features: np.ndarray,
    arm: str,
    policy: dict,
    *,
    c_policy: dict,
    split: str,
    checkpoint_role: str
) -> tuple[dict, ZcaTransform | None]:
    if split != "train":
        raise ValueError("Eligibility gate uses training features only")
    if checkpoint_role != "final":
        raise ValueError(
            "Intermediate/diagnostic smoke checkpoints cannot enter candidate eligibility"
        )
    if arm not in ("A", "B", "C", "Aprime"):
        raise ValueError("Unknown gate arm")
    settings = GatePolicy.from_dict(policy)
    c_settings = CTransformPolicy.from_dict(c_policy)
    raw = feature_statistics(features)
    effective, transform = raw, None
    if arm == "C" and not _moments_pass(raw, c_settings):
        if min(raw["covariance_eigenvalues"]) >= c_settings.eigenvalue_floor:
            transform = fit_zca(
                features, split="train", eigenvalue_floor=c_settings.eigenvalue_floor
            )
            effective = feature_statistics(transform.apply(features))
    coordinate_kurtosis = [
        v for v in effective["coordinate_excess_kurtosis"] if v is not None
    ]
    median = float(np.median(coordinate_kurtosis)) if coordinate_kurtosis else None
    shape_pass = True
    if arm in ("A", "Aprime"):
        shape_pass = (
            median is not None and abs(median) <= settings.gaussian_kurtosis_limit
        )
    elif arm == "B":
        shape_pass = median is not None and median >= settings.laplace_kurtosis_min
    c_policy_pass = arm != "C" or _moments_pass(effective, c_settings)
    eligible = c_policy_pass and (
        not settings.enabled or (_moments_pass(effective, settings) and shape_pass)
    )
    return {
        "eligible": bool(eligible),
        "policy": policy,
        "arm": arm,
        "c_transform_policy": c_policy,
        "c_transform_policy_pass": bool(c_policy_pass),
        "raw_statistics": raw,
        "effective_statistics": effective,
        "coordinate_shape_pass": shape_pass,
        "median_coordinate_kurtosis": median,
        "transformed": transform is not None,
        "scope": "training-only;proposed-policy-is-not-production-approval",
    }, transform
