"""Fixed-pair, tie-aware goal metrics; diagnostics never enter weight selection."""

import hashlib
import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[
        0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)
    ]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    a, b = average_ranks(first), average_ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def goal_correlations(
    features: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    *,
    split: str,
    pairs: int = 4096,
    seed: int = 20261126
) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("Goal-metric development never uses test rows")
    if len(features) < 2 or targets.shape != (len(features), 6) or pairs < 1:
        raise ValueError(
            "Goal metric requires aligned features/pose labels and positive pair count"
        )
    generator = np.random.default_rng(seed)
    first = generator.integers(0, len(features), pairs)
    second = (first + generator.integers(1, len(features), pairs)) % len(features)
    distance = np.linalg.norm(
        features[first].astype(np.float64) - features[second], axis=1
    )
    report = {
        "seed": seed,
        "pairs": pairs,
        "split": split,
        "pair_sha256": hashlib.sha256(
            np.column_stack((first, second)).astype("<i8").tobytes()
        ).hexdigest(),
    }
    for name, columns, mask in [
        ("T_position", slice(0, 2), np.ones(pairs, dtype=bool)),
        ("T_angle", slice(2, 4), np.ones(pairs, dtype=bool)),
        ("pusher_position", slice(4, 6), visible[first] & visible[second]),
    ]:
        pose_distance = np.linalg.norm(
            targets[first, columns] - targets[second, columns], axis=1
        )
        report[name] = {
            "spearman": _spearman(distance[mask], pose_distance[mask]),
            "valid_pairs": int(mask.sum()),
        }
    report["angle_distance"] = "unstandardized-sin-cos-chord"
    return report
