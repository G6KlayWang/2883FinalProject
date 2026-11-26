"""Measurement-only pose targets, visibility and the shared normalized group score."""

import numpy as np
import torch


def pose_targets(state: np.ndarray) -> np.ndarray:
    if state.ndim != 2 or state.shape[1] != 7 or not np.isfinite(state).all():
        raise ValueError("Pose labels require finite(N,7) simulator states")
    return np.column_stack(
        (state[:, 2:4], np.sin(state[:, 4]), np.cos(state[:, 4]), state[:, :2])
    )


def visibility_mask(state: np.ndarray, margin: float, convention: str) -> np.ndarray:
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Visibility margin must be finite and nonnegative")
    pusher = state[:, :2]
    if convention == "expanded_box":
        return ((pusher >= -margin) & (pusher <= 512 + margin)).all(axis=1)
    if convention == "circle_intersection":
        closest = np.clip(pusher, 0, 512)
        return ((pusher - closest) ** 2).sum(axis=1) <= margin**2
    raise ValueError(
        "Visibility convention must explicitly be expanded_box or circle_intersection"
    )


def group_error_tensors(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    train_variance: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    error = (prediction - targets).square()
    t_position = (error[:, :2] / train_variance[:2]).mean()
    angle = error[:, 2:4].sum(1).mean() / train_variance[2:4].sum()
    pusher = (
        (error[visible, 4:] / train_variance[4:]).mean()
        if visible.any()
        else error[:, 4:].sum() * 0
    )
    return t_position, angle, pusher


def group_scores(
    prediction: np.ndarray,
    targets: np.ndarray,
    train_variance: np.ndarray,
    visible: np.ndarray,
) -> dict:
    if prediction.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError("Pose score requires matching(N,6) predictions and targets")
    if visible.shape != (len(targets),) or not visible.any():
        raise ValueError("Pusher scoring requires at least one visible row")
    if not all(np.isfinite(v).all() for v in (prediction, targets, train_variance)):
        raise ValueError("Pose score inputs must be finite")
    if np.any(train_variance[[0, 1, 4, 5]] <= 0) or train_variance[2:4].sum() <= 0:
        raise ValueError("Training variance denominators must be positive")
    tensors = [
        torch.from_numpy(np.array(v, copy=True))
        for v in (
            prediction.astype(np.float64),
            targets.astype(np.float64),
            train_variance.astype(np.float64),
            visible,
        )
    ]
    terms = group_error_tensors(*tensors)
    return {
        **{
            name: value.item()
            for name, value in zip(("T_position", "T_angle", "pusher_position"), terms)
        },
        "score": sum(t.item() for t in terms) / 3,
        "visible_rows": int(visible.sum()),
        "rows": len(targets),
    }
