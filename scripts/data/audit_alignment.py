"""Measurement-only audit against the pinned upstream PushT pusher controller.

This reconstructs measured dynamics; it does not run a simulator or train a model.
The historical archive converter remains unidentified. See README for evidence.
"""
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from scripts.data.file_io import sha256_file, write_json_new
from scripts.data.schema import read_layout

MAX_POSITION_ERROR_PIXELS = 1e-4
MAX_LAG0_TO_SHIFTED_RMSE_RATIO = 0.01
SIMULATOR_REVISION = '21446f1ede6d5284e981bd7b47f432b994e6d812'


def _errors_for_lag(state: np.ndarray, action: np.ndarray, indices: np.ndarray, lag: int) -> dict:
    if not len(indices):
        return {'count': 0, 'rmse_pixels': None, 'median_pixels': None, 'p99_pixels': None, 'max_pixels': None}
    position = state[indices, :2].astype(np.float64)
    velocity = state[indices, 5:7].astype(np.float64)
    target = position + 100 * action[indices + lag].astype(np.float64)
    # Pinned env.py: dt=.01, k_p=100, k_v=20, control_hz=10.
    for _ in range(10):
        velocity += (100 * (target - position) - 20 * velocity) * .01
        position += velocity * .01
    errors = np.linalg.norm(position - state[indices + 1, :2], axis=1)
    return {
        'count': int(len(indices)), 'rmse_pixels': float(np.sqrt(np.mean(errors**2))),
        'median_pixels': float(np.median(errors)), 'p99_pixels': float(np.quantile(errors, .99)),
        'max_pixels': float(errors.max()),
    }


def _lag_is_separated(correct: dict, shifted: dict) -> bool:
    if not correct['count'] or not shifted['count']:
        return False
    if shifted['rmse_pixels'] <= 0:
        return False
    return correct['rmse_pixels'] <= MAX_LAG0_TO_SHIFTED_RMSE_RATIO * shifted['rmse_pixels']


def audit_alignment(path: Path, *, code_revision: str) -> dict:
    """Bind every-transition controller checks and explicit pass/fail to file bytes."""
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError('code_revision must be a nonempty string')
    path = Path(path)
    before = path.stat()
    with h5py.File(path, 'r') as handle:
        episodes = read_layout(handle)
        state, action = handle['state'][:], handle['action'][:]
        proprio = handle['proprio'][:]
        if not all(np.isfinite(values).all() for values in (state, action, proprio)):
            raise ValueError('Alignment measurements must contain finite state, action and proprio values')
        step, episode_ids = handle['step_idx'][:], handle['episode_idx'][:]
        lengths = np.array([ep['length'] for ep in episodes])
        valid = np.flatnonzero(step[1:] > 0)
        lag_indices = {
            -1: valid[step[valid] > 0], 0: valid,
            1: valid[step[valid + 1] < lengths[episode_ids[valid + 1]] - 1],
        }
        report = {
            'simulator_revision': SIMULATOR_REVISION,
            'code_revision': code_revision,
            'selection': ('all within-episode adjacent transitions; '
                          'shifted comparisons exclude invalid/terminal actions'),
            'lags': {str(lag): _errors_for_lag(state, action, indices, lag) for lag, indices in lag_indices.items()},
            'proprio_matches_state_0_1_5_6': bool(np.array_equal(proprio, state[:, [0, 1, 5, 6]])),
            'angle_min_radians': float(state[:, 4].min()),
            'angle_max_radians': float(state[:, 4].max()),
        }
    maximum_error = report['lags']['0']['max_pixels']
    expected_transitions = sum(ep['length'] - 1 for ep in episodes)
    checks = {
        'all_transitions_checked': report['lags']['0']['count'] == expected_transitions,
        'transitions_exist': expected_transitions > 0,
        'lag0_position_within_tolerance': maximum_error is not None and maximum_error <= MAX_POSITION_ERROR_PIXELS,
        'lag0_far_below_shifted': all(
            _lag_is_separated(report['lags']['0'], report['lags'][lag]) for lag in ('-1', '1')
        ),
        'proprio_matches_state': report['proprio_matches_state_0_1_5_6'],
        'angles_in_radian_wrap_range': 0 <= report['angle_min_radians'] <= report['angle_max_radians'] < 2 * np.pi,
    }
    report.update(
        status='passed' if all(checks.values()) else 'failed', checks=checks,
        thresholds={
            'max_position_error_pixels': MAX_POSITION_ERROR_PIXELS,
            'max_lag0_to_shifted_rmse_ratio': MAX_LAG0_TO_SHIFTED_RMSE_RATIO,
        },
        source={'hdf5_sha256': sha256_file(path), 'hdf5_size_bytes': before.st_size},
    )
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('HDF5 changed during alignment audit')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = audit_alignment(args.hdf5, code_revision=args.code_revision)
    write_json_new(args.output, result)
    logging.info('ALIGNMENT %s', result)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
