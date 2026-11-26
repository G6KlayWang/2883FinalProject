"""Measurement-only audit against the pinned upstream PushT pusher controller.

This reconstructs measured dynamics; it does not run a simulator or train a model.
The historical archive converter remains unidentified. See README for evidence.
"""
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from scripts.data.file_io import write_json_new
from scripts.data.schema import read_layout

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


def audit_alignment(path: Path) -> dict:
    """Check every valid transition; compare shifted actions without boundary leaks."""
    with h5py.File(path, 'r') as handle:
        episodes = read_layout(handle)
        state, action = handle['state'][:], handle['action'][:]
        step, episode_ids = handle['step_idx'][:], handle['episode_idx'][:]
        lengths = np.array([ep['length'] for ep in episodes])
        valid = np.flatnonzero(step[1:] > 0)
        lag_indices = {
            -1: valid[step[valid] > 0], 0: valid,
            1: valid[step[valid + 1] < lengths[episode_ids[valid + 1]] - 1],
        }
        return {
            'simulator_revision': SIMULATOR_REVISION,
            'selection': 'all within-episode adjacent transitions; shifted comparisons exclude invalid/terminal actions',
            'lags': {str(lag): _errors_for_lag(state, action, indices, lag) for lag, indices in lag_indices.items()},
            'proprio_matches_state_0_1_5_6': bool(np.array_equal(handle['proprio'][:], state[:, [0, 1, 5, 6]])),
            'angle_min_radians': float(state[:, 4].min()),
            'angle_max_radians': float(state[:, 4].max()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    result = audit_alignment(args.hdf5)
    write_json_new(args.output, result)
    logging.info('ALIGNMENT %s', result)


if __name__ == '__main__':
    main()
