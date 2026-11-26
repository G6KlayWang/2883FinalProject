"""Inspect the observed flat PushT schema without fitting any statistics."""
import argparse
from collections import defaultdict
import hashlib
import logging
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the archive's Blosc filter.
import numpy as np

from scripts.data.file_io import sha256_file, write_json_new
from scripts.data.families import PUSHT_HDF5_SHA256, inspect_families, validate_families

LOG = logging.getLogger(__name__)
EXPECTED = {
    'pixels': ((224, 224, 3), np.dtype('uint8')),
    'action': ((2,), np.dtype('float32')),
    'state': ((7,), np.dtype('float32')),
    'proprio': ((4,), np.dtype('float32')),
    'episode_idx': ((), np.dtype('int64')),
    'step_idx': ((), np.dtype('int64')),
}


def read_layout(handle: h5py.File) -> list[dict]:
    """Validate stored axes and boundaries; row position is the stable episode ID."""
    required = {'ep_len', 'ep_offset', *EXPECTED}
    if missing := required - set(handle):
        raise ValueError(f'Missing datasets: {sorted(missing)}')
    for name in required:
        if not isinstance(handle[name], h5py.Dataset):
            raise ValueError(f'{name} must be a dataset')
    lengths, offsets = handle['ep_len'][:], handle['ep_offset'][:]
    if lengths.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu':
        raise ValueError('Episode lengths and offsets must be integer arrays')
    if lengths.ndim != 1 or not len(lengths) or offsets.shape != lengths.shape or np.any(lengths <= 0):
        raise ValueError('Episode lengths must be positive, nonempty and match offsets')
    lengths, offsets = lengths.astype(np.int64), offsets.astype(np.int64)
    wanted_offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    if not np.array_equal(offsets, wanted_offsets):
        raise ValueError('Episode offsets must be contiguous, starting at zero')
    total = int(lengths.sum())
    for name, (shape, dtype) in EXPECTED.items():
        dataset = handle[name]
        if dataset.shape != (total, *shape) or dataset.dtype != dtype:
            raise ValueError(f'Unexpected {name} shape/dtype: {dataset.shape}/{dataset.dtype}')
    episode_indices, step_indices = handle['episode_idx'][:], handle['step_idx'][:]
    episodes = []
    for episode_id, (offset, length) in enumerate(zip(offsets.tolist(), lengths.tolist())):
        end = offset + length
        if not np.all(episode_indices[offset:end] == episode_id):
            raise ValueError(f'episode_idx inconsistent in episode {episode_id}')
        if not np.array_equal(step_indices[offset:end], np.arange(length)):
            raise ValueError(f'step_idx inconsistent in episode {episode_id}')
        episodes.append({'episode_id': episode_id, 'offset': offset, 'length': length})
    return episodes


def _json_value(value):
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='backslashreplace')
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _describe(dataset: h5py.Dataset) -> dict:
    filters = dataset.id.get_create_plist()
    description = {
        'shape': list(dataset.shape), 'dtype': str(dataset.dtype),
        'chunks': list(dataset.chunks) if dataset.chunks else None,
        'filters': [_json_value(list(filters.get_filter(i))) for i in range(filters.get_nfilters())],
        'attributes': {k: _json_value(v) for k, v in dataset.attrs.items()},
    }
    # Numeric summaries describe raw data; they are never fitted model transforms.
    if dataset.name != '/pixels' and dataset.dtype.kind in 'iuf':
        low, high = None, None
        for start in range(0, len(dataset), 65536):
            values = dataset[start:start + 65536]
            if not np.isfinite(values).all():
                raise ValueError(f'{dataset.name} must contain finite values')
            chunk_low, chunk_high = values.min(axis=0), values.max(axis=0)
            low = chunk_low if low is None else np.minimum(low, chunk_low)
            high = chunk_high if high is None else np.maximum(high, chunk_high)
        description.update(min=_json_value(low), max=_json_value(high))
    return description


def duplicate_episode_groups(handle: h5py.File, episodes: list[dict]) -> list[list[int]]:
    """Check exact action-sequence + first-image duplicates without decoding all images.

    Images are decoded only for matching action hashes. This does not detect
    approximately equivalent trajectories or duplicates with altered actions.
    """
    actions = handle['action'][:]
    candidates = defaultdict(list)
    for ep in episodes:
        start, length = ep['offset'], ep['length']
        digest = hashlib.sha256(actions[start:start + length].tobytes()).hexdigest()
        candidates[(length, digest)].append(ep)
    duplicates = []
    for group in candidates.values():
        if len(group) < 2:
            continue
        images = defaultdict(list)
        for ep in group:
            digest = hashlib.sha256(handle['pixels'][ep['offset']].tobytes()).hexdigest()
            images[digest].append(ep['episode_id'])
        duplicates.extend(ids for ids in images.values() if len(ids) > 1)
    return sorted(duplicates)


def inspect_file(path: Path) -> dict:
    """Emit portable layout, ranges, exact duplicate audit and complete-file identity."""
    path = Path(path)
    before = path.stat()
    with h5py.File(path, 'r') as handle:
        episodes = read_layout(handle)
        # Check filter availability now, before a future loader starts reading.
        handle['pixels'][0]
        handle['pixels'][-1]
        report = {
            'schema_version': 1,
            **inspect_families(handle, episodes),
            'attributes': {k: _json_value(v) for k, v in handle.attrs.items()},
            'arrays': {name: _describe(dataset) for name, dataset in handle.items()},
            'episodes': episodes,
            'duplicate_episode_groups': duplicate_episode_groups(handle, episodes),
            'hdf5_size_bytes': before.st_size,
        }
    LOG.info('Hashing complete HDF5 (%d bytes)', before.st_size)
    report['hdf5_sha256'] = sha256_file(path)
    validate_families(episodes, report, report['hdf5_sha256'])
    report['verified_pusht_source'] = report['hdf5_sha256'] == PUSHT_HDF5_SHA256
    if not report['verified_pusht_source']:
        LOG.warning('Unverified HDF5 source %s: pinned PushT 185x101 structure assertion skipped', report['hdf5_sha256'])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('HDF5 changed during inspection')
    LOG.info('INSPECTED %d episodes | %d rows | %s', len(episodes), sum(ep['length'] for ep in episodes), report['hdf5_sha256'])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json_new(args.output, inspect_file(args.hdf5))


if __name__ == '__main__':
    main()
