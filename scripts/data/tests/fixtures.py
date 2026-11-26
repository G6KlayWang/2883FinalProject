"""Small real HDF5 files whose observations/actions encode their row indices."""
from pathlib import Path
import h5py
import numpy as np


def make_dataset(path: Path, lengths=(12, 6)) -> Path:
    lengths = np.asarray(lengths, dtype=np.int32)
    total = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    with h5py.File(path, 'w') as f:
        f['ep_len'], f['ep_offset'] = lengths, offsets
        f['episode_idx'] = np.repeat(np.arange(len(lengths)), lengths)
        f['step_idx'] = np.concatenate([np.arange(n) for n in lengths])
        f['action'] = np.column_stack((np.arange(total), -np.arange(total))).astype(np.float32)
        state = np.arange(total * 7, dtype=np.float32).reshape(total, 7)
        f['state'], f['proprio'] = state, state[:, [0, 1, 5, 6]]
        pixels = f.create_dataset('pixels', (total, 224, 224, 3), dtype='uint8', compression='gzip')
        for row in range(total):
            pixels[row] = np.full((224, 224, 3), row % 256, dtype=np.uint8)
    return path


DOWNLOAD = {'sha256': 'b' * 64, 'size_bytes': 70, 'dataset_revision': 'source-revision'}
