"""Image/action-only PushT readers; simulator measurements have a separate API."""
from bisect import bisect_right
from contextlib import contextmanager
import copy
from itertools import accumulate
import logging
from pathlib import Path
from collections.abc import Iterator, Sequence

import h5py
import hdf5plugin  # noqa: F401 -- explicitly registers the archive codec.
import numpy as np
from PIL import Image

from scripts.data.file_io import verify_file
from scripts.data.families import inspect_families
from scripts.data.schema import read_layout
from scripts.data.splits import validate_manifest

LOG = logging.getLogger(__name__)
FRAMESKIP = 5


def resize_rgb(image: np.ndarray) -> np.ndarray:
    """Full-image Pillow bilinear resize -> contiguous float32 RGB CHW [0,1]."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected uint8 HWC RGB image')
    resized = np.asarray(Image.fromarray(image).resize((96, 96), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)


def window_count(length: int, horizon: int, stride: int) -> int:
    """Number of phase-0 windows of K transitions, with stride in macro steps."""
    if any(type(v) is not int or v <= 0 for v in (length, horizon, stride)):
        raise ValueError('Length, horizon and stride must be positive integers')
    blocks = (length - 1) // FRAMESKIP
    return max(0, (blocks - horizon) // stride + 1)


def _locate(cumulative: Sequence[int], index: int) -> tuple[int, int]:
    size = cumulative[-1]
    if type(index) is not int:
        raise TypeError('Dataset index must be an integer')
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError(index)
    episode_position = bisect_right(cumulative, index) - 1
    return episode_position, index - cumulative[episode_position]


class PreparedData:
    """Verify content identity once before sharing lightweight readers with workers.

    No HDF5 handles survive a read. Full-file SHA validation happens here, not
    once per worker or sample. Subsequent reads reject changed file stat identity;
    the prepared HDF5 must remain immutable during use.
    """
    def __init__(self, path: Path, manifest: dict, *, allow_diagnostic_episode_split: bool = False):
        validate_manifest(manifest)
        self.diagnostic_episode_split = manifest['split_unit'] == 'episode'
        if self.diagnostic_episode_split and allow_diagnostic_episode_split is not True:
            raise ValueError('Episode splits are diagnostic only; require allow_diagnostic_episode_split=True')
        if self.diagnostic_episode_split:
            LOG.warning('Using diagnostic episode split; shared-start families may cross splits')
        self.path = Path(path).resolve()
        self.manifest = copy.deepcopy(manifest)
        self.episode_splits = {i: split for split, ids in manifest['splits'].items() for i in ids}
        source = manifest['source']
        verify_file(self.path, source['hdf5_size_bytes'], source['hdf5_sha256'])
        with h5py.File(self.path, 'r') as handle:
            if read_layout(handle) != manifest['episodes']:
                raise ValueError('HDF5 layout does not match split manifest')
            if manifest['split_unit'] == 'family':
                actual_families = inspect_families(handle, manifest['episodes'])
                if any(manifest.get(key) != value for key, value in actual_families.items()):
                    raise ValueError('HDF5 initial-state family membership does not match manifest')
        self._identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @contextmanager
    def _open(self) -> Iterator[h5py.File]:
        if self._file_identity() != self._identity:
            raise ValueError('Prepared HDF5 changed after verification')
        with h5py.File(self.path, 'r') as handle:
            yield handle
        if self._file_identity() != self._identity:
            raise ValueError('Prepared HDF5 changed during reading')

    def _episodes(self, split: str) -> list[dict]:
        if split not in self.manifest['splits']:
            raise ValueError(f'Unknown split: {split}')
        return [self.manifest['episodes'][i] for i in self.manifest['splits'][split]]

    def _metadata(self, episode_id: int, split: str) -> dict:
        return {'episode_id': episode_id, 'split': split, 'split_unit': self.manifest['split_unit'],
                'hdf5_sha256': self.manifest['source']['hdf5_sha256']}


class TrajectoryDataset:
    """Each sample: frames (K+1,3,96,96), raw actions (K,10), metadata.

    actions[j].reshape(5,2) contains outgoing actions t:t+5 in order and
    frames[j+1] is observation t+5. Initial conditioning is frames[0]. Use
    phase 0 only; discard incomplete tails and the final row's outgoing action.
    Simulator state/proprio are never read or returned on this path.
    """
    def __init__(self, prepared: PreparedData, split: str, horizon: int = 1, stride: int = 1):
        window_count(1, horizon, stride)  # Validate even for an empty split.
        self.prepared, self.split = prepared, split
        self.horizon, self.stride = horizon, stride
        self.episodes = prepared._episodes(split)
        counts = [window_count(ep['length'], horizon, stride) for ep in self.episodes]
        self.cumulative = list(accumulate(counts, initial=0))
        self.statistics = {
            'assigned_episodes': len(self.episodes),
            'episodes_without_windows': counts.count(0),
            'windows': self.cumulative[-1],
            'dropped_tail_transitions': sum((ep['length'] - 1) % FRAMESKIP for ep in self.episodes),
            'terminal_action_rows_ignored': len(self.episodes),
        }
        LOG.info('Trajectory split=%s horizon=%d stride=%d phase=0 frameskip=5 %s',
                 split, horizon, stride, self.statistics)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict:
        position, window = _locate(self.cumulative, index)
        episode = self.episodes[position]
        start = window * self.stride * FRAMESKIP
        frame_indices = tuple(start + i * FRAMESKIP for i in range(self.horizon + 1))
        offset = episode['offset']
        with self.prepared._open() as handle:
            frames = np.stack([resize_rgb(handle['pixels'][offset + i]) for i in frame_indices])
            actions = handle['action'][offset + start:offset + frame_indices[-1]].reshape(self.horizon, 10)
        metadata = self.prepared._metadata(episode['episode_id'], self.split)
        metadata['frame_indices'] = frame_indices
        return {'frames': frames, 'actions': actions, 'metadata': metadata}


class RawFrameDataset:
    """Explicit access to every raw image in a split, without an encoder policy."""
    def __init__(self, prepared: PreparedData, split: str):
        self.prepared, self.split = prepared, split
        self.episodes = prepared._episodes(split)
        self.cumulative = list(accumulate((ep['length'] for ep in self.episodes), initial=0))

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict:
        position, frame_index = _locate(self.cumulative, index)
        episode = self.episodes[position]
        with self.prepared._open() as handle:
            image = resize_rgb(handle['pixels'][episode['offset'] + frame_index])
        metadata = self.prepared._metadata(episode['episode_id'], self.split)
        metadata['frame_index'] = frame_index
        return {'image': image, 'metadata': metadata}


class MeasurementReader:
    """Simulator-derived arrays for probes/evaluation only; never model tokens.

    state columns: pusher x,y; T x,y; T angle in radians modulo 2pi;
    pusher velocity x,y. proprio contains state columns 0,1,5,6.
    """
    def __init__(self, prepared: PreparedData, *, purpose: str):
        if purpose not in ('probe', 'evaluation'):
            raise ValueError('Measurements require purpose=probe or evaluation')
        self.prepared, self.purpose = prepared, purpose

    def read(self, episode_id: int, frame_indices: Sequence[int]) -> dict:
        episodes = self.prepared.manifest['episodes']
        if type(episode_id) is not int or not 0 <= episode_id < len(episodes):
            raise IndexError(episode_id)
        episode = episodes[episode_id]
        indices = list(frame_indices)
        if not indices or any(type(i) is not int or not 0 <= i < episode['length'] for i in indices):
            raise IndexError('Measurement frame indices outside episode')
        with self.prepared._open() as handle:
            # Separate row reads also preserve caller order/repeated indices.
            measurements = {key: np.stack([handle[key][episode['offset'] + i] for i in indices])
                            for key in ('state', 'proprio')}
        measurements['metadata'] = self.prepared._metadata(episode_id, self.prepared.episode_splits[episode_id])
        return measurements
