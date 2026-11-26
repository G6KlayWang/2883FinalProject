"""Verify representative K2 windows in every split and publish a portable report."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.data.families import PUSHT_HDF5_SHA256
from scripts.data.file_io import sha256_file, write_json_new
from scripts.data.loader import MeasurementReader, PreparedData, RawFrameDataset, TrajectoryDataset

HORIZON = 2


def _check_array(values: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(values, np.ndarray) or values.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain finite float32 values')


def _check_sample(sample: dict, prepared: PreparedData, split: str) -> None:
    if set(sample) != {'frames', 'actions', 'metadata'}:
        raise ValueError('Unexpected model fields; only frames, actions and metadata are allowed')
    _check_array(sample['frames'], (3, 3, 96, 96), 'frames')
    _check_array(sample['actions'], (2, 10), 'actions')
    if sample['frames'].min() < 0 or sample['frames'].max() > 1:
        raise ValueError('frames must be in [0,1]')
    metadata = sample['metadata']
    if set(metadata) != {'episode_id', 'split', 'split_unit', 'hdf5_sha256', 'frame_indices'}:
        raise ValueError('Unexpected model metadata fields')
    if metadata['split'] != split or metadata['episode_id'] not in prepared.manifest['splits'][split]:
        raise ValueError(f'Sample episode does not belong to split {split}')
    if metadata['split_unit'] != prepared.manifest['split_unit']:
        raise ValueError('Sample split_unit differs from the manifest')
    if metadata['hdf5_sha256'] != prepared.manifest['source']['hdf5_sha256']:
        raise ValueError('Sample source hash differs from the manifest')
    indices = tuple(metadata['frame_indices'])
    if len(indices) != 3 or indices != tuple(range(indices[0], indices[0] + 11, 5)):
        raise ValueError('K2 frames must be five stored steps apart')
    length = prepared.manifest['episodes'][metadata['episode_id']]['length']
    if indices[0] < 0 or indices[0] % 5 or indices[-1] >= length:
        raise ValueError('K2 frames must be phase zero and contained in one episode')


def _check_measurements(reader: MeasurementReader, sample: dict) -> dict:
    metadata = sample['metadata']
    labels = reader.read(metadata['episode_id'], metadata['frame_indices'])
    if set(labels) != {'state', 'proprio', 'metadata'}:
        raise ValueError('Unexpected measurement fields')
    _check_array(labels['state'], (3, 7), 'measurement state')
    _check_array(labels['proprio'], (3, 4), 'measurement proprio')
    if not np.array_equal(labels['proprio'], labels['state'][:, [0, 1, 5, 6]]):
        raise ValueError('Measurement proprio does not match state columns 0,1,5,6')
    expected_metadata = {key: value for key, value in metadata.items() if key != 'frame_indices'}
    if labels['metadata'] != expected_metadata:
        raise ValueError('Measurement provenance differs from the model sample')
    return {key: list(labels[key].shape) for key in ('state', 'proprio')}


def run_smoke(hdf5: Path, manifest_path: Path, output: Path, *,
              allow_diagnostic_episode_split: bool = False) -> dict:
    """Check first/middle/last K2 windows; hash HDF5 once for all split readers.

    Each split must contain K2 windows. Selection roles may reference the same
    index for tiny fixtures. No simulator measurements are stored in the report.
    """
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    manifest = json.loads(Path(manifest_path).read_text())
    prepared = PreparedData(hdf5, manifest, allow_diagnostic_episode_split=allow_diagnostic_episode_split)
    reader = MeasurementReader(prepared, purpose='evaluation')
    report = {
        'report_version': 1, 'status': 'passed', 'horizon': HORIZON, 'stride': 1, 'frameskip': 5,
        'source': manifest['source'], 'manifest_sha256': sha256_file(manifest_path),
        'preparation_code_revision': manifest['code_revision'],
        'smoke_code_sha256': sha256_file(Path(__file__)),
        'split_unit': manifest['split_unit'], 'seed': manifest['seed'],
        'split_algorithm': manifest['split_algorithm'],
        'diagnostic_episode_split': prepared.diagnostic_episode_split,
        'verified_pusht_source': manifest['source']['hdf5_sha256'] == PUSHT_HDF5_SHA256,
        'splits': {},
    }
    for split in ('train', 'validation', 'test'):
        dataset = TrajectoryDataset(prepared, split, horizon=HORIZON)
        if not len(dataset):
            raise ValueError(f'Split {split} has no K2 windows')
        samples = []
        for selection, index in [('first', 0), ('middle', len(dataset) // 2), ('last', len(dataset) - 1)]:
            sample = dataset[index]
            _check_sample(sample, prepared, split)
            samples.append({
                'selection': selection, 'index': index, 'metadata': sample['metadata'],
                'model_fields': sorted(sample),
                'shapes': {key: list(sample[key].shape) for key in ('frames', 'actions')},
            })
            if selection == 'first':
                measurement_shapes = _check_measurements(reader, sample)
        report['splits'][split] = {
            **dataset.statistics, 'samples': samples,
            'raw_frames': len(RawFrameDataset(prepared, split)),
            'measurement_purpose': 'evaluation', 'measurement_shapes': measurement_shapes,
        }
    write_json_new(output, report)
    logging.info('SMOKE passed all splits; report=%s source=%s', output, report['source'])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-diagnostic-episode-split', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    run_smoke(args.hdf5, args.manifest, args.output,
              allow_diagnostic_episode_split=args.allow_diagnostic_episode_split)


if __name__ == '__main__':
    main()
