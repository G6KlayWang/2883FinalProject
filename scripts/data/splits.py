"""Pure deterministic episode splits and portable preparation manifests."""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import re
from collections.abc import Sequence

from scripts.data.file_io import write_json_new
from scripts.data.families import validate_families

ALGORITHM = 'sha256-rank-v1'


def split_episodes(episode_ids: Sequence[int], seed: int = 42) -> dict[str, list[int]]:
    """Rank SHA256('pusht-split-v1:<seed>:<id>'); reserve floor(N/20) each.

    SHA-256 ranking avoids dependence on a library RNG or caller iteration order.
    Lists within splits are sorted for locality; all IDs are split before filtering.
    """
    ids = list(episode_ids)
    if type(seed) is not int:
        raise ValueError('Split seed must be an integer')
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Episode IDs must be nonempty, unique nonnegative integers')
    ranked = sorted(ids, key=lambda i: (hashlib.sha256(f'pusht-split-v1:{seed}:{i}'.encode('ascii')).digest(), i))
    heldout = len(ids) // 20
    train_end, validation_end = len(ids) - 2 * heldout, len(ids) - heldout
    return {
        'train': sorted(ranked[:train_end]),
        'validation': sorted(ranked[train_end:validation_end]),
        'test': sorted(ranked[validation_end:]),
    }


def _validate_episodes(episodes: list[dict]) -> None:
    if not episodes:
        raise ValueError('Manifest requires episodes')
    next_offset = 0
    for episode_id, episode in enumerate(episodes):
        if (set(episode) != {'episode_id', 'offset', 'length'}
                or any(type(v) is not int for v in episode.values())
                or episode['episode_id'] != episode_id
                or episode['offset'] != next_offset or episode['length'] <= 0):
            raise ValueError('Invalid manifest episode IDs, offsets or lengths')
        next_offset += episode['length']


def _validate_source(source: dict) -> None:
    for key in ('archive_sha256', 'hdf5_sha256'):
        if not isinstance(source.get(key), str) or not re.fullmatch('[0-9a-f]{64}', source[key]):
            raise ValueError(f'Invalid source {key}')
    for key in ('archive_size_bytes', 'hdf5_size_bytes'):
        if type(source.get(key)) is not int or source[key] <= 0:
            raise ValueError(f'Invalid source {key}')
    if not isinstance(source.get('dataset_revision'), str) or not source['dataset_revision']:
        raise ValueError('Missing dataset revision')


def validate_manifest(manifest: dict) -> None:
    """Reject altered assignments, invalid provenance and cross-split duplicates."""
    if manifest.get('manifest_version') != 2 or manifest.get('split_algorithm') != ALGORITHM:
        raise ValueError('Unsupported split manifest version or algorithm')
    if manifest.get('seed') != 42 or type(manifest.get('seed')) is not int:
        raise ValueError('This milestone requires split seed 42')
    _validate_source(manifest['source'])
    if not isinstance(manifest.get('code_revision'), str) or not manifest['code_revision'].strip():
        raise ValueError('Missing code revision')
    _validate_episodes(manifest['episodes'])
    if manifest.get('split_unit') == 'family':
        validate_families(manifest['episodes'], manifest, manifest['source']['hdf5_sha256'])
        family_splits = split_episodes([f['family_id'] for f in manifest['families']], manifest['seed'])
        if manifest.get('family_splits') != family_splits:
            raise ValueError('Stored family splits do not reproduce the seed/algorithm')
        expected = {split: sorted(i for family_id in ids for i in manifest['families'][family_id]['episode_ids'])
                    for split, ids in family_splits.items()}
    elif manifest.get('split_unit') == 'episode':
        expected = split_episodes([ep['episode_id'] for ep in manifest['episodes']], manifest['seed'])
    else:
        raise ValueError('Unknown split unit')
    if manifest.get('splits') != expected:
        raise ValueError('Stored split assignments do not reproduce the seed/algorithm')
    membership = {episode_id: split for split, group in expected.items() for episode_id in group}
    for group in manifest['duplicate_episode_groups']:
        if any(type(i) is not int or i not in membership for i in group):
            raise ValueError('Invalid duplicate episode IDs')
        if len({membership[i] for i in group}) > 1:
            raise ValueError(f'Exact duplicate content crosses splits: {group}')


def build_manifest(schema: dict, download: dict, code_revision: str, *, split_unit: str = 'family') -> dict:
    """Bind all assignments to inspected content and the verified source archive."""
    schema = copy.deepcopy(schema)
    if schema.get('schema_version') != 1:
        raise ValueError('Unsupported schema report')
    manifest = {
        'manifest_version': 2, 'split_algorithm': ALGORITHM, 'seed': 42, 'split_unit': split_unit,
        'source': {
            'archive_sha256': download['sha256'], 'archive_size_bytes': download['size_bytes'],
            'dataset_revision': download['dataset_revision'],
            'hdf5_sha256': schema['hdf5_sha256'], 'hdf5_size_bytes': schema['hdf5_size_bytes'],
        },
        'code_revision': code_revision,
        'episodes': schema['episodes'],
        'duplicate_episode_groups': schema['duplicate_episode_groups'],
        'splits': split_episodes([ep['episode_id'] for ep in schema['episodes']]),
    }
    if split_unit == 'family':
        for key in ('grouping_method', 'families', 'episode_initial_state_sha256', 'family_structure'):
            manifest[key] = schema[key]
        manifest['family_splits'] = split_episodes([f['family_id'] for f in schema['families']])
        manifest['splits'] = {
            split: sorted(i for family_id in ids for i in schema['families'][family_id]['episode_ids'])
            for split, ids in manifest['family_splits'].items()
        }
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-unit', choices=('family', 'episode'), default='family')
    parser.add_argument('--schema', type=Path, required=True)
    parser.add_argument('--download-manifest', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    manifest = build_manifest(json.loads(args.schema.read_text()), json.loads(args.download_manifest.read_text()), args.code_revision, split_unit=args.split_unit)
    write_json_new(args.output, manifest)
    logging.info('SPLIT seed=%d algorithm=%s counts=%s source=%s code=%s', manifest['seed'], ALGORITHM,
                 {k: len(v) for k, v in manifest['splits'].items()}, manifest['source'], args.code_revision)


if __name__ == '__main__':
    main()
