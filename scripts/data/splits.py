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
from scripts.data.families import require_fields, validate_families

ALGORITHM = 'sha256-rank-v1'
DEFAULT_SEED = 42


def split_episodes(episode_ids: Sequence[int], seed: int = DEFAULT_SEED) -> dict[str, list[int]]:
    """Rank SHA256('pusht-split-v1:<seed>:<id>'); reserve floor(N/20) each.

    SHA-256 ranking avoids dependence on a library RNG or caller iteration order.
    Lists within splits are sorted for locality; all IDs are split before filtering.
    """
    ids = list(episode_ids)
    if type(seed) is not int:
        raise ValueError('Split seed must be an integer')
    if not ids:
        raise ValueError('Episode IDs must be nonempty')
    for episode_id in ids:
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError(f'Episode ID must be a nonnegative integer: {episode_id!r}')
    if len(ids) != len(set(ids)):
        raise ValueError('Episode IDs must be unique')
    ranked = sorted(ids, key=lambda i: (hashlib.sha256(f'pusht-split-v1:{seed}:{i}'.encode('ascii')).digest(), i))
    heldout = len(ids) // 20
    train_end, validation_end = len(ids) - 2 * heldout, len(ids) - heldout
    return {
        'train': sorted(ranked[:train_end]),
        'validation': sorted(ranked[train_end:validation_end]),
        'test': sorted(ranked[validation_end:]),
    }


def _validate_episodes(episodes: list[dict]) -> None:
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('manifest.episodes must be a nonempty list')
    next_offset = 0
    fields = {'episode_id', 'offset', 'length'}
    for episode_id, episode in enumerate(episodes):
        context = f'episode {episode_id}'
        require_fields(episode, fields, context)
        if set(episode) != fields:
            raise ValueError(f'{context}: unexpected fields {sorted(set(episode) - fields)}')
        for field in fields:
            if type(episode[field]) is not int:
                raise ValueError(f'{context}: {field} must be an integer')
        if episode['episode_id'] != episode_id:
            raise ValueError(f'{context}: episode_id must equal its ordered position')
        if episode['offset'] != next_offset:
            raise ValueError(f'{context}: offset must be {next_offset}')
        if episode['length'] <= 0:
            raise ValueError(f'{context}: length must be positive')
        next_offset += episode['length']


def _validate_source(source: dict) -> None:
    require_fields(source, {
        'archive_sha256', 'hdf5_sha256', 'archive_size_bytes', 'hdf5_size_bytes', 'dataset_revision',
    }, 'manifest.source')
    for key in ('archive_sha256', 'hdf5_sha256'):
        if not isinstance(source[key], str) or not re.fullmatch('[0-9a-f]{64}', source[key]):
            raise ValueError(f'manifest.source.{key} must be a SHA-256 hex digest')
    for key in ('archive_size_bytes', 'hdf5_size_bytes'):
        if type(source[key]) is not int or source[key] <= 0:
            raise ValueError(f'manifest.source.{key} must be a positive integer')
    if not isinstance(source['dataset_revision'], str) or not source['dataset_revision'].strip():
        raise ValueError('manifest.source.dataset_revision must be a nonempty string')


def _expected_splits(manifest: dict) -> dict[str, list[int]]:
    if manifest['split_unit'] == 'episode':
        return split_episodes([ep['episode_id'] for ep in manifest['episodes']], manifest['seed'])
    if manifest['split_unit'] != 'family':
        raise ValueError(f'Unknown split_unit: {manifest["split_unit"]!r}')
    validate_families(manifest['episodes'], manifest, manifest['source']['hdf5_sha256'])
    family_splits = split_episodes([f['family_id'] for f in manifest['families']], manifest['seed'])
    if manifest.get('family_splits') != family_splits:
        raise ValueError('family_splits do not reproduce the seed/algorithm')
    return {split: sorted(i for family_id in ids for i in manifest['families'][family_id]['episode_ids'])
            for split, ids in family_splits.items()}


def _validate_duplicates(groups: list[list[int]], splits: dict[str, list[int]]) -> None:
    if not isinstance(groups, list):
        raise ValueError('duplicate_episode_groups must be a list')
    membership = {episode_id: split for split, group in splits.items() for episode_id in group}
    for group_id, group in enumerate(groups):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError(f'duplicate_episode_groups[{group_id}] must list at least two episode IDs')
        for episode_id in group:
            if type(episode_id) is not int or episode_id not in membership:
                raise ValueError(f'duplicate_episode_groups[{group_id}]: invalid episode ID {episode_id!r}')
        if len({membership[i] for i in group}) > 1:
            raise ValueError(f'Exact duplicate content crosses splits: {group}')


def validate_manifest(manifest: dict) -> None:
    """Reject altered assignments, invalid provenance and cross-split duplicates."""
    require_fields(manifest, {
        'manifest_version', 'split_algorithm', 'seed', 'source', 'code_revision',
        'episodes', 'split_unit', 'splits', 'duplicate_episode_groups',
    }, 'manifest')
    if manifest['manifest_version'] != 2:
        raise ValueError('Unsupported manifest_version; expected 2')
    if manifest['split_algorithm'] != ALGORITHM:
        raise ValueError(f'Unsupported split_algorithm; expected {ALGORITHM}')
    if type(manifest['seed']) is not int or manifest['seed'] != DEFAULT_SEED:
        raise ValueError('This milestone requires integer seed 42')
    _validate_source(manifest['source'])
    if not isinstance(manifest['code_revision'], str) or not manifest['code_revision'].strip():
        raise ValueError('manifest.code_revision must be a nonempty string')
    _validate_episodes(manifest['episodes'])
    expected = _expected_splits(manifest)
    if manifest['splits'] != expected:
        raise ValueError('Stored split assignments do not reproduce the seed/algorithm')
    _validate_duplicates(manifest['duplicate_episode_groups'], expected)


def build_manifest(schema: dict, download: dict, code_revision: str, *, split_unit: str = 'family') -> dict:
    """Bind all assignments to inspected content and the verified source archive."""
    require_fields(schema, {
        'schema_version', 'hdf5_sha256', 'hdf5_size_bytes', 'episodes', 'duplicate_episode_groups',
    }, 'schema')
    require_fields(download, {'sha256', 'size_bytes', 'dataset_revision'}, 'download manifest')
    schema = copy.deepcopy(schema)
    _validate_episodes(schema['episodes'])
    if schema.get('schema_version') != 1:
        raise ValueError('Unsupported schema report')
    manifest = {
        'manifest_version': 2, 'split_algorithm': ALGORITHM, 'seed': DEFAULT_SEED, 'split_unit': split_unit,
        'source': {
            'archive_sha256': download['sha256'], 'archive_size_bytes': download['size_bytes'],
            'dataset_revision': download['dataset_revision'],
            'hdf5_sha256': schema['hdf5_sha256'], 'hdf5_size_bytes': schema['hdf5_size_bytes'],
        },
        'code_revision': code_revision,
        'episodes': schema['episodes'],
        'duplicate_episode_groups': schema['duplicate_episode_groups'],
        'splits': {},
    }
    if split_unit == 'family':
        require_fields(schema, {
            'grouping_method', 'families', 'episode_initial_state_sha256', 'family_structure',
        }, 'schema family evidence')
        validate_families(schema['episodes'], schema, schema['hdf5_sha256'])
        for key in ('grouping_method', 'families', 'episode_initial_state_sha256', 'family_structure'):
            manifest[key] = schema[key]
        manifest['family_splits'] = split_episodes([f['family_id'] for f in schema['families']], manifest['seed'])
        manifest['splits'] = {
            split: sorted(i for family_id in ids for i in schema['families'][family_id]['episode_ids'])
            for split, ids in manifest['family_splits'].items()
        }
    elif split_unit == 'episode':
        manifest['splits'] = split_episodes([ep['episode_id'] for ep in schema['episodes']], manifest['seed'])
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
    manifest = build_manifest(
        json.loads(args.schema.read_text()), json.loads(args.download_manifest.read_text()),
        args.code_revision, split_unit=args.split_unit,
    )
    write_json_new(args.output, manifest)
    logging.info('SPLIT seed=%d algorithm=%s counts=%s source=%s code=%s', manifest['seed'], ALGORITHM,
                 {k: len(v) for k, v in manifest['splits'].items()}, manifest['source'], args.code_revision)


if __name__ == '__main__':
    main()
