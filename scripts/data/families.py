"""Use initial simulator state solely to prevent shared-start split leakage."""
from collections import Counter
import hashlib
import re

import h5py
import numpy as np

GROUPING_METHOD = 'exact-initial-state-float32-le-v1'
PUSHT_HDF5_SHA256 = 'b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb'


def family_structure(episodes: list[dict], families: list[dict]) -> dict:
    sizes = Counter(len(family['episode_ids']) for family in families)
    return {
        'family_count': len(families),
        'size_histogram': {str(size): count for size, count in sorted(sizes.items())},
        'contiguous_members': all(
            f['episode_ids'] == list(range(f['episode_ids'][0], f['episode_ids'][-1] + 1))
            for f in families
        ),
        'equal_lengths_within_families': all(
            len({episodes[i]['length'] for i in f['episode_ids']}) == 1 for f in families
        ),
    }


def inspect_families(handle: h5py.File, episodes: list[dict]) -> dict:
    """Group bitwise-identical initial 7-D rows; stable IDs follow first occurrence.

    Canonical little-endian float32 hashes are portable across machine endianness.
    The initial-state values themselves are not persisted in the split manifest.
    """
    # Read the small numeric column once instead of 18k slow HDF5 fancy selections.
    initial_states = handle['state'][:][[ep['offset'] for ep in episodes]]
    if not np.isfinite(initial_states).all():
        raise ValueError('Initial family states must be finite')
    hashes = [hashlib.sha256(np.asarray(row, dtype='<f4').tobytes()).hexdigest() for row in initial_states]
    by_hash = {}
    for episode_id, digest in enumerate(hashes):
        by_hash.setdefault(digest, []).append(episode_id)
    families = [{'family_id': family_id, 'initial_state_sha256': digest, 'episode_ids': ids}
                for family_id, (digest, ids) in enumerate(by_hash.items())]
    return {
        'grouping_method': GROUPING_METHOD, 'families': families,
        'episode_initial_state_sha256': hashes,
        'family_structure': family_structure(episodes, families),
    }


def require_fields(value: dict, fields: set[str], context: str) -> None:
    """Report malformed manifest objects at their boundary, before indexing."""
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    missing = fields - value.keys()
    if missing:
        raise ValueError(f'{context} missing required fields: {", ".join(sorted(missing))}')


def _validate_family_record(family: dict, family_id: int, hashes: list[str]) -> None:
    context = f'family {family_id}'
    require_fields(family, {'family_id', 'initial_state_sha256', 'episode_ids'}, context)
    if type(family['family_id']) is not int or family['family_id'] != family_id:
        raise ValueError(f'{context}: family_id must equal its ordered position')
    ids = family['episode_ids']
    if not isinstance(ids, list) or not ids:
        raise ValueError(f'{context}: episode_ids must be a nonempty list')
    for episode_id in ids:
        if type(episode_id) is not int or not 0 <= episode_id < len(hashes):
            raise ValueError(f'{context}: invalid episode_id {episode_id!r}')
    if ids != sorted(set(ids)):
        raise ValueError(f'{context}: episode_ids must be sorted and unique')
    for episode_id in ids:
        if hashes[episode_id] != family['initial_state_sha256']:
            raise ValueError(f'{context}: initial_state_sha256 does not match episode {episode_id}')


def _validate_known_pusht_structure(episodes: list[dict], families: list[dict]) -> None:
    if len(families) != 185:
        raise ValueError('Verified PushT source requires 185 families')
    for family in families:
        start = 101 * family['family_id']
        if family['episode_ids'] != list(range(start, start + 101)):
            raise ValueError(f"family {family['family_id']}: verified PushT requires 101 contiguous episodes")
        if len({episodes[i]['length'] for i in family['episode_ids']}) != 1:
            raise ValueError(f"family {family['family_id']}: verified PushT requires equal episode lengths")


def validate_families(episodes: list[dict], evidence: dict, hdf5_sha256: str) -> None:
    """Reject inconsistent membership and verify this archive's observed 185x101 rule."""
    require_fields(evidence, {
        'grouping_method', 'families', 'episode_initial_state_sha256', 'family_structure',
    }, 'family evidence')
    if evidence['grouping_method'] != GROUPING_METHOD:
        raise ValueError(f'Unsupported grouping_method: {evidence["grouping_method"]!r}')
    families = evidence['families']
    if not isinstance(families, list) or not families:
        raise ValueError('families must be a nonempty list')
    hashes = evidence['episode_initial_state_sha256']
    if not isinstance(hashes, list) or len(hashes) != len(episodes):
        raise ValueError('episode_initial_state_sha256 must have one hash per episode')
    for episode_id, digest in enumerate(hashes):
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError(f'episode_initial_state_sha256[{episode_id}] must be a SHA-256 hex digest')
    seen, seen_hashes, previous_first = [], set(), -1
    for family_id, family in enumerate(families):
        _validate_family_record(family, family_id, hashes)
        ids, digest = family['episode_ids'], family['initial_state_sha256']
        if ids[0] <= previous_first:
            raise ValueError(f'family {family_id}: families must follow first-episode order')
        if digest in seen_hashes:
            raise ValueError(f'family {family_id}: initial_state_sha256 already belongs to another family')
        previous_first = ids[0]
        seen.extend(ids)
        seen_hashes.add(digest)
    if sorted(seen) != list(range(len(episodes))):
        raise ValueError('Family membership must cover every episode exactly once')
    if evidence['family_structure'] != family_structure(episodes, families):
        raise ValueError('family_structure summary does not match membership')
    if hdf5_sha256 == PUSHT_HDF5_SHA256:
        _validate_known_pusht_structure(episodes, families)
