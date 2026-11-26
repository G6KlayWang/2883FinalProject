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
        'contiguous_members': all(f['episode_ids'] == list(range(f['episode_ids'][0], f['episode_ids'][-1] + 1)) for f in families),
        'equal_lengths_within_families': all(len({episodes[i]['length'] for i in f['episode_ids']}) == 1 for f in families),
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


def validate_families(episodes: list[dict], evidence: dict, hdf5_sha256: str) -> None:
    """Reject inconsistent membership and verify this archive's observed 185x101 rule."""
    if evidence.get('grouping_method') != GROUPING_METHOD:
        raise ValueError('Unsupported family grouping method')
    families, hashes = evidence['families'], evidence['episode_initial_state_sha256']
    if not families or len(hashes) != len(episodes) or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes):
        raise ValueError('Missing or invalid family initial-state hashes')
    seen, seen_hashes, previous_first = [], set(), -1
    for family_id, family in enumerate(families):
        ids, digest = family['episode_ids'], family['initial_state_sha256']
        if (type(family['family_id']) is not int or family['family_id'] != family_id
                or not ids or any(type(i) is not int or not 0 <= i < len(episodes) for i in ids)
                or ids != sorted(set(ids)) or ids[0] <= previous_first
                or digest in seen_hashes or any(hashes[i] != digest for i in ids)):
            raise ValueError('Inconsistent family IDs, hashes or episode membership')
        previous_first = ids[0]
        seen.extend(ids)
        seen_hashes.add(digest)
    if sorted(seen) != list(range(len(episodes))):
        raise ValueError('Family membership must cover every episode exactly once')
    structure = family_structure(episodes, families)
    if evidence.get('family_structure') != structure:
        raise ValueError('Family structure summary does not match membership')
    if hdf5_sha256 == PUSHT_HDF5_SHA256:
        expected = [list(range(101 * i, 101 * (i + 1))) for i in range(185)]
        if ([f['episode_ids'] for f in families] != expected
                or not structure['equal_lengths_within_families']):
            raise ValueError('Verified PushT source requires 185 contiguous families of 101 equal-length episodes')
