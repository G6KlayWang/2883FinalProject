# PushT data tools

Downloads the research dataset `quentinll/lewm-pusht/pusht_expert_train.h5.zst`
to `/root/autodl-tmp/pusht`. The archive stays compressed. The remote filesystem's
reported free space, rather than the advertised host capacity, determines whether
it fits. Allow additional space before extracting or caching encoder features.

## Design and implementation plan

Use Hugging Face's resumable client, pin each download to a dataset commit,
then check file size and the published SHA-256. HTTP makes partial-file byte counts
observable; optional Xet mode uses parallel transfers and the client's progress bar.
Save provenance in a JSON manifest.
Keep credentials and data outside Git. Sync committed code using a Git bundle over
SSH, which does not require placing GitHub credentials on the GPU server.

1. Test acceptance of valid bytes and rejection of truncated/corrupt downloads.
2. Implement the downloader and AutoDL acceleration wrapper.
3. Commit only the data tools and ignore rules; transfer that commit to the server.
4. Run detached, monitor logs, and verify the archive before reporting success.

## Install and run on AutoDL

From the repository root, using your chosen Python environment:

```bash
source /etc/network_turbo
python3 -m pip install --index-url https://pypi.org/simple -r scripts/data/requirements.txt
mkdir -p /root/autodl-tmp/pusht
nohup bash scripts/data/download_pusht.sh \
  --destination /root/autodl-tmp/pusht \
  > /root/autodl-tmp/pusht/download.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/pusht/download.log
```

The wrapper runs `source /etc/network_turbo` before network requests. Set
`DATA_PYTHON=/path/to/python` if Python is not on the SSH shell's PATH, and use that
interpreter for installation too. This instance required acceleration and the
official PyPI index for installation; its default Aliyun index returned no packages.
AutoDL notes that the proxy can slow other package mirrors.

Progress appears every 15 seconds with bytes, percentage, speed, and estimated time.
Add `--transport xet` for Hugging Face's high-performance parallel transfer mode.
In Xet mode, read progress from the client's log, not the preallocated file size.
Use a fresh destination when changing transport with an incomplete download.
Rerun the same command after interruption to resume. A lock prevents concurrent
downloads to the same destination. Success requires a `VERIFIED` log line and
`download_manifest.json`, containing the source revision, code commit, size, and
SHA-256. A completed transfer alone is not verification. A corrupt existing archive
causes failure rather than being silently accepted or deleted.

## Verify locally

```bash
python3 -m unittest discover -s scripts/data/tests -v
bash -n scripts/data/download_pusht.sh
```

## Sync committed code

Create a bundle locally with `git bundle create /tmp/pusht-code.bundle HEAD`,
transfer it using `scp -P PORT`, then on the server use
`git clone /path/to/pusht-code.bundle /root/autodl-tmp/2883FinalProject`.
For subsequent bundles, fetch their `HEAD` and merge with `--ff-only`; never reset
remote work. Both checkouts should report the same `git rev-parse HEAD`.

## Prepare an existing archive (no training)

Run from the repository root with Python 3.12 or later in an isolated environment.
The existing downloader and wrapper retain their behavior. If the archive is
already available, use it directly; preparation makes no dataset network requests.
The verified server archive is `/root/autodl-tmp/pusht_expert_train.h5.zst`;
its download manifest may need to be supplied separately from the archive directory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/data/requirements.txt
# Set these to existing local files; output belongs in ignored storage.
PUSHT_ARCHIVE='/absolute/path/pusht_expert_train.h5.zst'
PUSHT_DOWNLOAD_MANIFEST='/absolute/path/download_manifest.json'
PUSHT_PREP_DIR='data/pusht'
mkdir -p "$PUSHT_PREP_DIR"
.venv/bin/python -m scripts.data.extract_pusht \
  --archive "$PUSHT_ARCHIVE" --download-manifest "$PUSHT_DOWNLOAD_MANIFEST" \
  --output "$PUSHT_PREP_DIR/pusht_expert_train.h5"
.venv/bin/python -m scripts.data.schema \
  --hdf5 "$PUSHT_PREP_DIR/pusht_expert_train.h5" \
  --output "$PUSHT_PREP_DIR/schema.json"
.venv/bin/python -m scripts.data.audit_alignment \
  --hdf5 "$PUSHT_PREP_DIR/pusht_expert_train.h5" \
  --output "$PUSHT_PREP_DIR/alignment.json"
# Set to the exact reviewed commit; do not reuse a prior candidate's revision.
PUSHT_CODE_REVISION='REPLACE_WITH_REVIEWED_COMMIT'
.venv/bin/python -m scripts.data.splits \
  --schema "$PUSHT_PREP_DIR/schema.json" \
  --download-manifest "$PUSHT_DOWNLOAD_MANIFEST" \
  --code-revision "$PUSHT_CODE_REVISION" \
  --output "$PUSHT_PREP_DIR/split_manifest.json"
```

On the server, use the data volume for the venv, HDF5 and reports, rather than the
30 GiB root overlay. The server coordinator runs these commands only after review
and exact-commit synchronization. No GPU or PyTorch is required for preparation.

Extraction verifies the full archive size and SHA-256 before decompressing, checks
free space for the declared output size plus a 1 GiB reserve, bounds the decoder
window to 128 MiB, and verifies frame completion, output size and HDF5 readability.
It hashes the input again while decompressing and prints an extraction receipt
with both hashes. Publication uses a unique sibling `.incomplete` file and an
atomic hard link that cannot replace an existing output. Ordinary exceptions and
interruptions clean up the temporary file; a hard kill can leave a stale partial.
A partial is never reused. Preserve the archive and investigate existing outputs;
all commands refuse to overwrite completed artifacts. Use a fresh output name
when intentionally regenerating a report. No decompression resume is attempted.

## Observed schema and provenance

Verified archive: 13,136,247,974 bytes, SHA-256
`7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`,
HF dataset revision `655cd446b9929369d7d406001da85c15d1457850`.
Extracted HDF5: 46,300,921,856 bytes, SHA-256
`b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
The split is carved from this one upstream `train` archive; no upstream validation
archive is mixed into it.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| pixels | (2336736,224,224,3) | uint8 | RGB, HWC |
| action | (2336736,2) | float32 | outgoing relative pusher target displacement /100 |
| state | (2336736,7) | float32 | pusher x,y; T x,y; T angle; pusher vx,vy |
| proprio | (2336736,4) | float32 | state columns 0,1,5,6 |
| episode_idx / step_idx | (2336736,) | int64 | episode ID / within-episode row |
| ep_len | (18685,) | int32 | positive episode lengths |
| ep_offset | (18685,) | int64 | contiguous episode start rows |

Episode lengths range 49–246, median 123. There are no root/dataset attributes.
Pixels use Blosc filter 32001 and chunks of 100 images; `hdf5plugin` registers the
codec. Numeric columns use 1000-row chunks. Schema inspection verifies axes,
dtypes, contiguous boundaries, episode/step indices, finite numeric values,
codec readability, full-file identity, families, and exact duplicate episodes.
The duplicate audit hashes complete action sequences and then first images for
matching candidates. It detects exact duplicates under that definition, not
near-equivalent trajectories; the family audit handles shared starts separately.

## User-approved whole-family split

On 2026-11-26 the user amended the initial random episode split to keep whole
families together. All 18,685 episodes belong to 185 families of 101 noisy replays.
Grouping hashes the exact initial seven-field state row in canonical little-endian
float32 bytes. Stable family IDs follow first occurrence in archive episode order;
the raw state values are not written to the manifest. For this HDF5 hash, the audit
requires 185 groups of 101 contiguous equal-length episodes and checks the observed
`episode_id // 101` structure against the inferred groups. Grouping is never
inferred from integer division alone.

`sha256-rank-v1` ranks IDs by SHA256 of ASCII
`pusht-split-v1:<seed>:<id>`, with seed 42. Validation/test each take floor(N/20)
families; train takes the remainder. Membership lists are sorted. This gives
167/9/9 families and 16,867/909/909 episodes. Split every episode before any length
filter. The manifest persists source archive/HDF5 hashes and sizes, source
revision, code revision, grouping method, per-episode initial-state hashes, family
membership, family assignments and episode assignments. Validation recomputes
assignments and rejects altered, omitted or overlapping members and cross-split
exact duplicates. `PreparedData` independently compares grouping with the HDF5.

Validation selects regularizer weights; test supplies final reported measurements.
Each held-out set has only 9 distinct starting configurations: evaluation over 50
or more episodes reuses starts and reports must state that limit. No statistics
or normalizers are fitted here. `--split-unit episode` retains the earlier
explicit option for diagnostics; it is not the approved research split.

## Loader interfaces and temporal alignment

```python
import json
from pathlib import Path
from scripts.data.loader import PreparedData, TrajectoryDataset, RawFrameDataset, MeasurementReader

manifest = json.loads(Path('data/pusht/split_manifest.json').read_text())
prepared = PreparedData(Path('data/pusht/pusht_expert_train.h5'), manifest)
windows = TrajectoryDataset(prepared, 'train', horizon=2, stride=1)
sample = windows[0]
# sample['frames']: (3,3,96,96), float32 RGB CHW [0,1]
# sample['actions']: (2,10), raw float32; reshape to (2,5,2) if desired
# sample['metadata']: episode_id, split, HDF5 hash, local frame_indices
print(windows.statistics)
raw_frames = RawFrameDataset(prepared, 'train')  # every raw frame; no encoder policy selected
measurements = MeasurementReader(prepared, purpose='evaluation')
labels = measurements.read(sample['metadata']['episode_id'], sample['metadata']['frame_indices'])
```

For K macro transitions starting at local row t, frames are
`t,t+5,…,t+5K`; action block j is the five outgoing actions
`action[t+5j:t+5j+5]`, paired with frame `t+5j+5`. Only phase 0 is used.
One stored step is one 0.1-second controller interval, not a physics substep.
Each window is contained in one episode. Incomplete trailing transitions and
the final action (which has no successor image) are discarded and counted.
Horizon/stride are positive integer macro steps. Too-short episodes stay in the
manifest and are counted as having no windows; empty splits have length 0.

One shared function performs a full-image Pillow bilinear resize with its downscale
antialiasing, then converts to contiguous float32 RGB CHW /255. There are no crops,
flips, rotations, action clipping, fitted normalization or learned transforms.
Raw actions exceed nominal [-1,1] in this archive and must remain unclipped.
The training readers return only images/actions plus provenance. State/proprio
are read solely for split-leakage prevention or through `MeasurementReader`
with explicit purpose `probe`/`evaluation`. No simulator label becomes a model token.

`PreparedData` hashes the full HDF5 once (about 20 seconds locally), checks layout
and family identity, and must be constructed before sharing readers with workers.
The file must then remain immutable; stat changes are rejected. Every sample opens
and closes its own HDF5 handle, so no handles cross fork/spawn boundaries. Tests
exercise two spawned processes without requiring PyTorch. Per-sample open and
100-image chunk decompression favor correctness over optimized throughput; measure
and optimize caching in a later training milestone if necessary.

## Evidence and remaining limits

Pinned primary source:
[stable-worldmodel PushT env.py](https://github.com/galilai-group/stable-worldmodel/blob/21446f1ede6d5284e981bd7b47f432b994e6d812/stable_worldmodel/envs/pusht/env.py)
(`step`, `_get_obs`) consumes relative targets as current pusher position + 100×action
with PD gains 100/20, dt 0.01, 10 physics substeps per stored interval. Angle is in
radians modulo2π; position is in 512-pixel world units and velocity in pixels/second.
[DINO-WM's loader](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/pusht_dset.py)
independently uses relative actions divided by 100 and identifies appended agent
velocities. Its hard-coded normalization statistics do not describe this exact
archive; future normalizers must be fitted on the approved training split.

`audit_alignment` reconstructs every 2,318,051 valid within-episode transition.
Outgoing action lag 0 gives pusher-position RMSE 1.38326e-5 pixels, max 6.26306e-5;
lag ±1 comparisons each cover 2,299,366 valid shifted-action pairs with RMSE 4.45696.
Proprio equals state[:,[0,1,5,6]] exhaustively. Independent reviewer image checks
accepted image/state alignment on 3,000 interior rows across 2,725 episodes, seed
20261126 (sampled, 0.128% of frames), not an exhaustive image proof.

The exact historical producer/converter remains unlocated. Current upstream
collection code is corroboration, not proof of how this archive was created.
Before planning, the eventual evaluation simulator (the proposal names DINO-WM)
must pass state/action replay equivalence, including block dynamics. This dataset
milestone neither runs that simulator nor authorizes training/planning.

Reviewer observation: the whole pusher disc is outside the image in 3,160 rows
(0.135%, 259 episodes); its centre is outside in 6,432 rows. These rows are retained,
not silently filtered; later pusher pose-probe reports must state their treatment.

## Verification

```bash
.venv/bin/python -m unittest discover -s scripts/data/tests -v
bash -n scripts/data/download_pusht.sh
.venv/bin/python -m compileall -q scripts/data
```

Tests use real synthetic HDF5/Zstandard files and cover corruption, truncation,
partial cleanup, no-overwrite behavior, schema inconsistencies, exact duplicates,
family and episode leakage, tampered manifests, index alignment, incomplete tails,
short episodes, resizing, state isolation and process-safe reads. Development CLIs
emit DEBUG-level logs; reports are atomically published and remain outside Git.

Local DATA-001 evidence lives in the executor worktree's ignored `data/pusht/`:
`schema_family.json`, `family_split_manifest.json`, `alignment_report.json` and
`family_smoke.json`. Earlier `schema.json`/`split_manifest.json` files are superseded
checkpoints. The assigned handoff in the original repository records candidate
identity and verification results; production manifests should use the final
reviewed commit instead of this uncommitted candidate revision.
