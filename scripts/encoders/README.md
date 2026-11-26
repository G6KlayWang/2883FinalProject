# Encoder pipeline

Four image-only, randomly initialized encoder arms share the reviewed PushT data
interfaces. **ENCODER-002:** the user approved the saved recommendation. Executable
`configs/production-approved.json` changes exactly five approval labels from the
preserved `production-proposed.json`; `configs/recipe-approval.json` records the
actual decision and reference hashes. Numerical settings below are the approved
recipe, retaining their research provenance and unvalidated performance caveats.
Merge remains separate; no entrypoint implements Diffusion Forcing or planning.

## One-command approved queue

After both reviews and exact sync, the Server coordinator uses the existing
visible Chrome/Jupyter worker. Executor runs only local synthetic checks.
Set `PY`, `SHA`, `H5`, `MAN`, `CACHE_ROOT`, `OUT`, and the four available GPU UUIDs
to the verified environment, checkout, prepared inputs and fresh artifact paths.
Use a new ENCODER-002 output directory, preserving ENCODER-001 diagnostics.

```sh
ENCODER_PYTHON="$PY" bash scripts/encoders/launch.sh --code-revision "$SHA" --hdf5 "$H5" --manifest "$MAN" --cache-root "$CACHE_ROOT" --output "$OUT" --gpu-uuids "$GPU1" "$GPU2" "$GPU3" "$GPU4"
```

This is a foreground, visible queue; Jupyter's terminal session persists when the
browser disconnects. For explicit detached supervision, use the same saved command
with `nohup env ENCODER_PYTHON="$PY" ... >"$OUT.supervisor.log" 2>&1 &`, then display
that saved log with `tail -f`. The supervisor ignores SIGHUP, child stdout/stderr go
to per-attempt log files, and a dead display disables itself without stopping
training. SIGINT/SIGTERM request clean update-boundary checkpoints. No silent SSH,
background remote fallback, or script generation through terminal heredocs.

The launcher verifies the exact clean source checkout and approved recipe, serially
prepares the train cache, then performs the following **production continuations**:

1. S1: A-rho2.5-seed0 alone to200 updates.
2. S2: same A checkpoint to400; B-rho2.5, C-rho0.25 and Aprime-rho0.210526 seed0
   each to200, on four distinct GPUs. Starts are staggered until each mandatory
   source/cache verification completes, then a shared start barrier releases
   actual concurrent training. Reports require measured overlapping model time.
3. After passing S2, build the validation cache serially while runs are paused.
   S3 resumes those four toward30000 and schedules the other six as slots free.

Every step counts toward the original30000; all runs retain the exact approved
configuration,24h cumulative model cap, environment, code and original GPU UUID.
The20min S1/S2 watchdog starts per trainer at `startup-ready`, excluding cold
verification/preparation. It requests a graceful checkpoint stop and stops the
queue. If exit is not observed after120s grace, record the owned PIDs for the
coordinator; **no automatic SIGKILL, rollback, retry or recipe change** occurs.

Capacity gates use observed history, excluding each segment's first20 updates:
S1 median(update+data-wait) projected over30000 must fit half the24h cap; S2 elapsed
model time plus projected remaining time must fit80%. Each peak GPU allocation
must be below85% of its actual card capacity. S2 requires actual four-run overlap,
cgroup working-set headroom, memory-event and CPU-throttling snapshots. Failed or
missing evidence stops before S3. Recorded failed S1/S2 assessments latch the
queue before phase entry on ordinary resume. Both failed and passed resource
records stay immutable; passed evidence is reused only with unchanged bound queue/
run identities and satisfied stage targets. No idle resampling, automatic reset
or override is provided. No batch/precision/worker/grid/budget fallback.

Preparation/status/explicit continuation use the same saved script:

```sh
# Optional train-cache preparation only; normal launch already does this.
ENCODER_PYTHON="$PY" bash scripts/encoders/launch.sh --code-revision "$SHA" --hdf5 "$H5" --manifest "$MAN" --cache-root "$CACHE_ROOT" --output "$OUT" --prepare-only
ENCODER_PYTHON="$PY" bash scripts/encoders/launch.sh --output "$OUT" --status
# Repeat the original full command with --resume to continue clean paused runs.
# Failed runs with unresolved updates stay parked; --retry cannot bypass them.
```

Queue resume requires unchanged code/config/source/cache/receipt/environment and
all originally assigned GPU UUIDs. Other display/queue controls stay outside
`RunConfig` and the model run identity. A failed attempt is never automatically
restarted. With exclusive original-GPU leases and idle hardware, explicit resume
reconciles each actual ledger and hash/identity-verified checkpoint, including
cleanly finished orphan trainers. Clean runs become paused/completed; unresolved
attempts remain failed and parked. Unaffected S3 jobs may continue, but the queue
returns an incomplete-matrix status and selection still requires all ten final
candidates. Parked runs never satisfy S1/S2 capacity gates. Nonfinite stops remain
preserved failures; no automatic checkpoint rollback is provided.

A later explicit decision may authorize **one failed run** to restart from zero:
repeat the original command with `--resume --restart-from-zero RUN_ID
--restart-decision "recorded later decision/reference"`. This is a capability,
not authorization to invoke it in the initial launch. It archives the entire
failed directory under `discarded/`, preserves prior attempts/history/checkpoints,
and records discarded completed/attempted counts and last-known runtime (an
unobserved hard-crash tail stays unknown). Controls and original GPU remain fixed.
Extra discarded-work compute must be part of that later decision. Between-update
resource-guard stops save a current checkpoint before reporting the guard failure.

Shared cache preparation holds an exclusive lock, reads at most one decoded
source image chunk (<=64MiB) through one persistent verified reader and applies
exactly the reviewed resize/order. New caches are atomically published read-only.
Reused cache metadata is checked during preparation; every trainer still performs
all existing full HDF5/cache checks. Startup-ready prevents competing hash storms;
no trust receipt or verification shortcut is introduced. A conservative13GiB
future-output allowance is checked in addition to missing caches and the40GiB
reserve. Existing source, caches, user files and running jobs are preserved.

Artifacts: `queue_record.json`, atomic `queue_state.json`, append-only
`queue_events.jsonl`, `logs/RUN-attempt.log`, startup-ready/barrier records,
capacity reports including actual overlap/resources, and `runs/RUN/` containing
resolved identity/config, attempts, per-step `metrics.jsonl`, checkpoints and
training reports. Each completed run exports `history.csv`; optional standalone
export can run on observed JSONL:

```sh
"$PY" -m scripts.encoders.history --history "$OUT/runs/A-rho2.5-seed0/metrics.jsonl" --csv "$EXPORT/history.csv"
```

CSV preserves absent gradient observations as empty fields. Logs include losses,
LR, gradient norms at unchanged intervals, step/data timing, observed source-frame
throughput, cumulative runtime and actual GPU allocated/reserved/peak memory.
Terminal ETA is an estimate, not a measured completion guarantee. Loss curves
never select weights or compare objectives across arms.

Optional PNG plotting is `python -m scripts.encoders.history --history FILE --plot
OUTPUT.png` with Matplotlib installed **in a separate plotting environment**.
Do not install plotting packages or change the encoder environment/Pillow after
training begins: all installed versions bind exact resume/cache identity.
CSV/progress require no new runtime dependencies; optional plots do not delay launch.

After all ten final candidates, use the reviewed extract/evaluate/selection tools
below. No automatic weight selection from training loss or test/planning results.
The three emitted seed1 configs and actual approved selection receipt may launch
in a separate fresh queue by adding `--repeat-config-directory "$REPEATS"
--selection-report "$SELECTION"` to the original command with a new `--output`.
No guessed repeats or extra A-prime run. Full training may take hours; a healthy
launch is not evidence that all thirteen jobs or scientific checks are finished.

## Environment and verification

Run commands from the repository root. Use an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/encoders/requirements.txt
.venv/bin/python -m scripts.encoders.environment
```

The Server coordinator instead installs `scripts/encoders/requirements-cu128.txt`
(torch 2.8.0+cu128, torchvision 0.23.0+cu128, six unchanged data pins), then runs:

```sh
.venv/bin/python -m pip check
.venv/bin/python -m scripts.encoders.environment --require-cuda
```

The self-check imports dependencies and records all installed versions; it does
not perform a model step. Runtime imports require no SciPy. Local implementation
verification uses Python 3.13.10/CPU. Server Python 3.12.3 dependency/import checks
passed separately; source execution on that environment follows review and sync.

Tests use standard-library unittest; pytest is an optional development runner:

```sh
.venv/bin/python -m unittest discover -s scripts/encoders/tests -v
.venv/bin/python -m unittest discover -s scripts/data/tests -v
.venv/bin/python -m compileall -q scripts/encoders
```

Tests cover actual ResNet forward/backward, analytic/reference losses, worker
parity, image/cache order and corruption, exact resume and failed-attempt limits,
eval/BatchNorm isolation, train-only whitening/probes, and candidate selection.
They use synthetic data; they do not establish batch-512 capacity or convergence.

## Authorized diagnostic smoke — coordinator only

After independent source review and exact sync, the Server coordinator uses the
existing visible Chrome worker. Executor does not run these GPU commands. Set
`PY`, `GPU_UUID`, `H5`, `MAN`, `RUN_ROOT`, and `SHA` to verified absolute environment,
leased GPU, prepared source, output and frozen code identities. Prepared source:

```sh
DATA=/root/autodl-tmp/pusht-prepared/data001-d4c1d5cd0935762ba6c7bd2fbacf995d9c63b66e
H5="$DATA/pusht_expert_train.h5"
MAN="$DATA/family_split_manifest.json"
RUN="$RUN_ROOT/diagnostic-armA-seed0"
```

Start from a new output directory; run sequentially, preserving visible output:

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.train \
  --config scripts/encoders/configs/smoke.json --hdf5 "$H5" --manifest "$MAN" \
  --output "$RUN" --code-revision "$SHA" --stop-at 10

CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.train \
  --config scripts/encoders/configs/smoke.json --hdf5 "$H5" --manifest "$MAN" \
  --output "$RUN" --code-revision "$SHA" \
  --resume "$RUN/diagnostic-step-000010.pt" --stop-at 20

CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.extract \
  --checkpoint "$RUN/diagnostic-step-000020.pt" --hdf5 "$H5" --manifest "$MAN" \
  --split train --output "$RUN/diagnostic-features" --code-revision "$SHA" \
  --diagnostic
```

This profile enforces A/seed0/rho10, batch16, full specified model and directions,
strict FP32, TF32 off, at most256 distinct train phase-0 frames spread across
families, at most20 lifetime attempted updates including resume, and600 cumulative
model seconds including extraction and the disposable train-mode comparison.
The diagnostic schedule has2 warmup updates; production has1000/30000. No AMP,
compilation, clipping, DDP, automatic restart or budget extension is implemented.

Inspect `metrics.jsonl`, `attempt-*-started.json`, `attempt-*-report.json`,
`training_report.json`, `diagnostic-frame-ids.json`, `run_state.json`, both atomic
checkpoints, and `diagnostic-features/extraction_report.json`. Reports bind source,
config, code, dependency versions, GPU UUID, checkpoint hashes, attempted/completed
counts, losses/gradient norms, parameter change, restored state, model time and
peak GPU memory. Extraction must report finite `(N,128)` features. Diagnostic
artifacts cannot become scientific candidates, fit candidate probes or enter
selection. A tiny-batch distribution gate is not a smoke success criterion.

The coordinator also keeps one attempts ledger across output directories; a new
directory does not authorize a fresh smoke budget. It must report every attempt.

Checkpoints include parameters/BatchNorm buffers, optimizer, scheduler count/LR,
logical sampler cursor, named and global RNG states, immutable identities and
runtime ledger. Resume requires the latest recorded checkpoint file hash and
matching counts/population/config/environment. An attempted but uncommitted update
blocks automatic recovery; route failures through Executor and review. Keep the
last two checkpoints. Load only trusted local checkpoints (`torch.load` includes
optimizer/Python state). Atomic publication refuses existing artifact directories.

## Input/cache contract

`ImageFrames` returns only resized RGB images. Actions/states are excluded from
encoder/model inputs; measurement labels are read explicitly later. Family IDs
are used only for leakage validation and diagnostic frame spreading. Whole-family
splits remain the user-approved DATA-001 split, with no test image/label access.
`phase0` means local frames0,5,10,...; `raw` means every frame. This encoder choice
never changes the reviewed trajectory loader's five-action blocks.

Build shared production caches once per split/policy (not needed for smoke):

```sh
"$PY" -m scripts.encoders.image_cache --hdf5 "$H5" --manifest "$MAN" \
  --split train --policy phase0 --output "$CACHE_ROOT/train-phase0"
"$PY" -m scripts.encoders.image_cache --hdf5 "$H5" --manifest "$MAN" \
  --split validation --policy phase0 --output "$CACHE_ROOT/validation-phase0"
```

Choose `--policy raw` only if that policy is approved. Caches store uint8 HWC96x96
mmap rows, lossless relative to the reviewed Pillow resize before float conversion.
Identity includes verified HDF5/manifest hashes, Pillow version, preprocessing,
policy, split, exact episode/frame IDs and both file hashes. Workers reopen mmap
files read-only per process. Phase-0 train/validation populations are433896/21715
frames; raw populations are2136352/107464. Each update draws exactly B distinct
frames from a seeded epoch permutation, dropping incomplete tails. Augmentations
are keyed by seed/update/frame/view, so worker prefetch cannot advance committed
sampler/RNG position. No production family reweighting is added.

## Individual-run and evaluation interfaces

Use a reviewed approved copy of the complete explicit config as `$CFG`, and a
predeclared candidate ID from the run sheet. The following are interfaces for
later authorized work, not a launch request:

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.train \
  --config "$CFG" --candidate A-rho10-seed0 --hdf5 "$H5" --manifest "$MAN" \
  --cache "$CACHE_ROOT/train-phase0" --output "$CANDIDATE_RUN" --code-revision "$SHA"

CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.extract \
  --checkpoint "$CANDIDATE_RUN/step-030000.pt" --hdf5 "$H5" --manifest "$MAN" \
  --cache "$CACHE_ROOT/train-phase0" --split train --output "$FEATURE_ROOT/train" \
  --code-revision "$SHA"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.extract \
  --checkpoint "$CANDIDATE_RUN/step-030000.pt" --hdf5 "$H5" --manifest "$MAN" \
  --cache "$CACHE_ROOT/validation-phase0" --split validation \
  --output "$FEATURE_ROOT/validation" --code-revision "$SHA"

"$PY" -m scripts.encoders.measure --features "$FEATURE_ROOT/train" \
  --output "$FEATURE_ROOT/train-statistics.json"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.evaluate \
  --checkpoint "$CANDIDATE_RUN/step-030000.pt" --train-features "$FEATURE_ROOT/train" \
  --validation-features "$FEATURE_ROOT/validation" --hdf5 "$H5" --manifest "$MAN" \
  --output "$EVALUATION" --run-id A-rho10-seed0

"$PY" -m scripts.encoders.selection --config "$CFG" \
  --records "$EVAL_ROOT"/*/report.json --output "$SELECTION" \
  --repeat-config-directory "$REPEAT_CONFIGS"
```

Extraction requires a completed final checkpoint and clean frozen `eval()`
forwards; it verifies unchanged parameters/buffers and reports a disposable-copy
train/eval comparison on at most two full batches sampled without replacement
from the complete population. Its seed is the configured probe seed (shared
across arms), and the report records that seed and the sampled episode/frame-row
digest. Evaluation verifies complete source/split/row identity,
fits train-only transformations/probes, and publishes a self-contained report,
linear/MLP probes, and optional C transform. Subsequent effective C extraction
must add `--transform "$EVALUATION/C-transform"` to both train/validation commands.
Its fit source/checkpoint/policy/training-row identity is checked. C uses an
explicit proposed `c_transform` policy: `if_moments_fail`, mean limit.1, covariance
interval[.5,2], raw eigenvalue floor.05. These are configurable proposals with
their own approval status, not acceptance of the gate amendment. Selection
requires all ten first-seed reports with one consistent protocol. It writes
three seed1 configurations only for approved policies and binds their config
hashes in the selection receipt, without launching them. A/B grids must match.
Seed0 production requires `--candidate`; arbitrary direct/off-grid weights and
extra seeds are refused. Run each selected repeat using its exact emitted config
and receipt, with no `--candidate`:

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -m scripts.encoders.train \
  --config "$REPEAT_CONFIGS/A-selected-seed1.json" --selection-report "$SELECTION" \
  --hdf5 "$H5" --manifest "$MAN" --cache "$CACHE_ROOT/train-phase0" \
  --output "$REPEAT_RUN" --code-revision "$SHA"
```
Test rows and planning results cannot affect this path.

## Explicit mathematical and proposed recipe settings

- Standard torchvision ResNet18 `weights=None`; projector512→2048→2048→128 with
  hidden BN/ReLU and plain output. Initialization runs in a seed-scoped fork of
  the global Torch RNG. Every arm forwards each view separately: BN running
  statistics/counters advance twice per update. This differs from concatenated
  LeJEPA view forwarding and is an explicit shared proposal.
- A/B share1024 random unit directions plus128 axes per update and average the
  two view SWDs; target batches are independent per view. Dedicated mutable CPU
  direction/target streams preserve identical A/B directions. B uses inverse-CDF
  product Laplace with scale1/sqrt2 and guarded uniform endpoints.
- C uses unbiased variance/covariance, eps1e-4, variance averaged across views,
  off-diagonal squared sum divided by128, and the two covariance terms summed/25.
  Off-diagonal entries are selected directly to avoid cancellation.
- Aprime implements the pinned LeJEPA SIGReg Epps–Pulley statistic from revision
  `c293d291ca87cd4fddee9d3fffe4e914c7272052`:17 knots0..3, symmetric Gaussian
  trapezoid weights, batch multiplier,1024 random slices and no added axes.
  Convention is explicit. Lambda is authoritative and inconsistent
  rho=4lambda/(1-lambda) is rejected; native loss is the common loss times
  (1-lambda)/4. Default proposed lambda.05 gives rho.2105263157894737.
- Augmentation numbers/order are explicit JSON fields: appearance jitter
  brightness/contrast.4, saturation.2, hue.1 at probability.8; mild3px Gaussian
  blur sigma.1–1 at probability.5; Gaussian noise std.02 at probability.5; clip
  to[0,1]. No geometric transformation or fitted input normalization.
- Proposed AdamW batch512,30000 updates,1000 linear-warmup updates from1% peak,
  cosine decay5e-4→5e-7, decay5e-4 on weights only. Biases/norm parameters have
  no decay. Separate term gradient norms use `autograd.grad` without touching
  optimizer `.grad`; `regularizer_unweighted_gradient_norm` explicitly measures
  the unweighted regularizer. Actual deterministic-algorithm/warn-only/cuDNN flags
  and matmul precision are persisted in the run record. TF32 choices are explicit; smoke always disables both.
- Measurements report exhaustive float64 train moments/eigenspectrum and both
  coordinate/fixed-projection excess kurtosis. EP uses at most4096 fixed sampled
  rows and128 directions; report distinguishes scope. Constant directions have
  null kurtosis. The proposed gate requires means±.1/eigenvalues[.5,2], Gaussian
  median coordinate kurtosis±.5 or B median≥1.5. C may use a symmetric train-only
  ZCA only when its independent C-policy moments fail and min raw eigenvalue≥.05;
  an unrepairable C fails that policy even with the eligibility gate disabled.
  No repair is implemented for other arms.
- Proposed probes share train feature standardization and train target variances:
  closed-form OLS; two-layer hidden256 MLP,50 fixed epochs, final epoch, AdamW
  lr.001/wd0, seed1729. No validation early stopping. Targets are Txy, sin/cosθ,
  pusherxy; angle uses one shared var(sin)+var(cos) denominator. Visibility is
  explicitly expanded box[-15,527]^2 (optional circle-intersection convention).
  Only pusher fitting/scoring is masked. Selection averages three normalized MLP
  group errors equally; linear probe and seeded goal-distance Spearman are
  diagnostics. Scores within2% of best tie; prefer nearest log2-distance to
  rho10(A/B) or1(C), then smaller rho and run ID. No eligible candidate stops.
- Runtime defaults: one job/GPU,4 workers/job,prefetch2, trainer2 intra-op/1
  inter-op threads, worker math1; cgroup working-set memory guard200GiB and disk reserve40GiB.
  Working set is max(0,memory.current-inactive_file), falling back conservatively
  to memory.current when memory.stat/inactive_file is unavailable. Separately,
  raw memory.current at95% of a finite memory.max stops execution. Errors record
  the accounting values; this protects the actual limit without counting the
  whole inactive file cache against the200GiB working-set budget.
  Production max86400 seconds is an engineering stop, not a runtime prediction.
  Full-cache and batch512 memory/throughput checks remain future execution work.

Remaining operational work: the coordinator owns cross-directory smoke attempt
accounting (M5); augmentation/label/cache throughput needs separately authorized
profiling (M6); final test-set reporting needs new reviewed tooling after choices
are locked (M7). Development entrypoints continue to refuse test rows.
