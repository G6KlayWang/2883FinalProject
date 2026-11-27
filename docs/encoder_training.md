# Encoder Training Specification

**Current execution status (ENCODER-002):** the user answered “Use the recommendation
recipe,” approving the saved reviewed numerical recipe and full encoder training.
The saved shell launcher, progress/history and exact staged queue are documented
in [scripts/encoders/README.md](../scripts/encoders/README.md). Follow the updated
[ENCODER-002 assignment](../doc/team/handoffs/encoder002-assignment.md), independent
reviews and exact source sync. Earlier draft/proposed/no-implementation wording
below is retained as historical design provenance; performance/convergence claims
remain unvalidated. Merge and Diffusion Forcing/planning remain separate decisions.

**Status:** Executor handoff draft, 2026-11-26. This document consolidates the research design, verified data preparation, and Research Reviewer recommendations. Writing it does **not** approve unresolved experimental choices or authorize training, including pilot runs.

**Later task authorization:** the user has now requested implementation and a bounded single-arm smoke test. [ENCODER-001](../doc/team/handoffs/encoder001-assignment.md) defines that authorized diagnostic scope. Full training remains gated; this does not convert proposed production settings into accepted experimental decisions.

Read the [research proposal](../doc/research_proposal.md), [implementation plan](../doc/implementation_plan.md), [semantic rulings](../doc/team/reviews/data001-semantics.md), and [encoder recommendations](../doc/team/handoffs/encoder-training-recommendations.md). Explicit user decisions take precedence over older plan text. In particular, the approved split is by **whole family**, superseding the original episode split. Settings marked **proposed** below still need a recorded decision before production runs.

## 1. Objective and run matrix

Test whether the distribution of frozen visual features affects Diffusion Forcing prediction and planning on PushT. Keep architecture, observations, augmentations, and optimization settings shared; vary the regularizer and its weight.

| Arm | Regularizer / intended target | First-seed candidates | Follow-up |
| --- | --- | --- | --- |
| A | Sliced Wasserstein / standard Gaussian | Three weights | Second seed at selected weight |
| B | Same sliced Wasserstein implementation / product Laplace | Three weights | Second seed at selected weight |
| C | VICReg variance and covariance / moments only | Three weights | Second seed at selected weight |
| A′ | Published SIGReg / standard Gaussian | One fixed weight | Encoder reference checks |

There are **four encoder variants, ten initial training runs, and thirteen total encoder runs** after the three second-seed runs. Each run produces separate parameters. Pose-probe training is additional. A′ does not automatically receive a Diffusion Forcing run; that is a later decision if its measurements differ materially from A.

Order: implement and review → authorize concrete configuration → train ten candidates → freeze encoders and evaluate/probe → select one weight each for A/B/C → train three second-seed encoders and repeat checks → report readiness for separately authorized Diffusion Forcing training. Do not extend grids or budgets automatically.

## 2. Verified data and access contracts

Use the prepared HDF5 and family manifest from DATA-001, not a new split or download:

```text
/root/autodl-tmp/pusht-prepared/data001-d4c1d5cd0935762ba6c7bd2fbacf995d9c63b66e/
    pusht_expert_train.h5
    family_split_manifest.json
```

Dataset identity: HDF5 SHA256 `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`, 46,300,921,856 bytes. It contains 18,685 episodes and 2,336,736 frames, grouped into 185 families of 101 related noisy replays.

**Approved split:** family-disjoint 90/5/5, seed 42: 167/9/9 families and 16,867/909/909 episodes. Validation and test each contain only nine distinct initial conditions. Never enable diagnostic episode splits for these experiments.

Reuse the reviewed `scripts.data.loader` interfaces from commit `d4c1d5cd0935762ba6c7bd2fbacf995d9c63b66e`:

- `PreparedData` verifies file and manifest identity and family membership.
- `RawFrameDataset` exposes every frame in a split; it does **not** implement an encoder sampling policy.
- `TrajectoryDataset` implements current phase-0, five-action-block semantics.
- `MeasurementReader` isolates simulator measurements for probes/evaluation.
- `resize_rgb` performs full-image bilinear resize to contiguous float32 RGB `(3, 96, 96)` in `[0, 1]`.

Encoder optimization consumes images only. Do not expose state, proprioception, actions, family IDs, or pose labels to the encoder loss. Family checks may read initial state; probe/evaluation paths may read labels. Preserve episode/frame IDs as metadata for alignment, never as model inputs.

## 3. Frame sampling: paired decision still open

| Policy | Encoder population | Relationship to trajectory loader |
| --- | --- | --- |
| Current R3-compatible proposal | Rows `0, 5, 10, …` in each assigned episode | Current phase-0 windows |
| Alternative requiring R3 amendment | Every raw frame in assigned episodes | All-offset windows, with frames five rows apart |

Phase-0 populations are 433,896 train / 21,715 validation / 18,887 test frames. Raw populations are 2,136,352 / 107,464 / 92,920. Implement the selected policy explicitly; do not flatten overlapping trajectory windows to obtain a supposedly frame-uniform encoder dataset.

The proposed encoder sampler is uniform over frames, shuffled without replacement and mixed across episodes, with no episode/family reweighting. Define last-batch handling and epoch cycling in the frozen configuration; fixed full batches matter for batch-dependent regularizers.

**Phase-0-only encoder training is a project proposal, not a published LeJEPA recommendation.** The nearest inspected stable-worldmodel code uses all starting offsets. Neither that precedent nor LeWorldModel establishes a standalone encoder recipe for this experiment. Matching the set of frame positions also does not imply matching Diffusion Forcing sampling weights: overlapping windows favor interior frames. Document the later window/token weighting and remeasure feature statistics under it.

## 4. Architecture and augmentations

Required contract: image `(B, 3, 96, 96)` → ResNet-18 → three-layer MLP projector → feature `(B, 128)`. Use the **projector output** for regularization, probes, feature caching, and downstream models.

- Initialize randomly, for example `resnet18(weights=None)`; no pretrained weights are required or allowed by this design.
- Proposed projector: `512 → 2048 → 2048 → 128`, BatchNorm/ReLU on hidden layers and a plain linear output. Hidden width and the ResNet stem choice must be recorded before implementation is frozen.
- Do not add output normalization, feature L2 normalization, or an arm-specific architecture.
- Proposed view count: two independent appearance-only views of the same frame, using color jitter, blur, and noise. No crops, flips, rotations, or translations.
- Freeze exact augmentation probabilities, strengths, operation order, clipping, and any input normalization. These numerical settings are not specified yet; never silently adopt a library preset.

## 5. Loss definitions and proposed weight grids

Proposed common invariance term: `L_inv = mean((z1 - z2)^2)` over samples and coordinates. Proposed common objective: `L = L_inv + rho * R`. Average the two per-view sliced Wasserstein values for A/B. C has different variance/covariance view reductions, specified explicitly below; A′ retains the reference SIGReg view reduction. Preserve these reductions when importing reference implementations.

### A and B: one implementation

Sample `B × 128` target vectors: independent standard Gaussian coordinates for A; independent Laplace coordinates with location zero and scale `1/sqrt(2)` for B. Both targets have identity covariance.

Each step draw 1,024 random unit directions, append all 128 coordinate axes, project both features and target vectors, sort each projection over the batch, and average squared quantile differences over samples and directions. Redraw directions and targets each step. Specify whether both views share draws; use the same policy for A and B.

Only the target sampler switches between A and B. In particular, sample the **128-dimensional product Laplace before projection**; a general projection of it is not itself a scalar Laplace variable. Keep coordinate axes because random high-dimensional projections alone weakly distinguish these targets.

Proposed shared grid: **`rho ∈ {2.5, 10, 40}`**. The anchor 10 comes from the official Rectified LpJEPA CIFAR-100 ResNet-18 configuration (invariance/regularizer weights 25/250); the paper's ImageNet-100 setting uses 25/125. Transfer to PushT appearance-only views is unvalidated.

### C: moments-only control

Proposed grid: **`rho ∈ {0.25, 1, 4}`**. For each view, center features over the batch and define:

```text
R_var(view) = mean_coordinates(relu(1 - sqrt(unbiased_variance + 1e-4)))
Cov(view)   = centered_features.T @ centered_features / (B - 1)
R_cov(view) = sum_off_diagonal(Cov(view)^2) / 128
R_C         = (R_var(view1) + R_var(view2)) / 2
              + (R_cov(view1) + R_cov(view2)) / 25
```

Official VICReg averages variance penalties but **sums** covariance penalties over views. This definition makes `rho=1` match its 25/25/1 balance after dividing the objective by 25. If expressed with a view-averaged covariance term instead, its coefficient must be `2/25`, not `1/25`.

C's numeric grid intentionally differs from A/B: the shared multipliers are `{1/4, 1, 4}` around each loss family's anchor. This interpretation of the plan's “same grid” condition needs user acknowledgement.

VICReg alone does not force zero mean or upper-bound coordinate variance. Measure achieved moments rather than assuming whitening, and handle C's permitted transform as described below.

### A′: SIGReg reference

Proposed single published `lambda = 0.05`, expressed on the common invariance scale as:

```text
rho_Aprime = 4 * lambda / (1 - lambda) = 0.2105263158
L_common  = MSE(z1, z2) + rho_Aprime * SIGReg
L_native  = (1 - lambda) / 4 * L_common
```

The factor assumes exactly **two views, both treated as global**, so the native consistency term is `MSE/4`. Common-form normalization versus native scaling remains a **user decision**. They share gradient directions, but absolute scale can affect optimization; do not claim exact AdamW equivalence. Retain the reference SIGReg batch-size multiplier, 17 knots, and 1,024 slices. The recommended integration setting is the released code's `[0, 3]` range with its symmetric weighting, rather than the paper's `[-5, 5]`; record the range, weights, native lambda, effective rho, and loss convention explicitly.

## 6. Proposed shared optimization recipe

| Setting | Recommendation, pending approval |
| --- | --- |
| Optimizer | AdamW; record exact betas and epsilon before freezing config |
| Weight decay | `5e-4` on convolution/linear weights; exclude biases and normalization parameters |
| Batch | 512 source frames, two views per frame |
| Budget | 30,000 optimizer updates per run; final checkpoint for selection |
| Learning rate | Peak `5e-4`; 1,000-step linear warmup from 1% of peak; cosine decay to `5e-7` |
| Numerics | FP32; no gradient clipping |
| Seeds | First seed 0; second seed 1 at selected A/B/C weights |

Share initial parameters **and buffers**, frame order, and augmentation streams across arms/weights within a seed. Separate RNG streams for initialization, data, augmentation, projections, targets, and probes so an arm's target draws cannot shift its frame order. Specify distributed/global batch semantics before using multiple GPUs.

Thirty thousand steps, batch feasibility, and throughput have not been validated on this workload. No GPU-hour estimate is established. Extra weights, budget doubling, intermediate-checkpoint selection, or clipping/rerun fallbacks require an explicit amendment and run approval.

## 7. Encoder checks and proposed eligibility gate

Freeze the encoder, switch to evaluation mode, and extract clean, unaugmented features. Check BatchNorm buffers remain unchanged. Report coordinate/projection excess kurtosis, mean, covariance spectrum, Epps–Pulley statistics, pose information, and goal-distance/pose-distance rank correlation on the chosen training frame set and validation set. Test remains untouched until all choices are locked. Use fixed evaluation samples/projections across arms and record whether statistics are exhaustive or sampled.

Include a train-mode versus eval-mode moment comparison to detect BatchNorm effects. Perform train-mode diagnostics on a disposable model copy with fixed batches and no gradients or optimizer updates; do not mutate the frozen checkpoint or use those outputs for downstream caches. Compare A′ with the range spanned by A's three weights, not only its selected candidate.

The following **proposed amendment is not yet approved**. Evaluate it on the selected **training** frame population, before validation-based weight selection:

- Every coordinate mean lies in `[-0.1, 0.1]`; every covariance eigenvalue lies in `[0.5, 2.0]`.
- A and A′: median coordinate excess kurtosis lies in `[-0.5, 0.5]`.
- B: median coordinate excess kurtosis is at least `1.5`. C has no shape threshold.
- A and B receive no post-hoc repair. Do not introduce an unapproved transform for A′.
- C uses raw features if moments pass. Otherwise, permit one train-fitted centering plus symmetric ZCA transform only if every raw covariance eigenvalue is at least `0.05`; otherwise the candidate is ineligible. Verify transformed moments. Save the transform as part of the effective encoder and use it consistently in probes, selection, caches, Diffusion Forcing, and planning.

These thresholds are heuristic diagnostics, not proofs of Gaussianity or independent Laplace coordinates. Preserve the full measurements. If an arm has no eligible candidate, stop before Diffusion Forcing and report; do not silently relax thresholds. The existing requirement to establish heavier tails for B than A remains applicable.

A training-free synthetic response table can check loss behavior before encoder training. The recommendation permits one threshold revision against that table, recorded before any candidate encoder is trained; calibration must not become retrospective tuning to observed results.

## 8. Pose probes and proposed selection score

Fit linear and two-layer MLP probes on frozen training features from the chosen frame set, using corresponding validation frames for scoring. Targets are T position, `(sin(theta), cos(theta))`, and pusher position. Use one fixed probe configuration across candidates. Width, optimizer, budget, seed, target preprocessing, and any probe checkpoint policy still need specification. Do not backpropagate probe loss into encoders. If the eligibility gate is approved, select only among candidates that pass it.

The recommended selection statistic is validation MLP error, equally averaging three normalized groups:

1. T position: mean of x and y MSE divided by their respective training variances.
2. T angle: `mean(error_sin^2 + error_cos^2) / (train_var_sin + train_var_cos)`; one shared denominator preserves equal orientation weighting.
3. Pusher position: the same coordinate-wise normalization, restricted to the agreed visible-row mask.

Compute all normalization constants on training rows only and reuse them unchanged. Specify degenerate-variance handling in advance. The proposed mask excludes pusher centers more than 15 pixels outside the 512-pixel image; finalize exact boundary/corner semantics and apply the same mask to pusher probe fitting and scoring across arms. Other target groups retain those rows.

Proposed tie rule: scores within 2% favor the weight closer to the published anchor (A/B: 10; C: 1). Freeze the relative-difference denominator, distance-to-anchor convention, and deterministic residual tie break before selection.

Report linear-probe and per-group errors, but exclude them from selection. Validation chooses weights; **test and planning outcomes never define the score, thresholds, or hyperparameters**. Reserve final test reporting until choices are locked. “Variance unexplained” uses training variance as a common scale; it does not guarantee a score of one for the train-mean predictor on a shifted validation distribution.

## 9. Implementation boundaries, artifacts, and tests

Suggested new package: `scripts/encoders/`, separating configuration, model, augmentations, target samplers/shared losses, training, clean-feature measurement, probes, and caching. These modules and training commands do **not** exist yet. Reuse data tooling rather than duplicating split or resize logic. Use Superpowers for implementation, with small functions, descriptive names, explicit validation, and meaningful tests.

Every run must retain its resolved configuration and hash, code revision, dependency versions, dataset/manifest hashes, arm, seed, selected frame population, loss reductions, RNG states, optimizer/scheduler state, and completed update count. Log per-term loss and gradient norms, learning rate, throughput, finite-value failures, and checkpoint identity. Enable verbose development logging without leaking credentials.

Feature caches must bind checkpoint and optional C-transform hashes to split, episode/frame order, preprocessing, dtype, and shape. Cache clean evaluation-mode outputs with no gradients. Keep data, features, checkpoints, credentials, and generated run artifacts outside Git.

Required focused tests include:

- Family isolation and exact selected-frame indexing/counts, without state leakage into encoder samples.
- A/B shared loss path, target unit variance, coordinate axes, reduction factors, and gradients; deterministic small examples rather than fragile statistical assertions alone.
- VICReg reductions, SIGReg native/common scaling, and finite gradients.
- Appearance-only augmentation contracts; frozen parameters and BatchNorm buffers during extraction/probing.
- Train-only whitening/normalization; C-transform persistence and identical downstream application.
- Probe masks, selection/ties, failed-gate behavior, cache identity/order, and checkpoint resume including RNG/scheduler state.

## 10. GPU execution and resource plan

This section incorporates the [Server coordinator's execution plan](../doc/team/handoffs/server-gpu-training-plan.md). Hardware snapshots are time-specific; check available devices and container limits again before launch. Adding GPUs changes available concurrency, not the thirteen-run design or scientific configuration. This documentation update does not start training.

**Latest verified inventory: GPU-PLAN-002, 2026-11-26 19:42:03–19:42:07 UTC.** The coordinator's read-only inspection completed successfully on the existing server. This supersedes the earlier two-GPU resource snapshot:

| Resource | Verified configuration |
| --- | --- |
| GPUs | **4 × RTX PRO 6000 Blackwell Workstation Edition**, each reporting 97,887 MiB VRAM |
| Activity | All four idle in the bounded snapshot; no GPU processes reported |
| Interconnect | PCIe; peer-read capability available, no NVLink; bandwidth not benchmarked |
| Container CPU quota | **64 CPU-equivalents**; 128 visible logical CPUs are not 128 dedicated CPUs |
| Container RAM / shared memory | **464 GiB** RAM limit / **232 GiB** `/dev/shm`; shared memory consumes the RAM budget |
| Data volume | **200 GiB** capacity, **143.20 GiB free** |
| Prepared data | Archive/HDF5 paths and sizes present; family-manifest and prior artifact-check report hashes unchanged |

The inspection did not rehash the full HDF5 or execute model steps. GPU UUIDs and detailed evidence are in the coordinator's report. Availability must be rechecked at launch.

### Scheduling and device ownership

Use **one independent training run per available GPU**, now **up to four simultaneous runs**, as the initial scheduling policy. This is a proposed starting point, not a measured optimum. Keep each candidate's full approved batch on its assigned GPU; do not divide a candidate across GPUs or increase its batch because more devices are available.

Process the ten first-seed candidates through a fixed queue, starting the next pending candidate when a GPU becomes free. GPU assignment and start time must not change initialization, frame order, augmentation draws, or regularizer RNG streams. After the approved checks and validation selection, schedule the three selected second-seed runs using the same policy. Extraction and probes are separate scheduled jobs; avoid unmeasured contention with active training. Test remains reserved until choices are locked.

The coordinator proposes the following nominal allocation. Groups are organizational, not synchronization barriers; `low/mid/high` refer to the eventual approved grids:

| Group | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | --- | --- | --- | --- |
| 1, seed 0 | A-low | B-low | C-low | A′ |
| 2, seed 0 | B-mid | C-mid | A-mid | B-high |
| 3, seed 0 | C-high | A-high | Available | Available |
| After selection, seed 1 | A-selected | B-selected | C-selected | Available |

The first three groups contain exactly ten candidates; the last adds three. An available slot does not authorize an additional encoder run. Initially allow at most two concurrent feature extractors and at most four GPU jobs total, counting extraction/probe jobs as well as training.

The Server coordinator pins jobs by GPU UUID using `CUDA_VISIBLE_DEVICES`, maintains an exclusive advisory lock and run ledger per device, and gives each run a unique output directory and visible terminal. Recheck for outside workloads before launch; do not kill unrelated processes. Run only the reviewed immutable checkout and resolved configuration.

Neither distributed data parallel nor multiple simultaneous training jobs on one GPU is the default. Extra concurrency requires measured aggregate throughput gains with resource headroom. Distributed training additionally requires review of global-batch SWD sorting, VICReg moments, SIGReg scaling, BatchNorm, and gradient correctness; averaging per-device regularizers does not reproduce the full-batch objective. NVLink is unnecessary for independent jobs.

### Input pipeline and capacity

The existing HDF5 reader reopens the file per sample; compressed image chunks contain 100 frames. Random access can repeatedly decode a roughly 14.36 MiB chunk for one frame. This is a code-based bottleneck hypothesis, not measured training performance. Read a source frame once and construct both augmented views from it. Preserve the logical random sample order when optimizing reads; do not group neighboring frames into batches merely for speed.

Prefer a reviewed, immutable **resized uint8 image cache**, shared through read-only memory mapping rather than copied into each worker. Preserve exact Pillow resize parity, frame ordering, family splits, and source/preprocessing hashes. Cache clean images, never fixed random augmentations. These calculated all-split payloads exclude metadata and temporary files:

| Population | 96×96 RGB uint8 images | FP32 128-d features per encoder |
| --- | ---: | ---: |
| Phase-0 | 12.22 GiB | 0.226 GiB |
| All raw frames | 60.17 GiB | 1.114 GiB |

Capacity calculations do not select the frame policy or authorize test extraction. Avoid the 240.68 GiB whole-raw float32 image cache. Build a cache once, validate it, and publish atomically; preserve the source archive/HDF5 and allow space for partial writes and checkpoints. The initial plan reserves at least 40 GiB disk; refresh this margin against the new inventory and actual artifact budget.

Budget DataLoader workers and parent/codec threads across **all concurrent jobs**, using cgroup CPU and memory limits rather than host totals. Use bounded prefetch (initial proposal: factor 2), ordered delivery, and persistent workers only after RNG/resume tests. Shared memory, pinned buffers, page cache, and worker heaps all count toward the same memory allowance. Stagger full-file verification, cache generation, extraction sweeps, and checkpoint writes. The current worker/concurrency budget must come from the latest inspected resource plan.

For the verified four-GPU configuration, start with **4 DataLoader workers per trainer (16 total)**, worker math/codec threads capped at one, **2 intra-op threads per trainer (8 total)**, and one inter-op thread per trainer. Target combined team CPU use of roughly 32 CPU-equivalents or less initially, leaving margin within the 64-CPU quota. At prefetch factor 2, the four loaders can queue 32 batches: about 3.375 GiB for two FP32 image views, excluding active/pinned copies and worker overhead. Retain an initial **200 GiB total cgroup-memory guard** and **40 GiB disk reserve**; more installed RAM does not justify automatic cache expansion. These are proposed operating budgets to validate, not measured utilization or reservations.

### Environment, precision, monitoring, and profiling

Create one reproducible training environment containing both GPU and data dependencies. The first inspection found PyTorch/torchvision in base Python and HDF5 dependencies in the separate preparation environment; neither was complete for training. Preserve existing environments and pin the training stack, including resize-library versions.

Record matrix-multiply and cuDNN TF32 flags explicitly. The first inspection found convolution TF32 enabled despite FP32 tensor storage; the scientific recipe must settle this before launch. Do not silently enable AMP, change batch size, use gradient accumulation, or switch compilation/numerical settings per arm to improve utilization.

Monitor aggregate completed updates/frames per second, data-wait and compute times, GPU utilization/VRAM, CPU throttling, RAM/shared memory, and disk headroom. Retain atomic resumable checkpoints with optimizer, scheduler, RNG, sampler position, and model buffers. Stop and report nonfinite values, OOM, data/configuration mismatch, unexpected device use, driver errors, or resource-reserve violations; do not silently change the experiment to recover.

After reviewed code and explicit run approval, use bounded profiling to compare one, two, and four **independent** single-GPU jobs and test input-pipeline settings. The updated proposal caps model profiling at 1,000 aggregate optimizer updates and 30 minutes, whichever comes first: four single-arm cases (400 updates), one four-job case (400), and an optional two-job case (200). Each model clone runs at most 20 warmup plus 80 measured updates. Profiling checkpoints do not become selected candidates. No throughput, speedup, or GPU-hour estimate is established yet.

No persistent inference-serving service is needed for encoder training. Frozen feature extraction uses scheduled local batch jobs; revisit serving only for a concrete later planning interface.

## 11. Decisions and handoff gate

Before a training launch, record the user's decisions on the scientific recipe:

- Phase-0 versus all-offset frame/window policy.
- Weight grids, shared optimization recipe, and A′ native versus common loss scaling.
- Distribution eligibility gate and thresholds, including C whitening details.
- Probe/score definition and numerical edge cases.

The Executor must also specify remaining shared implementation settings: stem/projector, augmentations, estimators, batching, RNG/view-draw policy, and probe configuration. Resolve routine engineering details through the existing implementation/review workflow; include experimental choices in the concrete configuration presented for training approval. Do not require a separate user confirmation for every implementation detail.

The baseline run matrix is thirteen encoder runs; optional all-weight second seeds or extra A′ lambdas are separate amendments. A documentation draft or an agent recommendation is not acceptance of an amendment.

Executor implements → Research Reviewer checks research fidelity → GitHub manager publishes/synchronizes reviewed code → independent PR Reviewer reports on exact head/base → orchestrator reports readiness → **user decides merge**. The orchestrator then presents the concrete run configuration and budget for **separate training approval**; only the Server coordinator launches authorized GPU work.

## 12. Reference provenance

The [recommendation report](../doc/team/handoffs/encoder-training-recommendations.md#sources) records source evidence and transfer limits. Relevant pinned upstream implementations are LeJEPA `c293d291ca87cd4fddee9d3fffe4e914c7272052`, Rectified LpJEPA `5ae61ab49a4ae489c68f2328ba614472163518d6`, and stable-worldmodel `21446f1ede6d5284e981bd7b47f432b994e6d812`; confirm imported source files and full revisions during implementation review. Published anchors motivate candidate settings, not claims that they are optimal or validated for this dataset.
