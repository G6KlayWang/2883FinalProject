# GPU-PLAN-001 / GPU-PLAN-002: encoder execution and resource plan

2026-11-26 · Server coordinator `pusht-server` / `t1f`. **Current snapshot is GPU-PLAN-002 at 19:42:03–19:42:07 UTC: four GPUs, 64 CPU-equivalents, 464 GiB RAM and 232 GiB shared memory. See the four-GPU addendum at the end for the current queue and aggregate budgets.** Recommend one independent encoder run per GPU, up to four active runs after reviewed code, profiling and explicit run approval. Scheduling is proposed, not measured optimality. No inference service is needed at this stage.

The following GPU-PLAN-001 evidence and two-slot plan are preserved as **dated history from 19:28:47–19:28:55 UTC**. Their two-GPU/32-CPU/232-GiB limits are superseded by GPU-PLAN-002, not current limits.

## Evidence and scope

Read `docs/encoder_training.md` (SHA-256 `23166a698adfcbf261071d73fd2ebd7be01235cfd924928d8b461c25d453f563`) and the latest server environment/execution handoffs. The user's acceptance of the explanation of thirteen runs does **not** resolve the scientific choices marked proposed in that specification.

Fresh inspection: **2026-11-26 19:28:47–19:28:55 UTC**, visible hc terminal **`t2g`**, exit **0**. The secure helper reread the current connection file; its endpoint matches `root@connect.bjb3.seetacloud.com:42348`, hostname `autodl-container-5k7exs50av-2a6a4cbb`. No password was exposed. Inspection used device/cgroup/filesystem metadata, imports and CUDA device-property queries; it ran no tensors/model steps, training, probes, benchmarks, installs, downloads, cache generation, Git mutations, sync, termination, or GPU-setting changes.

Bounded evidence log in the original repo: ignored `data/pusht/gpu-plan001-inspection.txt`, 10,964 bytes, SHA-256 `79ff7403892aadf3c78249a1fb9fa56ec5a407897a5869c7cc47f379fcfc615e`. Utilization evidence is three one-second samples, not a reservation or sustained performance measurement. No unrelated process command lines were inspected.

| Observed resource | Current evidence |
| --- | --- |
| GPUs | **2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, correcting the stale one-GPU role statement |
| VRAM | Each **97,887 MiB** reported by NVIDIA; torch reports 101,973,491,712 bytes addressable |
| Activity | Both 0% GPU/memory utilization, 2 MiB used, P8, 28°C; no GPU processes listed; sampled power about 11–14 W, configured cap 600 W |
| Driver / CUDA | Driver `595.104.02`; NVIDIA reports CUDA compatibility **13.2**, whereas installed torch is built with **CUDA 12.8**; these are different quantities |
| Interconnect | GPU0↔GPU1 **PXB** (PCIe bridges); peer-read capability reports OK; **no NVLink**. No transfer-bandwidth or NCCL test performed |
| NUMA | Both GPUs on NUMA node 0; CPU locality `0–31,64–95` |
| CPU visibility | 128 logical CPUs, two 32-core/64-thread Intel Xeon Gold 6530 sockets, affinity `0–127` |
| Actual CPU budget | cgroup `cpu.max = 3200000 100000`: **32 CPU-equivalents**, not 128 dedicated CPUs. Host load average about 17.3/15.0/12.9 is not a container utilization measurement |
| Memory budget | cgroup max **232 GiB**, high **230 GiB**, current **44.12 GiB**, including **43.68 GiB file cache**; zero recorded OOM/high events; no swap. Host MemTotal about 945 GiB is not the container allowance |
| Shared memory | `/dev/shm` **116 GiB**, unused at inspection; its pages also consume the cgroup memory budget |
| Data storage | XFS `/root/autodl-tmp`, 200 GiB capacity, **153,758,392,320 bytes / 143.20 GiB free**, project quota enabled; root overlay is only 30 GiB |
| File handles | Soft/hard per-process limit 1,024 / 65,535; avoid designs with many permanently open shards |

GPU UUIDs for future pinning: GPU0 `GPU-d667666c-e29d-a178-4793-653714ace3e9`; GPU1 `GPU-77a3c1e2-0dbf-f6e4-dedc-2201c54ad598`.

## Software and data readiness

Base `/root/miniconda3/bin/python`: Python **3.12.3**, torch **2.8.0+cu128**, torchvision **0.23.0+cu128**, numpy 2.3.2, Pillow 11.3.0, zstandard 0.22.0; **h5py and hdf5plugin absent**. CUDA metadata sees both GPUs, capability **12.0**, 188 SMs each; torch's compiled architecture list includes `sm_120`. These checks do not prove workload feasibility or throughput. Torch defaults expose 64 intra-op and 64 inter-op threads, excessive as a starting policy under the 32-CPU quota.

The isolated `/root/autodl-tmp/.venvs/pusht-prep/bin/python` has Python 3.12.3 and the reviewed data pins (numpy 2.5.5, Pillow 13.0.0, h5py 3.16.0, hdf5plugin 7.1.0, zstandard 0.25.0), but **no torch/torchvision**. Neither inspected environment is a complete encoder-training environment. Later create a reviewed, separately pinned training environment with GPU and data dependencies together; preserve these existing environments. Do not combine them through ad hoc `PYTHONPATH`, change resize-library versions per arm, or assume installation is authorized now.

Prepared directory:

```text
/root/autodl-tmp/pusht-prepared/data001-d4c1d5cd0935762ba6c7bd2fbacf995d9c63b66e/
```

Its HDF5 is present at **46,300,921,856 bytes**. Retained receipt records SHA-256 `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`. The source archive remains present at 13,136,247,974 bytes. Family manifest hash remains `c19df78b23a15207cec7892f73a8d4cdecfc0ea5ca157312564ae9d82d65171f`, with seed 42, family unit, and 16,867/909/909 episodes. `artifact_checks.json` still hashes to `a0e10ac001328392bddebf30615010ea7a51d488873ac79672d661d35bab43ee` and says passed. **This inspection checked metadata and small-report hashes, not a new full HDF5 checksum pass.** The reviewed data checkout remains available; an encoder implementation/CLI is not established by this plan.

## Launch prerequisites and numerical contract

Before any model execution, resolve the specification's frame population/window pairing, grids, A′ native/common scaling, distribution gate/C transform, probe recipe/score/mask/ties, architecture/stem, augmentation details, optimizer settings, batching/view reductions and RNG policy. Retain the **proposed 512 source frames × two views, FP32, shared recipe and fixed update budget** for planning; batch fit and the proposed 30,000 updates remain unvalidated and unapproved. The thirteen encoder runs are ten initial candidates plus three selected second seeds; probes add work.

A concrete numerical issue needs review: base torch currently reports matmul TF32 **false**, cuDNN TF32 **true**, and float32 matmul precision `highest`. FP32 tensor storage therefore does not settle convolution arithmetic. The reviewer/user must define the intended FP32 contract and freeze both TF32 policies. Strict full-FP32 would require disabling convolution TF32, which is an explicit reviewed numerical choice; no flag was changed here. AMP/BF16/FP16, TF32 changes, `torch.compile`, batch changes, gradient accumulation, clipping, and optimizer/schedule changes require numerical/research review and an approved configuration. Accumulating small microbatches does not reproduce batch-512 SWD/covariance/BatchNorm semantics automatically. See [PyTorch 2.8 CUDA precision controls](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

Implement → research review → publish/sync exact SHA → independent PR review → user's merge decision → concrete run/configuration/resource approval → Server coordinator launch. The run approval must explicitly cover profiling or probes if requested. Keep the DATA-001 checkout and prepared artifacts immutable.

## Scheduling recommendation and alternatives

| Strategy | Assessment for this workload |
| --- | --- |
| **One independent run per GPU** | Recommended initial policy: two full-batch jobs, no cross-GPU loss/BatchNorm changes, simple fault isolation. Maximizes useful candidate parallelism without assuming a speedup. Batch-512 FP32 VRAM fit still needs approved profiling |
| Multiple independent runs per GPU | Possible later if a single job leaves compute idle **after fixing input starvation**. Extra VRAM alone does not imply a gain: jobs contend for kernels, bandwidth, CPU, decompression and disk. Permit only if bounded measurements show higher aggregate completed-update throughput with headroom and unchanged data/RNG semantics |
| One run across both GPUs / DDP | Not the default for thirteen independent small-encoder jobs. It reduces simultaneous candidate count and adds communication/global-statistic work. P2P capability is not measured bandwidth; no NVLink is present. Reconsider only for demonstrated single-run constraints or measured aggregate benefit after a separately reviewed distributed implementation |

DDP synchronizes gradients but does not partition inputs or make a batch-dependent regularizer global for the user. With global batch 512, two local batches of 256 require differentiable global feature/statistic handling: SWD sorts over 512, VICReg uses global moments and the correct denominator, SIGReg keeps its global-batch multiplier, and projection/target draws and loss scaling must match. Local-loss averaging is a different estimator. BatchNorm/view semantics also need a deliberate global-statistics design and parity tests; ordinary buffer broadcasts do not supply it. See [PyTorch 2.8 DDP](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html). Do not silently use 512 per GPU, change SyncBatchNorm, or adopt DDP to fill devices.

**Ten initial runs:** use two work-conserving single-GPU slots. A concrete nominal pairing below spreads A/B across both devices; `low/mid/high` refer to the eventually approved arm-specific grids, not accepted values. Pairing is organizational, not a synchronization barrier or a measured duration estimate.

| Nominal pair | GPU0 | GPU1 |
| --- | --- | --- |
| 1 | A-low, seed 0 | B-low, seed 0 |
| 2 | B-mid, seed 0 | A-mid, seed 0 |
| 3 | A-high, seed 0 | B-high, seed 0 |
| 4 | C-low, seed 0 | A′, seed 0 |
| 5 | C-high, seed 0 | C-mid, seed 0 |

Start the next predeclared pending job when a slot frees; do not wait merely to preserve pairs or use interim validation outcomes to reorder budgets. Record physical UUID for every run. Within seed 0, preserve common initialization **and buffers**, exact frame order and augmentation streams across arms/weights; separate initialization/data/augmentation/projection/target/probe RNGs. Worker count, GPU index and start time must not alter sample/view draws. Prefer logical-step/sample/view keyed augmentation randomness or a tested equivalent, ordered delivery, and an explicit last-batch/epoch-cycling rule. See [PyTorch 2.8 reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html#dataloader).

**Checks, probes and second seeds:** freeze each final encoder and hash it. After initial candidates finish, use two slots for clean train/validation feature extraction if input profiling supports concurrent readers; otherwise serialize extraction. Fit identical linear/MLP probes from those frozen caches, with training-only fitted statistics and explicit measurement-reader labels. Small probes can use CPU only if the reviewed probe implementation/numerics specify it; avoid unmeasured GPU co-location. Apply the approved training eligibility gate before validation selection. Lock one weight per A/B/C, then launch A-selected/seed 1 and B-selected/seed 1 on the two GPUs; C-selected/seed 1 takes the next free slot. Repeat checks/probes. A′ has no automatic second seed. Spare slots may extract/probe a finished run only within approved CPU/I/O budgets; preserve encoder throughput first. Keep test untouched until choices are locked; no automatic Diffusion Forcing run.

## Input bottleneck, caches and headroom

The reviewed loader opens/closes HDF5 per sample. Pixels are Blosc-compressed in **100-image chunks** at 224×224×3. One decoded uint8 chunk is **14.36 MiB**. Uniform random frame access can decompress most of that for one selected image; reopening also loses per-handle chunk reuse. Two independent loaders can repeat this work. A warm filesystem page cache stores compressed file pages and does not eliminate codec/resize work. These are code/layout-based bottleneck hypotheses, **not measured encoder timings**. Do not turn the sampler into neighboring-frame batches to improve locality: batch composition affects the regularizers.

Read each source image once, then produce the two approved views; do not decode twice per view. Validate source/cache identity in the parent before workers, rather than hashing the full 46.3 GB independently in every worker. Stagger mandatory startup verification, cache construction, checkpoint writes and feature sweeps to limit shared-volume contention. Any reviewed batched chunk reader must restore the original logical sample order before augmentation/loss computation.

**Calculated payloads**, excluding headers, indices, labels, transforms and allocator overhead:

| Population, full-image resize to 96×96 RGB | Frames | uint8 payload | float32 payload |
| --- | ---: | ---: | ---: |
| Phase-0, all splits | 474,498 | 12.22 GiB | 48.87 GiB |
| All raw frames, all splits | 2,336,736 | 60.17 GiB | 240.68 GiB |

Phase-0 training alone is 11.17 GiB uint8; raw training alone 55.01 GiB. These alternatives do **not** decide the unresolved sampling policy. Prefer a later reviewed, shared, read-only **resized uint8 disk/mmap cache** of the approved population, with exact resize parity, immutable row mapping and source/split/preprocessing hashes. Cache clean images, not random augmentations; do not change their distribution. Generate nothing now.

Both uint8 alternatives fit current disk space, but a raw cache would leave only about 83 GiB before checkpoints/features. Reserve **at least 40 GiB disk** for runs, partial writes and recovery; never maintain two full 60 GiB cache copies during publication. Build once into a unique temporary file, validate, then atomically publish on the same filesystem. Whole-raw float32 images exceed available disk **and** the 232 GiB RAM cap; avoid them. Phase-0 float32 is possible by arithmetic but wastes headroom relative to uint8. Preserve archive/HDF5 and do not evict user data automatically.

In RAM, share mappings/page cache rather than copying a full cache into every trainer/worker. Proposed operating margin: keep total cgroup use below **200 GiB**, leaving ≥32 GiB beneath the hard limit; include HDF5 page cache, worker heaps, pinned buffers, tmpfs and probes. The existing 44 GiB is mostly reclaimable file cache, not a private allocation or guaranteed free capacity. `/dev/shm` is useful for batches but is neither extra RAM nor the default home for an entire dataset cache.

FP32 128-d features cost `N × 128 × 4`: all-split payload per encoder is **0.226 GiB phase-0** or **1.114 GiB raw**; thirteen such caches total **2.94 / 14.49 GiB**, before metadata. These are capacity upper-scenario calculations, not authorization to extract test features. Keep checkpoint/optional C-transform identity, exact row order, preprocessing and dtype in each cache. Extraction uses clean `eval()`/no-grad outputs and unchanged BatchNorm buffers.

## Worker, ownership and recovery policy

Proposed starting loader budget: **4 workers per trainer, prefetch factor 2**, ordered batches and persistent workers only after epoch/RNG/resume tests. Parent train processes: initially at most 4 intra-op threads and 1 inter-op thread each; worker math/codec threads capped to 1 where supported. Eight workers plus two bounded parents leave capacity within 32 CPU-equivalents for pinning, monitoring and I/O. This is a starting budget to profile, not an established optimum; never inherit 64-thread torch defaults into every worker. Consider NUMA-0 locality within allowed CPUs after profiling; topology affinity does not reserve those CPUs.

One FP32 512-image batch is **54 MiB**; sixteen prefetched batches across both jobs total **0.844 GiB** for one view, **1.688 GiB** for two, excluding active batches, pinned copies, decoded chunks and heaps. Budget several additional GiB per job and monitor actual cgroup use. Use pinned tensor batches/nonblocking transfer only through reviewed code; returning NumPy arrays alone does not establish pinning. [PyTorch 2.8 DataLoader](https://docs.pytorch.org/docs/2.8/data.html) defines prefetch as batches per worker and warns that unordered delivery can harm reproducibility.

Server coordinator owns a run ledger and an exclusive advisory lock per GPU UUID; future launch uses `CUDA_VISIBLE_DEVICES=<UUID>` so the job sees only its assigned device. Recheck device identity/processes before acquiring a slot; advisory locks coordinate this team, not external tenants. Unexpected use means defer/report, never kill another process. One immutable reviewed checkout, unique run/output directory, immutable resolved config hash and hc terminal per job; no code replacement under a running job. No GPU power/clock/MPS changes proposed.

Retain config, SHA, environment, source/manifest/cache hashes, frame policy, seed streams, completed updates and physical GPU identity. Log step/data-wait/H2D/compute times, term losses, gradient norms, LR, finite checks and memory; monitor GPU utilization/VRAM and cgroup CPU throttling/RAM/shm/disk at roughly 15–30-second intervals. Use rotating logs and atomic checkpoints, retaining an approved bounded checkpoint set and final checkpoint. A proposed engineering cadence is every 1,000 updates or 15 minutes, whichever first; freeze it with the run budget. Save parameters/buffers, optimizer/scheduler, all RNGs and exact sampler/cycle position, and test resume including prefetch replay. Resume only the same config/checkpoint identity; no automatic extra steps or early checkpoint selection.

Operational stop/report criteria to record before launch: nonfinite losses/gradients, OOM, identity mismatch or state leakage; disk below the 40 GiB reserve; cgroup approaching 200 GiB or new memory events; loader deadlock/no heartbeat for an agreed five-minute limit; unexpected GPU collision or driver/Xid error. Do not silently shrink batch, change precision, skip data, extend budgets, or kill unrelated processes. Persistent input starvation should trigger an engineering review at a safe checkpoint, not an experiment redesign. Finish at the approved update count regardless of validation curves. If no eligible candidate exists, follow the specification's stop/report gate rather than relaxing it.

## Later bounded profiling proposal — not authorized or run now

After reviewed code/environment and explicit approval of the exact profiling configuration: use fixed source rows and frozen RNG/view streams, batch 512/two views, and the approved FP32/TF32 contract. First compare at most 20 input-only batches with a 60-second cap for HDF5 versus an already reviewed/authorized cache path; preserve order and record cache-warm state. Cache construction itself needs separate preparation authorization.

For model steps, propose four single-run arm cases (A/B/C/A′), then an A/B pair with one job per GPU; optionally compare four independent jobs, two per GPU, only if memory/CPU margins permit. Cap each clone at **20 warmup + 80 measured updates**, all cases at **1,000 aggregate optimizer updates and 30 minutes wall time**, whichever limit comes first. Those are proposed future bounds, not current permission. Do not use profiling checkpoints for candidate selection or count them as production training without an explicitly tested/approved continuation design. DDP needs a separate numerical-parity review and profiling budget.

Record aggregate completed updates/source frames per second, p50/p95 step time, loading/H2D/compute split, VRAM peak, CPU throttling, host/shm memory, disk activity and stalls; distinguish simultaneous versus isolated measurements. Choose extra concurrency only for a repeatable aggregate gain with resource margins, not higher utilization alone. No measured throughput, speedup, batch feasibility, GPU-hour or completion-time estimate is available from this inspection.

## Serving and hardware comparison

Encoder training and feature extraction need local batch processes, not a persistent inference serving service. Launch frozen-encoder forwards as scheduled offline extraction jobs; an always-resident server would reserve resources without an identified client. Revisit serving only when a concrete later planner/inference interface requires it.

An **A800 configuration with NVLink could be preferable for communication-heavy DDP or model-parallel work**, but NVLink is not used by independent encoder jobs. For the recommended two independent runs, compare exact A800 variant/count/VRAM, numerical mode, input pipeline, measured aggregate throughput and rental cost against this two-GPU machine. The link name alone cannot establish which configuration is more powerful or cheaper for this workload. No A800 specifications or workload performance were measured here.

**Disposition:** report to `t1c` and hold. No model execution or resource changes are authorized by this plan.

## GPU-PLAN-002: four-GPU addendum — current at 19:42:07 UTC

Fresh bounded read-only SSH inspection ran at **2026-11-26 19:42:03–19:42:07 UTC**, hc **`t2h`**, exit **0**, after securely rereading the same connection endpoint. Hostname remains `autodl-container-5k7exs50av-2a6a4cbb`. Evidence: ignored `data/pusht/gpu-plan002-inspection.txt`, 10,494 bytes, SHA-256 `dfd3f0d7599d9aa8d2b627c917148ee034dcb6428d158d559ca3a7daae7b9903`. No model steps, benchmarks, installs, downloads, caches, settings changes, Git mutations/sync, process termination or nested agents ran. `docs/encoder_training.md` was not edited by the Server coordinator.

**Observed:** all four devices are NVIDIA RTX PRO 6000 Blackwell Workstation Edition, each **97,887 MiB**, **2 MiB used**, **0% GPU/memory utilization**, P8, 27–28°C. Three one-second samples showed no activity; no GPU processes were listed. Driver remains `595.104.02`, NVIDIA CUDA compatibility display 13.2. No new PyTorch import/workload check was performed in this snapshot; prior environment and numerical prerequisites still apply.

| Index / PCI address | UUID |
| --- | --- |
| 0 / `42:00.0` | `GPU-d667666c-e29d-a178-4793-653714ace3e9` |
| 1 / `48:00.0` | `GPU-77a3c1e2-0dbf-f6e4-dedc-2201c54ad598` |
| 2 / `4A:00.0` | `GPU-af16a90f-cd04-2423-d176-1f146632d8dc` |
| 3 / `59:00.0` | `GPU-fedd6aa9-8348-df92-9b29-58b8fb042dcb` |

All GPUs report NUMA-0 locality (`0–31,64–95`). GPU1↔GPU2 is **PIX**; other pairs are **PXB**. All off-diagonal peer-read checks report OK; none has NVLink. This establishes topology/capability, not link bandwidth or DDP performance. VRAM remains separate per independent job.

Current cgroup CPU quota is **`6400000 / 100000 = 64 CPU-equivalents`**, with 128 visible/allowed logical CPUs. RAM max **498,216,206,336 bytes = 464 GiB**, high **462 GiB**, current **0.356 GiB**; file cache only **0.041 GiB**, no recorded memory/throttling events, no swap. `/dev/shm` is **232 GiB**, unused. Do not assume the old snapshot's 43.7 GiB warm file cache persists. The data volume remains **200 GiB**, with **153,758,392,320 bytes / 143.20 GiB free**; more GPUs did not increase disk capacity.

Prepared HDF5/archive sizes remain 46,300,921,856 / 13,136,247,974 bytes at their existing paths. Family manifest and `artifact_checks.json` hashes remain `c19df78b23a15207cec7892f73a8d4cdecfc0ea5ca157312564ae9d82d65171f` / `a0e10ac001328392bddebf30615010ea7a51d488873ac79672d661d35bab43ee`; the manifest still specifies family/seed 42 and 16,867/909/909 episodes. The reviewed data checkout and both existing Python executable paths are present. No full HDF5 reread was performed.

**Recommended four-slot queue:** one independent, full-batch encoder job per UUID; no DDP or GPU co-location by default. Nominal groups below are not timing barriers: start the next predeclared pending job on an available slot, retaining common seed-specific parameters/buffers, frame order, augmentation streams and separate target/projection RNGs. Low/mid/high remain labels for pending approved grids; batch 512/two views and FP32 remain proposals with unresolved numerical settings.

| Nominal group | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | --- | --- | --- | --- |
| 1 | A-low / seed 0 | B-low / seed 0 | C-low / seed 0 | A′ / seed 0 |
| 2 | B-mid / seed 0 | C-mid / seed 0 | A-mid / seed 0 | B-high / seed 0 |
| 3 | C-high / seed 0 | A-high / seed 0 | Free slot | Free slot |

This is exactly ten initial candidates. After approved eligibility checks, identical probes and validation-only weight selection, launch **A-selected, B-selected and C-selected at seed 1** in three independent slots. The fourth slot may perform approved offline extraction/probe work without exceeding the shared CPU/I/O budget; it does not imply an extra encoder or A′ seed. Keep test locked until selection is final. Prefer at most **two concurrent feature extractors** initially, never adding GPU jobs beyond four total, and do not slow active encoder input with unprofiled background sweeps.

**Conservative aggregate starting budget:** 4 DataLoader workers per encoder × 4 encoders = **16 workers total**, worker math/codec threads capped at 1; trainer intra-op threads **2 each / 8 total**, inter-op **1 each**. Keep the combined team CPU demand around **32 CPU-equivalents or less initially**, leaving roughly half the quota as margin for pinning, I/O, monitoring and competing activity; thread counts are budgets, not measured utilization or CPU reservations. No inherited 64-thread pools. Prefetch factor **2** gives 32 prefetched batches total: calculated FP32 image payload **1.688 GiB one-view / 3.375 GiB two-view**, before pinned/active copies and worker heaps. Keep ordered, reproducible worker/sampler behavior and tested resume semantics. Increase workers only after approved profiling; do not infer that 64 CPUs calls for 64 workers.

The same XFS volume and random HDF5 100-image Blosc chunks now feed up to four readers. Stagger source validation/hash passes and checkpoint writes; retain the proposed shared resized-uint8 cache and **40 GiB disk reserve** strategy, without generating a cache now. More RAM does not remove decode costs or disk limits: a 240.68 GiB raw-frame FP32 image cache could fit RAM arithmetically but still cannot fit the current disk free space. The earlier conservative 200 GiB working-set guard can remain the initial operational ceiling; no automatic expansion of RAM caches or tmpfs use is proposed.

Later explicitly approved profiling should compare aggregate throughput of one, two and four independent jobs without changing batch/numerics/data streams. Keep the previous **1,000 aggregate-update / 30-minute** proposed cap: for example four single-arm cases (400 updates), the four-job case (400), and an optional two-job case (200), each clone at most 20 warmup + 80 measured updates. These bounds are a future proposal, not executed work or approval. Four idle GPUs do not establish a fourfold speedup or batch feasibility. All earlier scientific, environment, numerical, identity, monitoring, checkpoint and stopping prerequisites remain in force. **Reported to `t1c`; hold after this addendum.**
