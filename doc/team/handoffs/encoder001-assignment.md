# ENCODER-001: implementation and bounded GPU smoke test

## User authorization

2026-11-26: the user explicitly requested that the Executor write clean, organized, readable code according to `docs/encoder_training.md` and the GPU plan, then smoke-test one arm on a small data batch and report when ready for full training. This supersedes earlier blanket bans on model execution **only for implementation tests and this bounded smoke test**. Full candidate/probe/DF training, merges, and scientific changes remain outside this authorization.

Use the existing Herdr team and Superpowers implementation workflow. Do not reopen approval of doing this task or require another approval for necessary isolated environment setup, test artifacts, branch/PR preparation, or the scoped smoke. Research choices still marked proposed remain configurable and explicitly identified in the final production run sheet; implementation defaults are not claims of user acceptance.

## Required outcome

Implement the four-arm encoder pipeline against the verified dataset interfaces: validated configurations, randomly initialized ResNet-18/projector, appearance-only views, shared A/B SWD with distinct target samplers, correctly normalized VICReg and pinned SIGReg, deterministic frame/data/RNG handling, optimizer/schedule, clear metrics, finite checks, resumable atomic checkpoints, and clean feature extraction. Implement the specification's probe, measurement, eligibility, and selection tooling needed to use the encoders without manually reconstructing experiment logic. Keep provisional scientific policies explicit; never silently choose production settings based on smoke output.

Use cohesive modules with one responsibility, descriptive names, public type hints, explicit shapes and reductions, short functions and actionable errors. Avoid giant orchestration functions, speculative abstractions, broad exception swallowing, and tests that merely mirror code. Include documented runnable commands, dependency pins, small meaningful tests, and a reviewable production run sheet covering all thirteen candidate/repeat runs. Support the documented frame-policy choice without modifying the already reviewed Diffusion Forcing trajectory semantics. No DF model or planner implementation is requested here.

A shared resized-image cache is an implementation requirement for efficient production input where appropriate; validate source/preprocessing identity and resize/order parity. Full-cache generation is not necessary for the small smoke. Test rows must remain unused during the smoke and selection development.

## Bounded smoke configuration

- Arm A, seed 0, rho 10; full specified ResNet-18/projector and SWD direction count, no pretrained weights.
- Batch 16 source frames with two appearance views; at most 256 verified train-split phase-0 frames selected deterministically across multiple training episodes/families. This frame choice is diagnostic, not resolution of the production frame-policy decision.
- At most **20 total optimizer updates**, including the resumed portion; one GPU UUID, no DDP or co-located jobs. Use strict FP32 with matmul and convolution TF32 disabled explicitly for this diagnostic run. No AMP/compile/clipping.
- Use an explicit diagnostic schedule suitable for 20 updates, recorded separately from the proposed 30,000-step production schedule. No silent mutation of the production configuration.
- Save at an intermediate step, reload/resume to the total budget, and verify update count, optimizer/scheduler/RNG restoration plus changed model parameters. Local synthetic tests should additionally establish deterministic resume parity.
- Stop model execution after ten minutes wall time, or sooner on nonfinite values, OOM, identity/shape mismatch, or resource errors. Preparation/install/verification time is logged separately and is not a reason to raise the update cap.
- Report finite per-term losses/gradients, successful backward/optimizer updates, checkpoint identity/reload, finite clean features of shape `(N,128)`, observed runtime and peak GPU memory. Tiny-batch feature-distribution gate failure is not a failure of this pipeline smoke and is not evidence about scientific quality.
- A failure goes back to Executor → Research Reviewer before another changed-code run. Record all attempts and their update counts. No automatic training extension, grid search, or production run.

## Roles and order

1. GitHub manager prepares a clean isolated branch/worktree from the verified merged main baseline, preserving all original local changes. Copy only named finalized specs/evidence needed by the implementation; do not broadly stage root handoffs or user files.
2. Executor reads the spec/assignment, writes its concrete implementation plan using Superpowers, implements and runs local meaningful tests. The user already chose this execution method and authorized implementation; the team reviewer supplies independent checks. No nested agents, Git mutations, SSH, or browser server commands by Executor.
3. Research Reviewer checks frozen source identity and research fidelity, including loss factors, state exclusion, sampling/RNG controls, gate/selection and smoke safety. Fix/re-review until accepted.
4. GitHub manager freezes/publishes the exact accepted candidate and creates a draft PR; no merge. PR Reviewer checks exact live head/base independently. Manager and Server coordinator verify exact server checkout identity before execution.
5. Server coordinator owns visible remote preparation and the smoke. Use the already verified **Claude Chrome worker** `pusht-browser` / `t2j`, under coordinator direction, in its dedicated JupyterLab workspace `pusht-visible-check`. The coordinator may delegate browser typing to this existing worker; do not spawn further workers or fall back silently to background SSH. Leave commands, progress and outcome visible. Preserve existing notebooks/kernels/terminals. Necessary isolated training-environment creation and pinned dependencies are authorized as preparation; preserve base/preparation environments and data.
6. Once reviews and smoke pass, Orchestrator reports implemented scope, exact code/PR identity, actual smoke evidence, and any production decision or capacity check still outstanding. Full training waits for user approval. Do not equate a batch-16 smoke with demonstrated batch-512 feasibility or scientific convergence.

Maximum three workers actively working at once; persistent idle sessions do not count. Keep progress concise and evidence in `doc/team/handoffs/encoder001-*.md` / `doc/team/reviews/encoder001-*.md`. No credentials or generated datasets/checkpoints in Git.
