# PushT team task ledger

Plan: ../agent_team_plan.md (user-approved 2026-11-26).
Training authorization: NONE.

| Task | Owner | Status | Deliverable |
| --- | --- | --- | --- |
| TEAM-001 | Orchestrator | Complete: four roles acknowledged and model selections verified | Four project-specific Herdr workers and role briefs |
| DATA-001-GIT | GitHub manager | Complete: baseline `333dc0a`, isolated worktree ready | Baseline, isolated dataset branch/worktree |
| DATA-001-PREFLIGHT | Server coordinator | Complete: fresh archive checksum verified | Server/data/environment evidence |
| DATA-001-CODE | Executor | Frozen for independent code review; ownership released | Extraction, schema inspection, deterministic splits, loader and tests |
| DATA-001-REVIEW | Reviewer | Pending | Research/code review and any revision loop |
| DATA-001-SYNC | GitHub manager + Server coordinator | Pending | PR, reviewed SHA, verified remote checkout |
| DATA-001-RUN | Server coordinator | Pending | Dataset schema/splits/loader verification; no training |

Ruling: use fixed split seed 42 unless existing verified metadata requires another seed; this is a reproducibility default, not hyperparameter selection.
Ruling: preserve all pre-existing unrelated files and Capstone agents/templates.

## Active Herdr roles

- Orchestrator: `t1c`.
- Executor: `pusht-executor` / `t1k`, GPT-6.1 Sol; Superpowers required.
- Reviewer: `pusht-reviewer` / `t1g`, Claude Fable 5.2.
- GitHub manager: `pusht-github` / `t1d`, GPT-6.1 Sol.
- Server coordinator: `pusht-server` / `t1f`, GPT-6.1 Sol.
- Feature worktree: `/Users/charleswang/.codex/worktrees/0374/2883FinalProject`, branch `feat/dataset-tooling`.

Ruling: current server preflight reports two RTX PRO 6000 GPUs, superseding the previous single-GPU observation. No training is authorized, and future resource assignment will use fresh preflight evidence.

## DATA-001 review rulings (R1–R6)

- R1: Inspect the verified local HDF5 before implementing schema-specific loader behavior. Extraction/inspection tooling may be reviewed separately; all remote preparation waits for reviewed code and exact-SHA sync.
- R2: Establish stored action indexing and action meaning from upstream producer/consumer code plus data checks; correlation alone does not prove the convention. Preserve the canonical stored action representation that matches simulator consumption. Record uncertain state labels rather than guessing; unresolved semantics block scientific use.
- R3: Use phase 0 frames (0, 5, 10, …), drop incomplete terminal action blocks, and report drop counts. Verify whether storage is already temporally subsampled before treating five stored actions as five simulator actions.
- R4: Encoder training population selection is deferred to the encoder milestone. Current loader exposes explicit raw-frame and phase-0 trajectory interfaces; no encoder sampling defaults become an experimental decision here.
- R5: Model-facing samples include images/actions and provenance metadata only; every other simulator-derived field is available solely through an explicit measurement API.
- R6: Split all stable episode IDs before length filtering. Validation and test each receive floor(0.05*N); train receives remainder. Persist seed 42, deterministic algorithm, source hashes, ordered IDs and code revision. Validation selects weights; test supplies final reported measurements. Current milestone fits no statistics.
- Additional defaults: raw actions stay unnormalized; windows return K+1 frames and K action blocks with configurable K and no cross-episode boundaries. Keep existing unittest location scripts/data/tests for this milestone and document the command.

Ruling R2 evidence acceptance (independent semantics review complete; implementation review pending): pinned simulator code and near-float32-exact dynamics reconstruction support relative action ×100 and outgoing action[i] to observation[i+1]. Retain missing historical converter provenance as an explicit limitation. Reports must distinguish sampled checks from exhaustive checks. Source state layout plus exact proprio equality supports the measurement API.

## Resolved user decision: split families

Independent review found 185 groups of 101 noisy replays with bitwise-identical initial simulator states and strongly related action sequences. Exact-duplicate audit alone misses them. User was asked to choose whole-family 90/5/5 (recommended; 167/9/9 families), the original episode-level split with an in-distribution interpretation, or whole-family 80/10/10. The user selected whole-family 90/5/5 as recorded below; implementation review and dependent scientific use remain pending code approval. See `reviews/data001-semantics.md` for evidence.

User decision (2026-11-26): approved recommended whole-family 90/5/5, seed 42. Use 167/9/9 of the 185 families (16,867/909/909 episodes). Supersedes R6 random episode assignment and the pending-decision note above. Validate initial-state grouping and family disjointness; simulator state is used only for split integrity, never model input/target. Report the limit of nine distinct held-out starts per validation/test set.
