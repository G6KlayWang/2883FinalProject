# Herdr Research Agent Team Plan

Status: approved by the user on 2026-11-26. Team startup and the first dataset-preparation milestone are authorized; model training is not.

## Goal and management approach

Keep this conversation as the orchestrator. Use Herdr to run four interactive
workers, assign scoped tasks, collect evidence, enforce independent review, and
prepare reviewed code for the GPU server. The user decides when model training
starts.

The installed desktop version uses `hc` for agent management and `herdr-app` for
visible terminals. This orchestrator is already Herdr terminal `t1c`. Use the live
`HC_TERMINAL_ID` on subsequent sessions rather than assuming that ID never changes.
Use Herdr workers rather than the separate built-in subagent dispatcher.

Existing global templates are for a different Capstone project. Create separate
`pusht-*` templates; preserve the existing templates and the user's other terminals.

## Roles

| Role | Herdr name | Model | Responsibility |
| --- | --- | --- | --- |
| Orchestrator | Current session | Current model | Translate requests into tasks, assign one or several workers, enforce handoffs, resolve blockers, summarize readiness, and record user training approval. |
| Executor | `pusht-executor` | `gpt-6.1-sol` | Implement dataset processing, model training, feature caching, inference/serving, evaluation, configurations, and relevant tests. |
| Reviewer | `pusht-reviewer` | `claude-fable-5-2` | Independently check code and test evidence against the research proposal; report findings and approve or request changes. |
| GitHub manager | `pusht-github` | `gpt-6.1-sol` | Own branches/worktrees, commits, PRs, and synchronization of the approved commit to the server. |
| Server coordinator | `pusht-server` | `gpt-6.1-sol` | Own server inspection, environments, dataset preparation, GPU commands, job execution, monitoring, and artifact collection. |

The requested Executor and Reviewer model IDs are present in installed Herdr
templates. Verify actual model selection when starting each worker. Do not silently
fall back to another model if authentication or availability fails.

All workers remain interactive and inspectable. Codex workers must use
`--no-daemon`, as required by the installed Herdr manual. Workers must not create
additional agents. Dispatch at most three workers simultaneously; sequential
handoffs normally require only one or two working agents.

## Implementation and review workflow

1. **Assign:** The orchestrator writes a task brief with the objective, research
   requirements, allowed files, dependencies, acceptance criteria, and expected
   deliverables. The GitHub manager prepares the task branch/worktree when needed.
2. **Implement:** The Executor changes code and runs appropriate local checks.
   Its handoff identifies changed files, test results, configuration choices,
   remaining uncertainties, and the candidate commit or diff.
3. **Review:** The Reviewer reads the actual code and relevant proposal sections.
   It reports `APPROVED` or `CHANGES_REQUESTED`, identifying the exact reviewed
   revision, findings with file locations, research implications, and evidence.
4. **Revise:** Blocking findings return to the Executor. The revised code goes
   back to the Reviewer. Any later code/configuration change invalidates the
   previous approval until the affected change is reviewed again.
5. **Synchronize:** After approval, the GitHub manager pushes the branch, prepares
   a PR, and arranges synchronization of the reviewed commit to the server. The
   Server coordinator executes remote checkout commands on its behalf and returns
   the server commit SHA. The GitHub manager verifies the match. A PR need not be
   merged to test its reviewed commit; PR creation does not automatically merge it.
6. **Summarize:** The orchestrator reports what changed, review outcome, checks,
   PR/commit, server sync result, remaining issues, and the proposed training run.
7. **Authorize training:** The user explicitly chooses whether to start that run.
8. **Execute:** Only after that approval does the orchestrator instruct the Server
   coordinator to launch training and report its run ID, process/session, live
   log, checkpoint location, and eventual results or failure.

Research-design ambiguities go to the orchestrator and user; agents must not
silently change the experiment to make implementation easier or an arm perform better.

## Research review criteria

Use `doc/research_proposal.md` for scientific intent and
`doc/implementation_plan.md` for the intended implementation sequence. Flag
disagreements rather than selecting a convenient interpretation.

The Reviewer checks, where relevant:

- Episode-level 90/5/5 splits with a fixed seed and no cross-split leakage.
- Correct alignment of five 2-D actions with the following observation; no windows
  crossing episode boundaries; 96 × 96 inputs and 128-dimensional projector features.
- Simulator state used only for probes and evaluation, never encoder or Diffusion
  Forcing training inputs or targets.
- Appearance-only augmentations; shared settings across arms; A/B sharing one
  sliced-Wasserstein implementation with the target sampler as the experimental switch.
- Frozen encoders, traceable feature caches, and training-only fitting of
  normalizers, whitening transforms, and probes.
- Identical Diffusion Forcing settings and evaluation episodes/goals/seeds;
  separate action and feature losses and independent per-token noise levels.
- Regularizer selection by validation pose-probe error; measured feature shape
  and moments; prediction comparison in pose units rather than cross-arm feature loss.
- The A′ check, two-seed plan, reproducible run metadata, and the proposal's limits
  on attributing planning differences to feature distribution shape.

## Git and server ownership

The Executor owns implementation edits. The Reviewer does not edit implementation;
it writes review findings. The GitHub manager owns Git mutations and coordinates
any worktree handoff, avoiding simultaneous edits in the same checkout.

The Server coordinator is the single operator for remote commands, including
applying synchronization instructions from the GitHub manager. Use GitHub when
server authentication works; use Git bundles over SSH as the established fallback.
Both routes must produce the same reviewed SHA. Keep credentials, datasets,
caches, checkpoints, and run outputs out of Git.

Run training from an isolated checkout of its approved commit. Do not change
code underneath an active run. A run record includes the commit, configuration,
dataset checksum, split manifest, seeds, environment, resource/time budget, and
output paths. Approval applies to that specified run or explicitly approved batch;
changing its scientific configuration requires a new summary and approval.

Development tests and preparation can proceed within an assigned task. Do not
launch encoder, probe, or Diffusion Forcing training—including pilot training—until
the user approves it. The Executor implements training commands; the Server
coordinator runs them. Requested evaluation and serving work follows the same
code-review and synchronization path.

## Herdr operation

Store project role briefs and handoff formats in `doc/team/`; create corresponding
project-prefixed Herdr templates after this plan is approved. Keep a task ledger
and review reports in that directory. Store large execution artifacts outside Git.

Representative control commands, using explicit tasks and discovered addresses:

```bash
hc run --bg --name pusht-executor --cwd "$PROJECT_REPO" pusht-executor "<task brief>"
hc send pusht-reviewer "<review brief with candidate SHA and evidence paths>"
hc wait pusht-reviewer --timeout 45
hc read pusht-reviewer --lines 100
```

Include the current orchestrator address in each first task so workers can report
with `hc send <orchestrator-address>`. Check `first_message.sent` on startup. If it
is false, inspect readiness before retrying delivery. Use `hc wait` for status
changes, then inspect the report and actual artifacts; an idle/done badge is not
proof that tests passed or the implementation is correct.

Keep the four workers visible in the project's Herdr space, using app commands and
IDs discovered from `herdr-app ls`. Interrupt only this team's workers when needed;
do not restart the shared Herdr core. Before ending an agent, save its handoff so
another session can resume the role without reconstructing the conversation.

## Startup and first milestone

After approval of this plan:

1. Create the role briefs, task ledger, and `pusht-*` templates. Confirm model
   selection and role acknowledgment; no research training starts during onboarding.
2. Have the GitHub manager establish the repository baseline, including the
   currently untracked research documents, and prepare a dataset-preparation branch.
3. Have the Server coordinator reconfirm the single RTX PRO 6000, 200 GiB data
   disk, and verified archive at `/root/autodl-tmp/pusht_expert_train.h5.zst`.
4. Assign the Executor dataset extraction/inspection and split/loader tooling.
   The coordinator can inspect the environment in parallel, but preparation
   scripts are reviewed before running them on the server.
5. Run the review/fix/sync sequence, then use the coordinator to inspect the actual
   HDF5 schema, episode counts, action/state meanings, and sample alignment.
6. Deliver the schema report, deterministic split manifest, loader checks, PR,
   and synchronized commit. Then plan encoder implementation and its proposed
   training configuration for the user's later launch decision.

Approving this plan authorizes team setup and the first dataset-preparation
milestone. It does not authorize model training.
