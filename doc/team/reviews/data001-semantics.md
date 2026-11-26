# DATA-001-SEMANTICS: independent evidence review

Reviewer: `pusht-reviewer` (`t1g`, Claude Fable 5.2) · 2026-11-26 · Requested by `t1c`
Scope: evidence review of action/frame semantics and state meanings. This is separate from code approval; no Executor code was reviewed here.

## Verdict

| Question | Verdict |
| --- | --- |
| R2: stored `action[i]` is the outgoing action from `state[i]` to `state[i+1]`, relative, scale ×100 | **Evidence accepted**, exhaustively confirmed, with the provenance limitation below |
| `pixels[i]` depicts `state[i]` | **Accepted on a sample** of 3,000 frames (0.128% of rows) |
| State layout `[pusher x, y, block x, y, block angle, pusher vx, vy]` | **Accepted**: columns 0, 1, 5, 6 exhaustively; columns 2, 3, 4 on the image sample |
| Split design (R6) | **Design accepted, implementation review pending**: the file holds 185 families of 101 noisy replays each (Finding F1); the user selected whole-family 90/5/5, seed 42 (see addendum). The random episode-level split is superseded. |
| Closed-loop planning readiness | **Open**: a state-replay test on the chosen evaluation simulator is required before any planning run (Remaining uncertainty 2) |

## What was inspected

- File: `/Users/charleswang/.codex/worktrees/0374/2883FinalProject/data/pusht/inspection.h5` (renamed from `inspection.h5.incomplete` during this review), 46,300,921,856 bytes.
- Identity: I hashed the file once and hashed a fresh `zstd -dc` stream of the verified archive once. Both give SHA-256 `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`, equal to the Executor's `schema.json`. The archive hashes to `7cfbd6d9…212f318`, equal to the download manifest. Every result below is therefore about the verified bytes.
- Source evidence read (not executed): `data/source_evidence/env.py`, `stable_worldmodel_world_world.py`, `dino_pusht_dset.py`, `convert.py`, `collect_weak_pusht.py`.
- Interpreter: worktree `.venv` (Python 3.13.9, numpy 2.5.5, h5py 3.16.0, hdf5plugin, Pillow), run with `-I`. `pymunk` is absent, so block physics was not re-simulated.
- My scripts and raw outputs (session scratch, not durable):
  `/private/tmp/claude-501/-Users-charleswang-Desktop-Fall-2026-2883r-2883FinalProject/ca2145f8-6bac-4f71-a123-7cf31bc91a35/scratchpad/review/numeric_checks.py`,
  `/private/tmp/claude-501/-Users-charleswang-Desktop-Fall-2026-2883r-2883FinalProject/ca2145f8-6bac-4f71-a123-7cf31bc91a35/scratchpad/review/group_checks.py`,
  `/private/tmp/claude-501/-Users-charleswang-Desktop-Fall-2026-2883r-2883FinalProject/ca2145f8-6bac-4f71-a123-7cf31bc91a35/scratchpad/review/image_checks.py`, each with a matching `.out` file in the same directory.

## Check scope: exhaustive versus sampled

| Check | Scope |
| --- | --- |
| Structure, NaN/inf, proprio identity, value ranges | Exhaustive: all 2,336,736 rows, 18,685 episodes |
| Pusher dynamics reconstruction and lag/scale/substep alternatives | Exhaustive: all 2,318,051 within-episode transitions, boundary transitions included |
| Family structure, duplicate sequences, within-family divergence | Exhaustive on numeric columns |
| First-frame identity inside families | Sampled: 12 families, 36 pairs |
| Image versus state (pusher, block, angle, lag) | Sampled: 3,000 random interior rows across 2,725 episodes, seed 20261126 |
| Block dynamics | Not simulated; consistency statistics only |

The checkpoint handoff describes its dynamics check both as "tested indices (excluding boundary vicinity)" and as "all-transition". My run settles the scope question independently: the result holds on every transition.

## Findings

### F1 (resolved by user choice of option A; implementation review pending): 185 families of 101 noisy replays

Evidence, exhaustive:
- The 18,685 episodes have only **185 distinct initial state rows** (bitwise, all 7 columns). Each is shared by exactly **101 episodes**.
- Families are contiguous ID ranges: family `g` = episodes `101g … 101g+100`. All members of a family have the same length.
- **Zero** episodes share a bitwise-identical (state, action) sequence. These are distinct rollouts; an exact-sequence duplicate audit (as in `schema.json`, `duplicate_episode_groups: []`) cannot see the family structure.
- Members are noisy replays of one action plan: action-sequence correlation inside a family averages 0.935 (5th percentile 0.724) against 0.008 across families; the action residual around the family's per-step median has std 0.054 against an overall action std of 0.208.
- Divergence grows with time. Within-family std (px, 512-px frame):

| Step | Pusher position | Block position |
| --- | --- | --- |
| 1 | 1.6 | 0.03 |
| 10 | 6.8 | 0.36 |
| 25 | 10.8 | 4.9 |
| 50 | 15.4 | 19.0 |
| final | 23.9 | 39.9 |

- Sampled first frames inside a family are near-identical: 6 of 36 pairs bitwise equal, the rest differ by at most 35/255 in any pixel, consistent with lossy video encoding of the same scene.
- The goal drawn in inspected frames is the same fixed green T.

Characterization: identical starts and a shared demonstration plan with action noise; early segments are near-duplicates and late segments differ materially. Repeated full trajectories are absent.

Research impact: under a random episode-level split, each validation or test episode has about 91 training siblings with the same start frame and plan. Three planned measurements then read partly as recall of seen demonstrations: validation pose-probe error used for regularizer-weight selection (condition 9), held-out prediction error at fixed horizons, and planning from held-out starts and goals. Arms may benefit unequally from memorization, and pose probes cannot detect or remove that confound, which weakens attribution to distribution shape (H1 to H3). Condition 5's wording, "split by episode", is satisfied literally by either choice, so this is a design decision for the user.

Options:
- **A (recommended): family-level split, 90/5/5, seed 42.** floor(0.05 × 185) = 9 families each for validation and test, 167 for train: 16,867 / 909 / 909 episodes. The manifest records `family_id = episode_id // 101`, and the tooling asserts that rule against the initial-state hash. Cost: 9 distinct start configurations in each held-out split, so planning evaluation over 50 or more episodes reuses starts.
- **B: keep the episode-level split** and state in every report that held-out results are within-family.
- **C: family-level split with a larger held-out share** (for example 80/10/10, giving 149/18/18 families). This changes the documented 90/5/5.

Provenance of the grouping: the structure matches my recollection of DINO-WM's `pusht_noise` release (expert demonstrations each replayed with action noise). The pinned files in `data/source_evidence` contain no generator script, so this origin is unverified. The grouping itself is established from the data and needs no provenance to act on.

### F2: action indexing and meaning (R2) confirmed exhaustively

Model taken from pinned `env.py` `step`: target = pusher position + 100 × action; 10 substeps of `v += (100·(target − p) − 20·v)·0.01; p += v·0.01`. The pusher is a kinematic body (`env.py:623`), so its motion is independent of contacts and this closed form is complete.

| Hypothesis | Position RMSE (px) | Max (px) |
| --- | --- | --- |
| **`action[i]`, relative, ×100, 10 substeps** | **1.38e-5** | **6.3e-5** |
| `action[i−1]` | 4.46 | 50.4 |
| `action[i+1]` | 4.45 | 50.4 |
| scale ×50 / ×1 | 4.36 / 8.63 | 30.2 / 59.8 |
| absolute target ×100 / ×512 | 117.7 / 124.3 | 236.6 / 339.7 |
| 5 / 20 / 1 substeps | 5.79 / 8.60 / 10.57 | 38.4 / 63.1 / 63.9 |

- Zero transitions exceed 1e-4 px. Velocity is reconstructed too (RMSE 3.9e-6 px/s, max 5.2e-5), which confirms columns 5 and 6 as pusher velocity in px/s.
- First transition of each episode: RMSE 9.7e-6. Final transition: RMSE 1.4e-5. No boundary exception exists.
- 7,610 transitions have a target outside [0, 512]; their max error is 6.1e-5 px, so the generator applied no target clipping, matching the commented-out clip at `env.py:326`.
- One stored transition equals one 0.1 s control interval. Five stored actions span five control intervals, as R3 assumes; storage is not pre-subsampled.
- The terminal action of every episode is a finite real value (no NaN, no zeros) with a smaller spread (std 0.09) than interior actions (0.21). It has no successor frame and stays unused. Every episode starts with pusher velocity exactly 0.
- The executor's numbers (lag0 RMSE 1.386e-5, shifted about 4.4 px) reproduce.

The error sits at float32 rounding level across 2.3 million transitions while every alternative misses by orders of magnitude, so this is identification by exact dynamics, which satisfies R2's requirement beyond correlation.

### F3: frames and state are time-aligned (sampled)

Colour-segmented centroids from `pixels[i]`, compared with neighbouring state rows:

| Compared with | Pusher centroid RMSE (px) | Block centroid RMSE, block moving >3 px/step (px) |
| --- | --- | --- |
| `state[i−1]` | 11.76 | 6.48 |
| **`state[i]`** | **0.71** | **2.07** |
| `state[i+1]` | 11.56 | 6.67 |

- Among 2,192 frames with the pusher moving more than 5 px per step, lag 0 is the best match in 2,190.
- Axes: state x is the image column and state y is the image row, increasing downward (bias −0.03, −0.04 px; swapped or flipped axes give RMSE above 200 px). State units are pixels of a 512-px frame rendered at 224.
- Block: columns 2 and 3 are the T body origin and column 4 is its rotation under `world = pos + R(+θ)·c`. The fitted body-frame centroid offset is (0.43, 42.6) px against a geometric expectation of (0, 40.7) for the scale-30 tee; the opposite rotation sign gives a 42.8 px residual. The angle recovered from the image agrees with column 4 to a median 0.9° (99th percentile 4.0°) over the full [0, 2π) range.

Together F2 and F3 give the chain the loader needs: `pixels[i]` ↔ `state[i]`, `action[i]` drives `state[i]` → `state[i+1]`, so `action[i]` lies between `pixels[i]` and `pixels[i+1]`. The first link rests on a 0.128% sample; the second is exhaustive.

### F4: structure and identities (exhaustive)

- `sum(ep_len)` equals row count; `ep_offset` equals the cumulative sum; `episode_idx` equals `repeat(arange(N), ep_len)`; `step_idx` restarts at 0 in every episode. Episode IDs 0 to 18,684 are stable positional IDs.
- No NaN or inf in `action`, `state`, `proprio`. `proprio` equals `state[:, [0, 1, 5, 6]]` bitwise, so R5's treatment of `proprio` as simulator state is correct.
- Phase-0 frameskip-5 arithmetic for the Executor's drop report: 455,813 complete action blocks; 38,986 trailing transitions dropped; `(len − 1) mod 5` histogram is [3232, 4040, 3737, 3232, 4444].

### F5: observations with research consequences (non-blocking)

- **Pusher outside the image.** The pusher centre is outside [0, 512] in 6,432 rows (0.275%), and the whole disc is outside in 3,160 rows (0.135%) across 259 episodes. In those frames the pusher position is unrecoverable from pixels. This puts a small arm-independent floor under pusher pose-probe error; probe reports should state how such frames are handled.
- **Block motion without the pusher nearby.** With pusher and block origin more than 200 px apart, 3.5% of transitions still show block motion (up to 6.7 px). Wall-contact resolution after a hard push is the likely cause; without `pymunk` this stays unverified. It does not affect labels, since F3 ties columns 2 to 4 to the image.
- **Statistics differ from DINO-WM's hard-coded constants.** Full-file action mean (−0.0078, 0.0069) and std (0.208, 0.207), and state means such as pusher (228.8, 292.2), differ from `dino_pusht_dset.py` (`ACTION_STD` 0.2019/0.2002, `STATE_MEAN` 236.6/264.6). The archive is therefore not shown to be the exact dataset those constants describe. Normalizers must be fitted on our train split, as already ruled.
- **Stored actions exceed [−1, 1]** (range −1.49 to 2.04), agreed with the checkpoint: no clipping.

## Remaining uncertainty

1. **Historical producer and converter are unlocated.** The pinned `World.collect` rotates the action list so the reset-time entry lands last; the archive's terminal actions are ordinary finite values, and no pinned script generates the family structure. The archive's semantics are established from the data against the pinned environment's `step` and stand independent of the collector, which is the limitation to carry in reports.
2. **Evaluation simulator.** `implementation_plan.md` names the DINO-WM repository's PushT environment for evaluation; the evidence pins stable-worldmodel's `env.py`. Only DINO-WM's dataset loader is in evidence. Before any planning run, the chosen simulator must pass the same replay test: stored actions applied from a stored state reproduce the stored next states for pusher and block. That test also closes the block-dynamics gap.
3. **Image alignment is sampled.** A cheap exhaustive variant (pusher centroid at lag 0 for every in-frame row) fits the DATA-001-RUN server verification.
4. **Reproducibility of evidence.** The dynamics, family, and image checks currently exist as ad hoc runs by two agents. They belong in the reviewed tooling (schema report or tests) so the server run regenerates them from the exact synced commit.

## Addendum: split decision relayed after this report

`t1c` relayed, and then confirmed in a second message, that the user explicitly selected option A: whole-family 90/5/5, seed 42, 167/9/9 families, 16,867/909/909 episodes. I have this from the orchestrator's messages; the orchestrator's ledger is the record of the approval. Split status: design accepted, implementation review pending against the criteria below. The research hypotheses are unchanged; this amends the data split only.

Acceptance criteria I will apply to the frozen candidate's split code:

1. Family membership is derived from evidence in the file (exact initial-state row) and cross-checked against `episode_id // 101`; a mismatch fails loudly. Hard-coding `// 101` alone is insufficient.
2. The audit asserts 185 families of exactly 101 episodes on this dataset hash and fails on any other structure.
3. Splits are family-disjoint and exhaustive: every episode appears in exactly one split, and every family lies wholly inside one split. Tests assert both on a synthetic fixture with family structure.
4. Initial state is read only inside the split/audit path. Model-facing samples stay free of state (R5); a test proves the training API still emits images, actions, and provenance metadata only.
5. The manifest persists seed, algorithm, family IDs per split, episode IDs per split, family-defining hashes, dataset SHA-256, and code revision, and regeneration reproduces it.
6. Documentation states that validation and test each hold 9 distinct start configurations, and reports built on them carry that limit.

## Requests

- **Orchestrator/user:** F1 choice made (option A, see addendum).
- **Executor:** add `family_id` and an initial-state-hash family audit to the schema report regardless of the choice; label each reported check as exhaustive or sampled; carry limitations 1 and 2 in the provenance notes.
- **Code review (later):** I will verify that the loader encodes exactly the chain in F2 and F3, and that split tests assert family disjointness if A or C is chosen.
