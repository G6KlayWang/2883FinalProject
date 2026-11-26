# Encoder-training settings: recommendation for user decision

Research Reviewer (`pusht-reviewer` / `t1g`, Claude Fable 5.2) · 2026-11-26 · Requested by `t1c`
Status: recommendation only. Nothing here authorizes implementation, calibration runs, or training.

Agreed experiment preserved: arms A, B, C, A′; A and B share one loss with the target sampler as the only switch; shared architecture and appearance-only augmentations; no simulator state in training; whole-family 90/5/5, seed 42; weight selection by validation pose-probe error; test reserved for final measurements. Proposed amendments are labeled **AMENDMENT**.

Labels: **[P]** taken from a primary paper or official code (sources at the end, pinned). **[D]** my proposed default. **[C]** needs empirical calibration; a provisional value and a rule are given.

No published source trains ResNet-18 on 96 × 96 PushT frames with appearance-only views, so every [P] value is a transferred starting point.

## Recommended recipe

| Setting | Recommendation | Label |
| --- | --- | --- |
| Training frames | Phase-0 frames (rows 0, 5, 10, …) of the 16,867 train episodes: 433,896 frames | [D], decision 1 |
| Sampling weights | Uniform over frames, shuffled without replacement each epoch, batches mixed across episodes. No episode or family reweighting. Matching Diffusion Forcing's eventual window/token weighting is a later design item. | [D] |
| Views | 2 appearance-only views per frame | [D] |
| Invariance term | `L_inv = mse(z1, z2)` (mean over batch and coordinates) | [P] VICReg and Rectified LpJEPA code |
| Loss form, A, B, C | `L = L_inv + ρ · R`, regularizer applied to each view and averaged | [P] form |
| A and B weights | ρ ∈ {2.5, 10, 40} with `R` = sliced Wasserstein, 1,024 random directions + 128 axes, targets redrawn each step | anchor 10 [P]; grid [C] |
| C weights | ρ ∈ {0.25, 1, 4} with `R = var + cov/25` in official VICReg normalization (ε = 1e-4, γ = 1) | anchor 1 [P]; grid [C] |
| A′ weight | Same form as the other arms: `L = L_inv + ρ_A′ · SIGReg` with ρ_A′ = 4λ/(1 − λ) = 0.2105 for the published λ = 0.05, fixed in advance; official SIGReg (17 knots, 1,024 slices, statistic × batch size). No selection on A′. | λ [P]; common-form rescaling [D], decision 5 |
| Optimizer | AdamW, default betas | [P] LeJEPA |
| Weight decay | 5e-4, applied to conv and linear weights only (biases and norm parameters excluded) | [P] LeJEPA README (ResNets); exclusion per VICReg code |
| Peak learning rate | 5e-4, shared by all arms | [P] LeJEPA README; adequacy [C] via the convergence rule |
| Schedule | Linear warmup for 1,000 steps from 1% of peak, then cosine to peak/1000 | shape [P]; warmup length [D] |
| Batch size | 512 frames (1,024 views) | [D]; published guidance is ≥128 to 256 |
| Budget | 30,000 optimizer steps for every encoder run (about 35 epochs of the phase-0 train frames) | [C]; provisional, see the optional checks below |
| Checkpoint | Final step only; no early stopping or best-checkpoint choice | [D] |
| Precision, clipping | fp32; no gradient clipping; gradient norms logged | [D] |
| Seeds | 0 and 1. Within a seed every arm loads the same saved initial weights and uses the same data order, augmentation draws, and projection directions. | [D] |
| Held-out measurement | Validation phase-0 frames (21,715 frames, 9 families) for selection; test phase-0 frames (18,887 frames, 9 families) for final numbers; features from clean frames in eval mode | [D] |

Run cost is unmeasured. Wall-clock time per run and data-loading throughput at this batch size should be measured on the first authorized run before any schedule is promised.

## 1. Frame sampling

**Recommendation: train encoders on the set of frames Diffusion Forcing will use, sampled uniformly over frames.** Under ruling R3 Diffusion Forcing uses phase-0 frames only, so phase-0 training matches the set of frames exactly. A regularizer shapes the feature distribution of the batches it is given, so training on that set brings "mean 0, covariance I, target shape" close to where the experiment uses the features.

**Limit of that match.** Diffusion Forcing's sampler is not implemented or fully specified yet, so its frame weights are unknown, and I do not claim they equal frame-uniform weights. Two known sources of difference: uniform sampling over windows counts interior frames in more windows than frames near episode ends, and the initial conditioning frame of a window plays a different role from the frames that are next-feature targets. Frame-uniform encoder sampling is therefore a proposed policy, and matching weights across the two stages is a coupling that has to be designed: when the Diffusion Forcing window and token weighting is specified, it should be documented, and feature moments and shape should be re-measured under that weighting.

- **Precedent check (added after a user question).** I found no published reference for training an encoder on phase-0 frames only; that restriction comes from project ruling R3. The nearest precedent goes the other way: the stable-worldmodel dataset code that LeWorldModel builds on (`dataset.py` at the pinned revision `21446f1e`, in the worktree's `data/source_evidence`) spaces frames 5 apart inside a clip and starts clips at every row (`for start in range(length - span + 1)`), so every raw frame is used. I have not confirmed that LeWorldModel's published runs used exactly this code version. Neither reference trains a standalone encoder on this data: LeWorldModel trains its encoder jointly with the predictor, and DINO-WM uses a frozen pretrained DINOv2. The encoder frame choice should follow the Diffusion Forcing window choice: phase-0 windows pair with phase-0 encoder frames (current ruling, no precedent); all-offset windows pair with all raw frames (matches the reference practice, 5 times more windows, needs an amendment to R3 and a 65 GB frame cache). Decision 1 should be read as this paired choice.
- **Phase-0 versus all raw frames.** Raw frames give 5 times more images (2,136,352), but neighbours are 0.1 s apart and nearly identical, and they add no new families. They also shift the population slightly: initial frames are 3.9% of phase-0 frames and 0.8% of raw frames. Phase-0 costs less (a 13 GB uint8 cache for all splits) and removes that mismatch. My onboarding note leaned toward raw frames; the measured data facts changed my recommendation.
- **Episode or family weights.** Episode-uniform or family-uniform weights would move the encoder's training distribution further from any frame-based Diffusion Forcing sampler, and feature moments would then be further off where they matter. Repairing a mismatch with post-hoc whitening is costly for the science: a rotation destroys the axis-aligned Laplace structure of arm B. I propose frame-uniform sampling as the encoder policy and as the reference weighting when the Diffusion Forcing sampler is designed.
- **Batch mixing.** All three regularizers treat a batch as a sample of the marginal distribution. Batches of consecutive frames would bias them.
- **Known property, same in every arm.** The 16,867 initial train frames are 167 distinct scenes, so about 20 near-duplicate images appear per batch of 512.

**Measurement population (clarification, AMENDMENT A1).** The proposal measures shape and moments "on held-out features". Validation holds 9 families, so its marginal differs from the training marginal for sampling reasons alone, and frames within a family are strongly correlated. I recommend reporting shape and moments on two populations: clean eval-mode features of train phase-0 frames (did the arm reach its target) and validation (does it transfer). Test stays untouched until final reporting. Train-mode versus eval-mode moments should be logged too, since BatchNorm running statistics can move them.

## 2. Regularizer weights

**Numeric weights are comparable only inside one loss family.** A and B share code, so one numeric grid is meaningful. The sliced Wasserstein term is a mean squared quantile gap; the VICReg variance term is a hinge on standard deviation; SIGReg is a test statistic multiplied by batch size. No constant converts one into another. I therefore read "the same grid for each arm" as the same multipliers, {¼×, 1×, 4×}, around each family's published anchor.

- **A and B anchor, ρ = 10 [P].** The official Rectified LpJEPA CIFAR-100 ResNet-18 config uses invariance 25 and regularizer 250, with the same loss normalizations proposed here. Its ImageNet-100 setting (25 and 125, ρ = 5) sits inside the grid. That code also samples the product Laplace at scale 1/√2, matching arm B.
- **C anchor, ρ = 1 [P].** VICReg uses 25 / 25 / 1 for invariance / variance / covariance. The paper reports unstable training when the invariance and variance weights differ greatly, which is why the grid is a factor of 4 and stays close to the anchor. The covariance term is divided by the feature dimension, and that dimension is 128 here against 8,192 in the paper, so C's anchor transfers least well.
- **A′ policy.** One run at the published default λ = 0.05, fixed before any result. Tuning A′ toward A would make the check circular. "A behaves like published LeJEPA" is supported when A′'s encoder checks fall within the range spanned by A's three weights.
- **A′ and the shared view-consistency term (correction to my first version).** The plan requires every encoder to share one view-consistency loss. LeJEPA's native objective is `(1 − λ)·sim + λ·SIGReg`, and with two views its `sim` equals `mse(z1, z2)/4`. So the native loss is `((1 − λ)/4) · [mse(z1, z2) + (4λ/(1 − λ)) · SIGReg]`. Written on the common `L_inv = mse(z1, z2)`, the published relative weighting corresponds to ρ_A′ = 4λ/(1 − λ): 0.2105 at λ = 0.05, 0.0816 at 0.02, 0.4444 at 0.1.
  - **Recommended: objective normalization.** A′ uses `L = L_inv + 0.2105 · SIGReg`, so all four arms share the identical view-consistency term and coefficient, and A′ keeps LeJEPA's published balance between the two terms.
  - **What this departs from.** The native loss is smaller overall by the constant factor (1 − λ)/4 = 0.2375. Gradient directions are identical; magnitudes differ by that factor. Under AdamW the update is approximately invariant to such a factor, and the invariance is inexact: the ε term in the denominator, any gradient clipping threshold, and mixed-precision numerics all see absolute scale, and the balance against decoupled weight decay can shift when ε is non-negligible. The recommended recipe (fp32, no clipping, default ε) keeps these effects small, and they are unmeasured.
  - **Alternative: native scaling.** Use LeJEPA's loss exactly as published. This is the most literal "published LeJEPA", and it gives A′ a different overall loss scale and a different coefficient on the view-consistency term than arms A, B, and C.
  - This choice is a user decision (decision 5). Either way the run record should state both λ and ρ_A′. One further inherited property: official SIGReg multiplies the statistic by batch size, so its pressure at batch 512 is not the same as at the batch sizes in the sources; I keep the published code unchanged and flag it.
- **Why the grid is provisional [C].** Published ratios were tuned against crop-based views with a large invariance loss. Appearance-only views make the invariance term small, so the regularizer dominates more at every ρ than in the sources. Whether {¼×, 1×, 4×} brackets the useful range is unknown until runs exist.
- **Sampling-noise floor may differ between A and B (open question).** A finite batch never matches a sampled target exactly, so the sliced Wasserstein loss has a nonzero floor, and that floor could be larger for the heavier-tailed Laplace target. I have not verified its size or its dependence on batch size, and nothing in this recommendation rests on it. The response table below would measure it; logging the loss alongside its measured floor makes values interpretable.

### Calibration proposal (reproducible, uses no planning results)

Item 1 trains nothing. Items 2 to 4 are amendments. Items 3 and 4 create extra runs and extra probe fits, so they are **optional, capped, and never automatic**: each extension or rerun needs its own run summary and the user's explicit approval, like any other training run.

1. **Training-free response table.** Before any run, compute each regularizer on synthetic batches of 512 × 128: exact-target samples (the floor), Laplace under the Gaussian target and the reverse, coordinate scale 0.9 and 1.1, pairwise correlation 0.1. This records how each loss scores identical deviations and measures the floors. It needs code and a few seconds of compute, and no encoder.
2. **Admissibility gate (AMENDMENT A2).** A weight is eligible for selection only if its encoder reached the arm's intended distribution on train-population clean features. Provisional thresholds, to be confirmed against the response table and frozen before any candidate encoder is evaluated: |mean| ≤ 0.1 per coordinate; covariance eigenvalues within [0.5, 2]; median coordinate excess kurtosis within ±0.5 for A and A′ and at least 1.5 for B (target 3); no shape condition for C. Among eligible weights, the lowest validation pose-probe error wins, as already agreed. Reason: with easy invariance, probe error alone can favour a weak weight at which B never becomes heavy-tailed, which would erase the experimental variable.
3. **Grid extension (AMENDMENT A5, optional, user-approved per step).** Trigger: the selected weight is at a grid edge, or fewer than two weights are eligible. Proposed response: one further factor-of-4 weight in the indicated direction. A and B always share one grid, so a step triggered by either adds one run to each. **Cap:** at most two steps per loss family, which is at most 4 extra A/B runs and 2 extra C runs in total. When the cap is reached, or approval is withheld, the stage reports the edge result as a stated limitation and proceeds with the best eligible weight.
4. **Convergence check (AMENDMENT A6, optional).** Evaluate validation probe error and the shape statistics at steps 24,000 and 30,000 (one extra probe fit per run, on train features only, validation scoring only). The budget is judged adequate when both change by less than 5% between the two steps in every run. If a run fails the check, the proposed response is one repeat of the whole grid at 60,000 steps for all arms, **subject to user approval and capped at a single doubling**; without approval the results are reported as budget-limited. Without A6, the stage simply runs 30,000 steps and reports the final-step measurements, and budget adequacy stays an unverified assumption.

## 3–5. Optimizer, schedule, batch, budget, seeds

- **One optimizer for all arms.** LARS is the published choice for VICReg and Rectified LpJEPA ResNets; AdamW is the published choice for LeJEPA (ResNets included) and LeWorldModel. A shared optimizer is required by the one-variable condition. With AdamW the ratio ρ is the main quantity that distinguishes weights, since updates are approximately invariant to an overall loss scale. That invariance is approximate: ε, gradient clipping, weight-decay balance, and mixed-precision numerics all depend on absolute scale. For this reason every arm keeps the coefficient on `L_inv` at exactly 1, absolute loss and gradient norms are logged per term, and ε, precision, and clipping are shared settings. Limit: one shared learning rate may suit arms unequally; a per-arm learning-rate sweep would add selection freedom I advise against, and the optional convergence check (A6) is the available guard.
- **Weight decay.** It shrinks the final projector layer and pulls feature scale below the target, opposed by restoring forces of different shape in each loss family. A small value (5e-4) is intended to keep that arm-dependent bias small; the moment measurements show what remains.
- **Batch 512.** Rectified LpJEPA reports saturation at 128 to 256 for classification accuracy. Here the A/B contrast lives in the tails, and 512 gives twice as many tail samples per direction as 256 (about 5 above the 99th percentile instead of 2 to 3). I expect a less noisy sliced Wasserstein estimate from that and have not measured it. Memory and time cost at this batch size are unmeasured.
- **Budget in steps.** Fixed optimizer steps equalize updates across arms regardless of population choice. 30,000 is my estimate for simple scenes; published budgets (100 to 1,000 epochs) refer to harder data, and adequacy here is unverified.
- **Seeds.** Shared initial weights and shared random streams within a seed make arm differences paired. Two seeds indicate consistency of direction and support no significance claim, as the proposal already states.
- **fp32, no clipping.** Sorted-quantile tails and kurtosis are precision-sensitive, and the model is small. Clipping acts unequally on arms when active. Pre-registered fallback: if any run diverges, all arms rerun with clipping at 1.0 (LeWorldModel's value).

## Risks to keep in view

- **128 dimensions from about 5 scene degrees of freedom.** PushT frames vary almost only in pusher position and T pose. A full-rank isotropic 128-d target forces highly nonlinear features in every arm, which may show up as a gap between linear and MLP probes and a weak distance-to-pose correlation. Dimension is fixed by scope; this is context for reading results.
- **Projector output.** Recommended shared form: 512 → 2048 → 2048 → 128 with BatchNorm and ReLU on hidden layers and a plain linear output [P: LeJEPA minimal example]. A normalization layer on the output would enforce per-coordinate moments in all arms by construction and change what the regularizers have to do.

## Unresolved user decisions

1. **Frame population:** phase-0 train frames (recommended) or all raw train frames.
2. **AMENDMENT A2:** adopt the admissibility gate before probe-based selection, and confirm its thresholds.
3. **Selection scalar.** The plan lists linear and MLP probes over three quantities and needs one number. Everything below is frozen in writing before any candidate encoder is evaluated, and the test split plays no part in defining or tuning it.
   - **Proposal.** MLP probe fitted on train features; score on validation = mean of three group errors: T position, T angle, pusher position.
   - **Standardization from train only.** Position targets are standardized with means and standard deviations computed on the train split, and those constants are reused unchanged for validation and test.
   - **Angle tradeoff.** Standardizing sin θ and cos θ separately gives each component unit variance, and because T angles are unevenly distributed the two components get different scales, so the same angular error is penalized differently at different orientations. Leaving sin θ and cos θ unstandardized (chord error on the unit circle) penalizes all orientations equally, and the angle's weight relative to positions is then set by convention through the equal-weight group mean. I recommend the unstandardized chord error for that reason; the alternative is defensible and the choice should be explicit.
   - **Visibility mask.** Rows where the pusher disc lies fully outside the image (3,160 of 2,336,736 rows overall) are excluded from the pusher term by a fixed geometric rule on simulator state, defined once and applied identically to every arm and split. The mask uses state for measurement only.
   - **Linear probe.** Reported alongside, and excluded from selection.
4. **Run count and optional checks.** Plan: second seed only at the chosen weight (12 runs + A′). **AMENDMENT A3 (optional):** both seeds at all weights (18 runs + A′) and selection on the seed average, which reduces selection noise from one seed and 9 validation families. **AMENDMENTS A5 and A6 (optional):** the capped grid extension and the convergence check above, each needing approval when triggered.
5. **A′ form and breadth.** Form: common-`L_inv` normalization with ρ_A′ = 0.2105 (recommended) or LeJEPA's native scaling. Breadth: one run at λ = 0.05 (plan), or **AMENDMENT A4 (optional):** λ ∈ {0.02, 0.05, 0.1} (ρ_A′ = 0.0816, 0.2105, 0.4444), all published values, to show where A sits relative to SIGReg across strengths.

## Recommended options for the gate and the selection score (added at the user's request)

Both rules are written down and frozen before any candidate encoder is evaluated. Neither uses planning results or the test split. "Frame set" below means the Diffusion Forcing frame set chosen in decision 1.

### Distribution-quality gate: adopt (amends plan condition 9)

- **Where it is measured.** Final checkpoint, eval mode, unaugmented frames of the train split's frame set. Train is used because it is the population the regularizer acted on and it is large; validation has 9 families and is reported alongside without gating.
- **Moment conditions, all arms.** Every coordinate mean within ±0.1, and every eigenvalue of the feature covariance within [0.5, 2.0].
- **Shape conditions.** A and A′: median over the 128 coordinates of excess kurtosis within ±0.5. B: that median at least 1.5 (target 3). C: none, since C has no shape target.
- **A and B get no repair.** They must pass on raw features. Whitening is a linear map that can mix coordinates; a Gaussian survives that and an axis-aligned Laplace does not, so whitening would alter the experimental variable in one arm only.
- **C's permitted whitening.** If raw C features pass the moment conditions, C is used as is. If they fail, C is whitened once with mean subtraction and symmetric (ZCA) whitening fitted on those same train features, the choice closest to leaving features unchanged. This is allowed only when every raw covariance eigenvalue is at least 0.05, so the transform does not amplify near-empty directions by more than about 4.5 times. The frozen transform then becomes part of encoder C everywhere downstream: probes, the selection score, cached features, Diffusion Forcing, and planning. A C weight whose raw covariance has an eigenvalue below 0.05 is ineligible.
- **Selection.** Within each arm, only weights that pass are eligible; the lowest selection score among them wins.
- **If an arm has no eligible weight.** The stage stops before Diffusion Forcing and reports to the user, consistent with the plan's existing rule to stop when B is not heavier-tailed than A. Nothing is extended or rerun automatically.
- **Thresholds.** The numbers above are my proposed values. They may be revised once against the training-free response table and only before any encoder exists.

### One selection score: adopt (fills an unspecified detail of the plan)

- **Probe.** The plan's 2-layer MLP, fitted on train-split features of the frame set (eval mode, unaugmented), with one fixed configuration (width, epochs, learning rate, seed) shared by every arm and weight. Targets: T x, T y, sin θ, cos θ, pusher x, pusher y.
- **Score.** Mean of three group errors on validation, each expressed as the fraction of train variance left unexplained (1 means no better than predicting the train mean):
  - T position: squared error in x and y, each divided by its train variance, averaged.
  - T angle: squared error in sin θ and cos θ, averaged, divided by one shared constant, the train value of the same quantity for the mean predictor. A single constant for both components keeps the penalty equal at every orientation and puts the angle on the same footing as the positions.
  - Pusher position: as for T position, over visible rows only.
- **Constants from train only.** All means, variances, and the angle constant are computed on the train split and reused unchanged on validation and test.
- **Visibility mask.** A row is excluded from the pusher group, in probe fitting and scoring, when the pusher centre lies more than 15 px outside the 512-px image on any side (the whole disc is off-screen). Partially visible rows stay. The rule is identical for every arm and split and uses simulator state for measurement only.
- **Tie rule.** If two eligible weights differ by less than 2% in score, the one closer to the published anchor (ρ = 10 for A and B, ρ = 1 for C) is chosen.
- **Not used for selection.** The linear probe, the per-group errors, and pose errors in pixels and degrees are reported for every encoder. The test split is used only for final reporting with the frozen definition.

## Sources

| Source | Pin | What I took from it |
| --- | --- | --- |
| LeJEPA, arXiv 2511.08544 | paper HTML | Loss form; λ = 0.05; batch ≥ 128; 1,024 slices; 17 knots; SIGReg × batch size; AdamW with lr ∈ {5e-3, 5e-4}, wd ∈ {1e-1, 1e-2, 1e-5}; warmup + cosine |
| `rbalestr-lab/lejepa` | `c293d291` | README: lr 5e-4, wd 5e-4 for ResNets, final lr = peak/1000. MINIMAL.md: λ = 0.02 with 4 views, batch 256, projector [2048, 2048, d] with BatchNorm |
| Rectified LpJEPA, arXiv 2602.01456 | paper HTML | Sliced Wasserstein regularizer (Eq. 16); ImageNet-100 weights 25 / 125, 8,192 projections, batch 128; batch-size saturation at 128 to 256 |
| `YilunKuang/rectified-lp-jepa` | `5ae61ab4` | CIFAR-100 ResNet-18 config: 25 / 250, batch 256, LARS 0.03, wd 1e-4; loss code normalizations; Laplace scale 1/√2 |
| VICReg, arXiv 2105.04906 | paper HTML | 25 / 25 / 1; ε = 1e-4; instability when weights differ greatly; covariance divided by dimension |
| `facebookresearch/vicreg` | `4e12602f` | Loss normalization; bias and norm excluded from weight decay; warmup + cosine |
| LeWorldModel, arXiv 2603.19312 | paper HTML | PushT, frameskip 5; SIGReg weight robust over [0.01, 0.2]; projector with BatchNorm |
| `lucas-maes/le-wm` | `8edfeb33` | PushT config: AdamW lr 5e-5, wd 1e-3, batch 128, 100 epochs, SIGReg 0.09, clip 1.0 |

Unverified: the learning rate in the LeJEPA README carries no stated batch size; LeWorldModel's settings are for end-to-end training with a predictor on 224-px ViT inputs and transfer only loosely; the existence and size of a target-dependent noise floor, the benefit of batch 512, the adequacy of 30,000 steps, and all run costs are unmeasured. I ran no training, calibration, or loss computation for this document.

Correction (2026-11-26, found while reviewing `docs/encoder_training.md`): two statements in my recipe were under-specified.
- **C per-view reductions.** "Regularizer applied to each view and averaged" is exact for A, B, and A′. Official VICReg (`main_vicreg.py` at `4e12602f`) averages the variance term over the two views and **sums** the covariance term over them. The anchor ρ = 1 equals VICReg's 25 / 25 / 1 only with `R_C = mean_views(var) + sum_views(cov)/25`; if both terms are view-averaged, the equivalent coefficient is 2/25. The same code centres features, uses the unbiased variance with ε = 1e-4 inside the square root, and divides the covariance by (batch − 1).
- **SIGReg integration range.** The LeJEPA paper recommends 17 points on [−5, 5]. The released code at `c293d291` defaults to 17 points on [0, 3] with symmetric weights (equivalent to [−3, 3]), as do its minimal example and LeWorldModel. A′ must name one. I recommend the released code's default, since A′ stands for the published implementation; the paper reports negligible sensitivity to this range.

Revision note (2026-11-26, after `t1c`'s review): A′ is now expressed on the common view-consistency term with the derived coefficient, and the native-scaling alternative is a user decision; the claim that only ρ matters under AdamW is corrected; grid extension and the convergence check are optional, capped, approval-gated amendments; cost and noise-floor statements carry no quantitative claims; the selection scalar specifies train-only standardization, a frozen definition, the angle tradeoff, and no test involvement.
