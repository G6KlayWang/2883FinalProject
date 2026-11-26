# Research proposal: does a Gaussian feature target help Diffusion Forcing plan?

Charles Wang · Computer vision final project · Nov 26, 2026

This file explains what the research is and why, so that anyone writing code for it (including a coding agent) knows what the experiment must preserve. The step-by-step build is in [implementation_plan.md](implementation_plan.md). This version replaces the original course proposal, which compared LeJEPA against post-hoc normalized DINOv2 features; reviewer and instructor feedback moved the comparison to encoders that are trained toward different targets.

## Summary

LeJEPA trains visual encoders so that their features follow an isotropic Gaussian distribution, and argues this is the best distribution for downstream tasks. That argument is about static prediction from features (probing). This project tests whether the same property helps a different downstream use: learning action-conditioned trajectories and planning toward a goal image with Diffusion Forcing, on the PushT pushing task.

The experiment trains several encoders that are identical except for the distribution their features are pushed toward, freezes them, trains the same Diffusion Forcing model on each, and compares prediction and planning. It is a controlled comparison, not an attempt to build the best planner.

## Research question

> On PushT with frozen encoders, holding architecture, data, and the first two moments of the features fixed, does training toward a Gaussian target, rather than a Laplace target or a moments-only regularizer, improve Diffusion Forcing's trajectory prediction or planning? If it does, does the gain show up in prediction accuracy, in the goal distance used for planning, or in neither?

Competing hypotheses the experiment is built to tell apart:

| | Hypothesis | Predicts |
| --- | --- | --- |
| H0 | Only mean and covariance matter | A, B, and C perform alike |
| H1 | Gaussian shape helps | A beats B and C, with similar pose information in all arms |
| H2 | A full distribution target helps, whatever its shape | A and B alike, both beat C |
| H3 | Heavier tails help | B beats A |

Every outcome, including H0, is a result. The code should never be tuned to make one arm win.

## Why the question is open

- LeJEPA's optimality result concerns the bias and variance of probes on static embeddings. It says nothing about conditional dynamics, about diffusion in feature space, or about Euclidean distance to a goal feature.
- Diffusion Forcing adds isotropic noise, so each feature direction's signal-to-noise ratio depends only on its variance. Whitening alone should therefore capture most of the noise-schedule benefit, and shape would matter only through how hard the denoising function is to learn.
- A Gaussian marginal does not imply simple dynamics or a goal distance that tracks object pose.
- Regularizers do not always reach their target, so the achieved feature distribution has to be measured, not assumed.

## Design

Four encoders share architecture, data, augmentations, and the view-consistency loss. Only the regularizer differs.

| Arm | Regularizer | Target | Shape | Role |
| --- | --- | --- | --- | --- |
| A | Sliced Wasserstein | N(0, I) | Gaussian | LeJEPA's target |
| B | Sliced Wasserstein | Product Laplace, scale 1/√2 (unit variance) | Heavy-tailed | Same moments as A, different shape |
| C | VICReg variance and covariance terms | None | Unconstrained | Whitened features with no shape target |
| A′ | Original SIGReg | N(0, I) | Gaussian | Checks that A behaves like published LeJEPA |

Why these arms:

- **A against B** is the main test. They run the same loss code and differ only in the line that samples the target, so the designed difference is distribution shape.
- **A against C** asks whether a Gaussian target beats features that are only whitened.
- **A′** exists because SIGReg is written for a Gaussian target and cannot take a Laplace one directly. Arm A swaps in a sliced Wasserstein loss so that A and B can share code; A′ verifies the swap is harmless. A′ gets encoder checks only, unless it differs clearly from A.

## System

```
PushT frames ──> encoder (one per arm, trained then frozen) ──> cached 128-d features
                                                                    │
                     pose probes and encoder checks <───────────────┤
                                                                    │
cached features + actions ──> Diffusion Forcing (same config per arm)
                                                                    │
current frame + goal image ──> guided sampling ──> first action ──> simulator ──> replan
```

1. **Encoder.** ResNet-18 plus an MLP projector maps a 96 x 96 frame `o_t` to a 128-d feature `e_t`. Loss = view-consistency between augmented views of the same frame + weight × regularizer. The projector output is the feature, because it is the space the regularizer acts on.
2. **Checks and probes.** On held-out features: shape, moments, pose information, and goal-metric quality (see Measurements).
3. **Diffusion Forcing.** A small causal transformer over tokens `x_t = [ã_t, e_{t+1}]`, where `ã_t` is a normalized 10-d action block (5 stacked 2-D actions, frameskip 5). Each token gets its own noise level: `x_t^k = sqrt(ᾱ_k) x_t + sqrt(1 − ᾱ_k) ε`, with `ε ~ N(0, I)`. The model predicts the clean token from noisy tokens and history. Actions and features have separate mean-squared-error losses.
4. **Planner.** Sample a short future trajectory conditioned on history, with guidance that reduces the distance between the predicted final feature and the encoded goal image. Denoise near-future tokens before distant ones. Execute the first action block, observe, and plan again.

### Regularizers

Sliced Wasserstein, used by A and B:

```python
def sliced_wasserstein(z, target, n_proj=1024):
    # z, target: (batch, dim), same batch size
    d = z.shape[1]
    dirs = torch.randn(n_proj, d, device=z.device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    dirs = torch.cat([dirs, torch.eye(d, device=z.device)])  # add the coordinate axes
    pz = (z @ dirs.T).sort(dim=0).values
    pt = (target @ dirs.T).sort(dim=0).values
    return ((pz - pt) ** 2).mean()

target = torch.randn_like(z)                                                    # arm A
target = torch.distributions.Laplace(0., 2 ** -0.5).sample(z.shape).to(z.device)  # arm B
```

The coordinate axes are included on purpose. A random projection of 128 independent Laplace coordinates is almost Gaussian (excess kurtosis about 0.07 instead of 3), so random directions alone would barely separate A from B.

- **VICReg (C):** a variance term that keeps each coordinate's standard deviation at 1 or above, and a covariance term that penalizes squared off-diagonal covariance. No term looks at shape.
- **SIGReg (A′):** the Epps–Pulley characteristic-function test against a standard Gaussian, averaged over random 1-D projections, as in the LeJEPA release.

## What the code must guarantee

These are the conditions that make the comparison valid. Breaking one silently invalidates the result.

1. **One variable.** Encoder arms differ only in the regularizer and its weight. Architecture, data, augmentations, batch size, schedule, and feature dimension come from one shared config.
2. **Shared loss for A and B.** One function, with the target sampler as the only switch.
3. **Identical Diffusion Forcing.** Same architecture, horizon, noise schedule, step count, learning rate, and sampling budget for every arm.
4. **Frozen encoders.** Diffusion Forcing trains on cached features and never updates an encoder.
5. **No leakage.** Keep whole trajectory families together with seed 42 before fitting anything, following the user-approved family-split amendment in [implementation_plan.md](implementation_plan.md#data). This refines the original episode split to prevent noisy replay siblings crossing sets; the research hypotheses are unchanged. Normalizers, whitening transforms, and probes are fitted on training data only.
6. **Simulator state is for measurement only.** It is used for pose probes and evaluation, never as a training input or target for encoders or Diffusion Forcing.
7. **Pose-preserving augmentations.** Appearance changes only (colour jitter, blur, noise). No crops, flips, or rotations.
8. **Same evaluation.** One fixed set of episodes, goals, and planner seeds shared by all arms.
9. **Selection without planning results.** Regularizer weights are chosen by validation pose-probe error, from the same grid for each arm.
10. **Measured, not nominal, shape.** Log shape and moment statistics for every encoder; the analysis uses these values.
11. **Comparable units.** Training losses are not compared across arms. Predicted features are decoded to pose with that arm's probe, and arms are compared in pose error.
12. **Reproducibility.** Every run records its config, seed, and code version, and writes metrics to a file that the analysis reads.

## Measurements

Encoder level, on held-out features:

| Measure | What it tells us |
| --- | --- |
| Excess kurtosis per coordinate and per random projection; Epps–Pulley statistic | Whether each arm reached its intended shape (A near 0, B near 3 on coordinates) |
| Feature mean and covariance eigenvalues | Whether moments match across arms |
| Pose-probe error (linear and MLP) for T position, T angle, pusher position | Whether arms kept the same information |
| Rank correlation between feature distance and true pose distance | Whether the goal distance is meaningful |

Diffusion Forcing level, arms A, B, C:

| Measure | What it tells us |
| --- | --- |
| Prediction error in pose units at fixed horizons | Whether trajectory learning got easier |
| Closed-loop planning success and final pose error | Whether planning got better |
| Training and planning time | Cost |

A gap in prediction error points to trajectory learning. Equal prediction with unequal planning points to the goal distance. A gap in pose-probe error means the encoders kept different information, so a planning gap cannot be credited to shape.

## Scope

In scope: PushT only; frozen encoders; three trained arms plus one check arm; one Diffusion Forcing configuration; two encoder seeds.

Out of scope for now:

- DINOv2 or any pretrained encoder
- Post-hoc transforms as experimental arms (warping, Gaussianizing)
- A deterministic predictor or a search-based planner as a baseline
- End-to-end training of encoder and predictor
- Other environments, larger feature dimensions, or multiple Diffusion Forcing sizes
- Occlusion or frame-drop stress tests

Limits on what can be claimed:

- A regularizer changes what the encoder learns as well as the shape of its features. Probes detect this but cannot remove it.
- The result supports a claim about Gaussian against Laplace on PushT, not about Gaussian being the best shape.
- A Laplace target differs from a Gaussian in tail weight and in axis alignment; the design does not separate them.

## Open items to confirm in code

- The dataset repository name and file sizes come from pages that reference the dataset, not from the dataset page itself. Confirm on download.
- The image resolution, action shape, and state layout inside the HDF5 file are unverified. Inspect the file before building the loader, and confirm the state includes the T's position and angle and the pusher's position.

## Notation

| Symbol | Meaning |
| --- | --- |
| `o_t` | Image observation at step t |
| `e_t` | Frozen 128-d feature of `o_t` (projector output) |
| `ã_t` | Normalized 10-d action block between step t and t+1 |
| `x_t` | Diffusion Forcing token `[ã_t, e_{t+1}]` |
| `k_t`, `ᾱ_k` | Per-token noise level and its cumulative signal fraction |
| Arm | One encoder training condition (A, B, C, A′) |
| Pose | T position and angle, plus pusher position, from the simulator |

## References

- [LeJEPA](https://arxiv.org/abs/2511.08544): the Gaussian target and SIGReg
- [Rectified LpJEPA](https://arxiv.org/abs/2602.01456): the sliced Wasserstein regularizer and Laplace target
- [VICReg](https://arxiv.org/abs/2105.04906): the moments-only regularizer
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392): the sequence model and sampling scheme
- [DINO-WM](https://arxiv.org/abs/2411.04983): the PushT dataset and visual goal planning setup
- [LeWorldModel](https://arxiv.org/abs/2603.19312): SIGReg trained end to end with a predictor, the closest related system
