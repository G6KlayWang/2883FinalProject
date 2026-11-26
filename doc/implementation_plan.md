# Implementation plan: Gaussian feature targets for Diffusion Forcing planning

Nov 26, 2026 · Charles Wang

## Question

Does training a frozen visual encoder toward a Gaussian feature distribution make Diffusion Forcing better at predicting trajectories and planning on PushT, compared with a Laplace target or a moments-only regularizer? All arms share architecture, data, and the first two moments, so the designed difference is the shape of the feature distribution.

If there is a gain, the plan also asks where it shows up: in prediction accuracy, in the goal distance used for planning, or in neither.

## Data

PushT noisy expert replay, originally released with DINO-WM: about 18,500 to 20,000 episodes (sources differ) of 100 to 300 steps each, with images, 2-D pusher actions, and simulator states.

| Item | Source | Notes |
| --- | --- | --- |
| Dataset (primary) | [quentinll/lewm-pusht](https://huggingface.co/datasets/quentinll/lewm-pusht) on Hugging Face, file `pusht_expert_train.h5.zst` | 12.2 GB download, 46 GB unpacked |
| Dataset (alternative) | [DINO-WM OSF release](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28), folder `pusht_noise` | Same trajectories in DINO-WM's own format |
| Simulator for evaluation | [DINO-WM repository](https://github.com/gaoyuezhou/dino_wm), PushT environment | Needed to roll out plans and score success |

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('quentinll/lewm-pusht', 'pusht_expert_train.h5.zst', repo_type='dataset', local_dir='data')"
zstd -d data/pusht_expert_train.h5.zst   # needs the zstd command-line tool
```

Preparation:

1. Open the file and confirm the image resolution, the action shape, and that the state holds the T's position and angle and the pusher's position.
2. Split by episode into 90% train, 5% validation, 5% test with a fixed seed, before fitting anything.
3. Use a frameskip of 5: each step stacks 5 consecutive actions into one 10-d action and keeps the frame that follows them.
4. Resize frames to 96 x 96 for the encoders.
5. Use simulator states only for pose probes and evaluation, never for training.

## Arms

Four encoders share architecture, data, augmentations, and the view-consistency loss; only the regularizer differs.

| Arm | Regularizer | Target | Mean, covariance | Shape | Role |
| --- | --- | --- | --- | --- | --- |
| A | Sliced Wasserstein | N(0, I) | 0, I | Gaussian | Reference: LeJEPA's target |
| B | Sliced Wasserstein | Product Laplace, scale 1/√2 | 0, I | Heavy-tailed | Same moments as A, different shape |
| C | VICReg variance and covariance terms | None | About 0, about I | Unconstrained | Whitened features with no shape target |
| A′ | Original SIGReg | N(0, I) | 0, I | Gaussian | Checks that A behaves like published LeJEPA |

A and B run the same loss code and differ only in the line that samples the target. A′ gets the encoder checks but no Diffusion Forcing run unless it differs clearly from A.

## Models to train

Three kinds of model are trained: encoders, pose probes, and Diffusion Forcing. The planner is not trained.

| Model | Architecture | Input to output | Runs |
| --- | --- | --- | --- |
| Encoders A, B, C | ResNet-18 plus 3-layer MLP projector | 96 x 96 frame to 128-d feature | 3 regularizer weights per arm, then a second seed at the chosen weight: 12 |
| Encoder A′ | Same | Same | 1 |
| Pose probes | Linear and 2-layer MLP on frozen features | Feature to T position and angle (x, y, sin, cos) and pusher position | One pair per encoder |
| Diffusion Forcing | Small causal transformer, identical for every arm | Past features and actions to denoised tokens of (10-d action, 128-d next feature) | Arms A, B, C on 2 encoder seeds: 6 |

Encoder settings, shared by all arms:

- Loss: view-consistency between augmented views of the same frame, plus weight x regularizer.
- Augmentations: appearance only (colour jitter, blur, noise). No crops, flips, or rotations, because they change the pose.
- Regularizer in A and B: 1,024 random unit directions plus the 128 coordinate axes, with directions and target samples redrawn every step. Draw as many target samples as the batch holds.
- Batch size 256 or larger.
- Weight selection: lowest validation pose-probe error, never planning results.
- After training: freeze the encoder and cache features for all splits, one file per encoder.

Diffusion Forcing settings, identical for every arm:

- Each token joins a normalized 10-d action block and the 128-d feature of the frame that follows it.
- Every token gets an independently sampled noise level; the model predicts the clean token.
- Actions and features have separate mean-squared-error losses so the larger feature vector does not dominate.
- Training runs on cached features with the same architecture, horizon, step count, and learning rate.

Planner: sample a short future trajectory from the trained model, guided to bring the predicted final feature close to the encoded goal image. Execute the first action block, encode the new frame, and plan again.

## Evaluation

Evaluation has two stages: checks on all four encoders before any Diffusion Forcing run, then prediction and planning on arms A, B, and C.

Encoder checks, on held-out features:

| Check | Measurement | Expected |
| --- | --- | --- |
| Shape | Excess kurtosis per coordinate and per random projection; Epps-Pulley statistic | A near 0; B near 3 on coordinates |
| Moments | Feature mean and the eigenvalues of the covariance | Mean 0 and covariance I in every arm; if C is off, whiten it with a transform fitted on training features |
| Information | Pose-probe error, linear and MLP | Similar across arms; report the gap if not |
| Goal metric | Rank correlation between feature distance and true pose distance over random pairs of frames | Reported per arm |
| A against A′ | The four rows above | Close agreement; if not, give A′ a Diffusion Forcing run |

If B's coordinates do not come out heavier-tailed than A's, stop and fix the regularizer before training Diffusion Forcing.

Diffusion Forcing evaluation, arms A, B, C:

| Measure | How |
| --- | --- |
| Prediction error | Predict held-out trajectories, decode predicted features to pose with that arm's probe, and report pose error at fixed horizons |
| Planning success | Closed-loop planning in the simulator on one fixed set of 50 or more episodes shared by all arms; the goal is a later frame from a held-out trajectory; success by the simulator's criterion |
| Final pose error | Distance between the final and goal T pose, for partial credit |
| Cost | Training time to a fixed step count and planning time per step |

Training losses are not compared across arms, because a lower loss in one feature space does not mean better pose prediction. Pose error is the shared unit.

## Run order and compute

One RTX PRO 6000 (96 GB) is enough for every step; a second one shortens the Diffusion Forcing stage.

1. Download and inspect the data, then build the split and the frameskip loader.
2. Train encoders A, B, and C at three weights each, and A′ once. These are small and can share one GPU.
3. Cache features, train pose probes, run the encoder checks, and pick one weight per arm.
4. Train the second-seed encoders at the chosen weights and repeat the checks.
5. Train Diffusion Forcing on A, B, and C with the first encoder seed.
6. Evaluate prediction error and planning.
7. Repeat steps 5 and 6 on the second encoder seed.

Steps 1 to 6 with one seed are the minimum complete result.

## Reading the results

Each outcome answers the question, including a null result.

| Result | Conclusion |
| --- | --- |
| A, B, and C perform alike | Only the first two moments matter for Diffusion Forcing; shape does not |
| A is best, and pose-probe error is similar across arms | Gaussian shape helps |
| A and B are alike, and both beat C | A full distribution target helps, but its shape does not |
| B is best | Heavier-tailed features suit this planner better than Gaussian ones |
| Arms differ in planning but not in prediction error | The goal distance, not trajectory learning, explains the gap; check the goal-metric correlation |
| Arms differ in pose-probe error | The encoders kept different information, so the gap cannot be credited to shape |

## Limitations

- A regularizer changes what the encoder learns as well as the shape of its features. Pose probes detect this but cannot remove it.
- One non-Gaussian target, one task, and two seeds support a claim about Gaussian against Laplace on PushT, not about Gaussian being the best shape.
- A Laplace target differs from a Gaussian in tail weight and in axis alignment, and this design does not separate the two.
- Arm A enforces LeJEPA's target with a sliced Wasserstein loss, not the published SIGReg loss. A′ is the check on that substitution.

## Sources

- [LeJEPA](https://arxiv.org/abs/2511.08544): the Gaussian target and the SIGReg loss used in A′
- [Rectified LpJEPA](https://arxiv.org/abs/2602.01456): the sliced Wasserstein regularizer and the Laplace target used in A and B
- [VICReg](https://arxiv.org/abs/2105.04906): the regularizer in C
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392): the sequence model and sampling scheme
- [DINO-WM](https://arxiv.org/abs/2411.04983): the PushT dataset and planning setup
