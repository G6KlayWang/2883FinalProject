from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


class ProbeSelectionTests(unittest.TestCase):
    def test_pose_targets_mask_edges_and_shared_angle_denominator(self):
        probes = implementation("probes")
        state = np.array(
            [
                [0, 0, 20, 30, np.pi / 2, 0, 0],
                [-15, -15, 40, 50, 0, 0, 0],
                [-15.1, 100, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        targets = probes.pose_targets(state)
        np.testing.assert_allclose(targets[0], [20, 30, 1, 0, 0, 0], atol=1e-6)
        np.testing.assert_array_equal(
            probes.visibility_mask(state, 15, "expanded_box"), [True, True, False]
        )
        np.testing.assert_array_equal(
            probes.visibility_mask(state, 15, "circle_intersection"),
            [True, False, False],
        )
        truth = np.zeros((2, 6))
        prediction = truth.copy()
        prediction[:, 2] = 1
        prediction[1, 4:] = 999
        scores = probes.group_scores(
            prediction, truth, np.array([1, 1, 1, 9, 1, 1]), np.array([True, False])
        )
        self.assertAlmostEqual(scores["T_angle"], 0.1)
        self.assertAlmostEqual(scores["pusher_position"], 0.0)
        swapped = truth.copy()
        swapped[:, 3] = 1
        self.assertAlmostEqual(
            probes.group_scores(
                swapped, truth, np.array([1, 1, 1, 9, 1, 1]), np.ones(2, dtype=bool)
            )["T_angle"],
            0.1,
        )

    def test_linear_probe_train_only_fit_and_artifact_roundtrip(self):
        probes = implementation("probes")
        x = np.random.default_rng(4).normal(size=(64, 6))
        y = x * [2, 3, 1, 1, 4, 5] + [20, 30, 0, 0, 40, 50]
        visible = np.ones(64, dtype=bool)
        probe = probes.fit_linear_probe(
            x, y, visible, split="train", policy=run_config_dict()["probe"]
        )
        np.testing.assert_allclose(probe.predict(x), y, atol=1e-8)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "linear"
            probe.save(output, {"checkpoint_sha256": "a" * 64})
            restored = probes.load_probe(output, {"checkpoint_sha256": "a" * 64})
            np.testing.assert_allclose(restored.predict(x), y, atol=1e-8)
        with self.assertRaisesRegex(ValueError, "train"):
            probes.fit_linear_probe(
                x, y, visible, split="test", policy=run_config_dict()["probe"]
            )
        with self.assertRaisesRegex(ValueError, "variance"):
            probes.fit_linear_probe(
                x,
                np.ones((64, 6)),
                visible,
                split="train",
                policy=run_config_dict()["probe"],
            )

    def test_mlp_uses_fixed_final_epoch_and_finite_frozen_feature_training(self):
        probes = implementation("probes")
        x = np.random.default_rng(11).normal(size=(48, 6)).astype(np.float32)
        y = x * [2, 3, 1, 1, 4, 5] + [20, 30, 0, 0, 40, 50]
        settings = dict(run_config_dict()["probe"], epochs=3, hidden=16, batch_size=16)
        model = probes.fit_mlp_probe(
            x, y, np.ones(48, dtype=bool), split="train", policy=settings, device="cpu"
        )
        self.assertEqual(len(model.curve), 3)
        self.assertTrue(np.isfinite(model.curve).all())
        self.assertEqual(model.predict(x).shape, (48, 6))
        self.assertTrue(np.isfinite(model.predict(x)).all())

    def test_selection_ties_eligibility_and_diagnostic_rejection(self):
        selection = implementation("selection")

        def row(rho, score, eligible=True):
            return {
                "run_id": str(rho),
                "arm": "A",
                "rho": rho,
                "seed": 0,
                "score": score,
                "eligible": eligible,
                "checkpoint_role": "final",
                "score_split": "validation",
                "protocol_digest": "shared",
                "checkpoint_sha256": str(rho),
                "gate_policy_status": "proposed",
            }

        records = [row(2.5, 1.0), row(10, 1.015), row(40, 0.5, False)]
        chosen = selection.select_weight(records, "A", run_config_dict()["probe"])
        self.assertEqual(chosen["rho"], 10.0)
        with self.assertRaisesRegex(ValueError, "eligible"):
            selection.select_weight(
                [row(10, 1, False)], "A", run_config_dict()["probe"]
            )
        records[0]["checkpoint_role"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "diagnostic|final"):
            selection.select_weight(records, "A", run_config_dict()["probe"])
        records[0]["checkpoint_role"] = "final"
        records[0]["score_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation|test"):
            selection.select_weight(records, "A", run_config_dict()["probe"])

    def test_goal_metrics_use_fixed_pairs_and_tie_aware_ranks(self):
        metrics = implementation("goal_metrics")
        np.testing.assert_allclose(
            metrics.average_ranks(np.array([2.0, 1.0, 2.0])), [1.5, 0.0, 1.5]
        )
        x = np.arange(20, dtype=float).reshape(10, 2)
        pose = np.column_stack((x, np.sin(x[:, 0]), np.cos(x[:, 0]), x))
        result = metrics.goal_correlations(
            x, pose, np.ones(10, dtype=bool), split="validation", pairs=100, seed=1
        )
        self.assertAlmostEqual(result["T_position"]["spearman"], 1.0)
        self.assertEqual(
            result,
            metrics.goal_correlations(
                x, pose, np.ones(10, dtype=bool), split="validation", pairs=100, seed=1
            ),
        )
        with self.assertRaisesRegex(ValueError, "test"):
            metrics.goal_correlations(
                x, pose, np.ones(10, dtype=bool), split="test", pairs=100, seed=1
            )

    def test_complete_selection_refuses_mixed_protocols_and_does_not_approve_proposals(
        self,
    ):
        import copy
        from scripts.encoders.config import (
            RunConfig,
            run_matrix,
            shared_training_controls,
        )
        from scripts.encoders.artifacts import identity_digest

        selection = implementation("selection")
        value = run_config_dict()
        value.update(profile="production", recipe_status="proposed")
        config = RunConfig.from_dict(value)
        protocol = {
            "probe_policy": config.probe,
            "gate_policy": config.gate,
            "frame_policy": config.frame_policy,
            "shared_training_controls": shared_training_controls(config),
        }
        records = []
        for row in run_matrix()[:10]:
            coordinate = (
                2.0 if row["arm"] == "B" else (0.3 if row["arm"] == "Aprime" else 0.0)
            )
            if row["arm"] == "A":
                coordinate = {2.5: -0.4, 10.0: 0.0, 40.0: 0.4}[row["rho"]]
            records.append(
                {
                    **row,
                    "score": 1.0,
                    "score_split": "validation",
                    "eligible": True,
                    "checkpoint_role": "final",
                    "checkpoint_sha256": "fixture",
                    "protocol": protocol,
                    "protocol_digest": identity_digest(protocol),
                    "gate_policy_status": "proposed",
                    "gate": {
                        "median_coordinate_kurtosis": coordinate,
                        "raw_statistics": {
                            "mean": [0, 0],
                            "covariance_eigenvalues": [1, 1],
                            "coordinate_excess_kurtosis": [coordinate, coordinate],
                            "projection_excess_kurtosis": [0, 0],
                        },
                    },
                }
            )
        result = selection.select_all(records, config)
        self.assertEqual(result["status"], "provisional-policy")
        self.assertFalse(result["ready_for_separate_DF_decision"])
        self.assertEqual(
            result["Aprime_vs_all_A"]["median_coordinate_kurtosis"][
                "A_all_weight_range"
            ],
            [-0.4, 0.4],
        )
        changed = copy.deepcopy(records)
        changed[-1]["protocol_digest"] = "different"
        with self.assertRaisesRegex(ValueError, "protocol"):
            selection.select_all(changed, config)
        changed = copy.deepcopy(records)
        changed[3]["checkpoint_role"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "final|diagnostic"):
            selection.select_all(changed, config)
