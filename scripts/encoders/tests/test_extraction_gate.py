from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation
from scripts.encoders.tests.test_training_resume import SyntheticImages, tiny_model


class ExtractionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_clean_extraction_preserves_model_parameters_and_bn_buffers(self):
        extraction = implementation("extract")
        model = tiny_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features"
            extraction.extract_features(
                model,
                SyntheticImages(),
                path,
                {"checkpoint_sha256": "a" * 64, "arm": "A", "checkpoint_role": "final"},
                device="cpu",
                batch_size=16,
            )
            cached = extraction.FeatureCache(path)
            self.assertEqual(cached.features.shape, (32, 128))
            self.assertTrue(np.isfinite(cached.features).all())
            for key, value in before.items():
                torch.testing.assert_close(
                    value, model.state_dict()[key], rtol=0, atol=0
                )
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            diagnostic = extraction.train_mode_comparison(
                model, SyntheticImages(), batch_size=16, device="cpu"
            )
            self.assertEqual(diagnostic["scope"], "sampled-disposable-model-copy")
            for key, value in before.items():
                torch.testing.assert_close(
                    value, model.state_dict()[key], rtol=0, atol=0
                )

    def test_gate_c_train_only_zca_and_persistence(self):
        gate, transforms = implementation("gate"), implementation("transforms")
        features = np.random.default_rng(12).normal(size=(512, 3)) * [2.0, 3.0, 0.5] + [
            3.0,
            2.0,
            1.0,
        ]
        policy = run_config_dict()["gate"]
        report, transform = gate.assess_training_features(
            features, "C", policy, split="train", checkpoint_role="final"
        )
        self.assertTrue(report["eligible"])
        self.assertIsNotNone(transform)
        effective = transform.apply(features)
        np.testing.assert_allclose(effective.mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(np.cov(effective.T), np.eye(3), atol=1e-5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "zca"
            transform.save(path, {"checkpoint_sha256": "a" * 64})
            restored = transforms.ZcaTransform.load(
                path, {"checkpoint_sha256": "a" * 64}
            )
            np.testing.assert_array_equal(restored.apply(features), effective)
        with self.assertRaisesRegex(ValueError, "train"):
            transforms.fit_zca(features, split="validation", eigenvalue_floor=0.05)
        for arm in ("A", "B", "Aprime"):
            report, repair = gate.assess_training_features(
                features, arm, policy, split="train", checkpoint_role="final"
            )
            self.assertFalse(report["eligible"])
            self.assertIsNone(repair)

    def test_gate_rejects_rank_collapse_smoke_and_test_rows(self):
        gate = implementation("gate")
        policy = run_config_dict()["gate"]
        collapsed = np.ones((32, 3))
        report, transform = gate.assess_training_features(
            collapsed, "C", policy, split="train", checkpoint_role="final"
        )
        self.assertFalse(report["eligible"])
        self.assertIsNone(transform)
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            gate.assess_training_features(
                collapsed, "A", policy, split="train", checkpoint_role="diagnostic"
            )
        with self.assertRaisesRegex(ValueError, "train"):
            gate.assess_training_features(
                collapsed, "A", policy, split="test", checkpoint_role="final"
            )

    def test_statistics_report_coordinate_and_projection_kurtosis(self):
        measure = implementation("measure")
        z = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        report = measure.feature_statistics(z, projection_count=3, ep_sample_size=4)
        np.testing.assert_allclose(report["mean"], [0, 0])
        np.testing.assert_allclose(report["covariance_eigenvalues"], [4 / 3, 4 / 3])
        np.testing.assert_allclose(report["coordinate_excess_kurtosis"], [-2, -2])
        self.assertEqual(len(report["projection_excess_kurtosis"]), 3)
        self.assertEqual(report["moment_scope"], "exhaustive")
        self.assertEqual(report["epps_pulley_rows"], 4)
