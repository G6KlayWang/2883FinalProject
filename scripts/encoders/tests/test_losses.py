import importlib
from pathlib import Path
import unittest
import torch


def implementation(name):
    path = Path("scripts/encoders") / (name + ".py")
    if not path.exists():
        raise AssertionError(f"{name} is not implemented")
    return importlib.import_module("scripts.encoders." + name)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.losses = implementation("losses")

    def test_swd_quantiles_axes_and_gradients_by_value(self):
        z = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
        value = self.losses.sliced_wasserstein(z, target, torch.eye(2))
        self.assertAlmostEqual(value.item(), 2.0)
        value.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0.0)

    def test_vicreg_sums_two_view_covariances_instead_of_averaging(self):
        z = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], requires_grad=True)
        terms = self.losses.vicreg(z, z)
        self.assertAlmostEqual(terms["variance"].item(), 0.0)
        self.assertAlmostEqual(terms["covariance"].item(), 8.0)
        self.assertAlmostEqual(terms["regularizer"].item(), 0.32, places=6)
        terms["regularizer"].backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_sigreg_matches_pinned_zero_feature_reference_and_native_scaling(self):
        z = torch.zeros(2, 1, requires_grad=True)
        value = self.losses.sigreg(z, z, torch.ones(1, 1))
        self.assertAlmostEqual(value.item(), 0.8040951594633291, places=6)
        streams = implementation("rng").RandomStreams(0)
        common = self.losses.encoder_loss(
            z, z, "Aprime", 0.2105263157894737, streams, "common"
        )
        streams = implementation("rng").RandomStreams(0)
        native = self.losses.encoder_loss(
            z, z, "Aprime", 0.2105263157894737, streams, "native"
        )
        self.assertAlmostEqual(
            native["total"].item(), 0.2375 * common["total"].item(), places=6
        )
        common["total"].backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_ab_share_directions_and_dedicated_target_stream(self):
        rng = implementation("rng")
        a, b = rng.RandomStreams(7), rng.RandomStreams(7)
        da = self.losses.draw_directions(3, 4, a["directions"])
        self.losses.sample_target((16, 3), "B", b["targets"])
        db = self.losses.draw_directions(3, 4, b["directions"])
        torch.testing.assert_close(da, db, rtol=0, atol=0)
        torch.testing.assert_close(da[:, -3:], torch.eye(3), rtol=0, atol=0)
        torch.testing.assert_close(da.norm(dim=0), torch.ones(7))
        target_a = self.losses.sample_target((30000, 2), "A", a["targets"])
        target_b = self.losses.sample_target((30000, 2), "B", b["targets"])
        for values in (target_a, target_b):
            self.assertLess(abs(values.var().item() - 1), 0.035)
            self.assertTrue(torch.isfinite(values).all())
        self.assertGreater(
            (target_b**4).mean().item(), (target_a**4).mean().item() + 1.0
        )

    def test_bad_batches_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "batch"):
            self.losses.vicreg(torch.zeros(1, 2), torch.zeros(1, 2))

    def test_vicreg_retains_small_off_diagonals_beside_large_diagonals(self):
        z = torch.tensor([[-1e4, -1e-4], [1e4, 1e-4]], requires_grad=True)
        terms = self.losses.vicreg(z, z)
        # Each covariance has off-diagonal entries 2,2: squared sum /D = 4.
        self.assertAlmostEqual(terms["covariance"].item(), 8.0, places=5)

    def test_nondefault_aprime_lambda_requires_consistent_rho_and_explicit_convention(
        self,
    ):
        rng = implementation("rng")
        z = torch.tensor([[-1.0, 0.5], [0.25, -0.75]])
        rho = 4 * 0.02 / 0.98
        common = self.losses.encoder_loss(
            z, z, "Aprime", rho, rng.RandomStreams(0), "common", native_lambda=0.02
        )
        native = self.losses.encoder_loss(
            z, z, "Aprime", rho, rng.RandomStreams(0), "native", native_lambda=0.02
        )
        self.assertAlmostEqual(
            native["total"].item(), 0.98 / 4 * common["total"].item(), places=6
        )
        with self.assertRaisesRegex(ValueError, "rho|lambda"):
            self.losses.encoder_loss(
                z, z, "Aprime", rho, rng.RandomStreams(0), "native", native_lambda=0.05
            )

    def test_swd_dispatcher_averages_two_views_and_c_uses_vicreg_factors(self):
        from unittest.mock import patch

        rng = implementation("rng")
        z = torch.zeros(2, 1)
        for arm in ("A", "B"):
            with self.subTest(arm=arm), patch.object(
                self.losses,
                "sample_target",
                side_effect=[z.clone(), torch.full_like(z, 2)],
            ):
                terms = self.losses.encoder_loss(
                    z, z, arm, 3.0, rng.RandomStreams(0), "common", projections=1
                )
            self.assertAlmostEqual(terms["regularizer"].item(), 2.0)
            self.assertAlmostEqual(terms["total"].item(), 6.0)
        correlated = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        terms = self.losses.encoder_loss(
            correlated, correlated, "C", 2.0, rng.RandomStreams(0), "common"
        )
        self.assertAlmostEqual(terms["total"].item(), 0.64, places=6)

    def test_sigreg_nonzero_two_view_reference_fixture(self):
        first = torch.tensor([[-1.0, 0.5], [0.25, -0.75]], dtype=torch.float64)
        second = torch.tensor([[-0.5, 0.25], [1.0, -0.25]], dtype=torch.float64)
        statistic = self.losses.sigreg(first, second, torch.eye(2, dtype=torch.float64))
        self.assertAlmostEqual(statistic.item(), 0.2932515951892816, places=8)
