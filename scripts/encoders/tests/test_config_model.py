import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


class ConfigModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_configuration_requires_scientific_choices_and_caps_smoke(self):
        config = implementation("config")
        original = run_config_dict()
        validated = config.RunConfig.from_dict(original)
        self.assertEqual(validated.optimizer.batch_size, 16)
        self.assertEqual(
            validated.digest(),
            config.RunConfig.from_dict(copy.deepcopy(original)).digest(),
        )
        for field in (
            "frame_policy",
            "aprime_convention",
            "tf32_matmul",
            "gate",
            "c_transform",
            "probe",
            "grids",
        ):
            bad = copy.deepcopy(original)
            del bad[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                config.RunConfig.from_dict(bad)
        for section, field, value in [
            ("optimizer", "max_updates", 21),
            ("runtime", "max_seconds", 601),
            ("runtime", "max_frames", 257),
            ("optimizer", "batch_size", 8),
        ]:
            bad = copy.deepcopy(original)
            bad[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                config.RunConfig.from_dict(bad)
        proposed = copy.deepcopy(original)
        proposed.update(profile="production", recipe_status="proposed")
        with self.assertRaisesRegex(ValueError, "approval"):
            config.RunConfig.from_dict(proposed).require_execution_approval()

    def test_c_policy_requires_its_own_approval_even_when_gate_is_disabled(self):
        config = implementation("config")
        value = run_config_dict()
        value.update(profile="production", recipe_status="approved")
        for block in ("grids", "gate", "probe"):
            value[block]["status"] = "approved"
        value["gate"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "approval"):
            config.RunConfig.from_dict(value).require_execution_approval()
        value["c_transform"]["status"] = "approved"
        config.RunConfig.from_dict(value).require_execution_approval()

    def test_run_matrix_has_ten_candidates_and_three_selected_repeats(self):
        config = implementation("config")
        rows = config.run_matrix()
        self.assertEqual(len(rows), 13)
        self.assertEqual(sum(row["stage"] == "candidate" for row in rows), 10)
        self.assertEqual(
            [(r["arm"], r["seed"]) for r in rows if r["stage"] == "repeat"],
            [("A", 1), ("B", 1), ("C", 1)],
        )
        self.assertEqual(len({row["run_id"] for row in rows}), 13)

    def test_random_full_model_same_initial_buffers_and_separate_view_forward(self):
        model = implementation("model")
        a, b = model.build_encoder(0), model.build_encoder(0)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        images = torch.rand(2, 3, 96, 96)
        a.train()
        before = a.projector[1].num_batches_tracked.item()
        first, second = model.forward_views(a, images, images)
        self.assertEqual(first.shape, (2, 128))
        self.assertEqual(a.projector[1].num_batches_tracked.item(), before + 2)
        (first.square().mean() + second.square().mean()).backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in a.parameters()
                if p.grad is not None
            )
        )
        self.assertGreater(
            sum(
                p.grad.abs().sum().item() for p in a.parameters() if p.grad is not None
            ),
            0,
        )

    def test_production_requires_shared_grids_and_declared_weights_and_seeds(self):
        config = implementation("config")
        original = run_config_dict()
        original.update(profile="production", recipe_status="proposed")
        unequal = copy.deepcopy(original)
        unequal["grids"]["B"] = [1.0, 10.0, 40.0]
        with self.assertRaisesRegex(ValueError, "A/B|equal|shared"):
            config.RunConfig.from_dict(unequal)
        for changes in (
            {"rho": 7.0},
            {"seed": 2},
            {"arm": "Aprime", "rho": 4 * 0.05 / 0.95, "seed": 1},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "grid|seed|repeat"
            ):
                config.RunConfig.from_dict({**original, **changes})
