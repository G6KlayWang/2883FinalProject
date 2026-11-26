import copy
import unittest
import torch
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


class AugmentTests(unittest.TestCase):
    def test_appearance_views_are_keyed_without_consuming_global_rng(self):
        augment = implementation("augment")
        config = run_config_dict()["augment"]
        image = torch.linspace(0, 1, 3 * 96 * 96).reshape(3, 96, 96)
        state = torch.get_rng_state().clone()
        a = augment.appearance_view(image, config, seed=7, key=(0, 10, 0))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        torch.rand(100)
        shifted_state = torch.get_rng_state().clone()
        b = augment.appearance_view(image, config, seed=7, key=(0, 10, 0))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), shifted_state, rtol=0, atol=0)
        self.assertEqual(a.shape, image.shape)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        c = augment.appearance_view(image, config, seed=7, key=(0, 10, 1))
        self.assertFalse(torch.equal(a, c))
        torch.set_rng_state(state)

    def test_no_appearance_changes_preserve_every_spatial_pixel(self):
        augment = implementation("augment")
        config = copy.deepcopy(run_config_dict()["augment"])
        config.update(
            jitter_probability=0.0, blur_probability=0.0, noise_probability=0.0
        )
        image = torch.arange(3 * 96 * 96).reshape(3, 96, 96).float() / (3 * 96 * 96)
        torch.testing.assert_close(
            augment.appearance_view(image, config, seed=0, key=(1,)),
            image,
            rtol=0,
            atol=0,
        )
        config["hue"] = 0.9
        with self.assertRaisesRegex(ValueError, "hue"):
            augment.validate_augmentation(config)

    def test_spawned_workers_preserve_logical_batch_and_view_order(self):
        from torch.utils.data import DataLoader
        from scripts.encoders.frames import BatchOrder, KeyedBatches
        from scripts.encoders.tests.test_training_resume import SyntheticImages

        augment = implementation("augment")
        dataset = augment.AugmentedFrames(
            SyntheticImages(), run_config_dict()["augment"], 0
        )

        def batches(workers):
            kwargs = {"multiprocessing_context": "spawn"} if workers else {}
            return list(
                DataLoader(
                    dataset,
                    batch_sampler=KeyedBatches(BatchOrder(32, 16, 0), 0, 2),
                    num_workers=workers,
                    generator=torch.Generator().manual_seed(0),
                    **kwargs
                )
            )

        reference, parallel = batches(0), batches(2)
        for first, second in zip(reference, parallel):
            for key in ("index", "view1", "view2"):
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
