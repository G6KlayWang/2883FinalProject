import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from scripts.encoders.config import RunConfig
from scripts.encoders.tests.helpers import run_config_dict
from scripts.encoders.tests.test_losses import implementation


class SyntheticImages:
    def __init__(self):
        self.images = np.random.default_rng(13).random(
            (32, 3, 96, 96), dtype=np.float32
        )
        self.ids = np.column_stack((np.arange(32), np.zeros(32, dtype=np.int64)))
        self.identity = {
            "split": "train",
            "frame_policy": "phase0",
            "diagnostic_subset": True,
            "fixture": "images-v1",
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
        )


def short_config(logging_interval=1):
    value = run_config_dict()
    value["optimizer"].update(max_updates=4, warmup_updates=2)
    value["runtime"].update(
        workers=0, checkpoint_every=2, gradient_terms_every=logging_interval
    )
    return RunConfig.from_dict(value)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_learning_rate_step_indexing_and_decay_exclusion_by_effect(self):
        training = implementation("training")
        config = run_config_dict()
        config.update(profile="production", recipe_status="proposed")
        config["optimizer"].update(max_updates=30000, warmup_updates=1000)
        settings = RunConfig.from_dict(config).optimizer
        self.assertAlmostEqual(training.learning_rate_at(1, settings), 5e-6)
        self.assertAlmostEqual(training.learning_rate_at(1000, settings), 5e-4)
        self.assertAlmostEqual(training.learning_rate_at(30000, settings), 5e-7)
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        optimizer = training.make_optimizer(model, settings)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        weight, bias, norm = (
            model[0].weight.detach().clone(),
            model[0].bias.detach().clone(),
            model[1].weight.detach().clone(),
        )
        optimizer.step()
        self.assertFalse(torch.equal(model[0].weight, weight))
        torch.testing.assert_close(model[0].bias, bias, rtol=0, atol=0)
        torch.testing.assert_close(model[1].weight, norm, rtol=0, atol=0)

    def test_exact_resume_including_buffers_optimizer_rng_scheduler_and_stale_rejection(
        self,
    ):
        training = implementation("training")
        images = SyntheticImages()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            direct = training.Trainer(
                tiny_model(),
                short_config(),
                root / "direct",
                {"source": "synthetic", "code": "test"},
                "cpu",
            )
            direct.run(images, stop_at=4)
            partial = training.Trainer(
                tiny_model(),
                short_config(),
                root / "resumed",
                {"source": "synthetic", "code": "test"},
                "cpu",
            )
            checkpoint = partial.run(images, stop_at=2)
            resumed = training.Trainer(
                tiny_model(),
                short_config(),
                root / "resumed",
                {"source": "synthetic", "code": "test"},
                "cpu",
                resume=checkpoint,
            )
            resumed.run(images, stop_at=4)
            self.assertEqual(resumed.completed_updates, 4)
            self.assertEqual(resumed.attempted_updates, 4)
            for name, value in direct.model.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.model.state_dict()[name], rtol=0, atol=0
                )
            self.assertEqual(
                direct.optimizer.param_groups[0]["lr"],
                resumed.optimizer.param_groups[0]["lr"],
            )
            for name, state in direct.streams.state_dict().items():
                torch.testing.assert_close(
                    state, resumed.streams.state_dict()[name], rtol=0, atol=0
                )
            with self.assertRaisesRegex(ValueError, "(?i)stale|count"):
                training.Trainer(
                    tiny_model(),
                    short_config(),
                    root / "resumed",
                    {"source": "synthetic", "code": "test"},
                    "cpu",
                    resume=checkpoint,
                )

    def test_per_term_gradient_logging_does_not_change_updates(self):
        training = implementation("training")
        with tempfile.TemporaryDirectory() as folder:
            one = training.Trainer(
                tiny_model(),
                short_config(1),
                Path(folder) / "logged",
                {"fixture": "same"},
                "cpu",
            )
            two = training.Trainer(
                tiny_model(),
                short_config(1000),
                Path(folder) / "quiet",
                {"fixture": "same"},
                "cpu",
            )
            one.run(SyntheticImages(), stop_at=2)
            two.run(SyntheticImages(), stop_at=2)
            for name, value in one.model.state_dict().items():
                torch.testing.assert_close(
                    value, two.model.state_dict()[name], rtol=0, atol=0
                )
            metrics = [
                json.loads(line)
                for line in (Path(folder) / "logged" / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(all(row["gradient_norm"] > 0 for row in metrics))
            self.assertIn("regularizer_unweighted_gradient_norm", metrics[0])

    def test_smoke_population_caps_and_failed_update_cannot_be_retried(self):
        training = implementation("training")
        with tempfile.TemporaryDirectory() as folder:
            bad = SyntheticImages()
            bad.identity = dict(bad.identity, diagnostic_subset=False)
            trainer = training.Trainer(
                tiny_model(),
                short_config(),
                Path(folder) / "invalid",
                {"fixture": "caps"},
                "cpu",
            )
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                trainer.run(bad, stop_at=1)
            images = SyntheticImages()
            images.images[:] = float("nan")
            failed = training.Trainer(
                tiny_model(),
                short_config(),
                Path(folder) / "failed",
                {"fixture": "finite"},
                "cpu",
            )
            with self.assertRaises(FloatingPointError):
                failed.run(images, stop_at=1)
            state = json.loads((Path(folder) / "failed" / "run_state.json").read_text())
            self.assertEqual(
                (state["attempted_updates"], state["completed_updates"]), (1, 0)
            )
            with self.assertRaisesRegex(ValueError, "unresolved|attempted"):
                failed.run(SyntheticImages(), stop_at=1)

    def test_cli_rejects_stop_boundaries_before_gpu_or_data_access(self):
        import subprocess
        import sys

        for update in (0, -1, 21):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "diagnostic-no-execution"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.encoders.train",
                        "--config",
                        "scripts/encoders/configs/smoke.json",
                        "--hdf5",
                        "/not-read.h5",
                        "--manifest",
                        "/not-read.json",
                        "--output",
                        str(output),
                        "--code-revision",
                        "test",
                        "--stop-at",
                        str(update),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--stop-at", result.stderr)
                self.assertFalse(output.exists())

    def test_checkpoint_retention_is_bounded_and_fresh_outputs_do_not_mix_files(self):
        training = implementation("training")
        checkpoints = implementation("checkpoints")
        value = run_config_dict()
        value["optimizer"].update(max_updates=6, warmup_updates=2)
        value["runtime"].update(workers=0, checkpoint_every=2, keep_checkpoints=2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = training.Trainer(
                tiny_model(),
                RunConfig.from_dict(value),
                root / "bounded",
                {"fixture": "retention"},
                "cpu",
            )
            trainer.run(SyntheticImages(), stop_at=6)
            saved = list((root / "bounded").glob("diagnostic-step-*.pt"))
            self.assertEqual(
                sorted(
                    checkpoints.load_checkpoint(p)["completed_updates"] for p in saved
                ),
                [4, 6],
            )
            foreign = root / "foreign"
            foreign.mkdir()
            (foreign / "user.pt").write_bytes(b"untouched")
            with self.assertRaises(FileExistsError):
                training.Trainer(
                    tiny_model(), short_config(), foreign, {"fixture": "other"}, "cpu"
                )
            self.assertEqual((foreign / "user.pt").read_bytes(), b"untouched")

    def test_resume_rejects_modified_checkpoint_even_with_matching_metadata(self):
        training = implementation("training")
        checkpoints = implementation("checkpoints")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "integrity"
            trainer = training.Trainer(
                tiny_model(), short_config(), output, {"fixture": "integrity"}, "cpu"
            )
            path = trainer.run(SyntheticImages(), stop_at=2)
            state = checkpoints.load_checkpoint(path)
            next(
                value for value in state["model"].values() if value.is_floating_point()
            ).add_(1)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                training.Trainer(
                    tiny_model(),
                    short_config(),
                    output,
                    {"fixture": "integrity"},
                    "cpu",
                    resume=path,
                )

    def test_wall_clock_limit_interrupts_without_raising_the_budget(self):
        import time

        runtime = implementation("runtime")
        with self.assertRaises(TimeoutError):
            with runtime.model_time_limit(0.01):
                time.sleep(0.05)
        with runtime.model_time_limit(0.05):
            pass
