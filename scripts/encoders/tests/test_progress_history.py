"""Display/history must expose observations without changing optimization."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from scripts.encoders.tests.test_losses import implementation
from scripts.encoders.tests.test_training_resume import (
    SyntheticImages,
    tiny_model,
    short_config,
)


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_progress_shows_observed_metrics_and_plain_non_tty_output(self):
        progress = implementation("progress")
        row = {
            "update": 10,
            "total": 2.0,
            "invariance": 0.1,
            "regularizer": 0.19,
            "lr": 0.0005,
            "source_frames_per_second": 128.0,
            "model_seconds": 40.0,
        }
        for stream in (io.StringIO(), Terminal()):
            display = progress.ProgressDisplay(
                "A rho10 seed0", 20, stream=stream, refresh_seconds=0
            )
            display.update(row)
            display.close()
            output = stream.getvalue()
            for text in (
                "A rho10 seed0",
                "10/20",
                "50.0%",
                "loss=2",
                "inv=0.1",
                "reg=0.19",
                "lr=0.0005",
                "128.0frames/s",
                "ETA",
            ):
                self.assertIn(text, output)
            if not stream.isatty():
                self.assertNotIn("\x1b", output)
                self.assertNotIn("\r", output)

    def test_display_and_safe_stop_resume_preserve_exact_training_state(self):
        training, progress = implementation("training"), implementation("progress")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            direct = training.Trainer(
                tiny_model(),
                short_config(),
                root / "direct",
                {"fixture": "parity"},
                "cpu",
            )
            direct.run(SyntheticImages(), stop_at=4)
            display = progress.ProgressDisplay(
                "fixture", 4, stream=io.StringIO(), refresh_seconds=0
            )
            stopped = training.Trainer(
                tiny_model(),
                short_config(),
                root / "resumed",
                {"fixture": "parity"},
                "cpu",
            )
            checkpoint = stopped.run(
                SyntheticImages(),
                stop_at=4,
                observer=display.update,
                should_stop=lambda: stopped.completed_updates >= 1,
            )
            self.assertEqual(stopped.completed_updates, 1)
            resumed = training.Trainer(
                tiny_model(),
                short_config(),
                root / "resumed",
                {"fixture": "parity"},
                "cpu",
                resume=checkpoint,
            )
            resumed.run(SyntheticImages(), stop_at=4, observer=display.update)
            for name, value in direct.model.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.model.state_dict()[name], rtol=0, atol=0
                )
            for name, value in direct.streams.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.streams.state_dict()[name], rtol=0, atol=0
                )
            rows = [
                json.loads(line)
                for line in (root / "resumed/metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["update"] for r in rows], [1, 2, 3, 4])
            direct_rows = [
                json.loads(line)
                for line in (root / "direct/metrics.jsonl").read_text().splitlines()
            ]
            for observed, baseline in zip(rows, direct_rows):
                for key in (
                    "total",
                    "invariance",
                    "regularizer",
                    "gradient_norm",
                    "lr",
                ):
                    self.assertEqual(observed[key], baseline[key])
            self.assertTrue(all(r["source_frames_per_second"] > 0 for r in rows))
            self.assertTrue(all(r["gpu_memory_allocated_bytes"] is None for r in rows))
            self.assertIn("regularizer_unweighted_gradient_norm", rows[0])

    def test_csv_exports_missing_gradients_as_empty_without_inventing_values(self):
        history = implementation("history")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "metrics.jsonl"
            source.write_text(
                '{"update":1,"total":2,"gradient_norm":3}\n{"update":2,"total":1}\n'
            )
            history.export_csv(source, root / "history.csv")
            import csv

            with (root / "history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["gradient_norm"], "3")
            self.assertEqual(rows[1]["gradient_norm"], "")
            with self.assertRaises(FileExistsError):
                history.export_csv(source, root / "history.csv")
            source.write_text('{"update":1}\n{"update":1}\n')
            with self.assertRaisesRegex(ValueError, "update|history"):
                history.export_csv(source, root / "duplicate.csv")

    def test_dead_display_is_disabled_once_and_training_history_remains_complete(self):
        progress, training = implementation("progress"), implementation("training")

        class DeadStream(io.StringIO):
            def write(self, value):
                raise BrokenPipeError("terminal disconnected")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = training.Trainer(
                tiny_model(),
                short_config(),
                root / "run",
                {"fixture": "dead-display"},
                "cpu",
            )
            display = progress.ProgressDisplay(
                "fixture", 4, stream=DeadStream(), refresh_seconds=0
            )
            with self.assertLogs(
                "scripts.encoders.progress", level="WARNING"
            ) as warnings:
                trainer.run(SyntheticImages(), stop_at=4, observer=display.update)
            self.assertEqual(len(warnings.records), 1)
            self.assertEqual(trainer.completed_updates, 4)
            self.assertEqual(
                [
                    json.loads(line)["update"]
                    for line in (root / "run/metrics.jsonl").read_text().splitlines()
                ],
                [1, 2, 3, 4],
            )
            display.close()

    def test_absent_optional_metric_is_omitted_from_display(self):
        progress = implementation("progress")
        stream = io.StringIO()
        display = progress.ProgressDisplay(
            "fixture", 2, stream=stream, refresh_seconds=0
        )
        display.update({"update": 1, "total": None, "lr": 0.001})
        self.assertIn("lr=0.001", stream.getvalue())
        self.assertNotIn("loss=", stream.getvalue())

    def test_signal_after_logging_saves_current_update_instead_of_previous_checkpoint(
        self,
    ):
        import os, signal
        from scripts.encoders.interruptions import StopRequest, stop_signals
        from scripts.encoders.checkpoints import load_checkpoint

        training = implementation("training")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = training.Trainer(
                tiny_model(),
                short_config(),
                root / "run",
                {"fixture": "late-stop"},
                "cpu",
            )
            stop = StopRequest()

            def observer(row):
                if row["update"] == 3:
                    os.kill(os.getpid(), signal.SIGTERM)

            with stop_signals(stop):
                checkpoint = trainer.run(
                    SyntheticImages(), stop_at=4, observer=observer, should_stop=stop
                )
            self.assertEqual(trainer.completed_updates, 3)
            self.assertEqual(load_checkpoint(checkpoint)["completed_updates"], 3)
            resumed = training.Trainer(
                tiny_model(),
                short_config(),
                root / "run",
                {"fixture": "late-stop"},
                "cpu",
                resume=checkpoint,
            )
            resumed.run(SyntheticImages(), stop_at=4)
            self.assertEqual(resumed.completed_updates, 4)
