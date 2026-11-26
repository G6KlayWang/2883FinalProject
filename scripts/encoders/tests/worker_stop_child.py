"""Real-worker CPU fixture for graceful trainer-only signalling; synthetic images."""

import argparse
from dataclasses import replace
from pathlib import Path
import time
import torch
from scripts.data.file_io import write_json_new
from scripts.encoders.checkpoints import replace_json
from scripts.encoders.interruptions import StopRequest, stop_signals
from scripts.encoders.tests.test_training_resume import (
    SyntheticImages,
    short_config,
    tiny_model,
)
from scripts.encoders.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = short_config()
    config = replace(config, runtime=replace(config.runtime, workers=2))
    trainer = Trainer(
        tiny_model(),
        config,
        args.output,
        {"fixture": "real-worker-stop"},
        "cpu",
        resume=args.resume,
    )
    stop = StopRequest()

    def observer(row):
        if row["update"] == 1:
            write_json_new(args.marker, {"completed_update": 1})
            time.sleep(
                1
            )  # Parent signals while both real DataLoader workers are alive.

    with stop_signals(stop):
        checkpoint = trainer.run(
            SyntheticImages(), stop_at=4, observer=observer, should_stop=stop
        )
    replace_json(
        args.output / "training_report.json",
        {
            "status": "stopped" if stop() else "candidate",
            "completed_updates": trainer.completed_updates,
            "attempted_updates": trainer.attempted_updates,
            "checkpoint": str(checkpoint),
        },
    )
    raise SystemExit(130 if stop() else 0)


if __name__ == "__main__":
    main()
