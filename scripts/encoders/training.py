"""Finite, fixed-budget single-device updates with exact deterministic resume."""

import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.data.file_io import write_json_new, sha256_file
from scripts.encoders.augment import AugmentedFrames
from scripts.encoders.checkpoints import (
    capture_rng,
    load_checkpoint,
    replace_json,
    restore_rng,
    save_checkpoint,
    parameter_digest,
)
from scripts.encoders.config import OptimizerConfig, RunConfig
from scripts.encoders.frames import BatchOrder, KeyedBatches
from scripts.encoders.losses import encoder_loss
from scripts.encoders.model import forward_views
from scripts.encoders.rng import RandomStreams, keyed_seed
from scripts.encoders.runtime import (
    check_resources,
    configure_precision,
    model_time_limit,
    worker_init,
    runtime_record,
)


def learning_rate_at(update: int, settings: OptimizerConfig) -> float:
    """One-based used-update LR:1%=first, peak=warmup end, final=last update."""
    if not 1 <= update <= settings.max_updates:
        raise ValueError("LR update outside fixed budget")
    if update <= settings.warmup_updates:
        fraction = (
            (update - 1) / (settings.warmup_updates - 1)
            if settings.warmup_updates > 1
            else 1.0
        )
        return settings.peak_lr * (0.01 + 0.99 * fraction)
    fraction = (update - settings.warmup_updates) / (
        settings.max_updates - settings.warmup_updates
    )
    return settings.final_lr + 0.5 * (settings.peak_lr - settings.final_lr) * (
        1 + math.cos(math.pi * fraction)
    )


def make_optimizer(model: nn.Module, settings: OptimizerConfig) -> torch.optim.AdamW:
    decay_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    decayed, excluded = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decayed if id(parameter) in decay_ids else excluded).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": settings.weight_decay},
            {"params": excluded, "weight_decay": 0.0},
        ],
        lr=settings.peak_lr,
        betas=tuple(settings.betas),
        eps=settings.epsilon,
        foreach=False,
    )


def _gradient_norm(gradients) -> float:
    values = [g.detach() for g in gradients if g is not None]
    if any(not torch.isfinite(g).all() for g in values):
        raise FloatingPointError("Nonfinite gradient; run stopped")
    return math.sqrt(sum(g.double().square().sum().item() for g in values))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: RunConfig,
        output: Path,
        identity: dict,
        device: str,
        *,
        resume: Path | None = None,
    ):
        config.require_execution_approval()
        self.config, self.output, self.device = (
            config,
            Path(output),
            torch.device(device),
        )
        configure_precision(config)
        self.model = model.to(self.device)
        self.initial_parameter_digest = parameter_digest(self.model)
        self.optimizer = make_optimizer(model, config.optimizer)
        self.streams = RandomStreams(config.seed)
        self.identity = {"config_hash": config.digest(), "run_identity": identity}
        self.completed_updates = self.attempted_updates = 0
        self.model_seconds = 0.0
        self.data_identity = None
        self.last_checkpoint_name = self.last_checkpoint_sha256 = None
        if resume is None and self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(
                "Fresh training output must be empty; use explicit resume for an existing run"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        if resume is None:
            write_json_new(
                self.output / "run_record.json",
                {
                    **self.identity,
                    "config": config.to_dict(),
                    "checkpoint_profile": config.profile,
                    "torch": torch.__version__,
                    "device": device,
                    "runtime_controls": runtime_record(),
                },
            )
            self._write_ledger()
        else:
            self._restore(resume)

    def _write_ledger(self) -> None:
        replace_json(
            self.output / "run_state.json",
            {
                **self.identity,
                "completed_updates": self.completed_updates,
                "attempted_updates": self.attempted_updates,
                "model_seconds": self.model_seconds,
                "last_checkpoint_name": self.last_checkpoint_name,
                "last_checkpoint_sha256": self.last_checkpoint_sha256,
            },
        )

    def _restore(self, path: Path) -> None:
        ledger = json.loads((self.output / "run_state.json").read_text())
        if ledger.get("last_checkpoint_name") != Path(path).name or sha256_file(
            path
        ) != ledger.get("last_checkpoint_sha256"):
            raise ValueError("Stale checkpoint or checkpoint file hash mismatch")
        state = load_checkpoint(path)
        self.last_checkpoint_name = ledger["last_checkpoint_name"]
        self.last_checkpoint_sha256 = ledger["last_checkpoint_sha256"]
        if state["identity"] != self.identity or any(
            ledger.get(k) != v for k, v in self.identity.items()
        ):
            raise ValueError("Checkpoint/config/source/code identity mismatch")
        count = state["completed_updates"]
        if ledger["attempted_updates"] != count or ledger["completed_updates"] != count:
            raise ValueError(
                "Checkpoint is stale or an attempted update is unresolved; counts differ"
            )
        if (
            state["scheduler"]["completed_updates"] != count
            or state["sampler"]["next_update"] != count
        ):
            raise ValueError("Checkpoint scheduler/sampler count mismatch")
        self.initial_parameter_digest = state["initial_parameter_digest"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.streams.load_state_dict(state["streams"])
        restore_rng(state["global_rng"], self.device)
        self.completed_updates = self.attempted_updates = count
        self.model_seconds = max(state["model_seconds"], ledger["model_seconds"])
        self.data_identity = state["data_identity"]

    def _checkpoint(self) -> Path:
        prefix = "diagnostic-" if self.config.profile == "smoke" else ""
        path = self.output / f"{prefix}step-{self.completed_updates:06d}.pt"
        save_checkpoint(
            path,
            {
                "checkpoint_version": 1,
                "identity": self.identity,
                "config": self.config.to_dict(),
                "checkpoint_role": (
                    "diagnostic"
                    if self.config.profile == "smoke"
                    else (
                        "final"
                        if self.completed_updates == self.config.optimizer.max_updates
                        else "intermediate"
                    )
                ),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "initial_parameter_digest": self.initial_parameter_digest,
                "streams": self.streams.state_dict(),
                "global_rng": capture_rng(self.device),
                "completed_updates": self.completed_updates,
                "attempted_updates": self.attempted_updates,
                "model_seconds": self.model_seconds,
                "data_identity": self.data_identity,
                "scheduler": {
                    "completed_updates": self.completed_updates,
                    "last_used_lr": self.optimizer.param_groups[0]["lr"],
                },
                "sampler": {
                    "next_update": self.completed_updates,
                    "policy": "frame-uniform-drop-last-v1",
                },
            },
        )
        self.last_checkpoint_name, self.last_checkpoint_sha256 = path.name, sha256_file(
            path
        )
        self._write_ledger()
        owned = sorted(self.output.glob(prefix + "step-*.pt"))
        for old in owned[: -self.config.runtime.keep_checkpoints]:
            old.unlink()
        return path

    def _step(self, batch: dict) -> dict:
        settings = self.config.optimizer
        lr = learning_rate_at(self.completed_updates + 1, settings)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        transfer_started = time.monotonic()
        view1 = batch["view1"].to(self.device, non_blocking=self.device.type == "cuda")
        view2 = batch["view2"].to(self.device, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        h2d_seconds = time.monotonic() - transfer_started
        first, second = forward_views(self.model, view1, view2)
        terms = encoder_loss(
            first,
            second,
            self.config.arm,
            self.config.rho,
            self.streams,
            self.config.aprime_convention,
            projections=self.config.projections,
            native_lambda=self.config.native_lambda,
        )
        if any(not torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError(
                "Nonfinite loss; run stopped without an optimizer update"
            )
        metrics = {name: value.detach().item() for name, value in terms.items()}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.completed_updates % self.config.runtime.gradient_terms_every == 0:
            for name in ("invariance", "regularizer"):
                gradients = torch.autograd.grad(
                    terms[name], parameters, retain_graph=True, allow_unused=True
                )
                label = "regularizer_unweighted" if name == "regularizer" else name
                metrics[label + "_gradient_norm"] = _gradient_norm(gradients)
        terms["total"].backward()
        metrics["gradient_norm"] = _gradient_norm(p.grad for p in parameters)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Nonfinite updated parameter; stop and report")
        return {**metrics, "lr": lr, "h2d_seconds": h2d_seconds}

    def _loader(self, frames, stop_at: int) -> DataLoader:
        order = BatchOrder(
            len(frames), self.config.optimizer.batch_size, self.config.seed
        )
        kwargs = {}
        if self.config.runtime.workers:
            kwargs.update(
                prefetch_factor=self.config.runtime.prefetch,
                multiprocessing_context="spawn",
            )
        return DataLoader(
            AugmentedFrames(frames, self.config.augment, self.config.seed),
            batch_sampler=KeyedBatches(order, self.completed_updates, stop_at),
            num_workers=self.config.runtime.workers,
            worker_init_fn=worker_init,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
            generator=torch.Generator().manual_seed(
                keyed_seed(self.config.seed, "loader")
            ),
            **kwargs,
        )

    def run(self, frames, *, stop_at: int) -> Path:
        if self.attempted_updates != self.completed_updates:
            raise ValueError(
                "An attempted update is unresolved; no automatic retry is permitted"
            )
        if self.config.profile == "smoke":
            if not frames.identity.get("diagnostic_subset"):
                raise ValueError("Smoke requires an explicitly diagnostic frame subset")
            if (
                frames.identity.get("frame_policy") != "phase0"
                or len(frames) > self.config.runtime.max_frames
            ):
                raise ValueError("Smoke frame policy/cap violated")
        elif frames.identity.get("diagnostic_subset"):
            raise ValueError(
                "Diagnostic frame populations cannot train production candidates"
            )
        if not self.completed_updates < stop_at <= self.config.optimizer.max_updates:
            raise ValueError(
                "Stop update must advance within the immutable total budget"
            )
        if frames.identity.get("split") != "train":
            raise ValueError("Encoder updates require training images only")
        if self.data_identity is not None and self.data_identity != frames.identity:
            raise ValueError("Resume frame population/cache identity mismatch")
        self.data_identity = frames.identity
        loader = self._loader(frames, stop_at)
        self.model.train()
        initial_seconds, started = self.model_seconds, time.monotonic()
        waiting_since = started
        last_checkpoint = None
        try:
            with model_time_limit(self.config.runtime.max_seconds - initial_seconds):
                for batch in loader:
                    check_resources(self.output, self.config.runtime)
                    if self.attempted_updates >= self.config.optimizer.max_updates:
                        raise ValueError("Lifetime attempted update budget exhausted")
                    self.attempted_updates += 1
                    self._write_ledger()  # Reserve before any optimizer update, even if it later fails.
                    step_started = time.monotonic()
                    data_wait_seconds = step_started - waiting_since
                    metrics = self._step(batch)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    self.completed_updates += 1
                    self.model_seconds = initial_seconds + time.monotonic() - started
                    self._write_ledger()
                    metrics.update(
                        update=self.completed_updates,
                        model_seconds=self.model_seconds,
                        update_seconds=time.monotonic() - step_started,
                        data_wait_seconds=data_wait_seconds,
                    )
                    with (self.output / "metrics.jsonl").open("a") as log:
                        log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    if (
                        self.completed_updates % self.config.runtime.checkpoint_every
                        == 0
                        or self.completed_updates == stop_at
                    ):
                        last_checkpoint = self._checkpoint()
                    waiting_since = time.monotonic()
        finally:
            self.model_seconds = initial_seconds + time.monotonic() - started
            self._write_ledger()
        return last_checkpoint
