"""Explicit precision, thread budgets, wall-clock stops and resource guards."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
from collections.abc import Iterator
import torch
from scripts.encoders.config import RunConfig, RuntimeConfig


def configure_precision(config: RunConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = config.tf32_matmul
    torch.backends.cudnn.allow_tf32 = config.tf32_cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest" if not config.tf32_matmul else "high")
    torch.use_deterministic_algorithms(True)


def configure_process(config: RunConfig) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLOSC_NTHREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(config.runtime.intraop_threads)
    torch.set_num_interop_threads(config.runtime.interop_threads)
    configure_precision(config)


def worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"


def require_single_gpu() -> str:
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("Launch with exactly one CUDA_VISIBLE_DEVICES GPU UUID")
    if os.environ.get("WORLD_SIZE", "1") != "1" or torch.distributed.is_initialized():
        raise ValueError("Independent single-GPU training only; DDP is not implemented")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one available visible CUDA GPU")
    return uuid


def check_resources(output: Path, settings: RuntimeConfig) -> None:
    if shutil.disk_usage(output).free < settings.disk_reserve_gib * 1024**3:
        raise OSError(
            "Disk reserve violated; stop rather than changing experiment settings"
        )
    memory = Path("/sys/fs/cgroup/memory.current")
    if (
        memory.exists()
        and int(memory.read_text()) > settings.memory_limit_gib * 1024**3
    ):
        raise MemoryError("Configured aggregate cgroup memory guard exceeded")


@contextmanager
def model_time_limit(seconds: float) -> Iterator[None]:
    if seconds <= 0:
        raise TimeoutError("Cumulative model runtime budget exhausted")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expired(signum, frame):
        raise TimeoutError("Model wall-clock budget reached; no automatic extension")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
