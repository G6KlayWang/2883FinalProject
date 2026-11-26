"""CLI binding approved recipe, immutable inputs, preparation and queue ownership."""

import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.checkpoints import replace_json
from scripts.encoders.config import RunConfig, load_config, run_matrix
from scripts.encoders.devices import GpuLeases, verify_idle_gpus
from scripts.encoders.environment import environment_record
from scripts.encoders.interruptions import StopRequest, stop_signals
from scripts.encoders.preparation import prepare_caches
from scripts.encoders.queue import QueueRunner, render_status
from scripts.encoders.queue_state import QueueStore
from scripts.encoders.recipe import load_approved_recipe
from scripts.encoders.runs import resolve_run


CONFIGS = Path(__file__).parent / "configs"


def verify_code_revision(revision: str) -> None:
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if current != revision:
        raise ValueError("Requested code revision differs from checked-out HEAD")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "scripts/encoders",
            "scripts/data",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    extra = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "scripts/encoders",
            "scripts/data",
        ],
        text=True,
    )
    if extra.strip():
        raise ValueError(
            "Untracked source files violate the immutable reviewed checkout"
        )


def declarations(
    config: RunConfig,
    config_path: Path,
    revision: str,
    repeat_directory: Path | None,
    selection_path: Path | None,
) -> list[dict]:
    if bool(repeat_directory) != bool(selection_path):
        raise ValueError(
            "Repeats require both emitted config directory and selection receipt"
        )
    specs = []
    receipt = json.loads(selection_path.read_text()) if selection_path else None
    rows = run_matrix(config.grids, config.native_lambda)
    for row in rows[10:] if receipt else rows[:10]:
        path = (repeat_directory / f"{row['run_id']}.json") if receipt else config_path
        if receipt:
            candidate = load_config(path)
            expected = replace(config, arm=candidate.arm, rho=candidate.rho, seed=1)
            if candidate.to_dict() != expected.to_dict():
                raise ValueError("Repeat changed approved shared recipe controls")
            resolved, declaration = resolve_run(
                candidate, candidate=None, selection_report=receipt
            )
        else:
            resolved, declaration = resolve_run(
                config, candidate=row["run_id"], selection_report=None
            )
        specs.append(
            {
                **row,
                **declaration,
                "arm": resolved.arm,
                "rho": resolved.rho,
                "seed": resolved.seed,
                "status": "approved",
                "config_path": str(path.resolve()),
                "config_hash": resolved.digest(),
                "code_revision": revision,
                "max_updates": resolved.optimizer.max_updates,
                "max_seconds": resolved.runtime.max_seconds,
                "memory_limit_gib": resolved.runtime.memory_limit_gib,
            }
        )
    return specs


def storage_budget(cache_root: Path, output: Path, config: RunConfig) -> dict:
    missing = sum(
        count * (96 * 96 * 3 + 16)
        for split, count in (("train", 433896), ("validation", 21715))
        if not (cache_root / f"{split}-phase0").exists()
    )
    reserve = int(config.runtime.disk_reserve_gib * 1024**3)
    outputs = (
        13 * 1024**3
    )  # Conservative engineering allowance, not measured artifact sizes.
    cache_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache_root).free < missing + reserve:
        raise OSError("Insufficient cache disk capacity plus reserve")
    shared = cache_root.stat().st_dev == output.parent.stat().st_dev
    if shutil.disk_usage(output.parent).free < outputs + reserve + (
        missing if shared else 0
    ):
        raise OSError("Insufficient output/cache budget plus reserve")
    return {
        "estimated_missing_cache_bytes": missing,
        "reserved_output_allowance_bytes": outputs,
        "disk_reserve_bytes": reserve,
        "shared_filesystem": shared,
    }


def build_command(
    args: argparse.Namespace,
    run: dict,
    gpu: str,
    target: int,
    resume: Path | None,
    ready: Path,
    gate: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.encoders.train",
        "--config",
        run["config_path"],
        "--hdf5",
        str(args.hdf5),
        "--manifest",
        str(args.manifest),
        "--cache",
        str(args.cache_root / "train-phase0"),
        "--output",
        str(args.output / "runs" / run["run_id"]),
        "--code-revision",
        args.code_revision,
        "--stop-at",
        str(target),
        "--ready-file",
        str(ready),
        "--progress",
        "off",
    ]
    command.extend(
        ["--selection-report", str(args.selection_report)]
        if run["stage"] == "repeat"
        else ["--candidate", run["run_id"]]
    )
    if resume:
        command.extend(["--resume", str(resume)])
    if gate:
        command.extend(["--start-gate", str(gate)])
    return command


def _status(root: Path) -> None:
    state = json.loads((root / "queue_state.json").read_text())
    store = QueueStore(root, state["identity"], state["runs"], resume=True)
    render_status(store, sys.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIGS / "production-approved.json"
    )
    parser.add_argument(
        "--approval", type=Path, default=CONFIGS / "recipe-approval.json"
    )
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision")
    parser.add_argument("--gpu-uuids", nargs="+")
    parser.add_argument(
        "--lock-directory", type=Path, default=Path("/tmp/pusht-encoder-gpu-locks")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Preparation-only: build validation after train",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry", nargs="*", default=[])
    parser.add_argument("--repeat-config-directory", type=Path)
    parser.add_argument("--selection-report", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.status:
        _status(args.output)
        return
    signal.signal(
        signal.SIGHUP, signal.SIG_IGN
    )  # Terminal disconnect must not kill the supervisor.
    if args.resume and not (args.output / "queue_state.json").is_file():
        parser.error("--resume requires an existing queue state")
    if not args.resume and not args.prepare_only and args.output.exists():
        raise FileExistsError("Fresh queue output already exists; use --resume")
    if any(
        v is None
        for v in (args.hdf5, args.manifest, args.cache_root, args.code_revision)
    ):
        parser.error("provide --hdf5 --manifest --cache-root --code-revision")
    if args.include_validation and not args.prepare_only:
        parser.error(
            "--include-validation is preparation-only; normal queue builds it after capacity"
        )
    if args.prepare_only and args.resume:
        parser.error("--prepare-only cannot resume a model queue")
    if args.retry and not args.resume:
        parser.error("--retry requires explicit --resume")
    for key in (
        "hdf5",
        "manifest",
        "cache_root",
        "config",
        "approval",
        "repeat_config_directory",
        "selection_report",
    ):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    verify_code_revision(args.code_revision)
    config = load_approved_recipe(args.config, args.approval)
    specs = declarations(
        config,
        args.config,
        args.code_revision,
        args.repeat_config_directory,
        args.selection_report,
    )
    budget = storage_budget(args.cache_root, args.output, config)
    if args.prepare_only:
        report = prepare_caches(
            args.hdf5,
            args.manifest,
            args.cache_root,
            config,
            ("train", "validation") if args.include_validation else ("train",),
        )
        replace_json(
            args.cache_root / "preparation_report.json", {"budget": budget, **report}
        )
        return
    if not args.gpu_uuids:
        parser.error("provide leased --gpu-uuids for execution")
    # Supervisor lock lives outside the queue output so fresh-output collisions
    # remain detectable before the first run is created.
    lock_path = args.output.parent / ("." + args.output.name + ".supervisor.lock")
    with lock_path.open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with GpuLeases(args.lock_directory, args.gpu_uuids) as leases:
            inventory = verify_idle_gpus(args.gpu_uuids)
            if not args.resume and args.output.exists():
                raise FileExistsError("Fresh queue output already exists; use --resume")
            prep = prepare_caches(
                args.hdf5, args.manifest, args.cache_root, config, ("train",)
            )
            stat = args.hdf5.stat()
            identity = {
                "code_revision": args.code_revision,
                "config_hash": config.digest(),
                "config_file_sha256": sha256_file(args.config),
                "approval_sha256": sha256_file(args.approval),
                "hdf5_path": str(args.hdf5),
                "hdf5_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
                "manifest_sha256": sha256_file(args.manifest),
                "cache_root": str(args.cache_root),
                "train_cache_manifest_sha256": sha256_file(
                    args.cache_root / "train-phase0/manifest.json"
                ),
                "environment": environment_record(),
                "selection_sha256": (
                    sha256_file(args.selection_report)
                    if args.selection_report
                    else None
                ),
            }
            store = QueueStore(args.output, identity, specs, resume=args.resume)
            store.event(
                "preflight", inventory=inventory, budget=budget, preparation=prep
            )
            if args.resume:
                # A supervisor crash is never treated as a successful pause.
                for run in store.state["runs"]:
                    if (
                        run["gpu_uuid"] is not None
                        and run["gpu_uuid"] not in args.gpu_uuids
                    ):
                        raise ValueError("Resume leases omit an original GPU UUID")
                    if run["state"] == "running":
                        run["state"] = "failed"
                store.save()
                store.resume_failed(set(args.retry))
            elif args.selection_report:
                store.state["phase"] = "S3"
                store.save()

            def validation():
                report = prepare_caches(
                    args.hdf5, args.manifest, args.cache_root, config, ("validation",)
                )
                store.event("validation-cache-prepared", report=report)

            runner = QueueRunner(
                store,
                leases,
                lambda *values: build_command(args, *values),
                prepare_validation=validation,
            )
            stop = StopRequest()
            with stop_signals(stop):
                code = runner.run(stop)
            raise SystemExit(code)


if __name__ == "__main__":
    main()
