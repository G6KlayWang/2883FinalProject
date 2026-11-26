"""Observed capacity gates for the approved production-run continuation stages."""

import math
from pathlib import Path
import statistics
from scripts.encoders.runtime import memory_usage


def assess_run(
    rows: list[dict],
    report: dict,
    *,
    start: int,
    target: int,
    total: int,
    max_seconds: float,
    phase: str
) -> dict:
    selected = [r for r in rows if start < r["update"] <= target]
    measured = [r for r in selected if r["update"] > start + 20]
    finite = all(
        all(
            math.isfinite(r[k])
            for k in (
                "total",
                "invariance",
                "regularizer",
                "update_seconds",
                "data_wait_seconds",
            )
        )
        for r in selected
    )
    counts = (
        report.get("completed_updates") == target
        and report.get("attempted_updates") == target
    )
    median = (
        statistics.median(
            r["update_seconds"] + r["data_wait_seconds"] for r in measured
        )
        if measured
        else None
    )
    projection = (
        (
            median * total
            if phase == "S1"
            else report["model_seconds"] + median * (total - target)
        )
        if median is not None
        else None
    )
    fraction = 0.5 if phase == "S1" else 0.8
    memory = report.get("peak_gpu_memory_bytes_this_attempt")
    card = report.get("gpu_total_memory_bytes")
    memory_pass = memory is not None and card is not None and memory < 0.85 * card
    passed = (
        counts
        and len(selected) == target - start
        and finite
        and memory_pass
        and projection is not None
        and projection <= fraction * max_seconds
    )
    return {
        "passed": bool(passed),
        "phase": phase,
        "start": start,
        "target": target,
        "observed_rows": len(selected),
        "measured_rows": len(measured),
        "median_cycle_seconds": median,
        "projected_model_seconds": projection,
        "time_limit_seconds": fraction * max_seconds,
        "peak_gpu_memory_bytes": memory,
        "gpu_total_memory_bytes": card,
        "finite": finite,
        "counts_match": counts,
        "memory_pass": memory_pass,
    }


def overlap_seconds(reports: list[dict]) -> float:
    if not reports:
        return 0.0
    return max(
        0.0,
        min(r["training_finished_at"] for r in reports)
        - max(r["training_started_at"] for r in reports),
    )


def resource_snapshot(root: Path = Path("/sys/fs/cgroup")) -> dict:
    def counters(name):
        path = root / name
        return (
            {
                k: int(v)
                for k, v in (line.split() for line in path.read_text().splitlines())
            }
            if path.exists()
            else {}
        )

    return {
        "memory": memory_usage(root),
        "memory_events": counters("memory.events"),
        "cpu_stat": counters("cpu.stat"),
    }
