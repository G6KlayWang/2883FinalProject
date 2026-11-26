"""Rate-limited terminal output separated from machine-readable training history."""

import sys
import logging
import time
from typing import TextIO


class ProgressDisplay:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: TextIO | None = None,
        refresh_seconds: float | None = None,
    ):
        if total < 1:
            raise ValueError("Progress total must be positive")
        self.label, self.total = label, total
        self.stream = stream if stream is not None else sys.stderr
        self.disabled = False
        try:
            self.tty = self.stream.isatty()
        except (OSError, ValueError) as error:
            self.tty = False
            self._disable(error)
        self.refresh = (
            (1.0 if self.tty else 30.0) if refresh_seconds is None else refresh_seconds
        )
        self.last_refresh = float("-inf")
        self.active = False

    def _disable(self, error: Exception) -> None:
        if not self.disabled:
            self.disabled = True
            logging.getLogger(__name__).warning(
                "Display disabled; training/history continue: %s", error
            )

    def update(self, metrics: dict) -> None:
        if self.disabled:
            return
        try:
            self._update(metrics)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._disable(error)

    def _update(self, metrics: dict) -> None:
        now = time.monotonic()
        step = metrics["update"]
        if now - self.last_refresh < self.refresh and step != self.total:
            return
        self.last_refresh = now
        fraction = min(1.0, step / self.total)
        bar = "#" * int(20 * fraction) + "-" * (20 - int(20 * fraction))
        parts = [f"{self.label} [{bar}] {step}/{self.total} {100*fraction:.1f}%"]
        for key, label in (
            ("total", "loss"),
            ("invariance", "inv"),
            ("regularizer", "reg"),
            ("lr", "lr"),
        ):
            if metrics.get(key) is not None:
                parts.append(f"{label}={metrics[key]:.6g}")
        if metrics.get("source_frames_per_second") is not None:
            parts.append(f"{metrics['source_frames_per_second']:.1f}frames/s")
        if step and metrics.get("model_seconds") is not None:
            eta = max(0, self.total - step) * metrics["model_seconds"] / step
            parts.append(f"ETA estimate {eta/60:.1f}min")
        self.stream.write(
            ("\r\x1b[K" if self.tty else "")
            + " ".join(parts)
            + ("" if self.tty else "\n")
        )
        self.stream.flush()
        self.active = True

    def close(self) -> None:
        if self.disabled:
            return
        if self.tty and self.active:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except (OSError, ValueError) as error:
                self._disable(error)
            self.active = False
