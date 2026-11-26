#!/usr/bin/env python3
"""Download the PushT archive, report progress, and verify its published hash."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


REPO_ID = "quentinll/lewm-pusht"
FILENAME = "pusht_expert_train.h5.zst"
LOG = logging.getLogger("pusht")


def verify_file(path, expected_size, expected_sha256):
    """Reject incomplete or corrupt files, including same-size corruption."""
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected file size for {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    return actual


def progress(destination, total, stop):
    """Report actual HTTP partial-file bytes, not preallocated Xet file size."""
    previous = 0
    previous_time = time.monotonic()
    while not stop.wait(15):
        candidates = [destination / FILENAME]
        candidates += list((destination / ".cache/huggingface/download").glob("*.incomplete"))
        sizes = []
        for path in candidates:
            try:
                sizes.append(path.stat().st_size)
            except FileNotFoundError:
                pass  # The client may rename a completed partial between polls.
        current = max(sizes, default=0)
        now = time.monotonic()
        rate = max(0, current - previous) / (now - previous_time)
        eta = f"{(total - current) / rate / 60:.1f} min" if rate else "unknown"
        LOG.info("PROGRESS %.2f%% | %.3f / %.3f GiB | %.2f MiB/s | ETA %s",
                 100 * current / total, current / 2**30, total / 2**30,
                 rate / 2**20, eta)
        previous, previous_time = current, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/root/autodl-tmp/pusht"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--transport", choices=("http", "xet"), default="http")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # Xet reports received bytes itself; its preallocated file size is not progress.
    os.environ["HF_HUB_DISABLE_XET"] = "1" if args.transport == "http" else "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" if args.transport == "http" else "0"
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("TQDM_MININTERVAL", "15")
    os.environ.setdefault("HF_HOME", str(destination / ".hf-home"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, hf_hub_download
    import fcntl

    with (destination / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = HfApi().dataset_info(REPO_ID, revision=args.revision, files_metadata=True, timeout=60)
        entry = next(item for item in info.siblings if item.rfilename == FILENAME)
        if not entry.lfs or not entry.size:
            raise ValueError("Dataset metadata is missing its size or SHA-256")
        expected_sha256 = entry.lfs.sha256
        LOG.info("Dataset %s | revision %s | bytes %d | SHA-256 %s",
                 REPO_ID, info.sha, entry.size, expected_sha256)
        free = shutil.disk_usage(destination).free
        LOG.info("Destination %s | free %.2f GiB", destination, free / 2**30)
        target = destination / FILENAME
        if not target.exists() and free < entry.size + 1024**3:
            raise OSError("Insufficient free space for the archive plus 1 GiB reserve")

        stop = threading.Event()
        monitor = threading.Thread(target=progress, args=(destination, entry.size, stop), daemon=True)
        if args.transport == "http":
            monitor.start()
        else:
            LOG.info("Using Xet parallel transfer; progress comes from the Hugging Face client")
        try:
            for attempt in range(1, 6):
                try:
                    path = Path(hf_hub_download(
                        repo_id=REPO_ID, filename=FILENAME, repo_type="dataset",
                        revision=info.sha, local_dir=destination,
                    ))
                    break
                except Exception:
                    LOG.exception("Download attempt %d failed", attempt)
                    if attempt == 5:
                        raise
                    time.sleep(10 * attempt)
        finally:
            stop.set()
            if monitor.is_alive():
                monitor.join()

        LOG.info("Download finished; verifying all bytes against the published SHA-256")
        digest = verify_file(path, entry.size, expected_sha256)
        repo = Path(__file__).resolve().parents[2]
        code_revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "uncommitted"
        manifest = {
            "repo_id": REPO_ID, "dataset_revision": info.sha,
            "filename": FILENAME, "size_bytes": entry.size, "sha256": digest,
            "code_revision": code_revision, "transport": args.transport,
            "verified_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary = destination / "download_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(destination / "download_manifest.json")
        LOG.info("VERIFIED %s | SHA-256 %s", path, digest)


if __name__ == "__main__":
    main()
