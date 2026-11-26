"""Cross-queue advisory GPU UUID leases, inherited by owned trainer processes."""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess


class GpuLeases:
    def __init__(self, root: Path, uuids: list[str]):
        if (
            not 1 <= len(uuids) <= 4
            or len(uuids) != len(set(uuids))
            or any(not re.fullmatch(r"GPU-[A-Za-z0-9-]+", u) for u in uuids)
        ):
            raise ValueError("Provide1..4 distinct GPU UUIDs")
        self.root, self.uuids = Path(root), uuids
        self.handles = {}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for uuid in sorted(self.uuids):
                handle = (self.root / (uuid + ".lock")).open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise RuntimeError(
                        f"GPU already leased by another supervisor/trainer: {uuid}"
                    ) from None
                self.handles[uuid] = handle
                handle.seek(0)
                handle.truncate()
                json.dump({"owner_pid": os.getpid(), "gpu_uuid": uuid}, handle)
                handle.flush()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def fd(self, uuid: str) -> int:
        return self.handles[uuid].fileno()

    def __exit__(self, *args):
        # Close, never explicitly unlock: an inherited trainer FD keeps its lease
        # if a supervisor dies while the trainer is still alive.
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def verify_idle_gpus(uuids: list[str]) -> dict:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    )
    available = set(inventory.split())
    if set(uuids) - available:
        raise ValueError("Requested GPU UUID absent from current inventory")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if any(line.split(",")[0].strip() in uuids for line in processes.splitlines()):
        raise RuntimeError(
            "A requested GPU has existing compute processes; preserve them and stop"
        )
    return {"inventory": inventory, "compute_processes": processes}
