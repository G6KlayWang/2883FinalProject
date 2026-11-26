"""Identity hashing and atomic no-overwrite directory publication."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Iterator


def identity_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def new_directory(output: Path) -> Iterator[Path]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".lock")).open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=output.name + ".", suffix=".incomplete", dir=output.parent
            )
        )
        try:
            yield temporary
            os.rename(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
